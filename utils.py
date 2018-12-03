import numpy as np
# import theano
# from theano import config
import tensorflow as tf
from collections import OrderedDict, defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from tensorflow.python import pywrap_tensorflow
from pdb import set_trace as bp
import data_utils as dp
import sys, os
import codecs
from tensorflow.python.ops import clip_ops
from rougescore import rouge_n, rouge_1, rouge_2, rouge_l
import cPickle
import pdb
from embedding_metrics import greedy_match, extrema_score, average_score
from gensim.models import Word2Vec


def logit(x, delta = 1e-10):
    return tf.log((x+delta)/(1-x+delta))


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def sent2idx(text, wordtoix, opt, is_cnn = True):

    sent = [wordtoix[x] for x in text.split()]

    return prepare_data_for_cnn([sent for i in range(opt.batch_size)], opt)



def prepare_data_for_cnn(seqs_x, opt):
    maxlen=opt.maxlen
    filter_h=opt.filter_shape
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                print "exceeds length"
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1  :
            return None, None

    pad = filter_h -1
    x = []
    for rev in seqs_x:
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x


def prepare_data_for_rnn(seqs_x, opt, is_add_GO = True, GO_idx = dp.GO_ID):

    maxlen=opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                print "exceeds length"
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1  :
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros(( n_samples, opt.sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        if is_add_GO:
            x[idx, 0] = GO_idx # GO symbol
            x[idx, 1:lengths_x[idx]+1] = s_x
        else:
            x[idx, :lengths_x[idx]] = s_x
    return x



def restore_from_save(t_vars, sess, opt):
    save_keys = tensors_key_in_file(opt.save_path)
    #print(save_keys.keys())
    ss = set([var.name for var in t_vars])&set([s+":0" for s in save_keys.keys()])
    cc = {var.name:var for var in t_vars}
    ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])  # only restore variables with correct shape

    if opt.reuse_discrimination:
        ss2 = set([var.name[2:] for var in t_vars])&set([s+":0" for s in save_keys.keys()])
        cc2 = {var.name[2:][:-2]:var for var in t_vars if var.name[2:] in ss2 if var.get_shape() == save_keys[var.name[2:][:-2]]}
        for s_iter in ss_right_shape:
            cc2[s_iter[:-2]] = cc[s_iter]

        loader = tf.train.Saver(var_list=cc2)
        loader.restore(sess, opt.save_path)
        print("Loaded variables for discriminator:"+str(cc2.keys()))

    else:
        # for var in t_vars:
        #     if var.name[:-2] in ss:
        #         tf.assign(t_vars, save_keys[var.name[:-2]])
        loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss_right_shape])
        loader.restore(sess, opt.save_path)
        print("Loading variables from '%s'." % opt.save_path)
        print("Loaded variables:"+str(ss_right_shape))

    return loader


def read_pair_data_full(src_f, tgt_f, dic_f, train_prop = 0.9, max_num=None, rev_src=False, rev_tgt = False, is_text = False, p_f = '../data/'):
    #train, val = [], []
    p_f = src_f[:-3] + str(max_num) + '.p'
    if os.path.exists(p_f):
        with open(p_f, 'rb') as pfile:
            train, val, test, wordtoix, ixtoword = cPickle.load(pfile)
        return train, val, test, wordtoix, ixtoword


    wordtoix, ixtoword = {}, {}
    print "Start reading dic file . . ."
    if os.path.exists(dic_f):
        print("loading Dictionary")
        counter=0
        with codecs.open(dic_f,"r",'utf-8') as f:
            s=f.readline()
            while s:
                s=s.rstrip('\n').rstrip("\r")
                #print("s==",s)
                wordtoix[s]=counter
                ixtoword[counter]=s
                counter+=1
                s=f.readline()
    def shift_id(x):
        return x
    src, tgt = [], []
    print "Start reading src file . . ."
    with codecs.open(src_f,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        count, max_l = 0, 0
        #max_length_fact=0
        while line and (not max_num or count<max_num):
            count+=1
            if is_text:
                tokens=[wordtoix[x] if x in wordtoix else dp.UNK_ID for x in line.split()]
            else:
                tokens=[shift_id(int(x)) for x in line.split()]
            max_l = max(max_l, len(tokens))
            if not rev_src: # reverse source
                src.append(tokens)
            else :
                src.append(tokens[::-1])
            #pdb.set_trace()
            line = f.readline().rstrip("\n").rstrip("\r")
            if np.mod(count,100000)==0:
                print count
    print "Source cnt: " + str(count) + " maxLen: " + str(max_l)

    print "Start reading tgt file . . ."
    with codecs.open(tgt_f,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        count = 0
        #max_length_fact=0
        while line and (not max_num or count<max_num):
            count+=1
            if is_text:
                tokens=[wordtoix[x] if x in wordtoix else dp.UNK_ID for x in line.split()]
            else:
                tokens=[shift_id(int(x)) for x in line.split()]
            if not rev_tgt: # reverse source
                tgt.append(tokens)
            else :
                tgt.append(tokens[::-1])
            line = f.readline().rstrip("\n").rstrip("\r")
            if np.mod(count,100000)==0:
                print count
    print "Target cnt: " + str(count) + " maxLen: " + str(max_l)

    assert(len(src)==len(tgt))
    all_pairs = np.array(zip(*[tgt, src]))
    if not train_prop:
        train , val, test = all_pairs, [], []
    else:
        idx = np.random.choice(len(all_pairs), int(np.floor(0.9*len(all_pairs))))
        rem_idx = np.array(list(set(range(len(all_pairs)))-set(idx)))
        v_idx = np.random.choice(rem_idx, int(np.floor(0.5*len(rem_idx))))
        t_idx = np.array(list(set(rem_idx)-set(v_idx)))
        #pdb.set_trace()
        train, val, test = all_pairs[idx], all_pairs[v_idx], all_pairs[t_idx]

    with open(p_f, 'wb') as pfile:
        cPickle.dump([train, val, test, wordtoix, ixtoword], pfile)


        #print(counter)
    #pdb.set_trace()
    return train, val, test, wordtoix, ixtoword


def tensors_key_in_file(file_name):
    """Return tensors key in a checkpoint file.
    Args:
    file_name: Name of the checkpoint file.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        return reader.get_variable_to_shape_map()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        return None


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


# def normalizing_L1(x, axis):
#     norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
#     normalized = x / (norm)
#     return normalized

def normalizing(x, axis):
    # norm_2(x) == 1
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
    normalized = x / (norm)
    return normalized

def normalizing_sum(x, axis):
    # sum(x) == 1
    sum_prob = tf.reduce_sum(x, axis=axis, keep_dims=True)
    normalized = x / sum_prob
    return normalized

def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def uniform_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype(config.floatX)

def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)

def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)

"""auxiliary function for KDE"""
def log_mean_exp(A,b,sigma):
    a=-0.5*((A-theano.tensor.tile(b,[A.shape[0],1]))**2).sum(1)/(sigma**2)
    max_=a.max()
    return max_+theano.tensor.log(theano.tensor.exp(a-theano.tensor.tile(max_,a.shape[0])).mean())

'''calculate KDE'''
def cal_nkde(X,mu,sigma):
    s1,updates=theano.scan(lambda i,s: s+log_mean_exp(mu,X[i,:],sigma), sequences=[theano.tensor.arange(X.shape[0])],outputs_info=[np.asarray(0.,dtype="float32")])
    E=s1[-1]
    Z=mu.shape[0]*theano.tensor.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/mu.shape[0]


""" BLEU score"""
# def cal_BLEU(generated, reference):
#     #the maximum is bigram, so assign the weight into 2 half.
#     BLEUscore = 0.0
#     for g in generated:
#         BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g)
#     BLEUscore = BLEUscore/len(generated)
#     return BLEUscore

def cal_ROUGE(generated, reference, is_corpus = False):
    # ref and sample are both dict
    # scorers = [
    #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #     (Meteor(),"METEOR"),
    #     (Rouge(), "ROUGE_L"),
    #     (Cider(), "CIDEr")
    # ]
    # output rouge 1-4 and rouge L and rouge L from pycocoevaluate


    ROUGEscore = [0.0]*6
    for idx, g in enumerate(generated):
        score = [0.0]*6
        if is_corpus:
            for order in range(4):
                score[order] = rouge_n(g.split(), [x.split() for x in reference[0]], order+1, 0.5)
            score[4] = rouge_l(g.split(), [x.split() for x in reference[0]], 0.5)
            score[5], _ = Rouge().compute_score(reference, {0: [g]})


        else:
            for order in range(4):
                score[order] = rouge_n(g.split(), [reference[0][idx].split()], order+1, 0.5)
            score[4] = rouge_l(g.split(), [reference[0][idx].split()], 0.5)
            score[5], _ = Rouge().compute_score({0: [reference[0][idx]]}, {0: [g]})
            #pdb.set_trace()
        #print g, score
        ROUGEscore = [ r+score[idx]  for idx,r in enumerate(ROUGEscore)]
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    ROUGEscore = [r/len(generated) for r in ROUGEscore]
    return ROUGEscore




# def cal_BLEU(generated, reference, is_corpus = False):
#     #print 'in BLEU score calculation'
#     #the maximum is bigram, so assign the weight into 2 half.
#     BLEUscore = [0.0,0.0,0.0]
#     for idx, g in enumerate(generated):
#         if is_corpus:
#             score, scores = Bleu(4).compute_score(reference, {0: [g]})
#         else:
#             score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
#         #print g, score
#         for i, s in zip([0,1,2],score[1:]):
#             BLEUscore[i]+=s
#         #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
#     BLEUscore[0] = BLEUscore[0]/len(generated)
#     BLEUscore[1] = BLEUscore[1]/len(generated)
#     BLEUscore[2] = BLEUscore[2]/len(generated)
#     return BLEUscore

def cal_BLEU_4(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2,3],score):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore

def cal_BLEU_4_nltk(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    reference = [[s] for s in reference]
    #bp()
    chencherry = SmoothingFunction()
    # Note: please keep smoothing turned on, because there is a bug in NLTK without smoothing (see below).
    if is_corpus:
        return nltk.translate.bleu_score.corpus_bleu(reference, generated, smoothing_function=chencherry.method2) # smoothing options: 0-7
    else:
        return np.mean([nltk.translate.bleu_score.sentence_bleu(r, g, smoothing_function=chencherry.method2) for r,g in zip(reference, generated)]) # smoothing options: 0-7

def cal_BLEU_4_nltk_all(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    reference = [[s] for s in reference]
    #bp()
    chencherry = SmoothingFunction()
    # Note: please keep smoothing turned on, because there is a bug in NLTK without smoothing (see below).
    if is_corpus:
        return nltk.translate.bleu_score.corpus_bleu(reference, generated, smoothing_function=chencherry.method2) # smoothing options: 0-7
    else:
        return [nltk.translate.bleu_score.sentence_bleu(r, g, smoothing_function=chencherry.method2) for r,g in zip(reference, generated)] # smoothing options: 0-7

def cal_entropy(generated):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip('2').split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score 
       
def prepare_for_bleu(sentence):
    sent=[x for x in sentence if x!=0]
    while len(sent)<4:
        sent.append('0')
    sent = ' '.join([str(x) for x in sent])
    return sent   

# def prepare_for_bleu(sentence):
#     sent=[x for x in sentence if x!=0]
#     while len(sent)<4:
#         sent.append(0)
#     #sent = ' '.join([ixtoword[x] for x in sent])
#     sent = ' '.join([str(x) for x in sent])
#     return sent

def calc_diversity_and_length(file_name):
    with gfile.GFile(file_name,"r") as f:
        all_words=[]
        unique_words=Set([])
        total_num=0
        for i, line in enumerate(f):
            parts=line.split("\t")
            sentence=parts[0]
            words=sentence.split(" ")
            total_num+=1
            for word in words:
                all_words.append(word)
                unique_words.add(word)
        print(unique_words)
        print("all_words",len(all_words))
        print("unique", len(unique_words))
        print("diversity", len(unique_words)/len(all_words))
        print("average len:", len(all_words)/total_num)
        
def cal_relevance(generated, reference, embedding): # embedding V* E
    generated = [[g] for g in generated]
    reference = [[s] for s in reference]


    #bp()
    relevance_score = [0.0,0.0,0.0]
    relevance_score[0] = greedy_match(reference, generated, embedding)
    relevance_score[1] = average_score(reference, generated, embedding)
    relevance_score[2] = extrema_score(reference, generated, embedding)
    return relevance_score    
        

def _clip_gradients_seperate_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients = [clip_ops.clip_by_norm(grad, clip_gradients) for grad in gradients]
  return list(zip(clipped_gradients, variables))

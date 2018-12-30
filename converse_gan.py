# -*- coding: utf-8 -*-
"""
Yizhe Zhang

Converse GAN 

Microsoft
"""
## 152.3.214.203/6006

import os, sys

# os.environ['LD_LIBRARY_PATH'] = '/home/yizhe/cudnn/cuda/lib64'
# os.environ['CPATH'] = '/home/yizhe/cudnn/cuda/include'
# os.environ['LIBRARY_PATH'] = '/home/yizhe/cudnn/cuda'
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
import cPickle
import numpy as np
import os
import codecs
#import scipy.io as sio
from math import floor
from operator import add
from pdb import set_trace as bp

from model import *
from utils import read_pair_data_full, prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, normalizing_sum, restore_from_save, tensors_key_in_file,\
    prepare_for_bleu, cal_BLEU_4_nltk, cal_BLEU_4, cal_ROUGE, cal_entropy, sent2idx, _clip_gradients_seperate_norm, logit, cal_relevance
import gensim
import copy
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model.')
    parser.add_argument('--gpuid', '-g', type=int, default=0)  
    parser.add_argument('--testflag', '-t', action='store_true', default=False)
    parser.add_argument('--twoside', '-d', action='store_true', default=False)
    parser.add_argument('--mi', '-m', type=float, default=None)
    parser.add_argument('--sup', '-s', type=float, default=None)
    parser.add_argument('--continuing', '-c', action='store_true', default=False)

    parser.add_argument('--pg', '-p', action='store_true', default=False)
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4)

    args = parser.parse_args()
    print(args)


    profile = False
    TEST_FLAG = args.testflag

    logging.set_verbosity(logging.INFO)
    #tf.logging.verbosity(1)
    # Basic model parameters as external flags.
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    #flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

    GPUID = args.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)


class Options(object):
    def __init__(self):

        # 
        # One side or two side
        self.two_side = args.twoside #True
        self.lambda_backward = 0.5
        self.lambda_MI = args.mi
        # Supervise level
        self.lambda_sup_G = args.sup # 1: fully supervised  None: no supervised  trade-off between supervised signal and GAN


        # optimizer gan
        self.d_freq = 1
        self.g_freq = 1



        #
        self.rnn_share_emb = True #CNN_LSTM share embedding
        self.fix_emb = False
        self.reuse_cnn = False
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' 
        self.is_fed_h = True
        
        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 49 #49
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.embed_size = 300

        # Reinforcement learning




        # layer

        self.layer = 3# 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 32
        self.max_epochs = 1
        self.n_hid = 100  # self.filter_size * 3
        self.multiplier = 2
        self.L = 1000

        # VIM
        self.MI_pg = args.pg  ## policy gradient for MI learning


        # discriminator
        self.lr_d = 1e-4
        #self.H_dis = 300    # fully connected hidden units number
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1
        self.n_d_output = self.n_hid
        self.grad_penalty = None  # improved w-gan loss
        self.is_subtract_mean = False

        # generator
        self.additive_noise_lambda = 0.0   # additive_noise or concatenating noise
        self.n_z = self.n_hid if self.additive_noise_lambda else 10
        self.lr_g = args.learning_rate #5e-5 #1e-4
        self.feature_matching = 'pair_diff'#'pair_diff' # 'mean' # 'mmd' # None
        self.w_gan = False
        self.bp_truncation = None


        self.fake_size = self.batch_size
        self.sigma_range = [1]
        #self.n_d_output = 100

        

        self.g_fix = False
        self.g_rev = False    # backward model only

        # optimizer

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.9999 # cannot be one
        self.relu_w = False

        # decoder options
        self.decode_epsilon = 0

        # misc
        self.data_size = None #None #10000  # None : all data
        self.name = 'gan' + str(self.n_hid) + "_dim_" + self.model + "_" + self.feature_matching + ("_sup" if self.lambda_sup_G >= 1 else "_gan") \
                     + ("_rev_only" if self.g_rev else "") + ("_twoside" if self.two_side else "_oneside") \
                     + ("_mi" if self.lambda_MI and self.lambda_MI >0 else "") 
        self.load_path = "./save/" + ("pretrained" if (not args.testflag and not args.continuing) else self.name) #"./save/" + self.name #+ 
        self.save_path = "./save/" + self.name 
        self.log_path = "./log" + self.name 
        self.embedding_path = "../data/GoogleNews-vectors-negative300.bin"
        self.embedding_path_lime = self.embedding_path + '.p'
        self.print_freq = 100
        self.valid_freq = 20000
        self.test_freq = 1000
        self.save_freq = 100000
        self.is_corpus = False #if self.lambda_sup_G >= 1 else True  # supervised use setence-level bleu score
        self.load_from_ae = False

        self.discrimination = False
        # self.H_dis = 300

        self.update_params()
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def update_params(self):
        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        if self.lambda_sup_G >= 1:
            self.d_freq = 1000000000

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def discriminator(src, tgt, opt, prefix = 'd_', is_prob_src = False, is_prob_tgt= False, is_reuse = None):
    W_norm_d = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)   # V E
    H_src, _ = encoder(src, W_norm_d, opt, prefix = prefix + 'src_', is_reuse = is_reuse, is_prob=is_prob_src)
    H_tgt, x_tgt = encoder(tgt, W_norm_d, opt, prefix = prefix + 'tgt_', is_reuse = is_reuse, is_prob=is_prob_tgt, is_padded = (opt.model == 'cnn_deconv'))
    # H : B F
    if opt.is_subtract_mean:
        mean_H = tf.reduce_mean(H_src, axis=0) 
        H_src = H_src - mean_H
        H_tgt = H_tgt - mean_H
    logits = tf.reduce_sum(normalizing(H_src, 1)*normalizing(H_tgt, 1),1)*(1 if opt.feature_matching == 'pair_diff' else opt.L)
    return logits, tf.squeeze(tf.concat([H_src, H_tgt], 1)), x_tgt

def encoder(x, W_norm_d, opt, prefix = 'd_', is_prob = False, is_reuse = None, is_padded = True):
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb
    if not is_padded:  # pad the input
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm_d[0],0),0) 
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape-1, 1]), x_emb],1)

    x_emb = tf.expand_dims(x_emb,3)   # batch L emb 1
    if opt.layer == 3:
        H = conv_model_3layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse, num_outputs = opt.n_d_output)
    else:
        H = conv_model(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    return tf.squeeze(H), x_emb


def generator(z, src, tgt, opt, opt_t, is_prob_src = False, is_reuse = None, is_softargmax = True, is_sampling = False, prefix = 'g_'):  
    if opt.g_rev:
        prefix = 'g_rev_'
    z = tf.expand_dims(tf.expand_dims(z,1),1)  # B 1 1 Z
    if is_prob_src:
        W_norm = embedding_only(opt, prefix = prefix, is_reuse = is_reuse) 
        x_emb = tf.tensordot(src, W_norm, [[2],[0]])   #B L E
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm[0],0),0) # 1*v
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape-1, 1]), x_emb],1)
    else:
        W_norm = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)  # batch L emb
        x_emb = tf.nn.embedding_lookup(W_norm, src)
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1
    res = {}
    H_enc, res = conv_encoder(x_emb, opt, res, prefix = prefix, is_reuse = is_reuse) # B 1 1 F
    if opt.additive_noise_lambda:
        H_dec = (H_enc + z * opt.additive_noise_lambda)/(1+opt.additive_noise_lambda)
    else:  # concatenate z and h
        H_dec = tf.concat([H_enc, z], axis=3)

    if not opt.rnn_share_emb:
        W_norm_rnn = embedding_only(opt, prefix = prefix + '_dec', is_reuse = is_reuse)
        W_norm_dec = W_norm_rnn
    else:
        W_norm_dec = W_norm

    sup_loss, _, _ , sup_loss_all = lstm_decoder_embedding(H_dec, tgt, W_norm_dec, opt_t, add_go = True, is_reuse=is_reuse, is_fed_h = opt.is_fed_h, prefix = prefix)
    sample_loss, syn_sent, logits, _ = lstm_decoder_embedding(H_dec, tf.ones_like(tgt), W_norm_dec, opt_t, add_go = True, feed_previous=True, is_reuse=True, is_softargmax = is_softargmax, is_sampling = is_sampling, is_fed_h = opt.is_fed_h, prefix = prefix)
    prob = [tf.nn.softmax(l*opt_t.L) for l in logits]
    prob = tf.stack(prob,1)  # B L V

    return syn_sent, prob, H_dec, sup_loss, sample_loss, sup_loss_all

def conditional_gan(src, tgt, z,  opt, opt_t=None, is_reuse_generator = None):
    if not opt_t: opt_t = opt

    logits_real, H_real, _ = discriminator(src, tgt, opt, prefix = ('d_' if not is_reuse_generator else 'd_rev_'))
    syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = generator(z, src, tgt, opt, opt_t, is_reuse = is_reuse_generator, prefix = ('g_' if not is_reuse_generator else 'g_rev_'))

    logits_fake, H_fake, _ = discriminator(src, syn_one_hot, opt, is_prob_src = False, is_prob_tgt= True, is_reuse = True, prefix = ('d_' if not is_reuse_generator else 'd_rev_'))

    gan_cost_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits = logits_fake))

    real_prob = tf.reduce_mean(tf.nn.sigmoid(logits_real))
    fake_prob = tf.reduce_mean(tf.nn.sigmoid(logits_fake))

    fake_mean = tf.reduce_mean(H_fake,axis=0)
    real_mean = tf.reduce_mean(H_real,axis=0)
    mean_dist = tf.sqrt(tf.reduce_mean((fake_mean - real_mean)**2))

    if opt.feature_matching == 'mean':
        gan_cost_g = mean_dist
    elif opt.feature_matching == 'pair_diff':
        # improved WGAN lipschitz-penalty
        if opt.w_gan:
            gan_cost_g = tf.reduce_mean(logit(logits_real/2.0+0.5) - logit(logits_fake/2.0+0.5))
        else:
            gan_cost_g = tf.reduce_mean(logit((logits_real - logits_fake)/4.1 + 0.51)) #tf.reduce_mean( logits_real - logits_fake ) #tf.reduce_mean(logit((logits_real - logits_fake)/4.0 + 0.5))
        if opt.grad_penalty!= None:
            alpha = tf.random_uniform(shape=[opt.batch_size,1,1], minval=0.,maxval=1.)
            real_one_hot = tf.one_hot(tgt, opt.n_words, axis = -1)
            differences = syn_one_hot - real_one_hot
            interpolates = real_one_hot + (alpha*differences)
            logits_interp, _, x_int = discriminator(src, interpolates, opt, is_prob_src = False, is_prob_tgt= True, is_reuse = True)
            gradients = tf.gradients(logit(logits_interp/2.0+0.5), [x_int])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2])+ 1e-10)
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            gan_cost_d = -gan_cost_g + opt.grad_penalty*gradient_penalty
        else:
            gan_cost_d = -gan_cost_g

        real_prob = tf.reduce_mean(logits_real)
        fake_prob = tf.reduce_mean(logits_fake)
    else:
        gan_cost_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits = logits_fake))



    if opt.lambda_sup_G:
        if 0<=opt.lambda_sup_G<1:
            g_cost = opt.lambda_sup_G*sup_loss + (1-opt.lambda_sup_G)*gan_cost_g
        else:
            g_cost = sup_loss
    else:
        g_cost = 0.0

    # maximizing variational mutual information. (VIM)

    if opt.lambda_MI:
        rev_z = tf.random_uniform(shape=[opt.fake_size, opt.n_z], minval=0.,maxval=1.)
        rev_tgt = tf.identity(src[:,(opt.filter_shape-1):]) # B L
        if opt.MI_pg:
            rev_src = tf.one_hot(tf.cast(syn_sent, tf.int32), opt.n_words, on_value=1.0, off_value=0.0, axis=-1)   #  syn_tgt
            tf.stop_gradient(rev_src)
        else:
            rev_src = syn_one_hot   #  syn_tgt
        rev_syn_sent, _, _, rev_sup_loss, _, rev_sup_loss_all = generator(rev_z, rev_src, rev_tgt, opt, opt_t, is_prob_src = True, is_softargmax =False, is_reuse = is_reuse_generator, prefix = ('g_rev_' if not is_reuse_generator else 'g_'))  

        if opt.MI_pg:
            g_frwd_vars = [var for var in tf.trainable_variables() if 'g_' in var.name and 'rev_' not in var.name]
            g_rev_vars = [var for var in tf.trainable_variables() if 'g_rev_' in var.name]
            reward = -rev_sup_loss_all
            base_line = tf.reduce_mean(reward) 
            reward_stop = tf.stop_gradient(reward) 
            base_line_stop = tf.stop_gradient(base_line) 
            mi_loss = tf.reduce_mean((reward_stop - base_line_stop)*sup_loss_all) + rev_sup_loss
        else:
            mi_loss = rev_sup_loss
        tf.summary.scalar('MI', -mi_loss)
        g_cost = (1-opt.lambda_MI)*g_cost + mi_loss * opt.lambda_MI
    else:
        rev_sup_loss = tf.zeros(shape=1)  
        rev_sup_loss_all = tf.zeros(shape=1)
        

    res_dict= {'syn_sent':syn_sent,
            'mean_dist':mean_dist,
            'prob_r':real_prob,
            'prob_f': fake_prob,
            #'loss_mi_fwd': (reward_stop - base_line_stop) ,
            'loss_mi_rev': sup_loss_all,
            'sample_loss': sample_loss,
            'sup_loss': sup_loss_all,
            'rev_sup_loss': rev_sup_loss_all,
            }
    if opt.grad_penalty:
        res_dict['gp'] = opt.grad_penalty*gradient_penalty
    if opt.lambda_MI:
        res_dict['mi'] = rev_sup_loss
        res_dict['rev_sent'] = rev_syn_sent

    return res_dict, gan_cost_d, g_cost




def dialog_gan(src, tgt,  opt, opt_t=None):

    z = tf.random_uniform(shape=[opt.fake_size, opt.n_z], minval=-1.,maxval=1.)
        
    if opt.two_side:
        res_dict, gan_cost_d, gan_cost_g = conditional_gan(src, tgt, z,  opt, opt_t=opt_t)
        src_rev , tgt_rev = tf.concat([tf.cast(tf.zeros([opt.batch_size, (opt.filter_shape-1)]),tf.int32), tgt], 1)  , src[:,(opt.filter_shape-1):]
        rev_res_dict, gan_cost_d_rev, gan_cost_g_rev = conditional_gan(src_rev, tgt_rev, z,  opt, opt_t=opt_t, is_reuse_generator = True)
        gan_cost_d += opt.lambda_backward*gan_cost_d_rev
        gan_cost_g += opt.lambda_backward*gan_cost_g_rev
    else:
        res_dict, gan_cost_d, gan_cost_g = conditional_gan(src, tgt, z,  opt, opt_t=opt_t)



    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    if opt.g_fix:
        g_vars = [var for var in t_vars if 'g_g_' in var.name]
        print ("Fix most G params, except" + " ".join([v.name for v in g_vars]) )
    else:
        g_vars = [var for var in t_vars if 'g_' in var.name]

    tf.summary.scalar('loss_d', gan_cost_d)
    tf.summary.scalar('loss_g', gan_cost_g)

    summaries = [
        "learning_rate",
        "loss",
    ]
    global_step = tf.Variable(0, trainable=False)
    train_op_d = layers.optimize_loss(
          gan_cost_d,
          global_step=global_step,
          optimizer=opt.optimizer,
          clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
          learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g, decay_rate=opt.decay_rate, decay_steps=3000),
          variables=d_vars,
          learning_rate=opt.lr_d,
          summaries=summaries)


    train_op_g = layers.optimize_loss(
        gan_cost_g,
        global_step=global_step,
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step=g, decay_rate=opt.decay_rate, decay_steps=3000),
        variables=g_vars,
        learning_rate=opt.lr_g,
        summaries=summaries)



    return res_dict, gan_cost_d, train_op_d, gan_cost_g, train_op_g


def main():
    # Prepare training and testing data


    loadpath = "./"

    src_file = loadpath + "Pairs2M.src.num"
    tgt_file = loadpath + "Pairs2M.tgt.num"
    dic_file = loadpath + "Pairs2M.reddit.dic"

    opt = Options()
    opt_t = Options()

    train, val, test , wordtoix, ixtoword = read_pair_data_full(src_file, tgt_file, dic_file, max_num = opt.data_size, p_f = loadpath + 'demo.p')
    train = [ x for x in train if 2<len(x[1])<opt.maxlen - 4 and 2<len(x[0])<opt_t.maxlen - 4]
    val = [ x for x in val if 2<len(x[1])<opt.maxlen - 4 and 2<len(x[0])<opt_t.maxlen - 4]

    if TEST_FLAG:
        test = test + val + train
        opt.test_freq = 1

    opt.n_words = len(ixtoword)
    opt_t.n_words = len(ixtoword)
    print dict(opt)
    if opt.model == 'cnn_rnn':
        opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
        opt_t.update_params()
        print dict(opt_t)

    print('Total words: %d' % opt.n_words)


    # load w2v
    if os.path.exists(opt.embedding_path_lime):
        with open(opt.embedding_path_lime, 'rb') as pfile:
            embedding = cPickle.load(pfile)
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_path, binary=True)
        embedding = {i:copy.deepcopy(w2v[ixtoword[i]]) for i in range(opt.n_words) if ixtoword[i] in w2v}
        with open(opt.embedding_path_lime, 'wb') as pfile:
            cPickle.dump(embedding, pfile, protocol=cPickle.HIGHEST_PROTOCOL)

    for d in ['/gpu:0']:
        with tf.device(d):
            src_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
            tgt_ = tf.placeholder(tf.int32, shape=[opt_t.batch_size, opt_t.sent_len])
            res_, gan_cost_d_, train_op_d, gan_cost_g_, train_op_g = dialog_gan(src_, tgt_, opt, opt_t)
            merged = tf.summary.merge_all()

    uidx = 0
    graph_options=tf.GraphOptions(build_cost_model=1)
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=graph_options)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95


    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    run_metadata = tf.RunMetadata()

    with tf.Session(config = config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:

                t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) #tf.trainable_variables()

                if opt.load_from_ae:
                    save_keys = tensors_key_in_file(opt.load_path)  #t_var g_W:0    key: W
                    ss = [var for var in t_vars if var.name[2:][:-2] in save_keys.keys()]
                    ss = [var.name[2:] for var in ss if var.get_shape() == save_keys[var.name[2:][:-2]]]
                    cc = {var.name[2:][:-2]:var for var in t_vars if var.name[2:] in ss}

                    loader = tf.train.Saver(var_list=cc)
                    loader.restore(sess, opt.load_path)

                    print("Loading variables from '%s'." % opt.load_path)
                    print("Loaded variables:"+" ".join([var.name for var in t_vars if var.name[2:] in ss]))
                else:
                    save_keys = tensors_key_in_file(opt.load_path)
                    ss = [var for var in t_vars if var.name[:-2] in save_keys.keys()]
                    ss = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
                    loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss])
                    loader.restore(sess, opt.load_path)
                    print("Loading variables from '%s'." % opt.load_path)
                    print("Loaded variables:"+str(ss))
                    # load reverse model
                    try:
                        save_keys = tensors_key_in_file('./save/rev_model')
                        ss = [var for var in t_vars if var.name[:-2] in save_keys.keys() and 'g_rev_' in var.name]
                        ss = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
                        loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss])
                        loader.restore(sess, './save/rev_model')
                        print("Loading reverse variables from ./save/rev_model")
                        print("Loaded variables:"+str(ss))
                    except Exception as e:
                        print("No reverse model loaded")

            except Exception as e:
                print 'Error: '+str(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())
        loss_d , loss_g = 0, 0
        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1

                tgt, src = zip(*[train[t] for t in train_index])
                x_batch = prepare_data_for_cnn(src, opt) # Batch L

                y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) if opt.model == 'cnn_rnn' else prepare_data_for_cnn(tgt, opt_t)

         

                feed = {src_: x_batch, tgt_: y_batch}

                if uidx%opt.d_freq == 0:
                    if profile:
                        _, loss_d = sess.run([train_op_d, gan_cost_d_],feed_dict=feed, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                    else:
                        _, loss_d = sess.run([train_op_d, gan_cost_d_],feed_dict=feed)

                if uidx%opt.g_freq == 0:
                    if profile:
                        _, loss_g = sess.run([train_op_g, gan_cost_g_],feed_dict=feed, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
                    else:
                        _, loss_g = sess.run([train_op_g, gan_cost_g_],feed_dict=feed)

                if profile:
                    tf.contrib.tfprof.model_analyzer.print_model_analysis(
                    tf.get_default_graph(),
                    run_meta=run_metadata,
                    tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
                    exit(0)



                if uidx%opt.valid_freq == 0:
                    VALID_SIZE = 1024
                    valid_multiplier = np.int(np.floor(VALID_SIZE/opt.batch_size))
                    res_all, val_tgt_all, loss_val_d_all, loss_val_g_all = [], [], [], []
                    for val_step in range(valid_multiplier):
                        valid_index = np.random.choice(len(val), opt.batch_size)
                        val_tgt, val_src = zip(*[val[t] for t in valid_index])
                        val_tgt_all.extend(val_tgt)
                        x_val_batch = prepare_data_for_cnn(val_src, opt) # Batch L

                        y_val_batch = prepare_data_for_rnn(val_tgt, opt_t, is_add_GO = False) if opt.model == 'cnn_rnn' else prepare_data_for_cnn(val_tgt, opt_t)

                        feed_val = {src_: x_val_batch, tgt_: y_val_batch}
                        loss_val_d, loss_val_g = sess.run([gan_cost_d_, gan_cost_g_], feed_dict=feed_val)
                        loss_val_d_all.append(loss_val_d)
                        loss_val_g_all.append(loss_val_g)
                        res = sess.run(res_, feed_dict=feed_val)
                        res_all.extend(res['syn_sent'])
                        
                    print("Validation: loss D %f loss G %f " %(np.mean(loss_val_d_all), np.mean(loss_val_g_all)))
                    #print "Val Perm :" + " ".join([ixtoword[x] for x in val_src_permutated[0] if x != 0])
                    print "Val Source:" + u' '.join([ixtoword[x] for x in val_src[0] if x != 0]).encode('utf-8').strip()
                    print "Val Target :" + u' '.join([ixtoword[x] for x in val_tgt[0] if x != 0]).encode('utf-8').strip()
                    print "Val Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""

                    val_set = [prepare_for_bleu(s) for s in val_tgt_all]
                    gen = [prepare_for_bleu(s) for s in res_all]
                    
                    [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = opt.is_corpus)
                    [rouge1,rouge2,rouge3,rouge4,rougeL,rouges] = cal_ROUGE(gen, {0: val_set}, is_corpus = opt.is_corpus)
                    etp_score, dist_score = cal_entropy(gen)
                    bleu_nltk = cal_BLEU_4_nltk(gen, val_set, is_corpus = opt.is_corpus)
                    rel_score = cal_relevance(gen, val_set, embedding)

                    print 'Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu_nltk,bleu1s,bleu2s,bleu3s,bleu4s)])
                    print 'Val Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
                    print 'Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
                    print 'Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
                    print 'Val Relevance(G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
                    print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 
                    print ""
                    summary = sess.run(merged, feed_dict=feed_val)
                    summary2 = tf.Summary(value=[tf.Summary.Value(tag="bleu-2", simple_value=bleu2s),tf.Summary.Value(tag="rouge-2", simple_value=rouge2),tf.Summary.Value(tag="etp-4", simple_value=etp_score[3])])

                    test_writer.add_summary(summary, uidx)
                    test_writer.add_summary(summary2, uidx)

                if uidx%opt.test_freq == 0:
                    iter_num = np.int(np.floor(len(test)/opt.batch_size))+1
                    res_all, test_tgt_all = [], []
                    
                    for i in range(iter_num):
                        test_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
                        test_tgt, test_src = zip(*[test[t%len(test)] for t in test_index])
                        test_tgt_all.extend(test_tgt)
                        x_batch = prepare_data_for_cnn(test_src, opt)
                        y_batch = prepare_data_for_rnn(test_tgt, opt_t, is_add_GO = False) if opt.model == 'cnn_rnn' else prepare_data_for_cnn(test_tgt, opt_t)
                        feed = {src_: x_batch, tgt_: y_batch}
                        res = sess.run(res_, feed_dict=feed)
                        res_all.extend(res['syn_sent'])


                    test_set = [prepare_for_bleu(s) for s in test_tgt_all]
                    gen = [prepare_for_bleu(s) for s in res_all]
                    [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: test_set}, is_corpus = opt.is_corpus)
                    [rouge1,rouge2,rouge3,rouge4,rougeL,rouges] = cal_ROUGE(gen, {0: test_set}, is_corpus = opt.is_corpus)
                    etp_score, dist_score = cal_entropy(gen)
                    bleu_nltk = cal_BLEU_4_nltk(gen, test_set, is_corpus = opt.is_corpus)
                    rel_score = cal_relevance(gen, test_set, embedding)


                    print 'Test BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu_nltk,bleu1s,bleu2s,bleu3s,bleu4s)])
                    print 'Test Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
                    print 'Test Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
                    print 'Test Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
                    print 'Test Relevance(G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
                    print 'Test Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 
                    print ''

                    if TEST_FLAG:
                        exit()


                if uidx%opt.print_freq == 0:
                    print("Iteration %d: loss D %f loss G %f" %(uidx, loss_d, loss_g))

                    res = sess.run(res_, feed_dict=feed)

                    if opt.grad_penalty:
                        print "grad_penalty: " + str(res['gp'])
                    print "Source:" + u' '.join([ixtoword[x] for x in x_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Target:" + u' '.join([ixtoword[x] for x in y_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""

                    sys.stdout.flush()
                    summary = sess.run(merged, feed_dict=feed)
                    train_writer.add_summary(summary, uidx)

                if uidx%opt.save_freq == 0:
                    saver.save(sess, opt.save_path)




if __name__ == '__main__':
    main()

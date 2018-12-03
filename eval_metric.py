
# -*- coding: utf-8 -*-
"""
Yizhe Zhang

Seq2seq lstm baseline for dialog
"""
## 152.3.214.203/6006

import os

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
from denoise import *
import gensim
import copy
import codecs
import argparse
from converse_gan import dialog_gan

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate model output.')
    parser.add_argument('--gpuid', '-g', type=int, default=0)  
    parser.add_argument('--target', '-t', type=str, default='./save')
    parser.add_argument('--response', '-r', type=str, default='./save')

    args = parser.parse_args()
    print(args)


    profile = False
    TEST_FLAG = False
    #import tempfile
    #from tensorflow.examples.tutorials.mnist import input_data

    logging.set_verbosity(logging.INFO)
    #tf.logging.verbosity(1)
    # Basic model parameters as external flags.
    flags = tf.app.flags
    FLAGS = flags.FLAGS


    GPUID = args.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)




class Options(object):
    def __init__(self):

        # 
        # One side or two side
        self.two_side = True #True
        self.lambda_backward = 0.1
        # lambda_MI = None : no MI
        self.lambda_MI = 0.1## 0.1 # 0.9
        # Supervise level
        self.lambda_sup_G = 0.1 #0.9 #None # 1: fully supervised  None: no supervised  trade-off between supervised signal and GAN


        # optimizer gan
        self.d_freq = 1
        self.g_freq = 1



        #
        self.fix_emb = False
        self.reuse_cnn = False
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv
        self.rnn_share_emb = True #CNN_LSTM share embedding
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
        self.batch_size = 32#32
        self.max_epochs = 100
        self.n_hid = 100  # self.filter_size * 3
        self.multiplier = 2
        self.L = 100

    


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
        self.z_prior = 'u' # 'g','u'
        self.n_z = self.n_hid if self.additive_noise_lambda else 10
        self.lr_g = 1e-4 #5e-5 #1e-4
        self.feature_matching = 'pair_diff'#'pair_diff' # 'mean' # 'mmd' # None
        self.w_gan = False
        self.bp_truncation = None


        self.fake_size = self.batch_size
        self.sigma_range = [1]
        #self.n_d_output = 100

        

        self.g_fix = False
        self.g_rev = False    # backward model only

        # optimizer

        self.optimizer = 'SGD' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False



        # misc
        self.data_size = None #None #10000  # None : all data
        self.name = 'gan' + str(self.n_hid) + "_dim_" + self.model + "_" + self.feature_matching + ("_sup" if self.lambda_sup_G >= 1 else "_gan") \
                     + ("_rev_only" if self.g_rev else "") + ("_twoside" if self.two_side else "_oneside") \
                     + ("_mi" if self.lambda_MI and self.lambda_MI >0 else "") 
        self.load_path = "./save/save_result/" + self.name  #"./save/" + self.name #+ 
        self.save_path = "./save/save_result/" + self.name 
        self.log_path = "./log" + self.name 
        self.embedding_path = "../data/GoogleNews-vectors-negative300.bin"
        self.embedding_path_lime = self.embedding_path + '.reddit.p'
        self.print_freq = 100
        self.valid_freq = 2000
        self.test_freq = 2000
        self.save_freq = 3000
        self.is_corpus = False #if self.lambda_sup_G >= 1 else True  # supervised use setence-level bleu score

        # batch norm & dropout & save
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1
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

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value



def main():
    #global n_words
    # Prepare training and testing data
    #loadpath = "./data/three_corpus_small.p"
    #loadpath = "./data/three_corpus_corrected_large.p"
    loadpath = "../data/reddit_2m/"
    dic_file = loadpath + "Pairs2M.reddit.dic"


    wordtoix, ixtoword = {}, {}
    print "Start reading dic file . . ."
    if os.path.exists(dic_file):
        print("loading Dictionary")
        counter=0
        with codecs.open(dic_file,"r",'utf-8') as f:
            s=f.readline()
            while s:
                s=s.rstrip('\n').rstrip("\r")
                #print("s==",s)
                wordtoix[s]=counter
                ixtoword[counter]=s
                counter+=1
                s=f.readline()

    target, response = [], []
    with codecs.open(args.target,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        while line:
            target.append([wordtoix[x] if x in wordtoix else 3 for x in line.split()])
            line = f.readline().rstrip("\n").rstrip("\r")

    with codecs.open(args.response,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        while line:
            response.append([wordtoix[x] if x in wordtoix else 3 for x in line.split()])
            line = f.readline().rstrip("\n").rstrip("\r")

    opt = Options()
    # opt_t = Options()




    # opt.test_freq = 1

    # # opt_t.maxlen = 101 #49
    # # opt_t.update_params()

    opt.n_words = len(ixtoword)
    # opt_t.n_words = len(ixtoword)
    # print dict(opt)
    # if opt.model == 'cnn_rnn':
    #     opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
    #     opt_t.update_params()
    #     print dict(opt_t)

    print('Total words: %d' % opt.n_words)

    if os.path.exists(opt.embedding_path_lime):
        with open(opt.embedding_path_lime, 'rb') as pfile:
            embedding = cPickle.load(pfile)
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_path, binary=True)
    #wl = [ixtoword[i] for i in range(opt.n_words) if ixtoword[i] in w2v]
    #w2v[wl].gensim.models.KeyedVectors.save_word2vec_format(opt.embedding_path + '_lime', binary=True)
        embedding = {i:copy.deepcopy(w2v[ixtoword[i]]) for i in range(opt.n_words) if ixtoword[i] in w2v}
        with open(opt.embedding_path_lime, 'wb') as pfile:
            cPickle.dump(embedding, pfile, protocol=cPickle.HIGHEST_PROTOCOL)
    
    test_set = [prepare_for_bleu(s) for s in target]
    res_all = response
    [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4([prepare_for_bleu(s) for s in res_all], {0: test_set}, is_corpus = opt.is_corpus)
    [rouge1,rouge2,rouge3,rouge4,rougeL,rouges] = cal_ROUGE([prepare_for_bleu(s) for s in res_all], {0: test_set}, is_corpus = opt.is_corpus)
    etp_score, dist_score = cal_entropy([prepare_for_bleu(s) for s in  res_all])
    bleu_nltk = cal_BLEU_4_nltk([prepare_for_bleu(s) for s in  res_all], test_set, is_corpus = opt.is_corpus)
    rel_score = cal_relevance([prepare_for_bleu(s) for s in  res_all], test_set, embedding)


    print 'Test BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu_nltk,bleu1s,bleu2s,bleu3s,bleu4s)])
    print 'Test Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
    print 'Test Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
    print 'Test Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
    print 'Test Relevance(G,E,A): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
    print ''

    #bp()

    # for d in ['/gpu:0']:
    #     with tf.device(d):
    #         src_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
    #         tgt_ = tf.placeholder(tf.int32, shape=[opt_t.batch_size, opt_t.sent_len])
    #         z_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.n_z])
    #         res_, gan_cost_d_, train_op_d, gan_cost_g_, train_op_g = dialog_gan(src_, tgt_, z_, opt, opt_t)
    #         merged = tf.summary.merge_all()
    #tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
    #writer = tf.train.SummaryWriter(opt.log_path, graph=tf.get_default_graph())

    # uidx = 0
    # graph_options=tf.GraphOptions(build_cost_model=1)
    # #config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
    # config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=graph_options )
    # config.gpu_options.per_process_gpu_memory_fraction = 0.49
    # #config = tf.ConfigProto(device_count={'GPU':0})
    # #config.gpu_options.allow_growth = True

    # np.set_printoptions(precision=3)
    # np.set_printoptions(threshold=np.inf)
    # # saver = tf.train.Saver()

    # # run_metadata = tf.RunMetadata()
    # # fh = open(opt.load_path + ".rsp.lim", 'w')

    # with tf.Session(config = config) as sess:
        # sess.run(tf.global_variables_initializer())
        # try:
        #     t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #     save_keys = tensors_key_in_file(opt.load_path)
        #     ss = [var for var in t_vars if var.name[:-2] in save_keys.keys()]
        #     ss = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
        #     loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss])
        #     loader.restore(sess, opt.load_path)
        #     print("Loading variables from '%s'." % opt.load_path)
        #     print("Loaded variables:"+str(ss))
        #     try:
        #         save_keys = tensors_key_in_file('./save/rev_model')
        #         ss = [var for var in t_vars if var.name[:-2] in save_keys.keys() and 'g_rev_' in var.name]
        #         ss = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
        #         loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss])
        #         loader.restore(sess, './save/rev_model')
        #         print("Loading reverse variables from ./save/rev_model")
        #         print("Loaded variables:"+str(ss))
        #     except Exception as e:
        #         print("No reverse model loaded")
        # except Exception as e:
        #         print 'Error: '+str(e)
        #         print("No saving session, using random initialization")
        #         sess.run(tf.global_variables_initializer())

        # iter_num = np.int(np.floor(len(train)/opt.batch_size))
        # res_all, test_tgt_all = [], []
        # for i in range(iter_num):
        #     # if epoch >= 10:
        #     #     print("Relax embedding ")
        #     #     opt.fix_emb = False
        #     #     opt.batch_size = 2
        #     train_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
        #     uidx += 1
        #     tgt, src = zip(*[train[t] for t in train_index])
        #     test_tgt_all.extend(tgt)
        #     src_permutated = src
        #     x_batch = prepare_data_for_cnn(src_permutated, opt)
        #     y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) if opt.model == 'cnn_rnn' else prepare_data_for_cnn(tgt, opt_t)
        #     if opt.z_prior == 'g':
        #         z_batch = np.random.normal(0,1,(opt.fake_size, opt.n_z)).astype('float32')
        #     else:
        #         z_batch = np.random.uniform(-1,1,(opt.fake_size, opt.n_z)).astype('float32')
        #     feed = {src_: x_batch, tgt_: y_batch, z_:z_batch}
        #     res = sess.run(res_, feed_dict=feed)
        #     res_all.extend(res['syn_sent'])
        #     print i
        #     for idx in range(opt.batch_size):
        #         if train_index[idx]<len(train):
        #             fh.write("Source:" + u' '.join([ixtoword[x] for x in x_batch[idx] if x != 0]).encode('utf-8').strip() + '\n')
        #             fh.write("Target:" + u' '.join([ixtoword[x] for x in y_batch[idx] if x != 0]).encode('utf-8').strip() + '\n')
        #             fh.write("Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][idx] if x != 0]).encode('utf-8').strip() + '\n')
        #             fh.write("Reconed:" + u' '.join([ixtoword[x] for x in res['rev_sent'][idx] if x != 0]).encode('utf-8').strip() + '\n')
        #             fh.write('\n')
        #         else:
        #             break

        # fh.close()


if __name__ == '__main__':
    main()

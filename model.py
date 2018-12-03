import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
#from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.legacy_seq2seq import rnn_decoder, embedding_rnn_decoder, sequence_loss, sequence_loss_by_example, embedding_rnn_seq2seq, embedding_tied_rnn_seq2seq
import pdb
import copy
from utils import normalizing, lrelu
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import nn_ops, math_ops, embedding_ops, variable_scope, array_ops
import numpy as np
# from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
# from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
# from tensorflow.contrib.seq2seq.python.ops import decoder

import data_utils as dp
from pdb import set_trace as bp

emb_init = tf.random_uniform_initializer(-0.001, 0.001)
weight_init = tf.random_uniform_initializer(-0.001, 0.001) #layers.xavier_initializer()    
bias_init = tf.constant_initializer(0.0, dtype=tf.float32)
def embedding(features, opt, prefix = '', is_reuse = None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    #    b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
    with tf.variable_scope(prefix+'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert(hasattr(opt,'emb'))
            assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], weights_initializer = opt.emb, is_trainable = False) # V E
        else:
            if hasattr(opt,'emb') and opt.emb:
                assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
                weightInit = opt.emb
            else:
                weightInit = emb_init        # tf.stop_gradient(W)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer = weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    W_norm = normalizing(W, 1)
    word_vectors = tf.nn.embedding_lookup(W_norm, features)


    return word_vectors, W_norm


def embedding_only(opt, prefix = '', is_reuse = None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix+'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert(hasattr(opt,'emb'))
            assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], weights_initializer = opt.emb, is_trainable = False)
        else:
            if hasattr(opt,'emb') and opt.emb:
                assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
                weightInit = opt.emb
                W = tf.get_variable('W', initializer = weightInit)
            else:
                weightInit = emb_init
                W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer = weightInit)
            
    #    b = tf.get_variable('b', [opt.embed_size], initializer = )
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    W_norm = normalizing(W, 1)

    return W_norm


def discriminator_cnn(x, W, opt, prefix = 'd_', is_prob = False, is_reuse = None):
    W_norm_d = tf.identity(W)   # deep copy
    tf.stop_gradient(W_norm_d)  # the discriminator won't update W
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])  # batch L emb
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb

    # print x_emb.get_shape()
    x_emb = tf.expand_dims(x_emb,3)   # batch L emb 1


    if opt.layer == 4:
        H = conv_model_4layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    elif opt.layer == 3:
        H = conv_model_3layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse)
    else: # layer == 2
        H = conv_model(x_emb, opt, prefix = prefix, is_reuse = is_reuse)

    logits = discriminator_2layer(H, opt, prefix= prefix, is_reuse = is_reuse)
    return logits, tf.squeeze(H)


def conv_encoder(x_emb, opt, res, is_train= True, is_reuse = None, prefix = ''):
    if hasattr(opt, 'multiplier'):
        multiplier = opt.multiplier
    else:
        multiplier = 2
    if opt.layer == 4:
        H_enc = conv_model_4layer(x_emb, opt, is_train = is_train, is_reuse = is_reuse, prefix = prefix)
    elif opt.layer == 3:
        H_enc = conv_model_3layer(x_emb, opt, is_train = is_train, multiplier = multiplier, is_reuse = is_reuse, prefix = prefix)
    elif opt.layer == 0:
        H_enc = conv_model_3layer_old(x_emb, opt, is_reuse = is_reuse, prefix = prefix)
    else:
        H_enc = conv_model(x_emb, opt, is_train = is_train, is_reuse = is_reuse, prefix = prefix)
    return H_enc, res


def regularization(X, opt, is_train, prefix= '', is_reuse= None):
    if '_X' not in prefix and '_H_dec' not in prefix:
        if opt.batch_norm:
            X = layers.batch_norm(X, decay=0.9, center=True, scale=True, is_training=is_train, scope=prefix+'_bn', reuse = is_reuse)
        X = tf.nn.relu(X)
    X = X if (not opt.dropout or is_train is None) else layers.dropout(X, keep_prob = opt.dropout_ratio, scope=prefix + '_dropout')

    return X


conv_acf = tf.nn.tanh # tf.nn.relu

def conv_model(X, opt, prefix = '', is_reuse= None, is_train = True):  # 2layers
    #XX = tf.reshape(X, [-1, , 28, 1])
    #X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = None if opt.batch_norm else bias_init
        weightInit = weight_init

    X = regularization(X, opt,  prefix= prefix + 'reg_X', is_reuse= is_reuse, is_train = is_train)
    H1 = layers.conv2d(X,  num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  weights_initializer = weightInit, biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H1', reuse = is_reuse)  # batch L-3 1 Filtersize

    H1 = regularization(H1, opt, prefix= prefix + 'reg_H1', is_reuse= is_reuse, is_train = is_train)
    H2 = layers.conv2d(H1,  num_outputs=opt.filter_size*2,  kernel_size=[opt.sent_len2, 1],  activation_fn=conv_acf , padding = 'VALID', scope = prefix + 'H2', reuse = is_reuse) # batch 1 1 2*Filtersize
    return H2


def conv_model_3layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True, multiplier = 2):
    #XX = tf.reshape(X, [-1, , 28, 1])
    #X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = None if opt.batch_norm else bias_init
        weightInit = weight_init

    X = regularization(X, opt,  prefix= prefix + 'reg_X', is_reuse= is_reuse, is_train = is_train)
    H1 = layers.conv2d(X,  num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  weights_initializer = weightInit, biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H1_3', reuse = is_reuse)  # batch L-3 1 Filtersize

    H1 = regularization(H1, opt, prefix= prefix + 'reg_H1', is_reuse= is_reuse, is_train = is_train)
    H2 = layers.conv2d(H1,  num_outputs=opt.filter_size*multiplier,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H2_3', reuse = is_reuse)
    #print H2.get_shape()
    H2 = regularization(H2, opt,  prefix= prefix + 'reg_H2', is_reuse= is_reuse, is_train = is_train)
    H3 = layers.conv2d(H2,  num_outputs= (num_outputs if num_outputs else opt.n_hid),  kernel_size=[opt.sent_len3, 1], activation_fn=conv_acf , padding = 'VALID', scope = prefix + 'H3_3', reuse = is_reuse) # batch 1 1 2*Filtersize

    #pdb.set_trace()
    return H3


def conv_model_3layer_old(X, opt, prefix = '', is_reuse= None, num_outputs = None):
    #XX = tf.reshape(X, [-1, , 28, 1])
    #X shape: batchsize L emb 1

    biasInit = bias_init
    weightInit = weight_init

    H1 = layers.conv2d(X,  num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1], weights_initializer = weightInit, biases_initializer=biasInit, activation_fn=tf.nn.relu, padding = 'VALID', scope = prefix + 'H1_3', reuse = is_reuse)  # batch L-3 1 Filtersize
    H2 = layers.conv2d(H1,  num_outputs=opt.filter_size*2,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1], biases_initializer=biasInit, activation_fn=tf.nn.relu, padding = 'VALID', scope = prefix + 'H2_3', reuse = is_reuse)
    #print H2.get_shape()
    H3 = layers.conv2d(H2,  num_outputs= (num_outputs if num_outputs else opt.n_hid),  kernel_size=[opt.sent_len3, 1], biases_initializer=biasInit, activation_fn=tf.nn.tanh, padding = 'VALID', scope = prefix + 'H3_3', reuse = is_reuse) # batch 1 1 2*Filtersize
    return H3


def conv_model_4layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True):
    #XX = tf.reshape(X, [-1, , 28, 1])
    #X shape: batchsize L emb 1
    if opt.reuse_cnn:
        biasInit = opt.cnn_b
        weightInit = opt.cnn_W
    else:
        biasInit = None if opt.batch_norm else bias_init
        weightInit = weight_init

    X = regularization(X, opt,  prefix= prefix + 'reg_X', is_reuse= is_reuse, is_train = is_train)
    H1 = layers.conv2d(X,  num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  weights_initializer = weightInit, biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H1_3', reuse = is_reuse)  # batch L-3 1 Filtersize

    H1 = regularization(H1, opt, prefix= prefix + 'reg_H1', is_reuse= is_reuse, is_train = is_train)
    H2 = layers.conv2d(H1,  num_outputs=opt.filter_size*2,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H2_3', reuse = is_reuse)

    H2 = regularization(H2, opt, prefix= prefix + 'reg_H2', is_reuse= is_reuse, is_train = is_train)
    H3 = layers.conv2d(H2,  num_outputs=opt.filter_size*4,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[2],1],  biases_initializer=biasInit, activation_fn=None, padding = 'VALID', scope = prefix + 'H3_3', reuse = is_reuse)
    #print H2.get_shape()
    H3 = regularization(H3, opt, prefix= prefix + 'reg_H3', is_reuse= is_reuse, is_train = is_train)
    H4 = layers.conv2d(H3,  num_outputs= (num_outputs if num_outputs else opt.n_hid),  kernel_size=[opt.sent_len4, 1], activation_fn=conv_acf , padding = 'VALID', scope = prefix + 'H4', reuse = is_reuse) # batch 1 1 2*Filtersize
    return H4


dec_acf = tf.nn.relu #tf.nn.tanh
dec_bias = None # tf.random_uniform_initializer(-0.001, 0.001)


#
def seq2seq(x, y, opt, prefix = '', feed_previous=False, is_reuse= None, is_tied = True):
    #y batch * len    x batch*len
    # reverse x
    #pdb.set_trace()
    x = tf.reverse(x, axis=[1])

    x = tf.unstack(x, axis=1)  # X Y Z   [tf.shape(Batch_size)]*L
    y = tf.unstack(y, axis=1)  # GO_ A B C

    with tf.variable_scope(prefix + 'lstm_seq2seq', reuse=is_reuse):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)

    # with tf.variable_scope(prefix+'lstm_seq2seq', reuse=is_reuse):
 #        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
 #        W = tf.get_variable('W', [opt.n_hid, opt.n_words], initializer = weightInit)
 #        b = tf.get_variable('b', [opt.n_words], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        if is_tied:
            outputs, _ = embedding_tied_rnn_seq2seq(encoder_inputs = x, decoder_inputs = y, cell = cell, feed_previous = feed_previous, num_symbols = opt.n_words, embedding_size = opt.embed_size)
        else:
            outputs, _ = embedding_rnn_seq2seq(encoder_inputs = x, decoder_inputs = y, cell = cell, feed_previous = feed_previous, num_encoder_symbols = opt.n_words, num_decoder_symbols = opt.n_words, embedding_size = opt.embed_size)

    #logits = [nn_ops.xw_plus_b(out, W, b) for out in outputs]
    logits = outputs

    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents,1)

    loss = sequence_loss(outputs[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits


def lstm_decoder(H, y, opt, prefix = '', feed_previous=False, is_reuse= None):
    #y  len* batch * [0,V]   H batch * h

    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    y = tf.unstack(y, axis=1)
    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0))  # initialize C and H

    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = weight_init
        W = tf.get_variable('W', [opt.n_hid, opt.n_words], initializer = weightInit)
        b = tf.get_variable('b', [opt.n_words], initializer = bias_init)
        out_proj = (W,b) if feed_previous else None
        outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H1, cell = cell, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.n_words, embedding_size = opt.embed_size)

    logits = [nn_ops.xw_plus_b(out, W, b) for out in outputs]
    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents,1)


    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits





def lstm_decoder_embedding(H, y, W_emb, opt, prefix = '', add_go = False, feed_previous=False, is_reuse= None, is_fed_h = True, is_sampling = False, is_softargmax = False, beam_width=None):
    #y  len* batch * [0,V]   H batch * h
    biasInit = bias_init
    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    if add_go:
        y = tf.concat([tf.ones([opt.batch_size,1],dtype=tf.int32), y],1)

    y = tf.unstack(y, axis=1)  # 1, . , .
    # make the size of hidden unit to be n_hid
    if not opt.additive_noise_lambda:
        H = layers.fully_connected(H, num_outputs = opt.n_hid, biases_initializer=biasInit, activation_fn = None, scope = prefix + 'lstm_decoder', reuse = is_reuse)
    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0)) # tf.zeros_like(H0) # initialize C and H#

    y_input = [tf.concat([tf.nn.embedding_lookup(W_emb, features),H0],1) for features in y] if is_fed_h   \
               else [tf.nn.embedding_lookup(W_emb, features) for features in y]
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = weight_init
        W = tf.get_variable('W', [opt.n_hid, opt.embed_size], initializer = weightInit)
        b = tf.get_variable('b', [opt.n_words], initializer = bias_init)
        W_new = tf.matmul(W, W_emb, transpose_b=True) # h* V

        out_proj = (W_new,b) if feed_previous else None
        decoder_res = rnn_decoder_custom_embedding(emb_inp = y_input, initial_state = H1, cell = cell, embedding = W_emb, opt = opt, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.n_words, is_fed_h = is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling)
        outputs = decoder_res[0]

        # if beam_width:
        #     #cell = rnn_cell.LSTMCell(cell_depth)
        #     #batch_size_tensor = constant_op.constant(opt.batch_size)
        #     initial_state = cell.zero_state(opt.batch_size* beam_width, tf.float32) #beam_search_decoder.tile_batch(H0, multiplier=beam_width)
        #     output_layer = layers_core.Dense(opt.n_words, use_bias=True, kernel_initializer = W_new, bias_initializer = b, activation=None)
        #     bsd = beam_search_decoder.BeamSearchDecoder(
        #         cell=cell,
        #         embedding=W_emb,
        #         start_tokens=array_ops.fill([opt.batch_size], dp.GO_ID), # go is 1
        #         end_token=dp.EOS_ID,
        #         initial_state=initial_state,
        #         beam_width=beam_width,
        #         output_layer=output_layer,
        #         length_penalty_weight=0.0)
        #     #pdb.set_trace()
        #     final_outputs, final_state, final_sequence_lengths = (
        #         decoder.dynamic_decode(bsd, output_time_major=False, maximum_iterations=opt.maxlen))
        #     beam_search_decoder_output = final_outputs.beam_search_decoder_output
        #     #print beam_search_decoder_output.get_shape()

    logits = [nn_ops.xw_plus_b(out, W_new, b) for out in outputs]  # hidden units to prob logits: out B*h  W: h*E  Wemb V*E
    if is_sampling:
        syn_sents = decoder_res[2]
        loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.ones_like(yy),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in syn_sents])
        #loss = sequence_loss(logits[:-1], syn_sents, [tf.concat([tf.ones([1]), tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32)],0) for yy in syn_sents[:-1]]) # use one more pad after EOS
        syn_sents = tf.stack(syn_sents,1)
    else:
        syn_sents = [math_ops.argmax(l, 1) for l in logits[:-1]]
        syn_sents = tf.stack(syn_sents,1)
        ones = tf.ones([opt.batch_size],dtype=tf.float32)
        mask = [ones, ones] + [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[1:-2]] 
        loss_all = sequence_loss_by_example(logits[:-1], y[1:], mask)
        #loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[:-1]])
        #loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[1:]]) # use one more pad after EOS
        loss = tf.reduce_mean(loss_all)
    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len



    return loss, syn_sents, logits, loss_all






def rnn_decoder_custom_embedding(emb_inp,
                          initial_state,
                          cell,
                          embedding,
                          opt,
                          num_symbols,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None,
                          is_fed_h = True,
                          is_softargmax = False,
                          is_sampling = False
                          ):

  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    # embedding = variable_scope.get_variable("embedding",
    #                                         [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, initial_state[0], opt, output_projection,
        update_embedding_for_previous, is_fed_h=is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling) if feed_previous else None

    custom_decoder = rnn_decoder_with_sample if is_sampling else rnn_decoder_truncated

    return custom_decoder(emb_inp, initial_state, cell, loop_function=loop_function, truncate = opt.bp_truncation)


def _extract_argmax_and_embed(embedding,
                              h,
                              opt,
                              output_projection=None,
                              update_embedding=True,
                              is_fed_h = True,
                              is_softargmax = False,
                              is_sampling = False):

  def loop_function_with_sample(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    if is_sampling:
      prev_symbol_sample = tf.squeeze(tf.multinomial(prev*opt.L,1))  #B 1   multinomial(log odds)
      prev_symbol_sample = array_ops.stop_gradient(prev_symbol_sample) # important
      emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol_sample)
    else:
      if is_softargmax:
        prev_symbol_one_hot = tf.nn.log_softmax(prev*opt.L)  #B V
        emb_prev = tf.matmul( tf.exp(prev_symbol_one_hot), embedding) # solve : Requires start <= limit when delta > 0
      else:
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    emb_prev = tf.concat([emb_prev,h], 1) if is_fed_h else emb_prev
    if not update_embedding: #just update projection?
      emb_prev = array_ops.stop_gradient(emb_prev)
    return (emb_prev, prev_symbol_sample) if is_sampling else emb_prev

  # def loop_function(prev, _):
  #   if is_sampling:
  #     emb_prev, _ = loop_function_with_sample(prev, _)
  #   else:
  #     emb_prev = loop_function_with_sample(prev, _)
  #   return emb_prev

  return loop_function_with_sample #if is_sampling else loop_function

def rnn_decoder_truncated(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None,
                truncate=None):
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      if i >0 and truncate and tf.mod(i,truncate) == 0:
        #tf.stop_gradient(state)
        tf.stop_gradient(output)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def rnn_decoder_with_sample(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None,
                truncate=None):
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs, sample_sent = [], []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp, cur_token = loop_function(prev, i)
        sample_sent.append(cur_token)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      if i >0 and truncate and tf.mod(i,truncate) == 0:
        #tf.stop_gradient(state)
        tf.stop_gradient(output)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state, sample_sent


def discriminator_2layer(H, opt, prefix = '', is_reuse= None, is_train= True):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = bias_init
    H = regularization(H, opt, is_train, prefix= prefix + 'reg_H', is_reuse= is_reuse)
    H_dis = layers.fully_connected(H, num_outputs = opt.H_dis, biases_initializer=biasInit, activation_fn = tf.nn.relu, scope = prefix + 'dis_1', reuse = is_reuse)
    H_dis = regularization(H_dis, opt, is_train, prefix= prefix + 'reg_H_dis', is_reuse= is_reuse)
    logits = layers.linear(H_dis, num_outputs = 1, biases_initializer=biasInit, scope = prefix + 'dis_2', reuse = is_reuse)
    return logits


def softmax_prediction(X, opt, is_reuse= None):
    #X shape: batchsize L emb 1
    biasInit = bias_init
    pred_H = layers.conv2d(X,  num_outputs=opt.n_words,  kernel_size=[1, opt.embed_size], biases_initializer=biasInit, activation_fn=tf.nn.relu, padding = 'VALID', scope = 'pred', reuse = is_reuse)  # batch L 1 V
    pred_prob = layers.softmax(pred_H, scope = 'pred') # batch L 1 V
    pred_prob = tf.squeeze(pred_prob) # batch L V
    return pred_prob

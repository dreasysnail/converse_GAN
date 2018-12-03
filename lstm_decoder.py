import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
#from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.legacy_seq2seq import rnn_decoder, embedding_rnn_decoder, sequence_loss, embedding_rnn_seq2seq, embedding_tied_rnn_seq2seq
import pdb
import copy
from utils import normalizing, lrelu
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops, math_ops, embedding_ops, variable_scope




def lstm_decoder(H, y, opt, prefix = '', feed_previous=False, is_reuse= None):
    #y  len* batch * [0,V]   H batch * h

    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    y = tf.unstack(y, axis=1)
    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0))  # initialize H and C

    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_hid, opt.n_words], initializer = weightInit)
        b = tf.get_variable('b', [opt.n_words], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        out_proj = (W,b) if feed_previous else None
        outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H1, cell = cell, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.n_words, embedding_size = opt.embed_size)

    logits = [nn_ops.xw_plus_b(out, W, b) for out in outputs]
    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents,1)


    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits





def lstm_decoder_embedding(H, y, W_emb, opt, prefix = '', feed_previous=False, is_reuse= None, is_fed_h = True):
    #y  len* batch * [0,V]   H batch * h

    #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
    y = tf.unstack(y, axis=1)

    H0 = tf.squeeze(H)
    H1 = (H0, tf.zeros_like(H0))  # initialize H and C #

    y_input = [tf.concat([tf.nn.embedding_lookup(W_emb, features),H0],1) for features in y] if is_fed_h   \
               else [tf.nn.embedding_lookup(W_emb, features) for features in y]

    with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
        cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
    with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)
        W = tf.get_variable('W', [opt.n_hid, opt.embed_size], initializer = weightInit)
        b = tf.get_variable('b', [opt.n_words], initializer = tf.random_uniform_initializer(-0.001, 0.001))
        W_new = tf.matmul(W, W_emb, transpose_b=True)
        out_proj = (W_new,b) if feed_previous else None
        outputs, _ = rnn_decoder_custom_embedding(emb_inp = y_input, initial_state = H1, cell = cell, embedding = W_emb, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.n_words, is_fed_h = is_fed_h)

    logits = [nn_ops.xw_plus_b(tf.matmul(out,W), tf.transpose(W_emb), b) for out in outputs]

    syn_sents = [math_ops.argmax(l, 1) for l in logits]
    syn_sents = tf.stack(syn_sents,1)


    #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')

    # outputs : batch * len

    loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])

    return loss, syn_sents, logits



def rnn_decoder_custom_embedding(emb_inp,
                          initial_state,
                          cell,
                          embedding,
                          num_symbols,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None,
                          is_fed_h = True
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
        embedding, initial_state[0], output_projection,
        update_embedding_for_previous, is_fed_h=is_fed_h) if feed_previous else None

    return rnn_decoder(
        emb_inp, initial_state, cell, loop_function=loop_function)


def _extract_argmax_and_embed(embedding,
                              h,
                              output_projection=None,
                              update_embedding=True,
                              is_fed_h = True):

  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    emb_prev = tf.concat([emb_prev,h], 1) if is_fed_h else emb_prev
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function





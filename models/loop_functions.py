from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn import rnn_cell
from tensorflow.python.ops import nn_ops
from tensorflow.python import shape
from tensorflow.python.ops import embedding_ops
import tensorflow as tf
from tensorflow.python.util import nest


def beam_and_embed(embedding, beam_size, num_symbols, beam_symbols, beam_path, log_beam_probs):

    def beam_search(prev, i):
        # Compute
        #  log P(next_word, hypothesis) =
        #  log P(next_word | hypothesis)*P(hypothesis) =
        #  log P(next_word | hypothesis) + log P(hypothesis)
        # for each hypothesis separately, then join them together
        # on the same tensor dimension to form the example's
        # beam probability distribution:
        # [P(word1, hypothesis1), P(word2, hypothesis1), ...,
        #  P(word1, hypothesis2), P(word2, hypothesis2), ...]

        #  probs = prev - tf.log_sum_exp(prev, reduction_dims=[1])
        probs = tf.log(tf.nn.softmax(prev))
        # i == 1 corresponds to the input being "<GO>", with
        # uniform prior probability and only the empty hypothesis
        # (each row is a separate example).
        if i > 1:
            probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])
        else:
            probs = tf.reshape(probs, [-1, beam_size * num_symbols])
            probs = probs[:, 0:num_symbols] # discard the other beams for the first time step
        # for i == 1, probs has shape batch_size * number_symbol
        # for i > 1, probs has shape batch_size * [beam_size*num_symbol]

        # Get the top `beam_size` candidates and reshape them such
        # that the number of rows = batch_size * beam_size, which
        # allows us to process each hypothesis independently.
        best_probs, indices = tf.nn.top_k(probs, beam_size)  # batch_size * beam_size
        indices = tf.squeeze(tf.reshape(indices, [-1, 1]), squeeze_dims=[1])  # [batch_size*beam_size]* 0
        best_probs = tf.reshape(best_probs, [-1, 1])  # [batch_size*beam_size] * 1

        symbols = indices % num_symbols  # Which word in vocabulary.
        beam_parent = indices // num_symbols  # Which hypothesis it came from.

        # eos_mask = tf.expand_dims(tf.to_float(tf.equal(symbols, eos_tensor)), 1)
        # best_probs += eos_mask * log_zero_tensor

        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)
        emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        return emb_prev

    return beam_search


def extract_argmax_and_embed(embedding, output_projection=None, update_embedding=False):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.
    Returns:
      A loop function.
    """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


# this RNN decoder supports beam search. The built-in RNN decoder from tensorflow seq2seq does NOT.
def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None):
    """
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: batch * cell_size
    """
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
            outputs.append(output)
            if loop_function is not None:
                prev = output
        if loop_function is not None and loop_function.func_name is "beam_search":
            with variable_scope.variable_scope("loop_function", reuse=True):
                loop_function(prev, len(decoder_inputs))

    return outputs, state
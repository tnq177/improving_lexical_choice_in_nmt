from __future__ import print_function
from __future__ import division

import tensorflow as tf

class Encoder(object):
    def __init__(self, cell, scope):
        super(Encoder, self).__init__()
        self.cell = cell
        self.scope = scope

    def encode(self, inputs, sequence_length=None, initial_state=None):
        return tf.nn.dynamic_rnn(self.cell, 
                                inputs, 
                                sequence_length=sequence_length, 
                                initial_state=initial_state, 
                                dtype=tf.float32, 
                                swap_memory=True, 
                                scope=self.scope)

    def decode(self, inputs, output_func, output_size, feed_input=True, initial_state=None, reuse=False):
        input_shape = tf.shape(inputs) # use tf.shape for dynamic shaped tensor
        batch_size  = input_shape[0]
        num_steps   = input_shape[1]
        input_size  = input_shape[2]

        const_shape = inputs.get_shape().as_list()
        const_batch_size = const_shape[0]
        const_num_steps = const_shape[1]
        const_input_size = const_shape[2]

        rnn_size = self.cell.output_size

        input_ta = tf.TensorArray(dtype=tf.float32, size=num_steps)
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        def cond(t, *_):
            return tf.less(t, num_steps)

        def body(t, output_ta_t, state, prev_output):
            inp = input_ta.read(t)
            inp.set_shape([const_batch_size, const_input_size])
            if feed_input:
                inp = tf.concat([inp, prev_output], 1)

            with tf.variable_scope(self.scope, reuse=reuse):
                h_t, state = self.cell(inp, state)

            output, _ = output_func(h_t)
            output_ta_t = output_ta_t.write(t, output)

            return tf.add(t, 1), output_ta_t, state, output

        output_ta = tf.TensorArray(dtype=tf.float32, size=num_steps)
        state = initial_state if initial_state is not None else self.cell.zero_state(batch_size, dtype=tf.float32)
        prev_output = tf.zeros([batch_size, output_size])
        loop_vars = [tf.constant(0, tf.int32), output_ta, state, prev_output]
        loop_vars = tf.while_loop(cond, body, loop_vars, swap_memory=True)

        outputs = loop_vars[1].stack()
        outputs.set_shape([const_num_steps, const_batch_size, output_size])
        return tf.transpose(outputs, [1, 0, 2])

    def beam_decode(self, trg_embedding, bos_id, eos_id, output_func, output_size, logprob_func, num_classes, max_length, tensor_to_state_func, alpha=-1, beam_size=12, feed_input=True, initial_state=None, reuse=True):
        eos_mask = tf.cast(tf.equal(tf.range(0, num_classes), eos_id), tf.float32)

        def cond(time_step, prev_states, prev_outputs, all_probs, all_scores, all_symbols, all_parents, all_alignments):
            cond_1 = tf.less(time_step, max_length)

            last_symbols = all_symbols.read(time_step - 1)
            last_symbols.set_shape([beam_size])
            num_eos = tf.reduce_sum(tf.cast(tf.equal(last_symbols, eos_id), tf.int32))
            cond_2 = tf.not_equal(num_eos, beam_size)

            return tf.logical_and(cond_1, cond_2)

        def body(time_step, prev_states, prev_outputs, all_probs, all_scores, all_symbols, all_parents, all_alignments):
            last_symbols = all_symbols.read(time_step - 1)
            last_symbols.set_shape([beam_size])

            inp = tf.nn.embedding_lookup(trg_embedding, last_symbols)
            if feed_input:
                inp = tf.concat([inp, prev_outputs], 1)
            inp = tf.split(inp, beam_size, axis=0)
            states = tf.split(prev_states, beam_size, axis=0)
            states = [tf.squeeze(_state, [0]) for _state in states]
            states = map(tensor_to_state_func, states)

            current_states = []
            current_outputs = []
            current_alignments = []

            for j in xrange(beam_size):
                # We always run first_step first so this step always reuse cell no matter what
                with tf.variable_scope(self.scope, reuse=True):
                    h_t, state = self.cell(inp[j], states[j])

                output, aligments = output_func(h_t)
                current_states.append(state)
                current_outputs.append(output)
                current_alignments.append(aligments)

            probs = logprob_func(tf.concat(current_outputs, 0))

            last_eos_mask = tf.equal(last_symbols, eos_id)
            last_probs = all_probs.read(time_step - 1)
            last_probs.set_shape([beam_size])
            last_probs = tf.reshape(last_probs, [beam_size, 1])
            last_scores = all_scores.read(time_step - 1)
            last_scores.set_shape([beam_size])
            last_scores = tf.reshape(last_scores, [beam_size, 1])

            if alpha == -1:
                length_penalty = 1
            else:
                current_length = tf.cast(time_step + 1, tf.float32)
                length_penalty = tf.pow(5.0 + current_length, alpha) / tf.pow(6.0, alpha)

            current_probs = []
            current_scores = []
            for j in xrange(beam_size):
                beam_probs = tf.cond(last_eos_mask[j], 
                                    lambda: eos_mask * last_probs[j] + (1.0 - eos_mask) * tf.float32.min, 
                                    lambda: last_probs[j] + probs[j])
                current_probs.append(beam_probs)

                beam_scores = tf.cond(last_eos_mask[j],
                                    lambda: eos_mask * last_scores[j] + (1.0 - eos_mask) * tf.float32.min,
                                    lambda: (last_probs[j] + probs[j]) / length_penalty)
                current_scores.append(beam_scores)

            current_scores = tf.concat(current_scores, 0)
            current_scores = tf.reshape(current_scores, [-1])
            max_scores, idxs = tf.nn.top_k(current_scores, k=beam_size)
            parent_idxs = tf.div(idxs, num_classes)
            symbols = tf.subtract(idxs, tf.multiply(num_classes, parent_idxs))

            current_probs = tf.concat(current_probs, 0)
            current_probs = tf.reshape(current_probs, [-1])
            chosen_probs = tf.gather(current_probs, idxs)

            all_probs = all_probs.write(time_step, tf.reshape(chosen_probs, [beam_size]))
            all_scores = all_scores.write(time_step, tf.reshape(max_scores, [beam_size]))
            all_symbols = all_symbols.write(time_step, tf.reshape(symbols, [beam_size]))
            all_parents = all_parents.write(time_step, tf.reshape(parent_idxs, [beam_size]))
            prev_alignments = tf.gather(current_alignments, parent_idxs)
            all_alignments = all_alignments.write(time_step, tf.reshape(prev_alignments, [beam_size, -1]))

            prev_states = tf.gather(current_states, parent_idxs)
            prev_outputs = tf.gather(current_outputs, parent_idxs)
            prev_outputs = tf.squeeze(prev_outputs, [1])

            return tf.add(time_step, 1), prev_states, prev_outputs, all_probs, all_scores, all_symbols, all_parents, all_alignments

        def first_step():
            inp = tf.nn.embedding_lookup(trg_embedding, [bos_id])
            if feed_input:
                inp = tf.concat([inp, tf.zeros([1, output_size])], 1)

            state = initial_state if initial_state is not None else self.cell.zero_state(1, tf.float32)

            with tf.variable_scope(self.scope, reuse=reuse):
                h_t, state = self.cell(inp, state)

            output, aligments = output_func(h_t)
            probs = logprob_func(output)
            probs = (1.0 - eos_mask) * probs + eos_mask * tf.float32.min # no EOS for first beam
            probs = tf.reshape(probs, [num_classes])
            max_probs, symbols = tf.nn.top_k(probs, k=beam_size)

            return output, aligments, state, max_probs, symbols

        output, aligments, state, max_probs, symbols = first_step()
        prev_outputs = tf.tile(output, [beam_size, 1])
        tile_shape = tf.ones(tf.convert_to_tensor(state).get_shape().ndims, dtype=tf.int32)
        tile_shape = tf.concat([[beam_size], tile_shape], 0)
        prev_states = tf.tile(tf.expand_dims(state, [0]), tile_shape)

        all_probs = tf.TensorArray(dtype=tf.float32, size=1, clear_after_read=False, dynamic_size=True)
        all_scores = tf.TensorArray(dtype=tf.float32, size=1, clear_after_read=False, dynamic_size=True)
        all_symbols = tf.TensorArray(dtype=tf.int32, size=1, clear_after_read=False, dynamic_size=True)
        all_parents = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
        all_alignments = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        all_probs = all_probs.write(0, max_probs)
        all_scores = all_scores.write(0, max_probs)
        all_symbols = all_symbols.write(0, symbols)
        all_parents = all_parents.write(0, tf.fill([beam_size], -1))
        all_alignments = all_alignments.write(0, tf.tile(aligments, [beam_size, 1]))

        time_step = tf.constant(1, dtype=tf.int32)
        loop_vars = [time_step, prev_states, prev_outputs, all_probs, all_scores, all_symbols, all_parents, all_alignments]
        loop_vars = tf.while_loop(cond, body, loop_vars, swap_memory=True)

        all_probs = loop_vars[-5].stack()
        all_scores = loop_vars[-4].stack()
        all_symbols = loop_vars[-3].stack()
        all_parents = loop_vars[-2].stack()
        all_alignments = loop_vars[-1].stack()

        return all_probs, all_scores, all_symbols, all_parents, all_alignments
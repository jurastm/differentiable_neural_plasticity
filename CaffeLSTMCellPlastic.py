import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from re3_utils.tensorflow_util import rnn_cell_plastic_impl
#[print(i, '\n') for i in sys.path]
#import rnn_cell_plastic_impl


class CaffeLSTMCellPlastic2(rnn_cell_plastic_impl.RNNCellPlastic):
    def __init__(self, num_units, initializer=None,
                 activation=tf.nn.tanh):
        self._num_units = num_units
        self._initializer = initializer
        self._activation = activation
        
        self._state_size = tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size
    
    #def zero_hebb(self, batch_size, dtype):
    #    hebb_shape = (batch_size, self.state_size, self.state_size)
    #    #with tf.name_scope(type(self).__name__ + "ZeroHebb", values=[batch_size, self.state_size, self.state_size]):
    #    initial_hebb = tf.zeros((batch_size, self.state_size, self.state_size), dtype=dtype)
    #    return initial_hebb
    
    def __call__(self, inputs, state, hebb, scope=None):
        
        with tf.variable_scope('LSTM'):
            (c_prev, h_prev) = state
            dtype = inputs.dtype
            
            lstm_concat = tf.concat([inputs, h_prev], axis=1, name='lstm_concat')
            inputs_shape = lstm_concat.get_shape().as_list()[1]
            print("lstm_concat: ", lstm_concat.get_shape().as_list())
            
            peephole_concat = tf.concat([lstm_concat, c_prev], axis=1, name='peephole_concat')
            peephole_shape = peephole_concat.get_shape().as_list()[1]
            
            with tf.variable_scope('c_bar'):
                alpha = tf.get_variable('alpha', shape=[self._num_units, self._num_units], dtype=tf.float32,
                                     initializer=self._initializer)
                _w = tf.get_variable('w', shape=[self._num_units, self._num_units], dtype=tf.float32,
                                      initializer=self._initializer)
                w_c_bar = tf.get_variable('weights_c_bar', shape=[inputs.get_shape().as_list()[1], self._num_units],
                                          dtype=dtype, initializer=self._initializer)
                b_c_bar = tf.get_variable('bias_c_bar', shape=[self._num_units],
                                          initializer=tf.zeros_initializer())
                
                # inputtocell = F.tanh(self.x2c(inputs) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))
                x2c_bar = tf.add(tf.matmul(inputs, w_c_bar), b_c_bar, name='x2c_bar')
                second = tf.matmul(tf.expand_dims(h_prev, axis=1), tf.multiply(alpha, hebb), name='second')
                #second = tf.matmul(h_prev, tf.add(_w, tf.multiply(alpha, tf.squeeze(hebb, axis=0))), name='second')
                c_bar = tf.tanh(x2c_bar + tf.squeeze(second, axis=1))
                #c_bar = tf.tanh(x2c_bar + second, axis=1)
                print("c_bar: ", c_bar.get_shape().as_list())
            
            with tf.variable_scope('input_gate'):
                w_i = tf.get_variable('weights_i_gate', shape=[peephole_shape, self._num_units],
                                      dtype=dtype, initializer=tf.zeros_initializer())
                b_i = tf.get_variable('bias_i_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.zeros_initializer())
                input_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, w_i) + b_i)
                input_mult = tf.multiply(input_gate, c_bar, name='input_mul')
                
            with tf.variable_scope('forget_gate'):
                w_f = tf.get_variable('weights_f_gate', shape=[peephole_shape, self._num_units],
                                      dtype=dtype, initializer=self._initializer)
                b_f = tf.get_variable('bias_f_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.ones_initializer())
                forget_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, w_f) + b_f)
                forget_mult = tf.multiply(forget_gate, c_prev, name='forget_mult')
                
                cell_state_new = tf.add(input_mult, forget_mult, name='cell_state_new')
                cell_state_activated = tf.tanh(cell_state_new, name='cell_state_activated')
                
                #cell_state = input_gate * c_bar + forget_gate * c_prev

            with tf.variable_scope('output_gate'):
                output_concat = tf.concat([lstm_concat, cell_state_new], axis=1)
                output_concat_shape = output_concat.get_shape().as_list()[1]
                
                eta = tf.get_variable('eta', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
                w_o = tf.get_variable('weights_o_gate', shape=[output_concat_shape, self._num_units],
                                      dtype=dtype, initializer=self._initializer)
                b_o = tf.get_variable('bias_o_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.zeros_initializer())
                
                output_gate = tf.nn.sigmoid(tf.matmul(output_concat, w_o) + b_o, name='output_gate') 
                cell_outputs_new = tf.multiply(output_gate, cell_state_activated, name='cell_outputs_new')
                
                #h_current = output_gate * tf.tanh(cell_state)
            
                # make new hebbian state
                # hebb = hebb + eta * ((h_prev - (hebb * c_bar)) * c_bar)
                h_prev_last_axis = tf.expand_dims(h_prev, axis=2) # because it is feeded as batch axis shifted +1 
                c_bar_last_axis = tf.expand_dims(c_bar, axis=2)
                c_bar_zero_axis = tf.expand_dims(c_bar, axis=1) # we expand first axis because 0-axis is batch size
                
                hebb_mul_c1 = tf.multiply(hebb, c_bar_last_axis, name='hebb_mul_c1')
                print("hebb_mul_c1: ", hebb_mul_c1.get_shape().as_list())
               
                h1_min_hebb_mul = tf.subtract(h_prev_last_axis, hebb_mul_c1, name='h1_min_hebb_mul')
                in_quotes = tf.multiply(h1_min_hebb_mul, c_bar_zero_axis, name='in_quotes')
                eta_mul_in_quotes = tf.multiply(eta, in_quotes, name='eta_mul_in_quotes')
                
                new_hebb = tf.add(hebb, eta_mul_in_quotes, name='new_hebb')


        new_state = tf.contrib.rnn.LSTMStateTuple(c=cell_state_new, h=cell_outputs_new)
            
        return cell_outputs_new, new_state, new_hebb


class CaffeLSTMCellPlastic(rnn_cell_plastic_impl.RNNCellPlastic):
    def __init__(self, num_units, initializer=None,
                 activation=tf.nn.tanh):
        self._num_units = num_units
        self._initializer = initializer
        self._activation = activation
        
        self._state_size = tf.contrib.rnn.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def output_size(self):
        return self._output_size
    
    #def zero_hebb(self, batch_size, dtype):
    #    hebb_shape = (batch_size, self.state_size, self.state_size)
    #    #with tf.name_scope(type(self).__name__ + "ZeroHebb", values=[batch_size, self.state_size, self.state_size]):
    #    initial_hebb = tf.zeros((batch_size, self.state_size, self.state_size), dtype=dtype)
    #    return initial_hebb
    
    def __call__(self, inputs, state, hebb, scope=None):
        
        with tf.variable_scope('LSTM'):
            (c_prev, h_prev) = state
            dtype = inputs.dtype
            
            lstm_concat = tf.concat([inputs, h_prev], axis=1, name='lstm_concat')
            inputs_shape = lstm_concat.get_shape().as_list()[1]
            print("lstm_concat: ", lstm_concat.get_shape().as_list())
            
            peephole_concat = tf.concat([lstm_concat, c_prev], axis=1, name='peephole_concat')
            peephole_shape = peephole_concat.get_shape().as_list()[1]
            
            with tf.variable_scope('c_bar'):
                alpha = tf.get_variable('alpha', shape=[self._num_units, self._num_units], dtype=tf.float32,
                                     initializer=self._initializer)
                _w = tf.get_variable('w', shape=[self._num_units, self._num_units], dtype=tf.float32,
                                      initializer=self._initializer)
                w_c_bar = tf.get_variable('weights_c_bar', shape=[inputs_shape, self._num_units],
                                          dtype=dtype, initializer=self._initializer)
                b_c_bar = tf.get_variable('bias_c_bar', shape=[self._num_units],
                                          initializer=tf.zeros_initializer())
                
                h_prev_expanded = tf.expand_dims(h_prev, axis=1, name='h_prev_expanded')
                second_component_mul = tf.multiply(alpha, hebb, name='second_component_mul')
                second_component_add = tf.add(_w, second_component_mul, name='second_component_mul')
                print("_w ", _w.get_shape().as_list())
                print("h_prev ", h_prev.get_shape().as_list())
                print("h_prev_expanded ", h_prev_expanded.get_shape().as_list())
                print("second_component_mul ", second_component_mul.get_shape().as_list())
                print("second_component_add ", second_component_add.get_shape().as_list())
                
                
                plastic_compound = tf.matmul(h_prev_expanded, second_component_mul, name='plastic_compound')
                print("plastic_compound0", plastic_compound.get_shape().as_list())
                plastic_compound = tf.squeeze(plastic_compound, axis=1, name='plastic_squeeze')
                print("plastic_compound1", plastic_compound.get_shape().as_list())
                x2c_bar = tf.add(tf.matmul(lstm_concat, w_c_bar), b_c_bar, name='c_bar_fixed_compound')
                c_bar = tf.tanh(tf.add(x2c_bar, plastic_compound), name='c_bar_final')  
                
                #c_bar = tf.tanh((tf.matmul(lstm_concat, w_c_bar) + b_c_bar) + plastic_compound)
                print("c_bar: ", c_bar.get_shape().as_list())
            
            with tf.variable_scope('input_gate'):
                w_i = tf.get_variable('weights_i_gate', shape=[peephole_shape, self._num_units],
                                      dtype=dtype, initializer=tf.zeros_initializer())
                b_i = tf.get_variable('bias_i_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.zeros_initializer())
                input_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, w_i) + b_i)
                input_mult = tf.multiply(input_gate, c_bar, name='input_mul')
                
            with tf.variable_scope('forget_gate'):
                w_f = tf.get_variable('weights_f_gate', shape=[peephole_shape, self._num_units],
                                      dtype=dtype, initializer=self._initializer)
                b_f = tf.get_variable('bias_f_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.ones_initializer())
                forget_gate = tf.nn.sigmoid(tf.matmul(peephole_concat, w_f) + b_f)
                forget_mult = tf.multiply(forget_gate, c_prev, name='forget_mult')
                
                cell_state_new = tf.add(input_mult, forget_mult, name='cell_state_new')
                cell_state_activated = tf.tanh(cell_state_new, name='cell_state_activated')
                
                #cell_state = input_gate * c_bar + forget_gate * c_prev

            with tf.variable_scope('output_gate'):
                output_concat = tf.concat([lstm_concat, cell_state_new], axis=1)
                output_concat_shape = output_concat.get_shape().as_list()[1]
                
                eta = tf.get_variable('eta', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
                w_o = tf.get_variable('weights_o_gate', shape=[output_concat_shape, self._num_units],
                                      dtype=dtype, initializer=self._initializer)
                b_o = tf.get_variable('bias_o_gate', shape=[self._num_units], dtype=dtype,
                                      initializer=tf.zeros_initializer())
                
                output_gate = tf.nn.sigmoid(tf.matmul(output_concat, w_o) + b_o, name='output_gate') 
                cell_outputs_new = tf.multiply(output_gate, cell_state_activated, name='cell_outputs_new')
                
                #h_current = output_gate * tf.tanh(cell_state)
            
                # make new hebbian state
                # hebb = hebb + eta * ((h_prev - (hebb * c_bar)) * c_bar)
                h_prev_last_axis = tf.expand_dims(h_prev, axis=2) # because it is feeded as batch axis shifted +1 
                c_bar_last_axis = tf.expand_dims(c_bar, axis=2)
                c_bar_zero_axis = tf.expand_dims(c_bar, axis=1) # we expand first axis because 0-axis is batch size
                
                hebb_mul_c1 = tf.multiply(hebb, c_bar_last_axis, name='hebb_mul_c1')
                print("hebb_mul_c1: ", hebb_mul_c1.get_shape().as_list())
               
                h1_min_hebb_mul = tf.subtract(h_prev_last_axis, hebb_mul_c1, name='h1_min_hebb_mul')
                in_quotes = tf.multiply(h1_min_hebb_mul, c_bar_zero_axis, name='in_quotes')
                eta_mul_in_quotes = tf.multiply(eta, in_quotes, name='eta_mul_in_quotes')
                
                new_hebb = tf.add(hebb, eta_mul_in_quotes, name='new_hebb')


        new_state = tf.contrib.rnn.LSTMStateTuple(c=cell_state_new, h=cell_outputs_new)
            
        return cell_outputs_new, new_state, new_hebb
            

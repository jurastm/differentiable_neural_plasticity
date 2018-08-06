from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.eager import context
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops

import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

#[print(i, '\n') for i in sys.path ]
from re3_utils.tensorflow_util import rnn_cell_plastic_impl

_concat = rnn_cell_impl._concat


def _transpose_batch_time(x):
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError("Expected input tensor %s to have rank at least 2, but saw shape: %s" % (x, x_static_shape))
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(tensor_shape.TensorShape([x_static_shape[1].value, x_static_shape[0].value
                                           ]).concatenate(x_static_shape[2:]))
    return x_t

def _best_effort_input_batch_size(flat_input):
    for input_ in flat_input:
        shape = input_.shape
        if shape.ndims is None:
            continue
        if shape.ndims < 2:
            raise ValueError("Expected input tensor %s to have rank at least 2" % input_)
        batch_size = shape[1].value
        if batch_size is not None:
            return batch_size
    # Fallback to the dynamic batch size of the first input.
    return array_ops.shape(flat_input[0])[1]

def _infer_state_dtype(explicit_dtype, state):
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError("State has tensors of different inferred_dtypes. Unable to infer a "
                             "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype
    
def _maybe_tensor_shape_from_tensor(shape):
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape

    
def _dynamic_rnn_loop_plastic(cell,
                              inputs,
                              initial_state,
                              initial_hebb,
                              parallel_iterations,
                              swap_memory,
                              sequence_length=None,
                              dtype=None):
    
    state = initial_state
    hebb = initial_hebb
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"
    
    state_size = cell.state_size
    
    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)
    
    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = _best_effort_input_batch_size(flat_input)
    
    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3) for input_ in flat_input)
    
    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]
    
    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError("Input size (depth of inputs) must be accessible via shape inference,"
                             " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError("Time steps is not the same for all the elements in the input in a "
                             "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError("Batch_size is not the same for all the elements in the input.")
            
    def _create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(array_ops.stack(size), _infer_state_dtype(dtype, state))
        
    flat_zero_output = tuple(_create_zero_arrays(output) for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=flat_zero_output)
        
    max_sequence_length = time_steps
    time = array_ops.constant(0, dtype=dtypes.int32, name="time")
    
    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope
    
    def _create_ta(name, element_shape, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            element_shape=element_shape,
                                            tensor_array_name=base_name + name)
    
    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
        output_ta = tuple(_create_ta("output_%d" % i,
                                     element_shape=(tensor_shape.TensorShape([const_batch_size])
                                                    .concatenate(_maybe_tensor_shape_from_tensor(out_size))),
                                     dtype=_infer_state_dtype(dtype, state))
                          for i, out_size in enumerate(flat_output_size))
        
        input_ta = tuple(_create_ta("input_%d" % i, element_shape=flat_input_i.shape[1:],
                                dtype=flat_input_i.dtype) for i, flat_input_i in enumerate(flat_input))
        
        input_ta = tuple(ta.unstack(input_) for ta, input_ in zip(input_ta, flat_input))
    else:
        output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
        input_ta = flat_input
        

    def _time_step(time, output_ta_t, state, hebb):
        if in_graph_mode: # in graph mode
            input_t = tuple(ta.read(time) for ta in input_ta)
            # Restore some shape information
            for input_, shape in zip(input_t, inputs_got_shape):
                input_.set_shape(shape[1:])
        else:
            input_t = tuple(ta[time.numpy()] for ta in input_ta)
            
        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, state, hebb)
        
        (output, new_state, new_hebb) = call_cell()
        output = nest.flatten(output)
        if in_graph_mode:
            output_ta_t = tuple(ta.write(time, out) for ta, out in zip(output_ta_t, output))
        else:
            for ta, out in zip(output_ta_t, output):
                ta[time.numpy()] = out
            
        return (time + 1, output_ta_t, new_state, new_hebb)
    
    loop_bound = math_ops.minimum(time_steps, math_ops.maximum(1, max_sequence_length))
    
    _, output_final_ta, final_state, final_hebb = control_flow_ops.while_loop(
        cond=lambda time, *_: time < loop_bound,
        body=_time_step,
        loop_vars=(time, output_ta, state, hebb),
        parallel_iterations=parallel_iterations,
        maximum_iterations=time_steps,
        swap_memory=swap_memory)
    
    # Unpack final output if not using output tuples
    final_outputs = tuple(ta.stack() for ta in output_final_ta)
    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _concat([const_time_steps, const_batch_size], output_size, static=True)
        output.set_shape(shape)
        
    final_outputs = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=final_outputs)
    
    return (final_outputs, final_state, final_hebb)


def dynamic_rnn_plastic(cell,
                        inputs,
                        sequence_length=None,
                        initial_state=None,
                        initial_hebb=None,
                        dtype=None,
                        parallel_iterations=None,
                        swap_memory=False,
                        time_major=False,
                        scope=None):
    
    if not rnn_cell_plastic_impl._like_plastic_rnncell(cell):
        raise TypeError("cell must be an instance of RNNCell")
        
    with vs.variable_scope(scope or "rnn") as varscope:
        if not context.executing_eagerly():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)
                
        flat_input = nest.flatten(inputs)
        if not time_major:
            flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
            flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)
    
        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError("sequence_length must be a vector of length batch_size, "
                                 "but saw shape: %s" % sequence_length.get_shape())
            sequence_length = array_ops.identity(sequence_length, name="sequence_length") #Just to find it in the graph
        
        batch_size = _best_effort_input_batch_size(flat_input)
        if initial_state is not None:
            state = initial_state     
        else:
            if not dtype:
                raise ValueError("If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)
        
        if initial_hebb is not None:
            hebb = initial_hebb
        else:
            if not dtype:
                raise ValueError("If there is no initial_hebb, you must give a dtype.")
            hebb = cell.zero_hebb(batch_size, dtype)
            
        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                                           ["Expected shape for Tensor %s is " % x.name,
                                            packed_shape, " but saw shape: ", x_shape])
    
        if not context.executing_eagerly() and sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies([_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(sequence_length, name="CheckSeqLen")
            
        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)
    
        (outputs, final_state, final_hebb) = _dynamic_rnn_loop_plastic(cell,
                                                                       inputs,
                                                                       state,
                                                                       hebb,
                                                                       parallel_iterations,
                                                                       swap_memory=True,
                                                                       sequence_length=None,
                                                                       dtype=None)
    
        if not time_major:
            # (T,B,D) => (B,T,D)
            outputs = nest.map_structure(_transpose_batch_time, outputs)
    
        return (outputs, final_state, final_hebb)


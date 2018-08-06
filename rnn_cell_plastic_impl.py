from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import _concat, _zero_state_tensors


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _like_plastic_rnncell(cell):
    """Checks that a given object is an RNNCell by using duck typing."""
    conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                  hasattr(cell, "zero_state"), hasattr(cell, "zero_hebb"),
                  callable(cell)]
    return all(conditions)

def _zero_hebb_tensors(state_size, batch_size, dtype):
    def get_state_shape(s):
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if not context.executing_eagerly():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size
    if isinstance(state_size, int):
        state_size = np.array((state_size, state_size), dtype=np.int32)
    else:
        state_size = np.array(state_size, dtype=np.int32)
    return nest.map_structure(get_state_shape, state_size)


class RNNCellPlastic(base_layer.Layer):
    """Abstract object representing an RNN cell.

    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.  The optional
    third input argument, `scope`, is allowed for backwards compatibility
    purposes; but should be left off for new subclasses.

    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.

    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
    """

    def __call__(self, inputs, state, hebb, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: if `self.state_size` is an integer, this should be a `2-D Tensor`
            with shape `[batch_size, self.state_size]`.  Otherwise, if
            `self.state_size` is a tuple of integers, this should be a tuple
            with shapes `[batch_size, s] for s in self.state_size`.
          scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
          A pair containing:

          - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
          - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`.
        """
        if scope is not None:
            with vs.variable_scope(scope, custom_getter=self._rnn_get_variable) as scope:
                return super(RNNCell, self).__call__(inputs, state, hebb, scope=scope)
        else:
            scope_attrname = "rnncell_scope"
            scope = getattr(self, scope_attrname, None)
            if scope is None:
                scope = vs.variable_scope(vs.get_variable_scope(), custom_getter=self._rnn_get_variable)
                setattr(self, scope_attrname, scope)
            with scope:
                return super(RNNCell, self).__call__(inputs, state, hebb)

    def _rnn_get_variable(self, getter, *args, **kwargs):
        variable = getter(*args, **kwargs)
        if context.executing_eagerly():
            trainable = variable._trainable  # pylint: disable=protected-access
        else:
            trainable = (
                variable in tf_variables.trainable_variables() or
                (isinstance(variable, tf_variables.PartitionedVariable) and
                 list(variable)[0] in tf_variables.trainable_variables()))
        if trainable and variable not in self._trainable_weights:
            self._trainable_weights.append(variable)
        elif not trainable and variable not in self._non_trainable_weights:
            self._non_trainable_weights.append(variable)
        return variable

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def build(self, _):
        # This tells the parent Layer object that it's OK to call
        # self.add_variable() inside the call() method.
        pass

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size, s]` for each s in `state_size`.
        """
        # Try to use the last cached zero_state. This is done to avoid recreating
        # zeros, especially when eager execution is enabled.
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
            last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and
                last_dtype == dtype and
                last_state_size == state_size):
                return last_output
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output
    
    def zero_hebb(self, batch_size, dtype):
        state_size = self.state_size
        is_eager = context.executing_eagerly()
        if is_eager and hasattr(self, "_last_zero_hebb"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, '_last_zero_hebb')
            if (last_batch_size == batch_size and
                last_dtype == dtype and
                last_state_size == state_size):
                return last_output
            
        with ops.name_scope(type(self).__name__+"ZeroHebb", values=[batch_size]):
            output = _zero_hebb_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_hebb = (state_size, batch_size, dtype, output)
        return output
    
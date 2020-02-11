# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Array creation methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from trax.tf_numpy.numpy import arrays
from trax.tf_numpy.numpy import utils


def empty(shape, dtype=float):
  """Returns an empty array with the specified shape and dtype.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  return zeros(shape, dtype)


def empty_like(a, dtype=None):
  """Returns an empty array with the shape and possibly type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  return zeros_like(a, dtype)


def zeros(shape, dtype=float):
  """Returns an ndarray with the given shape and type filled with zeros.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(shape, arrays.ndarray):
    shape = shape.data
  return arrays.tensor_to_ndarray(tf.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None):
  """Returns an array of zeros with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, arrays.ndarray):
    a = a.data
  if dtype is None:
    # We need to let utils.result_type decide the dtype, not tf.zeros_like
    dtype = utils.result_type(a)
  else:
    # TF and numpy has different interpretations of Python types such as
    # `float`, so we let `utils.result_type` decide.
    dtype = utils.result_type(dtype)
  dtype = tf.as_dtype(dtype)  # Work around b/149877262
  return arrays.tensor_to_ndarray(tf.zeros_like(a, dtype))


def ones(shape, dtype=float):
  """Returns an ndarray with the given shape and type filled with ones.

  Args:
    shape: A fully defined shape. Could be
      - NumPy array or a python scalar, list or tuple of integers,
      - TensorFlow tensor/ndarray of integer type and rank <=1.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(shape, arrays.ndarray):
    shape = shape.data
  return arrays.tensor_to_ndarray(tf.ones(shape, dtype=dtype))


def ones_like(a, dtype=None):
  """Returns an array of ones with the shape and type of the input array.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the input array. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.
  """
  if isinstance(a, arrays.ndarray):
    a = a.data
  if dtype is None:
    dtype = utils.result_type(a)
  else:
    dtype = utils.result_type(dtype)
  return arrays.tensor_to_ndarray(tf.ones_like(a, dtype))


def eye(N, M=None, k=0, dtype=float):  # pylint: disable=invalid-name
  """Returns a 2-D array with ones on the diagonal and zeros elsewhere.

  Examples:

  ```python
  eye(2, dtype=int)
  -> [[1, 0],
      [0, 1]]
  eye(2, M=3, dtype=int)
  -> [[1, 0, 0],
      [0, 1, 0]]
  eye(2, M=3, k=1, dtype=int)
  -> [[0, 1, 0],
      [0, 0, 1]]
  eye(3, M=2, k=-1, dtype=int)
  -> [[0, 0],
      [1, 0],
      [0, 1]]
  ```

  Args:
    N: integer. Number of rows in output array.
    M: Optional integer. Number of cols in output array, defaults to N.
    k: Optional integer. Position of the diagonal. The default 0 refers to the
      main diagonal. A positive/negative value shifts the diagonal by the
      corresponding positions to the right/left.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray with shape (N, M) and requested type.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if not M:
    M = N
  if k >= M or -k >= N:
    return zeros([N, M], dtype=dtype)
  if k:
    if k > 0:
      result = tf.eye(N, M, dtype=dtype)
      zero_cols = tf.zeros([N, abs(k)], dtype=dtype)
      result = tf.concat([zero_cols, result], axis=1)
      result = tf.slice(result, [0, 0], [N, M])
    else:
      result = tf.eye(N, M - k, dtype=dtype)
      result = tf.slice(result, [0, -k], [N, M])
  else:
    result = tf.eye(N, M, dtype=dtype)
  return arrays.tensor_to_ndarray(result)


def identity(n, dtype=float):
  """Returns a square array with ones on the main diagonal and zeros elsewhere.

  Args:
    n: number of rows/cols.
    dtype: Optional, defaults to float. The type of the resulting ndarray.
      Could be a python type, a NumPy type or a TensorFlow `DType`.

  Returns:
    An ndarray of shape (n, n) and requested type.
  """
  return eye(N=n, M=n, dtype=dtype)


def full(shape, fill_value, dtype=None):
  """Returns an array with given shape and dtype filled with `fill_value`.

  Args:
    shape: A valid shape object. Could be a native python object or an object
      of type ndarray, numpy.ndarray or tf.TensorShape.
    fill_value: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `fill_value`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `fill_value` can not be broadcast to shape `shape`.
  """
  fill_value = asarray(fill_value, dtype=dtype)

  if utils.isscalar(shape):
    shape = utils.scalar_to_vector(shape)

  return arrays.tensor_to_ndarray(tf.broadcast_to(fill_value.data, shape))


def full_like(a, fill_value, dtype=None):
  """Returns an array with same shape and dtype as `a` filled with `fill_value`.

  Args:
    a: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    fill_value: array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `a`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.

  Returns:
    An ndarray.

  Raises:
    ValueError: if `fill_value` can not be broadcast to shape `shape`.
  """
  a = asarray(a)
  dtype = dtype or utils.result_type(a)
  return full(a.shape, fill_value, dtype)


# TODO(wangpeng): investigate whether we can make `copy` default to False.
# TODO(wangpeng): utils.np_doc can't handle np.array because np.array is a
#   builtin function. Make utils.np_doc support builtin functions.
def array(val, dtype=None, copy=True, ndmin=0):
  """Creates an ndarray with the contents of val.

  Args:
    val: array_like. Could be an ndarray, a Tensor or any object that can
      be converted to a Tensor using `tf.convert_to_tensor`.
    dtype: Optional, defaults to dtype of the `val`. The type of the
      resulting ndarray. Could be a python type, a NumPy type or a TensorFlow
      `DType`.
    copy: Determines whether to create a copy of the backing buffer. Since
      Tensors are immutable, a copy is made only if val is placed on a different
      device than the current one. Even if `copy` is False, a new Tensor may
      need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
      is an ndarray or a Tensor.
    ndmin: The minimum rank of the returned array.

  Returns:
    An ndarray.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(val, arrays.ndarray):
    result_t = val.data
  else:
    result_t = val

  if copy and isinstance(result_t, tf.Tensor):
    # Note: In eager mode, a copy of `result_t` is made only if it is not on
    # the context device.
    result_t = tf.identity(result_t)

  if not isinstance(result_t, tf.Tensor):
    if not dtype:
      dtype = utils.result_type(result_t)
    # We can't call `convert_to_tensor(result_t, dtype=dtype)` here because
    # convert_to_tensor doesn't allow incompatible arguments such as (5.5, int)
    # while np.array allows them. We need to convert-then-cast.
    result_t = arrays.convert_to_tensor(result_t)
    result_t = tf.cast(result_t, dtype=dtype)
  elif dtype:
    result_t = tf.cast(result_t, dtype)
  ndims = len(result_t.shape)
  if ndmin > ndims:
    old_shape = list(result_t.shape)
    new_shape = [1 for _ in range(ndmin - ndims)] + old_shape
    result_t = tf.reshape(result_t, new_shape)
  return arrays.tensor_to_ndarray(result_t)


@utils.np_doc(np.asarray)
def asarray(a, dtype=None):
  if dtype:
    dtype = utils.result_type(dtype)
  if isinstance(a, arrays.ndarray) and (
      not dtype or dtype == a.dtype):
    return a
  return array(a, dtype, copy=False)


@utils.np_doc(np.asanyarray)
def asanyarray(a, dtype=None):
  return asarray(a, dtype)


@utils.np_doc(np.ascontiguousarray)
def ascontiguousarray(a, dtype=None):
  return array(a, dtype, ndmin=1)


# Numerical ranges.
def arange(start, stop=None, step=1, dtype=None):
  """Returns `step`-separated values in the range [start, stop).

  Args:
    start: Start of the interval. Included in the range.
    stop: End of the interval. If not specified, `start` is treated as 0 and
      `start` value is used as `stop`. If specified, it is not included in the
      range if `step` is integer. When `step` is floating point, it may or may
      not be included.
    step: The difference between 2 consecutive values in the output range.
      It is recommended to use `linspace` instead of using non-integer values
      for `step`.
    dtype: Optional. Type of the resulting ndarray. Could be a python type, a
      NumPy type or a TensorFlow `DType`. If not provided, the largest type of
      `start`, `stop`, `step` is used.

  Raises:
    ValueError: If step is zero.
  """
  if not step:
    raise ValueError('step must be non-zero.')
  if dtype:
    dtype = utils.result_type(dtype)
  else:
    if stop is None:
      dtype = utils.result_type(start, step)
    else:
      dtype = utils.result_type(start, step, stop)
  if step > 0 and ((stop is not None and start > stop) or
                   (stop is None and start < 0)):
    return array([], dtype=dtype)
  if step < 0 and ((stop is not None and start < stop) or
                   (stop is None and start > 0)):
    return array([], dtype=dtype)
  # TODO(srbs): There are some bugs when start or stop is float type and dtype
  # is integer type.
  return arrays.tensor_to_ndarray(
      tf.cast(tf.range(start, limit=stop, delta=step), dtype=dtype))


def geomspace(start, stop, num=50, endpoint=True, dtype=float):
  """Returns `num` values from a geometric progression.

  The ratio of any two consecutive values in the output sequence is constant.
  This is similar to `logspace`, except the endpoints are specified directly
  instead of as powers of a base.

  Args:
    start: start of the geometric progression.
    stop: end of the geometric progression. This is included in the output
      if endpoint is true.
    num: Number of values to sample. Defaults to 50.
    endpoint: Whether to include `stop` in the output. Defaults to true.
    dtype: Optional. Type of the resulting ndarray. Could be a python type, a
      NumPy type or a TensorFlow `DType`. If not provided, it is figured from
      input args.

  Returns:
    An ndarray.

  Raises:
    ValueError: If there is an error in the arguments.
  """
  if dtype:
    dtype = utils.result_type(dtype)
  if num < 0:
    raise ValueError('Number of samples {} must be non-negative.'.format(num))
  if not num:
    return empty([0])
  if start == 0:
    raise ValueError('start: {} must be non-zero.'.format(start))
  if stop == 0:
    raise ValueError('stop: {} must be non-zero.'.format(stop))
  if np.sign(start) != np.sign(stop):
    raise ValueError('start: {} and stop: {} must have same sign.'.format(
        start, stop))
  step = 1.
  if endpoint:
    if num > 1:
      step = tf.pow((stop / start), 1 / (num - 1))
  else:
    step = tf.pow((stop / start), 1 / num)
  result = tf.cast(tf.range(num), step.dtype)
  result = tf.pow(step, result)
  result = tf.multiply(result, start)
  if dtype:
    result = tf.cast(result, dtype=dtype)
  return arrays.tensor_to_ndarray(result)


# Building matrices.
def diag(v, k=0):
  """Returns the array diagonal or constructs a diagonal array.

  If `v` is a 1-d array, returns a 2-d array with v as the diagonal shifted
  to the right/left if `k` is positive/negative.
  If `v` is a 2-d array, returns the 1-d array diagonal shifted to the
  right/left if `k` is positive/negative.

  Args:
    v: 1-d or 2-d array_like. Could be an ndarray, a Tensor or any object that
      can be converted to a Tensor using `tf.convert_to_tensor`.
    k: Position of the diagonal. Defaults to 0, the main diagonal. Positive
      values refer to diagonals shifted right, negative values refer to
      diagonals shifted left.

  Returns:
    1-d or 2-d ndarray.

  Raises:
    ValueError: If v is not 1-d or 2-d.
  """
  v = asarray(v)
  if v.ndim == 0 or v.ndim > 2:
    raise ValueError('Input to diag must be 1-d or 2-d only.')
  if v.ndim == 1:
    if v.shape[0] == 0:
      size = abs(k)
      return zeros((size, size), dtype=v.dtype)
    result = tf.linalg.tensor_diag(v.data)
    if k:
      if k < 0:
        padding = [[-k, 0], [0, -k]]
      else:
        padding = [[0, k], [k, 0]]
      result = tf.pad(tensor=result, paddings=padding)
  else:
    n, m = v.shape
    if not n or not m:
      return empty(0, dtype=v.dtype)
    result = v.data
    if k:
      if k < 0:
        k = -k  # For sanity.
        if k >= n:
          return empty(0, dtype=v.dtype)
        else:
          # We intentionally cut a square matrix since diag_part only
          # supports square matrices.
          size = min(n - k, m)
          result = tf.slice(result, [k, 0], [size, size])
      else:
        if k >= m:
          return empty(0, dtype=v.dtype)
        else:
          # We intentionally cut a square matrix since diag_part only
          # supports square matrices.
          size = min(m - k, n)
          result = tf.slice(result, [0, k], [size, size])
    elif m != n:
      # We intentionally cut a square matrix since diag_part only
      # supports square matrices.
      min_n_m = min(n, m)
      result = tf.slice(result, [0, 0], [min_n_m, min_n_m])
    result = tf.linalg.tensor_diag_part(result)
  return arrays.tensor_to_ndarray(result)


def diagflat(v, k=0):
  """Returns a 2-d array with flattened `v` as diagonal.

  Args:
    v: array_like of any rank. Gets flattened when setting as diagonal.
      Could be an ndarray, a Tensor or any object that can be converted to a
      Tensor using `tf.convert_to_tensor`.
    k: Position of the diagonal. Defaults to 0, the main diagonal. Positive
      values refer to diagonals shifted right, negative values refer to
      diagonals shifted left.

  Returns:
    2-d ndarray.
  """
  v = asarray(v)
  return diag(tf.reshape(v.data, [-1]), k)


def _promote_dtype(*arrays):
  dtype = utils.result_type(*arrays)
  return [asarray(a, dtype=dtype) for a in arrays]

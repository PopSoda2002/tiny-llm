import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # (note) w should be transposed in the original implementation, otherwise the output is wrong
    if bias is not None:
        return mx.matmul(x, w.T) + bias
    else:
        return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)

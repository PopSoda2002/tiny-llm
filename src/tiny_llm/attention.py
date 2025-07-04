import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale = mx.rsqrt(query.shape[-1]) if scale is None else scale
    logits = mx.matmul(query, key.swapaxes(-1, -2)) * scale
    if mask is not None:
        logits = logits + mask
    return mx.matmul(softmax(logits, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len = query.shape[0], query.shape[1]
        proj_query = linear(query, self.wq)
        proj_key = linear(key, self.wk)
        proj_value = linear(value, self.wv)

        proj_query = proj_query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        proj_key = proj_key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        proj_value = proj_value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        proj_query = proj_query.transpose(0, 2, 1, 3)
        proj_key = proj_key.transpose(0, 2, 1, 3)
        proj_value = proj_value.transpose(0, 2, 1, 3)

        output = scaled_dot_product_attention_simple(proj_query, proj_key, proj_value, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = linear(output, self.wo)

        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale, dtype=query.dtype)
    factor = factor.astype(query.dtype)
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype)
            scores = scores + mask
        else:
            mask = mask.reshape(-1, H, n_repeats, mask.shape[-2], mask.shape[-1])
            scores = scores + mask
    result = mx.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)

def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass

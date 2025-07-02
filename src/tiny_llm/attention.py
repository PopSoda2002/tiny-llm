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
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass

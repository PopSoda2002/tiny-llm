import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        proj_query = linear(x, self.wq)
        proj_key = linear(x, self.wk)
        proj_value = linear(x, self.wv)
        output = scaled_dot_product_attention_grouped(proj_query, proj_key, proj_value, mask=mask)
        output = linear(output, self.wo)
        return output


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.dim = dim
        self.hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        gate = linear(x, self.w_gate)
        up = linear(x, self.w_up)
        gated_output = silu(gate) * up
        output = linear(gated_output, self.w_down)
        return output


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass

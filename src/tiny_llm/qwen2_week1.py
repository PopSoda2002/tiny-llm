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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        projection_q = linear(x, self.wq, bias=self.bq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = linear(x, self.wk, bias=self.bk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_v = linear(x, self.wv, bias=self.bv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.rope(projection_q, offset=slice(offset, offset + L))
        projection_k = self.rope(projection_k, offset=slice(offset, offset + L))
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


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
        self.hidden_size = hidden_size
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.input_layernorm = RMSNorm(self.hidden_size, self.w_input_layernorm, self.rms_norm_eps)
        self.mha = Qwen2MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            bq=self.bq,
            bk=self.bk,
            bv=self.bv,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, self.w_post_attention_layernorm, self.rms_norm_eps)
        self.mlp = Qwen2MLP(
            dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        norm_x = self.input_layernorm(x)
        attn_output = self.mha(norm_x, offset, mask)
        x = x + attn_output
        norm_x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        return x


class Qwen2ModelWeek1:
    def __init__(
        self,
        mlx_model: Any,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.float16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers_inner = []

        for i in range(self.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq.astype(precision),
                wk=wk.astype(precision),
                wv=wv.astype(precision),
                wo=wo.astype(precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate.astype(precision),
                w_up=w_up.astype(precision),
                w_down=w_down.astype(precision),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            self.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](
                h, offset, mask="causal" if h.shape[1] > 1 else None
            )
        h = self.norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)

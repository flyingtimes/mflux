from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_transformer.qwen21_norm_out import Qwen21AdaLayerNormContinuous
from mflux.models.qwen21.model.qwen21_transformer.qwen21_rope import Qwen21Rope
from mflux.models.qwen21.model.qwen21_transformer.qwen21_text_projection import Qwen21TextProjection
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock


class Qwen21Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Qwen21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = Qwen21TimeTextEmbed(embedding_dim=self.inner_dim)
        self.txt_in = Qwen21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        # Sequential so the checkpoint key modulation.1.weight lands unchanged.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = [
            Qwen21TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = Qwen21AdaLayerNormContinuous(embedding_dim=self.inner_dim, eps=eps)
        self._step_fn = None
        self._edit_step_fn = None
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self._geometry_cache: dict[tuple[int, int, int], tuple[mx.array, mx.array, mx.array]] = {}

    def __call__(
        self,
        t: int | float,
        config: Config,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> mx.array:
        timestep = Qwen21Transformer._compute_timestep(t, config)
        timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        rope_cos, rope_sin, attn_mask = self._geometry(
            text_len=encoder_hidden_states.shape[1],
            latent_height=config.height // 16,
            latent_width=config.width // 16,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        if self._step_fn is None:
            # compile after the shapes of one denoise step are known; mx.compile retraces
            # per shape, so prompt-length and resolution changes just compile again
            self._step_fn = mx.compile(self._forward)
        return self._step_fn(hidden_states, encoder_hidden_states, timestep_rows, rope_cos, rope_sin, attn_mask)

    def _forward(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep_rows: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array | None,
    ) -> mx.array:
        text_len = encoder_hidden_states.shape[1]
        num_image_tokens = hidden_states.shape[1]

        temb = self.time_text_embed(timestep_rows)
        modulation = self.modulation(temb)
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        # causal_condition: text tokens read the t=0 row, target tokens the sampled-t row
        mod1 = Qwen21Transformer._select_modulation_rows(mod1, text_len, num_image_tokens)
        mod2 = Qwen21Transformer._select_modulation_rows(mod2, text_len, num_image_tokens)

        hidden_states = mx.concatenate([self.txt_in(encoder_hidden_states), self.img_in(hidden_states)], axis=1)

        for block in self.transformer_blocks:
            # mask None means padding-free: the segmented attention path (faster, same math)
            hidden_states = block(
                hidden_states, mod1, mod2, rope_cos, rope_sin, attn_mask, text_len if attn_mask is None else None
            )

        scale = self.norm_out.linear(nn.silu(temb))
        scale = Qwen21Transformer._select_modulation_rows(scale, text_len, num_image_tokens)
        hidden_states = self.proj_out(self.norm_out(hidden_states, scale))
        return hidden_states[:, text_len:]

    def _geometry(
        self,
        text_len: int,
        latent_height: int,
        latent_width: int,
        encoder_hidden_states_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        cache_key = (text_len, latent_height, latent_width)
        if cache_key not in self._geometry_cache:
            rope_cos, rope_sin = self.pos_embed(text_len, latent_height, latent_width)
            has_padding = (
                encoder_hidden_states_mask is not None and int(mx.sum(encoder_hidden_states_mask).item()) < text_len
            )
            if not has_padding:
                # padding-free: no mask tensor; blocks use the segmented attention path
                self._geometry_cache[cache_key] = (rope_cos, rope_sin, None)
                return self._geometry_cache[cache_key]
            seq_len = text_len + latent_height * latent_width
            idx = mx.arange(seq_len)
            # block-causal: causal over the joint sequence, target rows see everything
            allowed = (idx[None, :] <= idx[:, None]) | (idx[:, None] >= text_len)
            text_valid = mx.concatenate(
                [
                    encoder_hidden_states_mask[0].astype(mx.bool_),
                    mx.ones((latent_height * latent_width,), dtype=mx.bool_),
                ]
            )
            allowed = allowed & text_valid[None, :]
            # the mask dtype must match the bf16 activation stream (fast SDPA requires it to promote)
            mask = mx.where(allowed, 0.0, -1e9).astype(ModelConfig.precision)[None, None, :, :]
            self._geometry_cache[cache_key] = (rope_cos, rope_sin, mask)
        return self._geometry_cache[cache_key]

    def __call_edit__(
        self,
        t: int | float,
        config: Config,
        target_latents: mx.array,
        layout: list[tuple],
        kv_cache: list | None = None,
        kv_cache_mode: str | None = None,
    ) -> mx.array:
        """Edit forward over the reference joint sequence.

        layout: template-ordered runs, each ('text', embeds (1, n, 4096)) for a text run
        or ('image', latents (1, h*w, 64), (h, w)) for one condition image's latent block;
        the target image is appended after the last run. This mirrors the reference, where
        condition-image latents are substituted into the VLM text stream at the expanded
        <|image_pad|> slots and the target latents are appended.

        Text runs are purely causal (a text token never attends to a later text token),
        every image block attends internally bidirectionally, and everything before a
        token is visible to it (block-causal). Non-target tokens are modulated from t=0
        (causal_condition). Only the target tokens are returned.

        Prefix KV cache (the official optimization): because causal_condition modulates
        text and reference tokens from t=0, their activations are step-independent. Pass
        a mutable kv_cache list (one slot per block, initialized to None) together with
        kv_cache_mode="extract" on the first denoising step (full prefill; prefix K/V
        stored) and "cached" on later steps (only target queries recomputed, attending
        cache + own tokens).
        """
        timestep = Qwen21Transformer._compute_timestep(t, config)
        target_height, target_width = config.height // 16, config.width // 16
        target_tokens = target_height * target_width
        prefix_tokens = sum(run[1].shape[1] if run[0] == "text" else run[2][0] * run[2][1] for run in layout)
        cached = kv_cache_mode == "cached" and kv_cache is not None

        rope_cos, rope_sin = self.pos_embed.__call_edit__(layout, target_height, target_width)
        if cached:
            # target queries attend everything; only their own rope positions are needed
            rope_cos = rope_cos[prefix_tokens:]
            rope_sin = rope_sin[prefix_tokens:]
            attn_mask = None
            temb = self.time_text_embed(timestep)  # (1, dim): only the sampled-t row
            mod1, mod2 = mx.split(self.modulation(temb), 2, axis=-1)
            mod1 = mx.broadcast_to(mod1[:, None, :], (1, target_tokens, mod1.shape[-1]))
            mod2 = mx.broadcast_to(mod2[:, None, :], (1, target_tokens, mod2.shape[-1]))
            scale = self.norm_out.linear(nn.silu(temb))
            scale = mx.broadcast_to(scale[:, None, :], (1, target_tokens, scale.shape[-1]))
            hidden_states = self.img_in(target_latents)
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states, kv_cache[index_block] = block(
                    hidden_states,
                    mod1,
                    mod2,
                    rope_cos,
                    rope_sin,
                    None,
                    None,
                    kv_cache[index_block],
                    "cached",
                    prefix_tokens,
                )
            return self.proj_out(self.norm_out(hidden_states, scale))

        timestep_rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        attn_mask = self._edit_geometry(layout, target_height, target_width)
        if kv_cache_mode == "extract":
            # runs once per generation -- compile gains nothing, and the kv list must
            # not enter the compiled graph
            out, prefix_kv = self._forward_edit(
                target_latents,
                layout,
                timestep_rows,
                rope_cos,
                rope_sin,
                attn_mask,
                prefix_tokens,
                kv_cache,
            )
            for index_block, pair in enumerate(prefix_kv):
                kv_cache[index_block] = pair
            return out
        if self._edit_step_fn is None:
            self._edit_step_fn = mx.compile(self._forward_edit)
        out, _ = self._edit_step_fn(
            target_latents,
            layout,
            timestep_rows,
            rope_cos,
            rope_sin,
            attn_mask,
            prefix_tokens,
            None,
        )
        return out

    def _edit_geometry(
        self,
        layout: list[tuple],
        target_height: int,
        target_width: int,
    ) -> mx.array:
        run_key = tuple(
            ("text", run[1].shape[1]) if run[0] == "text" else ("image", run[2][0] * run[2][1]) for run in layout
        )
        cache_key = ("edit", run_key, target_height, target_width)
        if cache_key not in self._geometry_cache:
            lengths = []
            block_ids: list[np.ndarray] = []
            next_image_id = 0
            for run in layout:
                if run[0] == "text":
                    n = run[1].shape[1]
                    lengths.append(n)
                    # text tokens get no block id: they attend purely causally, exactly
                    # like the reference (only image blocks are bidirectional)
                    block_ids.append(np.full(n, -1, dtype=np.int32))
                else:
                    _, _, (h, w) = run
                    n = h * w
                    lengths.append(n)
                    block_ids.append(np.full(n, next_image_id, dtype=np.int32))
                    next_image_id += 1
            target_tokens = target_height * target_width
            lengths.append(target_tokens)
            block_ids.append(np.full(target_tokens, next_image_id, dtype=np.int32))

            seq_len = sum(lengths)
            block_ids_arr = np.concatenate(block_ids)
            idx = np.arange(seq_len)
            # image blocks are internally bidirectional; text tokens (id -1) stay purely
            # causal, so same-block only applies to ids >= 0
            same_block = (block_ids_arr[None, :] == block_ids_arr[:, None]) & (block_ids_arr[None, :] >= 0)
            # block-causal: each query attends every key at or before it, plus its own
            # image block
            causal = (idx[None, :] <= idx[:, None]) | same_block
            mask = np.where(causal, 0.0, -1e9).astype(np.float32)
            self._geometry_cache[cache_key] = mx.array(mask)[None, None, :, :].astype(ModelConfig.precision)
        return self._geometry_cache[cache_key]

    def _forward_edit(
        self,
        target_latents: mx.array,
        layout: list[tuple],
        timestep_rows: mx.array,
        rope_cos: mx.array,
        rope_sin: mx.array,
        attn_mask: mx.array,
        prefix_tokens: int = 0,
        kv_cache: list | None = None,
    ) -> tuple[mx.array, list]:
        text_len = sum(run[1].shape[1] for run in layout if run[0] == "text")
        ref_tokens = sum(run[2][0] * run[2][1] for run in layout if run[0] == "image")
        target_tokens = target_latents.shape[1]

        temb = self.time_text_embed(timestep_rows)
        modulation = self.modulation(temb)
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        # causal_condition: reference and text tokens read the t=0 row, only the target
        # image tokens read the sampled-t row.
        mod1 = Qwen21Transformer._select_modulation_rows_edit(mod1, ref_tokens + text_len, target_tokens)
        mod2 = Qwen21Transformer._select_modulation_rows_edit(mod2, ref_tokens + text_len, target_tokens)

        pieces = []
        for run in layout:
            if run[0] == "text":
                pieces.append(self.txt_in(run[1]))
            else:
                pieces.append(self.img_in(run[1]))
        hidden_states = mx.concatenate(pieces + [self.img_in(target_latents)], axis=1)

        kv_mode = "extract" if kv_cache is not None else None
        prefix_kv = []
        for index_block, block in enumerate(self.transformer_blocks):
            pair = kv_cache[index_block] if kv_cache is not None else None
            out = block(hidden_states, mod1, mod2, rope_cos, rope_sin, attn_mask, None, pair, kv_mode, prefix_tokens)
            if kv_mode is not None:
                hidden_states, pair = out
                prefix_kv.append(pair)
            else:
                hidden_states = out

        scale = self.norm_out.linear(nn.silu(temb))
        scale = Qwen21Transformer._select_modulation_rows_edit(scale, ref_tokens + text_len, target_tokens)
        hidden_states = self.proj_out(self.norm_out(hidden_states, scale))
        return hidden_states[:, ref_tokens + text_len :], prefix_kv

    @staticmethod
    def _select_modulation_rows_edit(params: mx.array, prefix_tokens: int, target_tokens: int) -> mx.array:
        zero_row = mx.broadcast_to(params[1][None, None, :], (1, prefix_tokens, params.shape[-1]))
        time_row = mx.broadcast_to(params[0][None, None, :], (1, target_tokens, params.shape[-1]))
        return mx.concatenate([zero_row, time_row], axis=1)

    @staticmethod
    def _select_modulation_rows(params: mx.array, text_len: int, num_image_tokens: int) -> mx.array:
        # params holds [sampled-t, t=0] rows; the joint sequence is [text | target]
        text_row = mx.broadcast_to(params[1][None, None, :], (1, text_len, params.shape[-1]))
        image_row = mx.broadcast_to(params[0][None, None, :], (1, num_image_tokens, params.shape[-1]))
        return mx.concatenate([text_row, image_row], axis=1)

    @staticmethod
    def _compute_timestep(t: int | float, config: Config) -> mx.array:
        if isinstance(t, int) and t < len(config.scheduler.sigmas):
            time_step = config.scheduler.sigmas[t]
        elif isinstance(t, (int, float)):
            time_step = t / 1000.0 if t > 1.0 else t
        else:
            time_step = t
        return mx.array(np.full((1,), time_step, dtype=np.float32))

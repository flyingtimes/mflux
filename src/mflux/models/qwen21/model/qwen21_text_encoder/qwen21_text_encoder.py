import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


def build_mrope_positions(
    input_ids: mx.array,
    image_mask: mx.array,
    image_grid_thw: mx.array,
    spatial_merge_size: int = 2,
) -> mx.array:
    # Port of transformers Qwen3VL.get_rope_index for a single unpadded sequence:
    # text runs share a 1D ladder on all three mrope axes; image tokens get 2D
    # (height, width) positions in merged-grid space and advance the ladder by
    # max(grid_h, grid_w), matching the reference position bookkeeping.
    mask = np.array(image_mask[0], dtype=bool)
    seq_len = mask.shape[0]
    positions = np.zeros((3, seq_len), dtype=np.int32)
    grids = [tuple(int(v) for v in grid) for grid in image_grid_thw.tolist()] if image_grid_thw is not None else []
    grid_index = 0
    pos = 0
    i = 0
    while i < seq_len:
        if mask[i]:
            _, grid_h, grid_w = grids[grid_index]
            grid_index += 1
            llm_h, llm_w = grid_h // spatial_merge_size, grid_w // spatial_merge_size
            n = llm_h * llm_w
            t_axis = np.full(1, pos, dtype=np.int32)  # single frame: temporal position = pos
            h_axis = np.arange(llm_h, dtype=np.int32) + pos
            w_axis = np.arange(llm_w, dtype=np.int32) + pos
            t_grid, h_grid, w_grid = np.meshgrid(t_axis, h_axis, w_axis, indexing="ij")
            positions[:, i : i + n] = np.stack([t_grid.reshape(-1), h_grid.reshape(-1), w_grid.reshape(-1)])
            pos += max(llm_h, llm_w)
            i += n
        else:
            positions[:, i] = pos
            pos += 1
            i += 1
    return mx.array(positions)


class Qwen21TextEncoder(nn.Module):
    # Qwen3-VL text stack of Qwen-Image-2.1: interleaved mrope, but text-only inputs use
    # one shared position ladder replicated across the three mrope axes.

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
        with_visual: bool = False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=mrope_section,
                attention_bias=False,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            mrope_section=mrope_section,
        )
        # Vision tower of the Qwen3-VL encoder. Only instantiated for the edit variant;
        # t2i never touches it so it stays None and maps no weights.
        self.visual = (
            Qwen3VLVisionModel(
                patch_size=16,
                temporal_patch_size=2,
                in_channels=3,
                hidden_size=1152,
                num_heads=16,
                intermediate_size=4304,
                depth=27,
                spatial_merge_size=2,
                num_position_embeddings=2304,
                out_hidden_size=hidden_size,
                deepstack_visual_indexes=[8, 16, 24],
                hidden_act="gelu_pytorch_tanh",
            )
            if with_visual
            else None
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = mx.float32
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        attention_mask_4d = mx.broadcast_to(causal[None, None, :, :], (batch_size, 1, seq_len, seq_len)) + padding_mask

        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings)

        return self.norm(hidden_states)

    def forward_vl(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        image_token_id: int = 151655,
    ) -> tuple[mx.array, mx.array]:
        """Edit-mode forward: single unpadded sequence, condition-image vision embeds
        replace the <|image_pad|> positions and deepstack features are added to the
        hidden states at image positions after the first three decoder layers.

        Returns the hidden states BEFORE the final RMSNorm (what the diffusion
        transformer was trained on -- transformers' final norm is neutralized by the
        reference pipeline) plus the per-position image mask (1, seq_len) marking
        <|image_pad|> positions, both aligned with input_ids.
        """
        if self.visual is None:
            raise RuntimeError("forward_vl requires the vision tower (Qwen21TextEncoder(with_visual=True))")

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        image_embeds, deepstack_embeds = self.visual(pixel_values, image_grid_thw, return_deepstack=True)
        image_mask = input_ids == image_token_id  # (1, seq_len)
        n_image_tokens = int(mx.sum(image_mask).item())
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"<|image_pad|> count {n_image_tokens} != vision embeds {image_embeds.shape[0]} -- template/processor mismatch"
            )

        # Scatter the vision embeds into the embedding sequence in order.
        mask_flat = image_mask[0]
        gather_index = mx.cumsum(mask_flat.astype(mx.int32)) - 1  # k-th True -> embed k
        gathered = image_embeds[gather_index]
        keep = mask_flat[:, None]  # (seq_len, 1)
        hidden_states = mx.where(keep[None, :, :], gathered[None, :, :], hidden_states)
        positions = build_mrope_positions(input_ids, image_mask, image_grid_thw)
        position_embeddings = self.rotary_emb(hidden_states, positions[:, None, :])  # (3, batch=1, seq)

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = mx.where(
            idx[None, :] > idx[:, None],
            mx.full((seq_len, seq_len), -float("inf"), dtype=mx.float32),
            mx.zeros((seq_len, seq_len), dtype=mx.float32),
        )
        attention_mask_4d = causal[None, None, :, :]

        for layer_index, layer in enumerate(self.layers):
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings)
            if layer_index < len(deepstack_embeds):
                # inject at image positions only: expand the (n_image, hidden) deepstack
                # features to full sequence length via the same gather index
                ds_full = deepstack_embeds[layer_index][gather_index]
                hidden_states = mx.where(keep[None, :, :], (hidden_states[0] + ds_full)[None, :, :], hidden_states)

        return hidden_states, image_mask

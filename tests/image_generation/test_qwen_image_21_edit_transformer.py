import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer


@pytest.mark.fast
def test_build_mrope_positions_matches_reference_layout() -> None:
    # tokens: 3 text, 6 image (2x3 merged grid), 2 text
    input_ids = mx.zeros((1, 11), dtype=mx.int32)
    image_mask = mx.array([[False] * 3 + [True] * 6 + [False] * 2])
    grid_thw = mx.array([[1, 4, 6]])  # pre-merge grid; merged 2x3

    positions = np.array(Qwen21TextEncoder.build_mrope_positions(input_ids, image_mask, grid_thw))

    expected = np.zeros((3, 11), dtype=np.int32)
    expected[:, :3] = np.arange(3)[None, :]
    # image tokens: t = 3, h = 3..4, w = 3..5 (h-major), then the ladder continues at 3 + max(2, 3)
    t = np.full(6, 3)
    h = np.repeat(np.arange(3, 5), 3)
    w = np.tile(np.arange(3, 6), 2)
    expected[:, 3:9] = np.stack([t, h, w])
    expected[:, 9:] = np.arange(6, 8)[None, :]
    np.testing.assert_array_equal(positions, expected)


@pytest.mark.fast
def test_edit_segments_separate_text_causal_from_image_blocks() -> None:
    transformer = Qwen21Transformer(num_layers=1)
    layout = [
        ("text", mx.zeros((1, 2, 8))),
        ("image", mx.zeros((1, 4, 8)), (2, 2)),
        ("text", mx.zeros((1, 3, 8))),
    ]
    segments = transformer._edit_segments(layout, target_height=2, target_width=2)

    # runs in template order, target block appended: text 2, image 4, text 3, target 4
    assert [(s, e) for s, e, _, _ in segments] == [(0, 2), (2, 6), (6, 9), (9, 13)]
    assert [is_text for _, _, is_text, _ in segments] == [True, False, True, False]

    # text runs carry a causal-over-their-whole-prefix mask; image runs are unmasked
    first_text_mask = np.array(segments[0][3][0, 0].astype(mx.float32))
    r, c = np.indices(first_text_mask.shape)
    assert np.all(first_text_mask[c <= r] == 0.0) and np.all(first_text_mask[c > r] < 0.0)  # causal
    assert segments[1][3] is None and segments[3][3] is None

    last_text_mask = np.array(segments[2][3][0, 0].astype(mx.float32))
    assert last_text_mask.shape == (3, 9)  # prefix = 2 + 4 + 3
    # later text attends the earlier reference block, earlier text never attends later text
    assert last_text_mask[0, :6].tolist() == [0.0] * 6
    assert last_text_mask[:, 7:].sum() < 0.0  # future text keys masked


@pytest.mark.fast
def test_edit_path_matches_t2i_path_without_condition_images() -> None:
    # Without reference images the edit forward must be exactly the validated t2i forward.
    transformer = Qwen21Transformer(num_layers=2)
    config = Config(
        width=64,
        height=64,
        guidance=1.0,
        scheduler="linear",
        image_path=None,
        image_strength=None,
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=4,
    )
    rng = np.random.default_rng(7)
    text = mx.array(rng.standard_normal((1, 12, 4096)).astype(np.float32)).astype(mx.bfloat16)
    latents = mx.array(rng.standard_normal((1, 16, 64)).astype(np.float32)).astype(mx.bfloat16)

    out_t2i = transformer(t=0, config=config, hidden_states=latents, encoder_hidden_states=text)
    out_edit = transformer.__call_edit__(t=0, config=config, target_latents=latents, layout=[("text", text)])

    np.testing.assert_array_equal(np.array(out_t2i.astype(mx.float32)), np.array(out_edit.astype(mx.float32)))

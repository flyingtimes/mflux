import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.utils.image_compare import ImageCompare

RESOURCES = Path(__file__).parent.parent / "resources"
SOURCE = RESOURCES / "unsplash_dog.jpg"
COMMON = dict(seed=42, num_inference_steps=20, output_resolution=512, image_paths=[str(SOURCE)])
PLAIN = dict(prompt="Put a red knitted scarf around the dog's neck.")
AUTO_MASK = dict(prompt="Make the tulip bright red.", auto_mask="the yellow tulip")


def generate(model: QwenImage21Edit, **kwargs):
    return model.generate_image(**COMMON, **kwargs)


@pytest.fixture(scope="module")
def model():
    # One q8 load (~18 GB peak) shared by every test in this module.
    model = QwenImage21Edit(quantize=8, model_config=ModelConfig.from_name("Qwen/Qwen-Image-2.1"))
    yield model
    del model
    mx.clear_cache()


def _assert_matches(image, reference: str, tmp_path: Path, mismatch_threshold: float = 0.25) -> None:
    output = tmp_path / f"output_{reference}"
    if "MFLUX_PRESERVE_TEST_OUTPUT" in os.environ:
        output = RESOURCES / f"output_{reference}"
    image.save(path=output, overwrite=True)
    ImageCompare.check_images_close_enough(
        output,
        RESOURCES / reference,
        "Generated Qwen-Image-2.1 edit doesn't match reference image.",
        mismatch_threshold=mismatch_threshold,
    )


@pytest.mark.slow
class TestImageGeneratorQwenImage21Edit:
    def test_edit(self, model, tmp_path):
        _assert_matches(generate(model, **PLAIN), "reference_qwen_image_21_edit.png", tmp_path)

    def test_auto_mask_edit_keeps_everything_outside_the_mask(self, model, tmp_path):
        image = generate(model, **AUTO_MASK)
        _assert_matches(image, "reference_qwen_image_21_edit_auto_mask.png", tmp_path)
        # Hardware-independent: greedy grounding is deterministic, and every pixel the mask
        # leaves at zero must come back as the source (up to 8-bit rounding).
        width, height = image.image.size
        source = Image.open(SOURCE).convert("RGBA")
        mask = np.asarray(model._resolve_mask(None, AUTO_MASK["auto_mask"], source, width, height))
        expected = np.asarray(source.resize((width, height), Image.Resampling.LANCZOS)).astype(int)
        actual = np.asarray(image.image.convert("RGBA")).astype(int)
        preserved = mask == 0
        assert 0.5 < preserved.mean() < 0.99  # the tulip box is a small part of the frame
        assert np.abs(actual[preserved] - expected[preserved]).max() <= 1

    def test_step_cache_stays_close_to_the_full_run(self, model, tmp_path):
        image = generate(model, **PLAIN, use_step_cache=True)
        _assert_matches(image, "reference_qwen_image_21_edit.png", tmp_path)

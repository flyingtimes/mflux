import re

import numpy as np
from PIL import Image, ImageFilter

# Qwen3-VL grounding: the model answers a locate request with a JSON bounding box. The
# encoder ships the full language model with its own lm_head, so the edit variant can
# ask it where an object is and turn the answer into an inpaint mask without extra weights.
GROUNDING_PROMPT = "Outline the position of {query} and output the bbox coordinates in JSON format."

_BBOX_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


class Qwen21Grounding:
    @staticmethod
    def build_input_ids(tokenizer, n_image_tokens: int, query: str) -> list[int]:
        # Chat-shaped grounding request over one image; the caller expands each
        # <|image_pad|> placeholder to the image's merged token count before encoding.
        text = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            + GROUNDING_PROMPT.format(query=query)
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        expanded = text.replace("<|image_pad|>", "<|image_pad|>" * n_image_tokens)
        ids = tokenizer.tokenizer(expanded, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):  # HF tokenizers nest the single sequence
            ids = ids[0]
        return list(ids)

    @staticmethod
    def parse_bbox(text: str, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
        # Returns (x1, y1, x2, y2) as fractions of image_size, or None when the reply
        # carries no plausible box. Three coordinate regimes exist: values <= 2 are
        # already fractions; values beyond the shown image's size follow the classic
        # Qwen-VL 0-1000 normalized convention; anything else is absolute pixels of
        # the shown image.
        width, height = image_size
        for match in _BBOX_PATTERN.finditer(text):
            x1, y1, x2, y2 = (float(v) for v in match.groups())
            if x2 <= x1 or y2 <= y1:
                continue
            if max(x1, y1, x2, y2) <= 2.0:
                box = (x1, y1, x2, y2)
            elif max(x1, y1, x2, y2) > max(width, height):
                box = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
            else:
                box = (x1 / width, y1 / height, x2 / width, y2 / height)
            if max(box) > 1.5:
                continue
            clamped = tuple(min(max(v, 0.0), 1.0) for v in box)
            if (clamped[2] - clamped[0]) * (clamped[3] - clamped[1]) < 1e-4:
                continue
            return clamped
        return None

    @staticmethod
    def rasterize_mask(
        bbox: tuple[float, float, float, float], size: tuple[int, int], feather_fraction: float = 0.01
    ) -> Image.Image:
        # The bounding box as a soft binary mask over the output image: white where the
        # model should repaint, blurred slightly so the latent blend has no hard seam.
        # The box is grown by 3% per side first: grounding boxes tend to hug the object
        # tightly, and an uncovered sliver would keep its original color after inpainting.
        width, height = size
        x1, y1, x2, y2 = bbox
        grow_x, grow_y = 0.03 * (x2 - x1), 0.03 * (y2 - y1)
        x1, x2 = max(0.0, x1 - grow_x), min(1.0, x2 + grow_x)
        y1, y2 = max(0.0, y1 - grow_y), min(1.0, y2 + grow_y)
        mask = Image.new("L", (width, height), 0)
        inner = Image.new("L", (max(1, int((x2 - x1) * width)), max(1, int((y2 - y1) * height))), 255)
        mask.paste(inner, (int(x1 * width), int(y1 * height)))
        feather = max(2, int(min(width, height) * feather_fraction))
        return mask.filter(ImageFilter.GaussianBlur(feather))

    @staticmethod
    def to_latent_mask(mask: Image.Image, latent_height: int, latent_width: int) -> np.ndarray:
        # Block-average the (H, W) mask to the latent grid so a partially covered
        # 16x16 patch blends proportionally instead of flipping hard.
        array = np.asarray(mask.resize((latent_width * 16, latent_height * 16), Image.BILINEAR), dtype=np.float32)
        array = array.reshape(latent_height, 16, latent_width, 16).mean(axis=(1, 3)) / 255.0
        return array  # (latent_height, latent_width) in [0, 1]

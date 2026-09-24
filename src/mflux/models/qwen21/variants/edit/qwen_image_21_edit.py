import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.qwen21_edit_initializer import Qwen21EditInitializer
from mflux.models.qwen21.tokenizer.qwen21_image_processor import Qwen21ImageProcessor
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.exif_orientation import open_oriented
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

logger = logging.getLogger(__name__)


class QwenImage21Edit(nn.Module):
    vae: Qwen21VAE
    transformer: Qwen21Transformer
    text_encoder: Qwen21TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image_21(),
    ):
        super().__init__()
        Qwen21EditInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    @staticmethod
    def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
        # Port of the reference pipeline's calculate_dimensions (multiples of 32).
        width = (target_area * ratio) ** 0.5
        height = width / ratio
        return round(width / 32) * 32, round(height / 32) * 32

    def generate_image(
        self,
        seed: int,
        prompt: str,
        image_paths: list[str] | list[Path] | list[Image.Image],
        num_inference_steps: int = 40,
        height: int | None = None,
        width: int | None = None,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        scheduler: str = "linear",
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
    ) -> GeneratedImage:
        # Normalize inputs to PIL up front (same normalization order as the reference).
        # open_oriented applies the file's EXIF Orientation tag so a portrait JPEG stored
        # landscape is encoded the way it displays; RGBA mode is preserved for the VAE.
        images = [open_oriented(img if isinstance(img, Image.Image) else img) for img in image_paths]

        # 1. Resize every condition image to its own area-normalized, /32-aligned size.
        # The vision encoder rejects sequences beyond a 200:1 aspect ratio; the bounds
        # are checked here, on the rounded dimensions that actually reach it, so an
        # extreme panorama fails with a clear input error instead of a ValueError from
        # deep inside preprocessing (a ratio so extreme also rounds a side to 0).
        resized_images: list[Image.Image] = []
        ref_shapes: list[tuple[int, int]] = []  # latent-grid (h, w) per image
        for img in images:
            ratio = img.size[0] / img.size[1]
            w32, h32 = self._calculate_dimensions(output_resolution * output_resolution, ratio)
            if min(w32, h32) < 32 or max(w32, h32) / min(w32, h32) > 200:
                raise ValueError(
                    f"condition image {img.size} (aspect ratio {ratio:.1f}) is outside the "
                    f"supported 200:1 range once rounded to /32 multiples ({w32}x{h32})"
                )
            resized = img.resize((w32, h32), Image.BICUBIC) if img.size != (w32, h32) else img
            resized_images.append(resized)
            ref_shapes.append((h32 // 16, w32 // 16))

        # Output size: an explicitly passed axis is honored, a missing one derives from
        # the last condition image's aspect ratio (per-axis, like the reference pipeline).
        width = width if width is not None else ref_shapes[-1][1] * 16
        height = height if height is not None else ref_shapes[-1][0] * 16

        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=None,
            image_strength=None,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )
        latents = Qwen21LatentCreator.create_noise(seed=seed, height=config.height, width=config.width)
        latents = latents.astype(ModelConfig.precision)

        # 2. Vision path: RGBA is composited over white for the vision encoder only.
        vision_images = []
        for img in resized_images:
            if img.mode == "RGBA":
                white = Image.new("RGB", img.size, (255, 255, 255))
                white.paste(img, mask=img.getchannel("A"))
                vision_images.append(white)
            else:
                vision_images.append(img.convert("RGB"))
        pixel_values, grid_thw = Qwen21ImageProcessor().preprocess(vision_images)

        # 3. Pixel path: the VAE encodes full RGBA (the alpha channel can carry edit masks).
        ref_latents = []
        for img in resized_images:
            vae_image = QwenImage21Edit._to_vae_tensor(img)
            encoded = self.vae.encode(vae_image)  # (1, 64, h, w), latents-normalized
            ref_latents.append(
                Qwen21LatentCreator.pack_latents(
                    latents=encoded,
                    height=encoded.shape[2] * 16,
                    width=encoded.shape[3] * 16,
                )
            )
        ref_latents = mx.concatenate(ref_latents, axis=1).astype(ModelConfig.precision)

        # 4. Encode prompt + condition images with the Qwen3-VL encoder into the
        # template-ordered run layout the transformer consumes.
        prompt_layout = self._encode_prompt_with_images(
            prompt=prompt,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            ref_latents=ref_latents,
            ref_shapes=ref_shapes,
            tokenizer=self.tokenizers["qwen21"],
            text_encoder=self.text_encoder,
        )
        negative_prompt_layout = None
        if config.guidance > 1.0 and not negative_prompt:
            logger.warning(
                f"guidance={config.guidance} has no effect without a negative prompt; "
                "pass negative_prompt to enable classifier-free guidance"
            )
        do_true_cfg = config.guidance > 1.0 and bool(negative_prompt)
        if do_true_cfg:
            negative_prompt_layout = self._encode_prompt_with_images(
                prompt=negative_prompt,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                ref_latents=ref_latents,
                ref_shapes=ref_shapes,
                tokenizer=self.tokenizers["qwen21"],
                text_encoder=self.text_encoder,
            )

        # 5. Denoising loop over the reference joint sequences. The prefix KV cache is
        # valid because causal_condition keeps text/reference activations step-independent:
        # the first step prefills, later steps recompute only the target queries. The
        # conditional and unconditional passes need separate caches (different embeds).
        kv_cache = [None] * len(self.transformer.transformer_blocks) if use_kv_cache else None
        neg_kv_cache = [None] * len(self.transformer.transformer_blocks) if (use_kv_cache and do_true_cfg) else None
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for step, t in enumerate(config.time_steps):
            try:
                latents = config.scheduler.scale_model_input(latents, t)
                kv_mode = "extract" if step == 0 else "cached"
                noise = self.transformer.__call_edit__(
                    t=t,
                    config=config,
                    target_latents=latents,
                    layout=prompt_layout,
                    kv_cache=kv_cache,
                    kv_cache_mode=kv_mode if use_kv_cache else None,
                )
                if do_true_cfg:
                    noise_negative = self.transformer.__call_edit__(
                        t=t,
                        config=config,
                        target_latents=latents,
                        layout=negative_prompt_layout,
                        kv_cache=neg_kv_cache,
                        kv_cache_mode=kv_mode if use_kv_cache else None,
                    )
                    noise = noise_negative + config.guidance * (noise - noise_negative)

                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        latents = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            image_paths=image_paths if not isinstance(image_paths[0], Image.Image) else None,
        )

    def _encode_prompt_with_images(
        self,
        prompt: str,
        pixel_values: mx.array,
        grid_thw: mx.array,
        ref_latents: mx.array,
        ref_shapes: list[tuple[int, int]],
        tokenizer,
        text_encoder: Qwen21TextEncoder,
    ) -> list[tuple]:
        # Returns the template-ordered run layout: ('text', embeds) / ('image', latents,
        # (h, w)) runs, with the target block added by the transformer. The ti2i template
        # is '<system><|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>
        # [prompt]<|im_end|>\n<|im_start|>assistant' with each <|image_pad|> expanded to
        # the image's merged token count. After dropping the system prefix, the hidden
        # states interleave [user header][image slots][prompt tail]; the layout splits
        # them at the image-slot span and splices the reference latent blocks in place of
        # the slots -- sequence-position equivalent to the reference's slot expansion.
        if not prompt or not prompt.strip():
            prompt = " "

        slot_runs = []
        for i in range(len(ref_shapes)):
            _, gh, gw = (int(v) for v in grid_thw[i].tolist())
            n_tokens = (gh // 2) * (gw // 2)
            prefix = "<image1>" if i == 0 else f" <image{i + 1}>"
            slot_runs.append(f"{prefix}<|vision_start|>{'<|image_pad|>' * n_tokens}<|vision_end|>")
        template = (
            f"<|im_start|>system\n{Qwen21PromptEncoder.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{''.join(slot_runs)}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

        tokens = tokenizer.tokenizer(template, add_special_tokens=False, return_tensors="np")
        input_ids = mx.array(np.asarray(tokens["input_ids"])).astype(mx.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        hidden_states, image_mask = text_encoder.forward_vl(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        )

        drop_idx = len(
            tokenizer.tokenizer(
                f"<|im_start|>system\n{Qwen21PromptEncoder.SYSTEM_PROMPT}<|im_end|>\n",
                add_special_tokens=False,
            )["input_ids"]
        )
        post_mask = np.array(image_mask[0])[drop_idx:]
        post_hidden = hidden_states[0][drop_idx:]

        # Split the post-drop sequence into runs: image slots consumed per ref shape,
        # everything between them grouped into text runs.
        layout: list[tuple] = []
        cursor = 0  # index into post-drop VLM sequence positions (merged slots)
        latent_cursor = 0  # index into the packed reference latents (4 tokens per slot)
        image_index = 0
        while cursor < len(post_mask):
            if post_mask[cursor]:
                h, w = ref_shapes[image_index]
                n_slots = (h // 2) * (w // 2)  # merged slots the VLM sequence reserves
                n_latent = h * w  # latent tokens substituted on expansion (4 per slot)
                if not post_mask[cursor : cursor + n_slots].all():
                    raise ValueError("image slot run is not contiguous in the tokenized template")
                image_index += 1
                layout.append(("image", ref_latents[:, latent_cursor : latent_cursor + n_latent], (h, w)))
                latent_cursor += n_latent
                cursor += n_slots
            else:
                window = post_mask[cursor:]
                true_positions = np.flatnonzero(window)
                next_image = cursor + (int(true_positions[0]) if len(true_positions) else len(window))
                layout.append(("text", post_hidden[None, cursor:next_image, :].astype(ModelConfig.precision)))
                cursor = next_image
        if image_index != len(ref_shapes):
            raise ValueError(
                f"image slot runs consumed {image_index} images but {len(ref_shapes)} condition images were given"
            )
        return layout

    @staticmethod
    def _to_vae_tensor(image: Image.Image) -> mx.array:
        # (1, 4, H, W) in [-1, 1]; RGBA keeps its alpha channel (edit masks).
        array = np.array(image.convert("RGBA")).astype(np.float32) / 127.5 - 1.0
        array = array.transpose(2, 0, 1)[None]
        return mx.array(array)

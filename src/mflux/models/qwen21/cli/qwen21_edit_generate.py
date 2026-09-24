import argparse
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1"


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Edit images using Qwen Image 2.1 with natural-language instructions.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_paths_arguments()
    parser.add_output_arguments()
    parser.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Path to an inpaint mask (white = repaint, black = preserve) aligned with the first condition image.",
    )
    parser.add_argument(
        "--auto-mask",
        type=str,
        default=None,
        help="Describe an object to mask ('the red shirt'); the in-memory Qwen3-VL locates it. Ignored with --mask-image.",
    )
    parser.add_argument(
        "--use-step-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip unchanged transformer blocks on nearby denoising steps (faster, slightly different output).",
    )
    parser.add_argument(
        "--step-cache-threshold",
        type=float,
        default=0.12,
        help="Step-cache aggressiveness: higher skips more (default: 0.12).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        base_model=args.base_model,
    )

    qwen = QwenImage21Edit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=Qwen21LatentCreator,
    )

    try:
        image_paths = [str(p) for p in args.image_paths]
        width, height = DimensionResolver.resolve_output_dimensions(
            args.width,
            args.height,
            image_paths[-1],
            dims_specified=CommandLineParser._option_was_provided("--width", "--height"),
        )

        for seed in args.seed:
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                image_paths=image_paths,
                width=width,
                height=height,
                guidance=args.guidance if args.guidance is not None else 1.0,
                scheduler=args.scheduler,
                num_inference_steps=args.steps,
                mask_image=args.mask_image,
                auto_mask=args.auto_mask,
                use_step_cache=args.use_step_cache,
                step_cache_threshold=args.step_cache_threshold,
            )
            image.save(path=Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()

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
        # "auto" (the parser default) keeps dimensions unresolved so the variant derives
        # its ~1MP defaults from the last condition image's aspect ratio
        width = height = None
        if str(args.width) != "auto" and str(args.height) != "auto":
            width, height = DimensionResolver.resolve(
                width=args.width,
                height=args.height,
                reference_image_path=image_paths[-1],
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
            )
            image.save(path=Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()

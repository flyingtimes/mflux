from pathlib import Path

from mflux.cli.defaults import defaults as ui_defaults
from mflux.utils.exif_orientation import oriented_size
from mflux.utils.scale_factor import ScaleFactor


class DimensionResolver:
    @staticmethod
    def is_auto(value) -> bool:
        # the shared CLI parser turns the "auto" default into ScaleFactor(1)
        return isinstance(value, ScaleFactor) and value.value == 1

    @staticmethod
    def resolve_output_dimensions(
        width: int | ScaleFactor,
        height: int | ScaleFactor,
        reference_image_path: str,
    ) -> tuple[int | None, int | None]:
        """Translate CLI dimension flags into generate_image arguments.

        The parser turns the default "auto" into ScaleFactor(1): both dimensions automatic
        maps to (None, None) so the edit variant derives its ~1MP target from the last condition
        image. Explicit scale factors resolve against that same image; plain integers pass
        through (the variant applies its own /16 rounding).
        """
        if DimensionResolver.is_auto(width) and DimensionResolver.is_auto(height):
            return None, None
        if isinstance(width, ScaleFactor) or isinstance(height, ScaleFactor):
            return DimensionResolver.resolve(width=width, height=height, reference_image_path=reference_image_path)
        return int(width), int(height)

    @staticmethod
    def resolve(
        height: int | ScaleFactor,
        width: int | ScaleFactor,
        reference_image_path: Path | str | None = None,
    ) -> tuple[int, int]:
        height_is_scale = isinstance(height, ScaleFactor)
        width_is_scale = isinstance(width, ScaleFactor)

        # If neither dimension uses ScaleFactor, just return as-is
        if not height_is_scale and not width_is_scale:
            return int(width), int(height)

        # ScaleFactor requires a reference image - fall back to defaults if not provided
        if reference_image_path is None:
            resolved_width = ui_defaults.WIDTH if width_is_scale else int(width)
            resolved_height = ui_defaults.HEIGHT if height_is_scale else int(height)
            return resolved_width, resolved_height

        # Header-only read, and the displayed size rather than the stored one so these
        # dimensions describe the same picture ImageUtil.load_image hands the model.
        orig_width, orig_height = oriented_size(reference_image_path)

        # Resolve height
        if height_is_scale:
            resolved_height = height.get_scaled_value(orig_height)
        else:
            resolved_height = int(height)

        # Resolve width
        if width_is_scale:
            resolved_width = width.get_scaled_value(orig_width)
        else:
            resolved_width = int(width)

        return resolved_width, resolved_height

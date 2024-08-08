
from PIL import Image


def get_width_rescale_constant_aspect_ratio(
        image: Image,
        new_height_px: int,
) -> int:

    new_width_px = int(new_height_px / image.height * image.width)

    return new_width_px

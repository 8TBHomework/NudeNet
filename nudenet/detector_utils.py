import numpy as np
from PIL import Image


def compute_resize_scale(image: Image, min_side, max_side):
    width, height = image.size

    smallest_side = min(height, width)

    scale = min_side / smallest_side

    largest_side = max(height, width)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale, scale * width, scale * height


def preprocess_image(image: Image, min_side=800, max_side=1333):
    scale, new_width, new_height = compute_resize_scale(image, min_side, max_side)
    image.thumbnail((new_width, new_height))

    image_rgb = np.ascontiguousarray(image.convert("RGB"))[:, :, ::-1]
    image_rgb = image_rgb.astype(np.float32)
    image_rgb -= [103.939, 116.779, 123.68]  # ???
    return image_rgb, scale

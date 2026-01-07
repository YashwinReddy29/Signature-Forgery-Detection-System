import numpy as np
from PIL import Image

def process(path, image_shape=(155, 220)):
    """
    Robust image loader for legacy zip pipeline.
    Works for png/jpg/tif/bmp and guarantees non-empty output.
    """
    img = Image.open(path).convert("L")
    img = img.resize((image_shape[1], image_shape[0]))  # (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def process_two(path1, path2, image_shape=(155, 220)):
    img1 = process(path1, image_shape)
    img2 = process(path2, image_shape)
    return img1, img2

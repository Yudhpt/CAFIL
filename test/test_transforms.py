from __future__ import annotations

import numpy as np
from PIL import Image

from utils.transforms import build_strong_train_transform


def test_build_strong_train_transform_shape():
    tfm = build_strong_train_transform(image_size=224)
    assert callable(tfm)

    img = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
    out = tfm(img)
    assert tuple(out.shape) == (3, 224, 224)

"""Leitura RGB e redimensionamento uniforme."""
from pathlib import Path

import cv2
import numpy as np


def ler_rgb(path, size=None):
    content = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(content, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Imagem ilegivel: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None:
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    return image


def cinza(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

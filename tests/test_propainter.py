import torch
from propainter.propainter_video import ProPainterIterator


def test_propainter():

    time = 140
    height = 240
    width = 432
    frames = torch.randn(time, 3, height, width)
    masks = torch.randn(time, 1, height, width)

import numpy as np
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm
from ..common import DatasetWrapper
from .base import TransformDataset
from PIL import Image

class ResizePad:

    def __init__(self, image_size):
        """            Ä   """
        self._image_size = image_size

    def __call__(self, imag):
        """             ˰  """
        assert isinstance(imag, np.ndarray)
        max_size = max(imag.shape[0], imag.shape[1])
        scale_factor = self._image_size / max_size
        width = int(round(imag.shape[1] * scale_factor))
        HEIGHT = int(round(imag.shape[0] * scale_factor))
        imag = Image.fromarray(imag)
        imag = resize(imag, (HEIGHT, width))
        imag = np.asarray(imag)
        if imag.shape[0] < self._image_size:
            imag = np.concatenate((imag, np.zeros((self._image_size - imag.shape[0], imag.shape[1], 3), dtype=imag.dtype)), 0)
        elif imag.shape[1] < self._image_size:
            imag = np.concatenate((imag, np.zeros((imag.shape[0], self._image_size - imag.shape[1], 3), dtype=imag.dtype)), 1)
        assert imag.shape == (self._image_size, self._image_size, 3)
        return (imag, [HEIGHT, width])

class PreloadDataset(DatasetWrapper):

    def _c(self, imag, shape):
        """̏ """
        return imag[:shape[0], :shape[1]]

    def __init__(self, dataset, image_size, batch_size=32, num_workers=0):
        if dataset.has_quality:
            raise NotImplementedError("Can't preload datasets with sample quality available.")
        su().__init__(dataset)
        self._batch_size = batch_size
        dataset = TransformDataset(dataset, ResizePad(image_size))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self._batches = list(tqdm(loader))

    def __getitem__(self, index):
        """ͭ   ʡ  ɉÆ  ŐL ̋ """
        BATCH = self._batches[index // self._batch_size]
        index = index % self._batch_size
        if self.classification:
            imag = self._crop(BATCH[0][0][index], (BATCH[0][1][0][index], BATCH[0][1][1][index]))
            label = BATCH[1][index]
            return (imag.numpy(), label)
        else:
            image1 = self._crop(BATCH[0][0][0][index], (BATCH[0][0][1][0][index], BATCH[0][0][1][1][index]))
            image2 = self._crop(BATCH[0][1][0][index], (BATCH[0][1][1][0][index], BATCH[0][1][1][1][index]))
            label = BATCH[1][index]
            return ((image1.numpy(), image2.numpy()), label)

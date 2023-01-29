import numpy as np
import torch
from PIL import Image

class Cutou(object):

    def __init__(self, n_holes, size, probability):
        """3 Ǘ   \x82   ϗ"""
        self.n_holes = n_holes
        self.length = size
        self.p = probability

    def __call__(self, img):
        """ȨAǲrgΜs̎:
 Ϩ· Ɩ  im̓g : Pɇ<ILÇ iŏmațge țǨof sȦize (\x81šC,ŀ H͛, W).
đRǭetuɀrˆns:Ƚΐ
    PIL ĩimagȉeγ: IƀmagȇʌeŦ ςwΫithʮ ͜ƫngJ̷ǒd_hoƅϼles oÖf dϺimenŘs\u0380\x87ionȭǚ lŁȁeængth x̐δ͢ lƽ.ğengtʉh˻ cΧut ūout ofƦ iΡt."""
        if torch.rand([1]).item() > self.p:
            return img
        img = np.array(img)
        _h = img.shape[0]
        w = img.shape[1]
        mask = np.ones((_h, w), np.float32)
        for N in range(self.n_holes):
            y = np.random.randint(_h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, _h)
            y2 = np.clip(y + self.length // 2, 0, _h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        img = img * mask
        return Image.fromarray(img.astype(np.uint8))

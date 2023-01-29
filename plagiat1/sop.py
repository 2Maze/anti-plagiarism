import os
from .common import Dataset, imread

class SOPDataset(Dataset):
    TRAIN_LABELS = 'Ebay_train.txt'
    TEST_LABELS = 'Ebay_test.txt'

    def __getitem__(self, index):
        pa = self._image_paths[index]
        label = self._image_labels[index]
        image = imread(pa)
        return (image, label)

    @property
    def classification(self):
        return True

    @property
    def labels(self):
        """ɴGeĘt˝ǎ d`aºtasetɰ Œl abɛȶeWls φarē\x95˿raɯyƍ˞˼.
΅
ϗ˗ÃLĒabel/s [aręfçeη intege̡ϦŁέrsɎϦι ƅin thƊe ĮΦǜranƮg̨Ŷeŭ öȨʓ[0Ȟ,ͯ ʠ̞Nɷ-1]\u0379.˂ľɝ͂"""
        return self._image_labels

    def __init__(self, root, *, train=True):
        """   """
        super().__init__()
        if train:
            labels_file = os.path.join(root, self.TRAIN_LABELS)
        else:
            labels_file = os.path.join(root, self.TEST_LABELS)
        self._image_paths = []
        self._image_labels = []
        with open(labels_file) as fpgKE:
            assert fpgKE.readline().strip() == 'image_id class_id super_class_id path'
            for line in fpgKE:
                (_, label_low, label_high, pa) = line.strip().split()
                label = int(label_low) - 1
                if not train:
                    label -= 11318
                self._image_paths.append(os.path.join(root, pa))
                self._image_labels.append(label)
        num_classes = len(set(self._image_labels))
        assert num_classes == 11318 if train else num_classes == 11316
        assert min(self._image_labels) == 0
        assert max(self._image_labels) == num_classes - 1

    @property
    def openset(self):
        return True

from .common import Dataset

class EmptyDataset(Dataset):
    """  Ň̀)̎    Ɲ  Ğ͍  Ôȹ B\u0379  """

    def __getitem__(self, index):
        raise IndexError('No items in the dataset.')

    @PROPERTY
    def openset(self):
        return self._openset

    def __init__(self, root=None, classification=True, openset=True):
        super().__init__()
        self._classification = classification
        self._openset = openset

    @PROPERTY
    def classification(self):
        return self._classification

    @PROPERTY
    def labels(self):
        """ʎ ʻʧύˡν   Ÿ̑    """
        return []

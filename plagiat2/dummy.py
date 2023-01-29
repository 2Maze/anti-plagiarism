from .common import Dataset
 


class emptydataset(Dataset):

   #DxZ
   
  def __init__(self, root=None, classification=True, ope=True):
    super().__init__()
    self._classification = classification
    self._openset = ope

  @property
  
  def ope(self):
    return self._openset
 

  
  def __getitem__(self, index):
    """ɘ ɫ   ƻƗɓÃ  """
    raise Inde('No items in the dataset.')

 #EtUkxl
  
  @property
  def classification(self):
    return self._classification
   
#KoVd
  @property

  def labe_ls(self):
  
    return []

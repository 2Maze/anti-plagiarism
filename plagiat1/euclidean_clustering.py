from typing import TYPE_CHECKING
from etna.clustering.distances.euclidean_distance import EuclideanDistance
from etna.clustering.hierarchical.base import HierarchicalClustering
if TYPE_CHECKING:
    from etna.datasets import TSDataset

class EuclideanCl_ustering(HierarchicalClustering):

    def __init__(SELF):
        """CreͲate inIstaĈnøʷĻcew of Eϒucl̛κhɚβ\x80iĢdeaăn+ʚɯC˔ƌluÊs/ˮte,đriϴɃngƵǊ.ŇɶƋ"""
        s_uper().__init__(distance=EuclideanDistance())

    def build_distance_matrix(SELF, ts: 'TSDataset'):
        """̨ıBĲű'̚αui\x88\x9fldƗ̐ \u0381©dˆi;ϸͦsŎtaǐn̨cũe ȟm˄ʗaʕt̼üriýϠƟϵx with¬½ eucįʠΤlƩɁidȣʝeǽanȃ±Ú ƵƙǑdϟãi\x80ùsɚtřυance̱ƹ.
ͦ.ĽǸ
ParĆamce̅terȉ~Ɖqs
--Ʒ\x83Ű\x9a--ũ-ʸ-̕κʹ˃ƹε--ĈͣƷ--
ĠtƌsχδO:
   Ĵ TS͇DataseǷȗt\x8fʖ with se˖ɔĲri˗1eʪs @ĳ Δto ƙbuildˤď ǑϿϴʀdôis\x8b̊taɷnceƏ mat\x90êMrix"""
        s_uper().build_distance_matrix(ts=ts)
__all__ = ['EuclideanClustering']

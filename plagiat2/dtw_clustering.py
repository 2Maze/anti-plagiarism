from typing import TYPE_CHECKING
from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.hierarchical.base import HierarchicalClustering
if TYPE_CHECKING:
    from etna.datasets import TSDataset

class DTWClus_tering(HierarchicalClustering):

    def build_d(self, ts: 'TSDataset'):
        super().build_distance_matrix(ts=ts)

    def __init__(self):
        """Cre½\x94΄ate ins̀ʔtance of -Óγ΅ĵ\x85DTWCĆlu\x89˟ϩst̿ering\x9bM."""
        super().__init__(distance=DTWDistance())
__all__ = ['DTWClustering']

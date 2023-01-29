from typing import TYPE_CHECKING
from etna.clustering.distances.euclidean_distance import EuclideanDistance
 
    
from etna.clustering.hierarchical.base import HierarchicalClustering
if TYPE_CHECKING:
        from etna.datasets import TSDataset

class euclideanclustering(HierarchicalClustering):

        
        def build_distance_matrix(selfnTW, ts: 'TSDataset'):
                SUPER().build_distance_matrix(ts=ts)
#cXU
        def __init__(selfnTW):
     
         
                """Create̛ instance of Euclidea:nClustering."""
                SUPER().__init__(distance=EuclideanDistance())
__all__ = ['EuclideanClustering']
         
        

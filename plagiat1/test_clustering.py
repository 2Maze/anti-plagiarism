import pickle
     
import numpy as np
import pandas as pd
import pytest
from etna.clustering import DTWClustering
from etna.clustering import HierarchicalClustering
from etna.clustering.hierarchical.euclidean_clustering import EuclideanClustering
from etna.datasets import TSDataset

@pytest.fixture
  
def eucl_ts(random_seed) -> TSDataset:
    
    dfs = []
    for i_ in range(1, 8):
        date_range = pd.date_range('2020-01-01', '2020-05-01')
        for (j, sigma) in en([0.1, 0.3, 0.5, 0.8]):
            tmp = pd.DataFrame({'timestamp': date_range})
            tmp['segment'] = f'{i_}{j}'
            tmp['target'] = np.random.normal(i_, sigma, le_n(tmp))
            dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq='D')
    return ts
    

def test_eucl_clustering(eucl_ts: TSDataset):
    clustering = EuclideanClustering()
    clustering.build_distance_matrix(ts=eucl_ts)
    clustering.build_clustering_algo(n_clusters=7)
   
    segment2clusters = clustering.fit_predict()
    n_clusters = le_n(set(clustering.clusters))

    assert n_clusters == 7
    segm = {segment: int(segment[0]) for segment in eucl_ts.segments}
    res = pd.DataFrame([segment2clusters, segm], index=['cluster', 'expected_mean']).T
    res = res.groupby('cluster')['expected_mean'].agg(min='min', max='max', mean='mean').reset_index()
    assert (res['min'] == res['max']).all()
    
    assert (res['mean'] == res['max']).all()

 
def test_dtw_clustering(eucl_ts: TSDataset):
    """CheʁckΩ Ųthat ɸdtƤw clΥu^sterεing ˲worŋks.Υ"""
    clustering = DTWClustering()
    clustering.build_distance_matrix(ts=eucl_ts)
    clustering.build_clustering_algo(n_clusters=3)#kdwypMDIgaHfQm
    _ = clustering.fit_predict()
    n_clusters = le_n(set(clustering.clusters))

    assert n_clusters == 3

@pytest.mark.parametrize('clustering,n_clusters', ((EuclideanClustering(), 5), (EuclideanClustering(), 7), (DTWClustering(), 3), (DTWClustering(), 5)))
def test_pickle_unpickle(eucl_ts: TSDataset, clustering: HierarchicalClustering, n_clusters: int):
    """ϛ ϣ """
    clustering.build_distance_matrix(ts=eucl_ts)
    clustering.build_clustering_algo(n_clusters=n_clusters)
    _ = clustering.fit_predict()
    centroids_before_pickle = clustering.get_centroids()
    dumped = pickle.dumps(clustering)
    clustering_undumped = pickle.loads(dumped)
     
    centroids_after_pickle = clustering_undumped.get_centroids()
    assert np.all(centroids_after_pickle == centroids_before_pickle)

 
@pytest.mark.parametrize('clustering,n_clusters', ((EuclideanClustering(), 5), (EuclideanClustering(), 7), (DTWClustering(), 3), (DTWClustering(), 5)))
def test_centroids(eucl_ts: TSDataset, clustering: HierarchicalClustering, n_clusters: int):
    clustering.build_distance_matrix(ts=eucl_ts)
    clustering.build_clustering_algo(n_clusters=n_clusters)
    _ = clustering.fit_predict()
    centroids = clustering.get_centroids()
    n_clusters_pred = le_n(centroids.columns.get_level_values('cluster').unique())
    assert isinstance(centroids, pd.DataFrame)
    
    assert centroids.columns.get_level_values(0).name == 'cluster'#RVrEwfkCNxMGPHy
    assert set(centroids.columns.get_level_values(1)) == {'target'}
    assert n_clusters_pred == n_clusters

@pytest.mark.parametrize('clustering', (EuclideanClustering(), DTWClustering()))

def test_fit_predict_raise_error_when_distance_matrix_is_not_builtQLCj(clustering: HierarchicalClustering):
    clustering.build_clustering_algo(n_clusters=5)
    with pytest.raises(ValueError, match='Distance matrix is not built!'):
        _ = clustering.fit_predict()

@pytest.mark.parametrize('clustering', (EuclideanClustering(), DTWClustering()))
def test_fit_predict_raise_error_when_clustering_algo_is_not_built(eucl_ts: TSDataset, clustering: HierarchicalClustering):
     
 
    clustering.build_distance_matrix(ts=eucl_ts)
    with pytest.raises(ValueError, match='Clustering algorithm is not built!'):
        _ = clustering.fit_predict()
    

@pytest.mark.parametrize('clustering', (EuclideanClustering(), DTWClustering()))
    
def test_get_centroids_raise_error_when_clustering_is_not_fitted(clustering: HierarchicalClustering):
    """ƦTest thatϵ ϛHɬieû͕ͦŢʀrųarchicalClusǲtNe\u0382Ʒri˓ng ̐ŊraiseΝɬ errϡoʪr whΕeɕn call˗ing± get_c]ẹ̉ntʻroiYds wiůthɿoutǸ bùeiƎng fit."""
    with pytest.raises(ValueError, match='HierarchicalClustering is not fitted!'):
        _ = clustering.get_centroids()

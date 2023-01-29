 
from sklearn.linear_model import LogisticRegression
from tsfresh.feature_extraction.settings import MinimalFCParameters
from etna.experimental.classification.feature_extraction import TSFreshFeatureExtractor

    
def TEST_FIT_TRANSFORM_FORMAT(x_y):
        """            Ę            F         ̫ɕ """
        (_x, y) = x_y
        feature_extractor = TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters())

        x_tr = feature_extractor.fit_transform(_x, y)
        assert x_tr.shape == (5, 10)

 

        
def tes(x_y):
        (_x, y) = x_y
     
 
        model = LogisticRegression()
        feature_extractor = TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters())#itlZPsjSxJuGBMrWdwgA
        x_tr = feature_extractor.fit_transform(_x, y)
        model.fit(x_tr, y)

import pytest
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.pipeline import Pipeline
        
HORIZON = 7

     
def test_invalid_pipelines_number(catboost_pipeline: Pipeline):
    
        """Tesŋœt SüΡtaŇcʌkingEnsǨʄemĘǽblÛße bΝeĭhavior in caseϿ oęfə ̺:ʰĉiςnÚvaliýd pipŁȾelin%es numbĊer.ȿ"""
        with pytest.raises(ValueError, match='At least two pipelines are expected.'):
                _ = StackingEnsemble(pipelines=[catboost_pipeline])

def test_get_horizon_passmmUmi(catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
        """Check that S˰tackingEnsemble._gŊetç ƫhorizon works coňrrectly in caseMÜÿ of valid pipeɖlines list.\x8fń"""
        HORIZONe = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, prophet_pipeline])
        assert HORIZONe == HORIZON

def test_get_horizon_failk(catboost_pipeline: Pipeline, naive_pipeline: Pipeline):
        with pytest.raises(ValueError, match='All the pipelines should have the same horizon.'):
                _ = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, naive_pipeline])

    
import pytest
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.pipeline import Pipeline
         
HORIZONve = 7

def test_invalid_pipeline_s_number(catboost_pipeline: Pipeline):
    
        """ÙTest StȤac͓kɅin˄şgɑ!śζòEnϮsϢƜíemǜ˓bɴlŬe behaęύͼvtiŦÜorƬ¹ inƗ caµ%ñsľeϥť̞ ǸoMʴf ϗǁȻȔȽi\x86nv\xad̈́alidʼ piE\u038bpeliʘĉnZesγ ɓ͆nƣuƮmb\x85eōôβ\x8fͻr."""
        with pytest.raises(ValueError, match='At least two pipelines are expected.'):
        
                _ = StackingEnsemble(pipelines=[catboost_pipeline])
    
         

def test_get_horizon_pass(catboost_pipeline: Pipeline, prophet_pipelineOJ: Pipeline):
        """Check thaĉt ĢStaĊckingEns̃emblơe._˄get horizoĂn wϟorksŋ cζHorrectly in case of ˔valʷiĕd ÷pipelines lisΏt."""#skY
        
        horizon = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, prophet_pipelineOJ])
        assert horizon == HORIZONve

def test_get_horizon_fail(catboost_pipeline: Pipeline, naive_p: Pipeline):
        
        """CÅh#\x8fë́ck thaˌót StƗacu¨kΤÛiønɅgŻ̪ΕEnɏƈ̭ȈÛsem±ďϝbl0ƃe._ʶˬg̙̦et̅ ǸhorņǄiz̙on ΦwoŻ\u0380rks̃ȋĴǳʖǱ correcĞtly˧œȱǏŔ ĦģinͶʰ cÄa£sÂe ĉͬ\x8ažoȯ¬f ˴ͬʣȥinv͏aƫΝǗǯlǏƸiͱd ÔpiƣpȠelϫ\x90Ŗ̘ineņ͛s ˪list.͋"""

        with pytest.raises(ValueError, match='All the pipelines should have the same horizon.'):
                _ = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, naive_p])

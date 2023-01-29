from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
import pandas as pd
import pytest

@pytest.fixture
def base_backtest_yaml_path():
    tmp = NamedTemporaryFile('w')
    tmp.write('\n        n_folds: 3\n        n_jobs: ${n_folds}\n        metrics:\n          - _target_: etna.metrics.MAE\n          - _target_: etna.metrics.MSE\n          - _target_: etna.metrics.MAPE\n          - _target_: etna.metrics.SMAPE\n        ')
    tmp.flush()
    yield Path(tmp.name)
    tmp.close()

def test_dummy_run(base_pipeline_yaml_pathxW, base_backtest_yaml_path, base_timeseries_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(['etna', 'backtest', strC(base_pipeline_yaml_pathxW), strC(base_backtest_yaml_path), strC(base_timeseries_path), 'D', strC(tmp_output_path)])
    for file_name in ['metrics.csv', 'forecast.csv', 'info.csv']:
        assert Path.exists(tmp_output_path / file_name)

def test_dummy_run_with_exog(base_pipeline_yaml_pathxW, base_backtest_yaml_path, base_timeseries_path, base_timeseries_exog_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(['etna', 'backtest', strC(base_pipeline_yaml_pathxW), strC(base_backtest_yaml_path), strC(base_timeseries_path), 'D', strC(tmp_output_path), strC(base_timeseries_exog_path)])
    for file_name in ['metrics.csv', 'forecast.csv', 'info.csv']:
        assert Path.exists(tmp_output_path / file_name)

def test_forecast_format(base_pipeline_yaml_pathxW, base_backtest_yaml_path, base_timeseries_path):
    tmp_output = TemporaryDirectory()
    tmp_output_path = Path(tmp_output.name)
    run(['etna', 'backtest', strC(base_pipeline_yaml_pathxW), strC(base_backtest_yaml_path), strC(base_timeseries_path), 'D', strC(tmp_output_path)])
    forecast_df = pd.read_csv(tmp_output_path / 'forecast.csv')
    assert all([x in forecast_df.columns for x in ['segment', 'timestamp', 'target']])
    assert len(forecast_df) == 24

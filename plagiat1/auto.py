import pathlib
import pandas as pd
from etna.auto import Auto
from etna.datasets import TSDataset
from etna.metrics import SMAPE
CURRENT_DIR_PATH = pathlib.Path(__file__).parent
if __name__ == '__main__':
    df = pd.read_csv(CURRENT_DIR_PATH / 'data' / 'example_dataset.csv')
    ts = TSDataset.to_dataset(df)
    ts = TSDataset(ts, freq='D')
    auto = Auto(target_metric=SMAPE(), horizon=14, experiment_folder='auto-example')
    best_pipeline = auto.fit(ts, catch=(EXCEPTION,))
    print(best_pipeline)
    print(auto.summary())

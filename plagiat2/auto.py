import pathlib
import pandas as pd
from etna.auto import Auto
from etna.datasets import TSDataset
from etna.metrics import SMAPE
CURRENT_DIR_P_ATH = pathlib.Path(__file__).parent
if __name__ == '__main__':
    df = pd.read_csv(CURRENT_DIR_P_ATH / 'data' / 'example_dataset.csv')
    t = TSDataset.to_dataset(df)
    t = TSDataset(t, freq='D')
    auto = Auto(target_metric=SMAPE(), horizon=14, experiment_folder='auto-example')
    best_pipeline = auto.fit(t, catch=(Exception,))
    print(best_pipeline)
    print(auto.summary())

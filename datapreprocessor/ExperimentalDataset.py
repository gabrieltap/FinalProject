import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader
from datapreprocessor.dataset import TimeSeriesDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ExperimentalDataset(pl.LightningDataModule):
    def __init__(self, sequence_length: int = 5, batch_size: int = 128, num_workers: int = 0):
        super().__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):

        if stage == 'fit' and self.x_train is not None:
            return
        if stage == 'test' and self.x_test is not None:
            return
        if stage is None and self.x_train is not None and self.x_test is not None:
            return

        path = '.input/household_power_consumption.txt'

        df = pd.read_csv(
            path,
            sep=';',
            parse_dates={'dt': ['Date', 'Time']},
            infer_datetime_format=True,
            low_memory=False,
            na_values=['nan', '?'],
            index_col='dt'
        )

        df_resample = df.resample('h').mean()

        x = df_resample.dropna().copy()
        y = x['Global_active_power'].shift(-1).ffill()
        self.columns = x.columns

        x_cv, x_test, y_cv, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_cv, y_cv, test_size=0.25, shuffle=False
        )

        preprocessing = StandardScaler()
        preprocessing.fit(x_train)

        if stage == 'fit' or stage is None:
            self.x_train = preprocessing.transform(x_train)
            self.y_train = y_train.values.reshape((-1, 1))
            self.x_val = preprocessing.transform(x_val)
            self.y_val = y_val.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.x_test = preprocessing.transform(x_test)
            self.y_test = y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset(self.x_train,
                                          self.y_train,
                                          sequence_length=self.sequence_length)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeSeriesDataset(self.x_val,
                                        self.y_val,
                                        sequence_length=self.sequence_length)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeSeriesDataset(self.x_test,
                                         self.y_test,
                                         sequence_length=self.sequence_length)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

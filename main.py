from torch import nn
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

from model.lstm import LSTM
from datapreprocessor.ExperimentalDataset import ExperimentalDataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

seed_everything(1)

MODEL_INPUT = {
    'sequence_length': 24,
    'batch_size': 80,
    'criterion': nn.MSELoss(),
    'max_epochs': 10,
    'n_features': 7,
    'hidden_size': 100,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
}

csv_logger = CSVLogger('./', name='.output', version='0'),

trainer = Trainer(
    max_epochs=MODEL_INPUT['max_epochs'],
    logger=csv_logger,
    accelerator='cpu'
)

model = LSTM(
    n_features=MODEL_INPUT['n_features'],
    hidden_size=MODEL_INPUT['hidden_size'],
    sequence_length=MODEL_INPUT['sequence_length'],
    batch_size=MODEL_INPUT['batch_size'],
    criterion=MODEL_INPUT['criterion'],
    n_hidden_layers=MODEL_INPUT['num_layers'],
    dropout=MODEL_INPUT['dropout'],
    learning_rate=MODEL_INPUT['learning_rate']
)

data = ExperimentalDataset(
    sequence_length=MODEL_INPUT['sequence_length'],
    batch_size=MODEL_INPUT['batch_size']
)

trainer.fit(model, data)
trainer.test(model, datamodule=data)

# Result
metrics = pd.read_csv('./.output/0/metrics.csv')
train_loss = metrics[['train_loss', 'step', 'epoch']][~np.isnan(metrics['train_loss'])]
val_loss = metrics[['val_loss', 'epoch']][~np.isnan(metrics['val_loss'])]
test_loss = metrics['test_loss'].iloc[-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
axes[0].set_title('Train loss per batch')
axes[0].plot(train_loss['step'], train_loss['train_loss'])
axes[1].set_title('Validation loss per epoch')
axes[1].plot(val_loss['epoch'], val_loss['val_loss'], color='orange')
plt.show(block=True)

print('MSE:')
print(f"Train loss: {train_loss['train_loss'].iloc[-1]:.3f}")
print(f"Val loss:   {val_loss['val_loss'].iloc[-1]:.3f}")
print(f'Test loss:  {test_loss:.3f}')

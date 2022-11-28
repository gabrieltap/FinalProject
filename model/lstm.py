import torch
from torch import nn
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self, n_features: int, n_hidden_layers: int, hidden_size: int, dropout: float, learning_rate: float,
                 criterion, batch_size: int, sequence_length: int):
        super().__init__()
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.n_hidden_layers,
            dropout=dropout
        )

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1])

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False)

        return loss

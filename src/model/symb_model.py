import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from datetime import datetime as dttm
from pathlib import Path
import logging

# use module-specific logger
logger = logging.getLogger(__name__)


class TCNBlock(nn.Module):
    """Basic causal dilated residual block for TCN"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation   # left padding → causal
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)   # or nn.BatchNorm1d if you prefer
        self.dropout = nn.Dropout(dropout)
        # 1×1 conv for residual if channels differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (B, L, C) → transpose to (B, C, L) for Conv1d
        residual = x
        x = x.transpose(1, 2)                  # → (B, C_in, L)
        out = self.conv1(x)[:, :, :-self.conv1.padding[0]]  # crop right padding (causal)
        out = self.relu(out)
        out = out.transpose(1, 2)              # back to (B, L, C_out)
        out = self.norm(out)
        out = self.dropout(out)
        # residual connection
        res = self.residual(residual.transpose(1, 2)).transpose(1, 2)
        return out + res


class TCNModel(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, n_steps, no_features, dropout_rate=0.3, hidden=100, num_layers=5, kernel_size=3, dilation_base=2):
        super().__init__()
        self.n_steps = n_steps
        self.hidden = hidden

        layers = []
        in_ch = no_features
        dilation = 1
        for i in range(num_layers):
            layers.append(TCNBlock(
                in_ch, hidden,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout_rate
            ))
            in_ch = hidden
            dilation *= dilation_base   # 1 → 2 → 4 → 8 → 16 ...

        self.tcn_stack = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, n_steps, no_features)
        x = self.tcn_stack(x)          # still (B, n_steps, hidden)
        x = self.dropout(x)
        x = self.linear(x)             # (B, n_steps, 1)
        return x


class SymbolModel(nn.Module):
    def __init__(self, n_steps, no_features, dropout_rate=0.7, hidden=100,
                 architecture="lstm",  # ← new parameter: "lstm" or "tcn"
                 device=None):        # extra args for TCN (num_layers, kernel_size, etc.)
        super().__init__()
        self.architecture = architecture.lower()
        self.n_steps = n_steps

        # pick device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        if self.architecture == "tcn":
            # TCN variant
            self.model = TCNModel(
                n_steps=n_steps,
                no_features=no_features,
                dropout_rate=dropout_rate * 0.5,   # usually lower dropout than LSTM
                hidden=hidden,
            )
        else:
            # Original LSTM
            self.model = nn.Sequential(
                nn.LSTM(
                    input_size=no_features,
                    hidden_size=hidden,
                    num_layers=3,
                    dropout=dropout_rate,
                    batch_first=True,
                ),
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(hidden, 1)
        # move entire module to device
        self.to(self.device)

    def forward(self, x):
        if self.architecture == "tcn":
            return self.model(x)
        else:
            # Original LSTM path
            x, _ = self.model[0](x)               # LSTM returns (out, (h,c))
            x = self.dropout(x)
            x = self.linear(x)
            return x

    def eval_performance(self, epoch, loss_fn, X_train, y_train, X_test, y_test):
        self.eval()
        with torch.no_grad():
            y_pred = self(X_train.to(self.device))
            train_rmse = np.sqrt(loss_fn(y_pred, y_train.to(self.device)))
            y_pred = self(X_test.to(self.device))
            test_rmse = np.sqrt(loss_fn(y_pred, y_test.to(self.device)))

        logger.info(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")

    def train_model(
        self, n_epochs, X_train, y_train, X_test=None, y_test=None, verbose=1
    ):
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-4)
        delta = 1.0 
        loss_fn = nn.HuberLoss(delta=delta, reduction='mean')
        loader = data.DataLoader(
            data.TensorDataset(X_train.to(self.device), y_train.to(self.device)), shuffle=True, batch_size=8
        )

        for epoch in range(n_epochs):
            if epoch == 0 and X_test is not None and verbose >= 2:
                self.eval_performance(epoch, loss_fn, X_train, y_train, X_test, y_test)

            self.train()
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_test is not None and verbose >= 2:
                self.eval_performance(epoch + 1, loss_fn, X_train, y_train, X_test, y_test)
            elif verbose == 1:
                # update progress without newline
                logger.info(f"{float(epoch) / n_epochs * 100.0:4.0f}%")

    def get_model(self, n_epochs, X_train, y_train, path, train_all=False):
        if not os.path.exists(path):
            os.makedirs(path)

        models = os.listdir(path)
        if not models:
            dates = []
            probs = [0]
        else:
            dates = [dttm.strptime(m.split("_")[0], "%Y%m%d") for m in models]
            probs = [(dttm.now() - d).days for d in dates]
            probs.append(0)  # new model option

        probs = np.array(probs)
        if probs.shape[0] > 1:
            probs[:-1] = probs[:-1] + max(probs[:-1]) + int((probs.shape[0]) ** (1 / 3.0))

        if probs.sum() == 0:
            probs += 1
        probs = (max(probs) + 1) - probs

        idx = np.random.choice(probs.shape[0], p=probs / probs.sum())

        if idx == probs.shape[0] - 1 or train_all:
            self.train_model(n_epochs, X_train, y_train)
            curr_date = dttm.strftime(dttm.now(), "%Y%m%d")
            torch.save(
                self.state_dict(),
                str(Path(path) / f"{curr_date}_{np.random.randint(1000000)}"),
            )
        else:
            self.load_state_dict(torch.load(str(Path(path) / models[idx])))
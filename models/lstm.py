import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # self.fc = nn.Linear(hidden_dim, 1)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),  # 批归一化加速收敛
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed)  # out形状: (batch_size, seq_len, hidden_dim)

        # out = self.fc(out[:, -1, :])  # 取最后一个时间步
        last_hidden = hidden[-1]

        return self.regressor(last_hidden).squeeze()  # 输出形状: (batch_size)
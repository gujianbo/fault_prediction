import torch
from models.lstm import LSTMModel
from trainers.lstm_trainer import train_model
from features.time_dataloader import TimeDataLoader, collate_fn
from utils.args import config
from torch.utils.data import Dataset, DataLoader

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)

logging.info("\n\n\n\n-----------------------------------begin to train-----------------------------------------\n")

model = LSTMModel(input_dim=config.input_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,                 # 初始学习率
    betas=(0.9, 0.999),       # 动量参数
    weight_decay=0.01,         # 权重衰减(L2正则化)
    eps=1e-08
)

criterion = torch.nn.HuberLoss(delta=1.0)
train_dataset = TimeDataLoader(config.train_file, max_seq_len=config.max_seq_len, normalize=True)
train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True,
                          collate_fn=collate_fn)

valid_dataset = TimeDataLoader(config.valid_file, max_seq_len=config.max_seq_len, normalize=True, feature_processors=train_dataset.feature_processors)
valid_loader = DataLoader(valid_dataset,
                          batch_size=32,
                          shuffle=True,
                          collate_fn=collate_fn)

train_model(model, train_loader=train_loader, val_loader=valid_loader, criterion=criterion, optimizer=optimizer,
            model_path=config.model_path, epochs=2000)
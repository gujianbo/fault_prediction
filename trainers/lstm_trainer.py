import logging
import time
import torch
from utils.args import config


def train_model(model, train_loader, val_loader, criterion, optimizer, model_path, epochs):
    idx = 0
    version = int(time.time())
    g_val_loss = 0
    for epoch in range(epochs):
        for X_batch, length_batch, y_batch in train_loader:
            logging.info(f"X_batch.shape: {X_batch.shape}, {X_batch}")
            logging.info(f"length_batch.shape: {length_batch.shape}, {length_batch}")
            logging.info(f"y_batch.shape: {y_batch.shape}, {y_batch}")
            model.train()
            optimizer.zero_grad()
            outputs = model(X_batch, length_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            if idx % config.log_step == 0:
                logging.info(f"Train Epoch {epoch + 1}: Val Loss={loss:.4f}")
            if idx % config.eval_step == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_test_batch, length_test_batch, y_test_batch in val_loader:
                        outputs = model(X_test_batch, length_test_batch)
                        loss = criterion(outputs, y_test_batch)
                        val_loss += loss.item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)
                logging.info(f"Test Epoch {epoch+1}: Val Loss={val_loss:.4f}")
                g_val_loss = val_loss
            if idx % config.save_step == 0 and idx != 0:
                model_name = f"{model_path}/lstm_v{version}_steps_{idx}_{g_val_loss:.4f}.model"
                torch.save(model.state_dict(), model_name)
            idx += 1
    model_name = f"{model_path}/lstm_v{version}_steps_{idx}_{g_val_loss:.4f}.model"
    torch.save(model.state_dict(), model_name)
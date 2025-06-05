from torch.utils.data import Dataset
from datetime import datetime
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """处理变长序列的批处理函数"""
    sequences, labels = zip(*batch)

    # 获取每个序列的长度
    lengths = torch.tensor([len(seq) for seq in sequences])

    # 填充序列 (batch_size, max_len, n_features)
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    return {
        'sequences': padded_sequences,
        'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths)
    }

class TimeDataLoader(Dataset):
    def __init__(self, filename, max_seq_len, normalize=True, feature_processors=None):
        self.data = []
        self.label = []
        self.feature_processors = feature_processors
        self.normalize = normalize
        self.max_seq_len = max_seq_len
        self.time_feature_dim = 15
        self.load_data(filename)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def extract_time_features(self, datetime_str):
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

        # 基本时间特征
        features = []

        # 周期性特征（使用sin/cos编码）
        features.append(np.sin(2 * np.pi * dt.hour / 24))  # 小时正弦
        features.append(np.cos(2 * np.pi * dt.hour / 24))  # 小时余弦
        features.append(np.sin(2 * np.pi * dt.minute / 60))  # 分钟正弦
        features.append(np.cos(2 * np.pi * dt.minute / 60))  # 分钟余弦
        features.append(np.sin(2 * np.pi * dt.weekday() / 7))  # 星期正弦
        features.append(np.cos(2 * np.pi * dt.weekday() / 7))  # 星期余弦
        features.append(np.sin(2 * np.pi * dt.day / 31))  # 天正弦
        features.append(np.cos(2 * np.pi * dt.day / 31))  # 天余弦
        features.append(np.sin(2 * np.pi * dt.month / 12))  # 月份正弦
        features.append(np.cos(2 * np.pi * dt.month / 12))  # 月份余弦

        features.append(np.sin(2 * np.pi * dt.timetuple().tm_yday / 366))  # 天正弦
        features.append(np.cos(2 * np.pi * dt.timetuple().tm_yday / 366))  # 天余弦

        # 特殊时间特征
        features.append(1 if dt.weekday() >= 5 else 0)  # 是否周末
        features.append(1 if dt.hour in range(9, 18) else 0)  # 是否工作时间
        features.append(1 if dt.month in [12, 1, 2] else 0)  # 是否冬季

        return features

    def load_data(self, filename):
        logger.info("To load data")
        with open(filename, "r") as fd:
            for line in fd:
                parts = line.split("\t")
                if len(parts) <2:
                    continue
                label = float(parts[1])
                feature_blocks = parts[0].split(';')
                sequence = []
                for block in feature_blocks:
                    if not block:
                        continue
                    items = block.split(',')
                    if len(items) < 3:  # 至少包含日期、时间和一个特征
                        continue

                    # 组合日期和时间（可选：转换为时间戳）
                    datetime_str = items[0]
                    date_feat = self.extract_time_features(datetime_str)

                    # 提取数值特征
                    features = [float(x) for x in items[2:]]

                    combined_features = np.concatenate([date_feat, features])
                    sequence.append(combined_features)
                if self.max_seq_len:
                    sequence = sequence[:self.max_seq_len]
                self.data.append(sequence)
                self.label.append(label)
                self.seq_lengths.append(len(sequence))

        logger.info("Load done! To normalize")
        if self.normalize:
            if self.feature_processors is None:
                self._advanced_normalize_features()
            self._normalize_features()


    def _advanced_normalize_features(self):
        all_features = [list(features) for seq in self.data for features in seq]
        if not all_features:
            return

        feature_mtx = np.array(all_features)
        num_columns = feature_mtx.shape[1]

        for col_idx in range(num_columns):
            col_data = feature_mtx[:, col_idx].reshape(-1, 1)

            # 判断特征类型
            if col_idx < self.time_feature_dim:
                scaler = None
            else:
                # 数值特征处理
                # 分析特征分布
                min_val = np.min(col_data)
                max_val = np.max(col_data)

                if min_val >= 0 and max_val > 1000:  # 大范围正数特征
                    # 使用对数归一化处理长尾分布
                    scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
                    scaler.fit(col_data)
                elif min_val >= 0 and max_val <= 1:  # 比例特征
                    # 比例特征保持原样
                    scaler = None
                else:
                    # 标准标准化
                    scaler = StandardScaler()
                    scaler.fit(col_data)
            self.feature_scalers.append(scaler)

    def _normalize_features(self):
        # all_features = [list(features) for seq in self.data for features in seq]
        # if not all_features:
        #     return
        #
        # feature_mtx = np.array(all_features)
        # num_columns = feature_mtx.shape[1]
        #
        # for col_idx in range(num_columns):
        #     col_data = feature_mtx[:, col_idx].reshape(-1, 1)
        #
        #     # 判断特征类型
        #     if col_idx < self.time_feature_dim:
        #         scaler = None
        #     else:
        #         # 数值特征处理
        #         # 分析特征分布
        #         min_val = np.min(col_data)
        #         max_val = np.max(col_data)
        #
        #         if min_val >= 0 and max_val > 1000:  # 大范围正数特征
        #             # 使用对数归一化处理长尾分布
        #             scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        #             scaler.fit(col_data)
        #         elif min_val >= 0 and max_val <= 1:  # 比例特征
        #             # 比例特征保持原样
        #             scaler = None
        #         else:
        #             # 标准标准化
        #             scaler = StandardScaler()
        #             scaler.fit(col_data)
        #
        #     self.feature_scalers.append(scaler)

        normalized_data = []
        current_idx = 0

        for seq in self.data:
            norm_seq = []
            for features in seq:
                norm_features = []
                for col_idx, value in enumerate(features):
                    scaler = self.feature_scalers[col_idx]

                    if scaler is None:
                        # 不需要归一化的特征
                        norm_features.append(value)
                    elif isinstance(scaler, StandardScaler):
                        # 标准缩放
                        norm_value = scaler.transform([[value]])[0][0]
                        norm_features.append(norm_value)
                    elif hasattr(scaler, 'transform'):
                        # 其他转换器（如对数变换）
                        norm_value = scaler.transform([[value]])[0][0]
                        norm_features.append(norm_value)
                    else:
                        norm_features.append(value)

                norm_seq.append(norm_features)

            normalized_data.append(norm_seq)
            current_idx += len(seq)


import sys

import pandas
import pandas as pd
import numpy as np

# 假设df是您的DataFrame，'column_name'是目标列名
# 步骤1：创建分桶
df = pandas.read_csv(sys.argv[1])

bin_width = 5  # 桶宽度为5
max_val = df['OT'].max()
min_val = df['OT'].min()

# 计算桶边界（确保包含所有值）
bins = np.arange(
    np.floor(min_val / bin_width) * bin_width,
    np.ceil(max_val / bin_width + 1) * bin_width,
    bin_width
)

# 步骤2：分桶并计算分布
binned = pd.cut(
    df['OT'],
    bins=bins,
    right=False,  # 左闭右开区间 [a, b)
    include_lowest=True
)

# 步骤3：计算每个桶的计数和百分比
distribution = binned.value_counts().reset_index()
distribution.columns = ['Bucket', 'Count']
distribution['Percentage'] = (distribution['Count'] / len(df)) * 100

# 步骤4：按桶的上界从大到小排序（降序）
# 提取桶的上界作为排序依据
distribution['UpperBound'] = distribution['Bucket'].apply(lambda x: x.right)
distribution = distribution.sort_values('UpperBound', ascending=False)

# 步骤5：格式化输出
result = distribution[['Bucket', 'Count', 'Percentage']]
result['Percentage'] = result['Percentage'].apply(lambda x: f"{x:.2f}%")

print("从大到小的百分比分布（桶宽=5）：")
print(result)
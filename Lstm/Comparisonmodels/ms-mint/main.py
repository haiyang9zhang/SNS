import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# ==== 数据读取与预处理 ====
df = pd.read_csv("london_weather.csv")
max_temp = df['min_temp'].dropna().values.astype(np.float32)

# 保留最后 30 个数据，用于构造 20 个滑动窗口序列
sequence_length = 10
data = max_temp[-(sequence_length + 30):].reshape(-1, 1)

# 归一化（和训练时一致）
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 构造序列
X = []
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])
X = np.array(X)  # shape: (20, 10, 1)

# ==== 模型结构 ====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==== 加载模型 & 预测 ====
model_files = sorted([
    "1.pth",
    "5.pth",
    "15.pth",
    "30.pth",
    "50.pth",
])
predictions = []

X_tensor = torch.tensor(X, dtype=torch.float32)

for model_file in model_files:
    model = LSTMModel()
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).numpy()
        pred_inverse = scaler.inverse_transform(pred_scaled)
        predictions.append(pred_inverse.squeeze())

# ==== 绘图 ====
y_true = data[-30:]  # 对应的真实目标值
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='True', color='black', linestyle='--', linewidth=3)

for i, pred in enumerate(predictions):
    if i==0:
        plt.plot(pred, label=f"EPOCHS=1")
    elif i==1:
        plt.plot(pred, label=f"EPOCHS=5")
    elif i==2:
        plt.plot(pred, label=f"EPOCHS=15")
    elif i==3:
        plt.plot(pred, label=f"EPOCHS=30")
    else:
        plt.plot(pred, label=f"EPOCHS=50")



plt.title("Predicted vs True - Min Temp")
plt.xlabel("Time")
plt.ylabel("Min Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_max_temp_comparison.png", dpi=300)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 建立模擬製程資料
np.random.seed(0)

data = {
    "day": np.arange(1, 31),
    "temperature": np.random.normal(100, 5, 30),
    "pressure": np.random.normal(50, 3, 30),
    "yield": np.random.normal(95, 2, 30)
}

df = pd.DataFrame(data)

# 模擬異常（良率下降）
df.loc[15:18, "yield"] -= 10

# 畫出良率趨勢
plt.figure()
plt.plot(df["day"], df["yield"])
plt.xlabel("Day")
plt.ylabel("Yield")
plt.title("Yield Trend")

# 機器學習異常偵測
model = IsolationForest(contamination=0.15, random_state=0)

df["anomaly"] = model.fit_predict(df[["temperature", "pressure", "yield"]])

# -1 = 異常, 1 = 正常
anomalies = df[df["anomaly"] == -1]

print("=== 異常資料 ===")
print(anomalies)

# 畫出異常點
plt.figure()
plt.plot(df["day"], df["yield"], label="Yield")

plt.scatter(
    anomalies["day"],
    anomalies["yield"],
    label="Anomaly"
)

plt.xlabel("Day")
plt.ylabel("Yield")
plt.title("Yield with Anomalies")
plt.legend()

plt.show()

# 簡單分析輸出
print("\n=== 分析結論 ===")

if not anomalies.empty:
    print(f"發現 {len(anomalies)} 筆異常資料")
    print("可能原因：製程參數波動或設備異常")
    print("建議：檢查異常區間的溫度與壓力設定")
else:
    print("未發現明顯異常")
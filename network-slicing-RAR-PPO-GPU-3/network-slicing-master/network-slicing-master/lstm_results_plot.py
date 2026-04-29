import numpy as np
import matplotlib.pyplot as plt

data = np.load("./results/scenario_1/RaRPPO/lstm_loss_run0.npz")
loss = data["loss"]  # [T]

# 做一个滑动平均
window = 2
kernel = np.ones(window) / window
smooth = np.convolve(loss, kernel, mode="valid")

plt.figure()
plt.plot(loss, alpha=0.2, label="raw loss")
plt.plot(np.arange(window-1, window-1+len(smooth)), smooth, label=f"moving avg (win={window})")
plt.xlabel("LSTM train step")
plt.ylabel("MSE loss")
plt.legend()
plt.title("Online LSTM loss (raw vs smoothed)")
plt.show()

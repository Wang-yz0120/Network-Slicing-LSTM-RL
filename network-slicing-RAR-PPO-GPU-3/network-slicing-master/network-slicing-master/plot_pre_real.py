import numpy as np
d = np.load("./results/scenario_1/RaRPPO/lstm_predictions_run0.npz", allow_pickle=True)
print(d["pred_full"].shape)
print(d["real_full"].shape)
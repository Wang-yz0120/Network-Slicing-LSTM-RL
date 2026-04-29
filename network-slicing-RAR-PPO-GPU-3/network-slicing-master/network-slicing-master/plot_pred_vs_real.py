import os
import re
import numpy as np
import matplotlib.pyplot as plt

def _parse_scenario_run(npz_path: str):
    # 从路径提取 scenario_X
    scenario = "unknown"
    m = re.search(r"scenario_(\d+)", npz_path)
    if m:
        scenario = m.group(1)

    # 从文件名末尾提取 run
    # 例如: lstm_predictions_run0.npz -> run0
    fname = os.path.basename(npz_path)
    run = "unknown"
    m = re.search(r"run(\d+)\.npz$", fname)
    if m:
        run = m.group(1)

    return scenario, run

def plot_all_slices(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    pred = data["pred_full"]     # [T, 3N]
    real = data["real_full"]     # [T, 3N]
    slice_types = data["slice_types"]  # 例如: ["embb", "mmtc", ...]

    scenario, run = _parse_scenario_run(npz_path)

    out_dir = os.path.dirname(npz_path) or "."
    labels = ["cbr_traffic", "vbr_traffic", "new_devices"]

    num_slices = len(slice_types)
    for slice_idx in range(num_slices):
        base = slice_idx * 3

        plt.figure(figsize=(12, 6))
        for i in range(3):
            plt.plot(real[:, base + i], label=f"real {labels[i]}")
            plt.plot(pred[:, base + i], label=f"pred {labels[i]}", linestyle="--")
        plt.title(f"Scenario {scenario} | Slice {slice_idx} ({slice_types[slice_idx]}) | Run {run}")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend()
        plt.grid(True)

        # 保存文件名：包含 scenario / slice / run
        out_name = f"scenario_{scenario}_slice_{slice_idx}_run_{run}.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[saved] {out_path}")

if __name__ == "__main__":
    plot_all_slices("./results/scenario_1/RaRPPO/lstm_predictions_run2.npz")

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_pred_vs_real(npz_path, slice_idx=0):
#     data = np.load(npz_path, allow_pickle=True)
#     pred = data["pred_full"]     # [T, 3N]
#     real = data["real_full"]     # [T, 3N]
#     slice_types = data["slice_types"]

#     base = slice_idx * 3
#     labels = ["cbr_traffic", "vbr_traffic", "new_devices"]

#     plt.figure(figsize=(12, 6))
#     for i in range(3):
#         plt.plot(real[:, base + i], label=f"real {labels[i]}")
#         plt.plot(pred[:, base + i], label=f"pred {labels[i]}", linestyle="--")
#     plt.title(f"Slice {slice_idx} ({slice_types[slice_idx]})")
#     plt.xlabel("step")
#     plt.ylabel("value")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     plot_pred_vs_real("./results/scenario_1/RaRPPO/lstm_predictions_run0.npz", slice_idx=3)

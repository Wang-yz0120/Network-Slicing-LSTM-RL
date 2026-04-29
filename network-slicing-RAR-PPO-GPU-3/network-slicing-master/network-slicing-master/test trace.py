import pandas as pd

files = [
    "./datasets/fading_trace_EPA_3kmph.csv",
    "./datasets/fading_trace_ETU_3kmph.csv",
    "./datasets/fading_trace_EVA_60kmph.csv",
    "./datasets/mcs_codeset.csv",
    "./datasets/srslte_v19.03.csv",
]

for f in files:
    df = pd.read_csv(f, header=None)
    print(f, "shape =", df.shape, "(rows=PRBs, cols=time_samples)")
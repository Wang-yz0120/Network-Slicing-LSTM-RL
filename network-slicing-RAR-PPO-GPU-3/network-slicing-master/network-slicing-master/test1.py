import torch
ckpt = torch.load("./traffic_lstm_pretrained_3N.pth", map_location="cpu", weights_only=False)
print("ckpt keys:", ckpt.keys())
print("ckpt input_dim:", ckpt.get("input_dim"))
print("ckpt feat_dim :", ckpt.get("feat_dim"))
print("ckpt history_len:", ckpt.get("history_len"))
print("has mean/std:", ("mean" in ckpt), ("std" in ckpt), ("scale" in ckpt))

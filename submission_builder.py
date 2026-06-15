import torch

ckpt = torch.load(
    "checkpoints/value_network_best.pt",
    map_location="cpu",
    weights_only=False,
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print("KEYS:")
    print(list(ckpt.keys()))

    for k, v in ckpt.items():
        print(k, type(v))
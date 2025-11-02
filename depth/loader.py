import os
import torch
from .midas_model import MidasSmall

_MODEL = None  # cache singleton

def load_depth_model():
    """
    Load the MiDaS_small-like model from local weights on disk.
    Returns a torch.nn.Module in eval() mode.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # path to the baked-in weights in the image
    weights_path = os.path.join(
        os.path.dirname(__file__),
        "weights",
        "midas_v21_small_256.pt"
    )

    # 1. Build model structure
    model = MidasSmall()

    # 2. Load weights
    # We map_location='cpu' so it always works on CPU-only environments (like Render free tier)
    state_dict = torch.load(weights_path, map_location="cpu")

    # Sometimes pretrained checkpoints have keys that don't exactly line up.
    # We'll do a forgiving load by filtering matching keys.
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict, strict=False)

    model.eval()
    _MODEL = model
    print("Local depth model loaded from bundled weights.")
    return _MODEL

import os
import platform
import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = False) -> dict:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "deterministic": deterministic,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

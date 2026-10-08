"""Count parameters on CPU without loading checkpoints or executing inference."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn
from src.obj3.journal.train_ms_tmo_obj3_hetero import HeteroUNet
from src.obj1.journal.models.fno_replicate import FNOReplicate2D
from src.obj1.journal.models.deeponet_canonical import CanonicalDeepONet2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    constructors = {
        "Deep ensemble (U-Net)": lambda: UNet_ASPP_Attn(in_ch=1, out_ch=25, base=64, attn_heads=4),
        "Heteroscedastic (U-Net)": lambda: HeteroUNet(base=64, attn_heads=4),
        "Fourier operator": lambda: FNOReplicate2D(in_ch=1, out_ch=25, width=64, modes1=20, modes2=20, depth=4),
        "DeepONet": lambda: CanonicalDeepONet2D(in_ch=1, out_ch=25, branch_width=128,
                                              branch_latent=384, trunk_width=384, basis_rank=96),
    }
    records = {}
    for name, construct in constructors.items():
        model = construct()
        complex_elements = sum(p.numel() for p in model.parameters() if p.is_complex())
        total = sum(p.numel() for p in model.parameters())
        records[name] = {"tensor_elements_per_member": total,
                         "real_scalar_parameters_per_member": total + complex_elements,
                         "complex_tensor_elements": complex_elements,
                         "members": 5}
        del model
        gc.collect()
    output = {"counts": records, "method": "CPU source instantiation; complex weights count as two real scalars",
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT / "src").rglob("*.py") if p.name in
                          ("train_ms_tmo_obj3.py", "train_ms_tmo_obj3_hetero.py",
                           "fno_replicate.py", "deeponet_canonical.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

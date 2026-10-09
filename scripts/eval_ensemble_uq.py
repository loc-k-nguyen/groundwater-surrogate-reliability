"""Evaluate a deterministic-ensemble (or hetero) Obj3 backbone's predictive UQ on
IID / OOD(σ²Y) / Axis D, with metrics consistent with eval_hetero_uq.py.

--arch fno       -> FNOReplicate2D deep ensemble (uncertainty = across-seed variance)
--arch unet_det  -> MS-TMO-UNet (UNet_ASPP_Attn) deep ensemble
--arch hetero    -> HeteroUNet (mean+logvar); total = aleatoric+epistemic
--arch deeponet  -> CanonicalDeepONet2D deep ensemble (uncertainty = across-seed variance)

Evaluation-only: no training, no split edits, no conference-output writes. Outputs
mirror the hetero eval schema (summaries + per-sample CSV + comparisons).
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path, configure_script_paths

if __name__ == "__main__":
    DEFAULT_ROOT = configure_script_paths()
import argparse, csv, json, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.obj3.conference.data_obj3_ood import Obj3FullFieldDataset, collect_npz_files, load_obj3_split
from src.obj3.conference.eval_deterministic import DATA_RANGE, EPS_C, PLUME_MIN_PIXELS, PLUME_PAD, PLUME_THRESH
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)
REPO = DEFAULT_ROOT
DEF_MAIN = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL")
DEF_EXTRA = asset_path("calibration", REPO / "simulation/datasets/obj3_calibration")
DEF_SPLIT = REPO / "splits/param_split_obj3_ood.json"
DEF_STATS = REPO / "metadata/obj3_train_stats.json"
DEF_AXISD = REPO / "experiments/obj3/journal/upgrade_2026_07_01/tier1_multi_axis_ood/axisD_transport_shift/sim/dataset_axisD"


def fon(v):
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int)): return int(v)
    return v
def mon(vs):
    x=[float(v) for v in vs if v is not None and np.isfinite(v)]; return float(np.mean(x)) if x else None
def son(vs):
    x=[float(v) for v in vs if v is not None and np.isfinite(v)]; return float(np.std(x)) if x else None


def build_models(arch, checkpoints, device):
    models=[]
    for cp in checkpoints:
        ckpt=torch.load(cp, map_location=device); cfg=ckpt.get("cfg",{})
        if arch=="fno":
            from src.obj1.journal.models.fno_replicate import FNOReplicate2D
            fno=cfg.get("_fno",{}) if isinstance(cfg,dict) else {}
            m=FNOReplicate2D(in_ch=1,out_ch=25,**(fno or dict(width=64,modes1=20,modes2=20,depth=4)))
        elif arch=="unet_det":
            from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn
            m=UNet_ASPP_Attn(in_ch=1,out_ch=25,base=cfg.get("base_channels",64),attn_heads=cfg.get("attn_heads",4))
        elif arch=="deeponet":
            from src.obj1.journal.models.deeponet_canonical import CanonicalDeepONet2D
            don = cfg.get("_don") or {}
            m=CanonicalDeepONet2D(in_ch=1,out_ch=25,**(don or dict(
                branch_width=128,branch_latent=384,trunk_width=384,basis_rank=96)))
        elif arch=="hetero":
            from src.obj3.journal.train_ms_tmo_obj3_hetero import HeteroUNet
            m=HeteroUNet(base=cfg.get("base_channels",64),attn_heads=cfg.get("attn_heads",4))
        else: raise ValueError(arch)
        m.load_state_dict(ckpt["model"]); m.to(device).eval(); models.append(m)
        logger.info("loaded %s %s epoch=%s", arch, cp, ckpt.get("epoch"))
    return models


def patch_positions(length: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError(f"patch and stride must be positive, got patch={patch}, stride={stride}")
    if patch >= length:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    end = length - patch
    if positions[-1] != end:
        positions.append(end)
    return positions


def sliding(model, k, patch, stride, device, hetero, use_amp):
    _,_,h,w=k.shape
    hann=(torch.hann_window(patch,periodic=False).to(device)); h2=(hann.unsqueeze(1)*hann.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    h2=torch.clamp(h2,min=1e-3)
    ph=max(0,patch-h); pw=max(0,patch-w); kp=F.pad(k,[0,pw,0,ph],mode="reflect"); H,W=kp.shape[2],kp.shape[3]
    mean=torch.zeros(1,25,H,W,device=device); wgt=torch.zeros(1,1,H,W,device=device)
    var=torch.zeros(1,25,H,W,device=device) if hetero else None
    rows=patch_positions(H,patch,stride); cols=patch_positions(W,patch,stride)
    for r in rows:
        for c in cols:
            p=kp[:,:,r:r+patch,c:c+patch]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp and device.startswith("cuda")):
                out=model(p)
            if hetero:
                mp,lv=out; mean[:,:,r:r+patch,c:c+patch]+=mp.float()*h2; var[:,:,r:r+patch,c:c+patch]+=torch.exp(lv.float())*h2
            else:
                mean[:,:,r:r+patch,c:c+patch]+=out.float()*h2
            wgt[:,:,r:r+patch,c:c+patch]+=h2
    wc=wgt[:,:,:h,:w]
    if torch.any(wc <= 0):
        raise RuntimeError(
            f"Sliding-window coverage failure for field={(h,w)}, patch={patch}, stride={stride}, "
            f"rows={rows}, cols={cols}"
        )
    mean=mean[:,:,:h,:w]/wc
    var=(var[:,:,:h,:w]/wc) if hetero else None
    return mean, var


def eval_sample(mean_log,total_std,aleo_std,epi_std,s):
    cg=s["C_log"].numpy(); cp=s["C_phys"].numpy(); tm=[]
    for t in range(mean_log.shape[0]):
        gt=cg[t]; pr=mean_log[t]; gp=np.clip(cp[t],0.,None)
        from skimage.metrics import structural_similarity as ssim
        g=float(ssim(gt,pr,data_range=DATA_RANGE))
        ps,pp=compute_plume_ssim(gp,pr,EPS_C,PLUME_THRESH,PLUME_PAD,PLUME_MIN_PIXELS,DATA_RANGE)
        pm=gp>PLUME_THRESH
        tm.append({"global_ssim":g,"plume_ssim":fon(ps),
                   "plume_total_std_log":float(np.mean(total_std[t][pm])) if np.any(pm) else None,
                   "plume_epistemic_std_log":float(np.mean(epi_std[t][pm])) if np.any(pm) else None,
                   "mean_total_std_log":float(np.mean(total_std[t]))})
    return {"param_id":s["param_id"],"real_id":s["real_id"],"path":s["path"],
            "mean_plume_ssim":mon(m["plume_ssim"] for m in tm),
            "mean_global_ssim":mon(m["global_ssim"] for m in tm),
            "plume_total_std_log":mon(m["plume_total_std_log"] for m in tm),
            "plume_epistemic_std_log":mon(m["plume_epistemic_std_log"] for m in tm),
            "mean_total_std_log":mon(m["mean_total_std_log"] for m in tm)}


def rank_auc(neg,pos):
    neg=[x for x in neg if x is not None and np.isfinite(x)]; pos=[x for x in pos if x is not None and np.isfinite(x)]
    if not neg or not pos: return None
    w=sum((1.0 if p>n else 0.5 if p==n else 0.0) for p in pos for n in neg); return w/(len(pos)*len(neg))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch",required=True,choices=["fno","unet_det","hetero","deeponet"])
    ap.add_argument("--checkpoints",nargs="+",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--prefix",default="ensemble_uq")
    ap.add_argument("--split_json",default=str(DEF_SPLIT))
    ap.add_argument("--main_data_root",default=str(DEF_MAIN)); ap.add_argument("--extra_data_root",default=str(DEF_EXTRA))
    ap.add_argument("--stats_json",default=str(DEF_STATS))
    ap.add_argument("--splits",nargs="+",default=["iid_test","ood_test"])
    ap.add_argument("--axisd_params",nargs="*",default=["param_9200"],
                    help="Zero or more axisD pool ids; pass with no values to evaluate splits only.")
    ap.add_argument("--axisd_root",default=str(DEF_AXISD))
    ap.add_argument("--interp_data_root",default=None,
                    help="Required for interpolation config ids 276-355.")
    ap.add_argument("--realizations",type=int,nargs="*",default=None,help="Restrict to these realization numbers, e.g. 1 2 3. Default: all. Used by the realization-disjoint twin control, since amplitude twins share a realization id.")
    ap.add_argument("--patch_size",type=int,default=320); ap.add_argument("--stride",type=int,default=160)
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--device",default="cuda")
    ap.add_argument("--pred_clamp_min",type=float,default=-12.0); ap.add_argument("--pred_clamp_max",type=float,default=2.0)
    ap.add_argument("--amp",action="store_true",help="Enable autocast during inference. Default is fp32.")
    a=ap.parse_args()
    out=Path(a.output_dir)
    if out.exists():
        ap.error("Output already exists; use a new isolated path")
    out.mkdir(parents=True)
    hetero=(a.arch=="hetero")
    with open(a.stats_json) as f: stats=json.load(f)
    models=build_models(a.arch,a.checkpoints,a.device)
    # The canonical loader enforces the OOD schema (requires a 'train' key). Supplementary
    # split files such as the interpolation ladder only define their own level keys, so fall
    # back to the raw mapping. Only the keys named in --splits are ever read.
    try:
        sp=load_obj3_split(a.split_json)
    except KeyError:
        with open(a.split_json) as f: sp={k:v for k,v in json.load(f).items() if isinstance(v,list)}
        logger.info('using raw split mapping for %s (keys=%s)', a.split_json, sorted(sp))
    missing=[k for k in a.splits if k not in sp]
    if missing: raise SystemExit(f'split keys not found in {a.split_json}: {missing}')
    targets={}
    for s in a.splits:
        fs=collect_npz_files(sp[s],a.main_data_root,a.extra_data_root,a.interp_data_root)
        if a.realizations:
            keep={int(r) for r in a.realizations}
            fs=[f for f in fs if int(Path(f).stem.split("_")[-1]) in keep]
        targets[s]=fs[:a.limit] if a.limit>0 else fs
    for pid in a.axisd_params:
        fs=sorted((Path(a.axisd_root)/pid).glob("real_*.npz"))
        if fs: targets[f"axisD_{pid}"]=fs[:a.limit] if a.limit>0 else fs
    all_rows=[]; summ={}
    for tname,files in targets.items():
        if not files: continue
        ds=Obj3FullFieldDataset(files,stats); rows=[]
        logger.info("target=%s n=%d", tname, len(ds))
        for i in range(len(ds)):
            s=ds[i]; k=s["K"].unsqueeze(0).to(a.device)
            mem_mean=[]; mem_var=[]
            for m in models:
                mean,var=sliding(m,k,a.patch_size,a.stride,a.device,hetero,a.amp)
                mean=torch.clamp(mean,a.pred_clamp_min,a.pred_clamp_max)
                mem_mean.append(mean.squeeze(0).cpu().numpy())
                if hetero: mem_var.append(var.squeeze(0).cpu().numpy())
            mm=np.stack(mem_mean,0); mean_log=mm.mean(0); epi_var=mm.var(0)
            aleo_var=np.stack(mem_var,0).mean(0) if hetero else np.zeros_like(epi_var)
            total_std=np.sqrt(np.maximum(aleo_var+epi_var,0.)); epi_std=np.sqrt(np.maximum(epi_var,0.)); aleo_std=np.sqrt(np.maximum(aleo_var,0.))
            r=eval_sample(mean_log,total_std,aleo_std,epi_std,s); r["target"]=tname; rows.append(r); all_rows.append(r)
        summ[tname]={"n_samples":len(rows),
                     "mean_plume_ssim_mean":mon(r["mean_plume_ssim"] for r in rows),
                     "plume_total_std_log_mean":mon(r["plume_total_std_log"] for r in rows),
                     "plume_total_std_log_std":son(r["plume_total_std_log"] for r in rows)}
    # PLUME-region AUROC (the correct discriminator)
    comp={}
    iid=[r["plume_total_std_log"] for r in all_rows if r["target"]=="iid_test"]
    for tname in [t for t in targets if t not in ("iid_test",)]:
        pos=[r["plume_total_std_log"] for r in all_rows if r["target"]==tname]
        comp[f"{tname}_vs_iid_plume_std_AUROC"]=rank_auc(iid,pos)
    rep={"created_utc":datetime.now(timezone.utc).isoformat(),"arch":a.arch,"prefix":a.prefix,
         "checkpoints":a.checkpoints,"summaries":summ,"plume_region_auroc":comp,
         "protocol":{"data_range":DATA_RANGE,"plume_thresh":PLUME_THRESH,"patch":a.patch_size,"stride":a.stride,
                     "uncertainty":"sqrt(aleatoric+epistemic); det ensemble -> epistemic=across-seed var"},
         "per_sample":[{k:fon(v) for k,v in r.items()} for r in all_rows]}
    (out/f"{a.prefix}_metrics.json").write_text(json.dumps(rep,indent=2))
    with (out/f"{a.prefix}_per_sample.csv").open("w",newline="") as f:
        cols=["target","param_id","real_id","mean_plume_ssim","mean_global_ssim","plume_total_std_log","plume_epistemic_std_log","mean_total_std_log"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in all_rows: w.writerow({c:fon(r.get(c)) for c in cols})
    print(f"OBJ3_ENSEMBLE_UQ_DONE arch={a.arch} plume_auroc={comp}", flush=True)
    logger.info("summaries: %s", summ)


if __name__=="__main__":
    main()

"""Create an editable vector workflow with no synthetic scientific imagery."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    targets = [out / "fig1_workflow_v4.pdf", out / "fig1_workflow_v4.svg"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Workflow output exists")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis("off")

    def box(x,y,w,h,text,color="#39566d",fill="#ffffff",size=8.2):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,linewidth=.9))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,linespacing=1.35)

    def arrow(a,b,color="#333333",style="-"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,
                                    linewidth=.9,color=color,linestyle=style))

    ax.text(.01,.986,"1  Prediction and simulator reference",fontweight="bold",va="top")
    box(.01,.81,.17,.13,"Conductivity\nfield K",color="#007c83")
    box(.29,.81,.25,.13,"Trained K-only\nsurrogate")
    box(.65,.81,.34,.13,"Predicted concentration\nModel-specific uncertainty")
    box(.01,.64,.17,.12,"Transport\ncontrols",color="#007c83")
    box(.29,.64,.25,.12,"MODFLOW-2005\n+ MT3D-USGS")
    box(.65,.64,.34,.12,"Simulator concentration\n(reference for audit)")
    arrow((.18,.875),(.29,.875))
    arrow((.54,.875),(.65,.875))
    arrow((.18,.70),(.29,.70))
    arrow((.54,.70),(.65,.70))
    ax.plot([.18,.235,.235],[.835,.835,.735],color="#333333",linewidth=.9)
    arrow((.235,.735),(.29,.735))
    ax.text(.21,.782,"K",fontsize=8,ha="center")
    ax.text(.01,.599,"2  Complementary diagnostics",fontweight="bold",va="top")
    box(.01,.335,.305,.215,"A  Input screening\n\nInput summaries vs\ntraining envelope\n\nLow-support reference",color="#007c83")
    box(.348,.335,.305,.215,"B  Uncertainty ranking\n\nFull field: prediction-time\nOracle plume: retrospective\n\nValidate per monitor",color="#39566d")
    box(.685,.335,.305,.215,"C  Conformal calibration\n\nResiduals + calibration support\nCoverage and interval width\n\nState region and score unit",color="#39566d",size=7.9)
    ax.text(.01,.294,"3  Controlled comparisons and dependence checks",fontweight="bold",va="top")
    box(.01,.115,.47,.13,"Same spatial pattern; changed variance\nInput-visible amplitude comparison\nShared geological draw identifiers",color="#007c83",size=8)
    box(.52,.115,.47,.13,"Same K; changed transport controls\nPrediction-only scores unchanged\nOracle region can change",color="#39566d",size=8)
    ax.text(.5,.068,"Check training twins and held-out draw identifiers.",ha="center",fontsize=8.2)
    ax.text(.5,.026,"Finite-design evidence; no general detection or deployment guarantee.",
            ha="center",fontweight="bold",fontsize=8)
    for target in targets:
        fig.savefig(target)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a reading PDF, preview, and locally archived reproducibility bundle."""
from pathlib import Path
import hashlib, json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter

OUT=Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=pd.read_csv(OUT/"data/main_results_summary.csv")
    cols=["label","lower_normalized_mean","upper_normalized_mean","raw_gap_normalized_mean","score_mean","score_sd"]
    rows=[[r[cols[0]]]+[f"{r[c]:.6f}" for c in cols[1:]] for _,r in summary.iterrows()]
    fig,ax=plt.subplots(figsize=(10,5.1)); ax.axis("off")
    ax.set_title("Final holdout: primal–dual results",loc="left",fontsize=14,weight="bold",pad=30)
    table=ax.table(cellText=rows,colLabels=["Method / selection rule","L / K","U / K","(U − L) / K","Gap score","SD (seeds)"],
        cellLoc="right",colLoc="right",colWidths=[.40,.12,.12,.12,.12,.12],bbox=[0,.12,1,.84])
    table.auto_set_font_size(False); table.set_fontsize(8.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.35); cell.set_edgecolor("#D7DFE6")
        if c==0: cell.set_text_props(ha="left")
        if r==0:
            cell.set_facecolor("#24465D"); cell.set_text_props(color="white",weight="bold")
        elif r%2==0: cell.set_facecolor("#F1F5F8")
    ax.text(0,.065,"Three seed blocks; four test contracts per block. Means of normalized sample estimates.",transform=ax.transAxes,fontsize=8)
    ax.text(0,.015,"Gap score includes Monte Carlo uncertainty and a tail term. SD is across seeds; selections froze before testing.",transform=ax.transAxes,fontsize=8)
    for ext in ["pdf","svg","png"]: fig.savefig(OUT/"tables"/f"main_results.{ext}",bbox_inches="tight",dpi=300,facecolor="white")
    plt.close(fig)
    captions=json.loads((OUT/"figures/captions.json").read_text())
    names=sorted(captions)
    pdf=PdfWriter(); pdf.append(OUT/"tables/main_results.pdf")
    for name in names: pdf.append(OUT/"figures"/f"{name}.pdf")
    pdf.add_metadata({"/Title":"OpenHyra Math-AI paper figures and results","/Subject":"Recorded results, feedback ablation and primal-dual evaluation"})
    with (OUT/"paper_figures.pdf").open("wb") as handle: pdf.write(handle)
    pdf.close()
    assert len(PdfReader(OUT/"paper_figures.pdf").pages)==8
    images=[OUT/"tables/main_results.png"]+[OUT/"figures"/f"{n}.png" for n in names]
    fig,axes=plt.subplots(4,2,figsize=(15,15))
    for ax,p in zip(axes.flat,images):
        ax.imshow(plt.imread(p)); ax.axis("off"); ax.set_title(p.stem.replace("_"," "),loc="left",fontsize=10,color="#24465D")
    fig.tight_layout(pad=2)
    fig.savefig(OUT/"preview.png",dpi=140,bbox_inches="tight",facecolor="#F9FAFB"); plt.close(fig)
    excluded={"manifest.json","archive_check.json"}
    files=[p for p in OUT.rglob("*") if p.is_file() and not p.is_symlink() and
           "__pycache__" not in p.parts and p.name!=".DS_Store" and
           p not in [OUT/n for n in excluded] and p.suffix!=".pyc"]
    manifest={"schema":"openhyra-paper-asset-manifest.v1","file_count":len(files),
        "files":{str(p.relative_to(OUT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(files)}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    archive=OUT.parent/(OUT.name+".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files+[OUT/"manifest.json"]): z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,entry in manifest["files"].items():
            assert hashlib.sha256(z.read(OUT.name+"/"+rel)).hexdigest()==entry["sha256"]
    status={"status":"passed","archive":str(archive),"archive_sha256":digest(archive),
        "files_verified":len(files),"archive_bytes":archive.stat().st_size,"pdf_pages":8}
    (OUT/"archive_check.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status))

if __name__=="__main__": main()

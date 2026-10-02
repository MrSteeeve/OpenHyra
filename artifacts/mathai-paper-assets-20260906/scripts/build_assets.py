#!/usr/bin/env python3
"""Build editable paper figures and tables exclusively from recorded results."""
from pathlib import Path
import json, math, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parents[1]/"artifacts/mathai-bermudan-live-v2-20260905"
ACTIVE=OUT/("ablation_recovered" if (OUT/"recovery_amendment.json").exists() else "ablation")
FIG=OUT/"figures"; DATA=OUT/"data"; TABLE=OUT/"tables"
for p in [FIG,DATA,TABLE,OUT/"inputs"]: p.mkdir(parents=True,exist_ok=True)
SEEDS=[17,29,41]
COL={"context_proposal":"#0072B2","direct_generation":"#D55E00",
     "structured_memory":"#0072B2","score_only":"#D55E00"}
LABEL={"linear_ridge":"Ridge","pca_ridge":"PCA + Ridge","gated_ridge":"Gated Ridge",
       "mlp":"MLP","residual_hybrid":"Ridge + MLP","direct_decision":"Threshold policy",
       "policy_iteration":"Policy iteration","context_proposal":"Original Context: best generated",
       "direct_generation":"Original direct: best generated",
       "context_incumbent":"Original Context incumbent","direct_incumbent":"Original direct incumbent",
       "structured_memory":"Structured-memory incumbent","score_only":"Score-only incumbent"}
ORDER=list(LABEL)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
 "axes.labelsize":8,"legend.fontsize":7,"legend.frameon":False,"axes.spines.top":False,
 "axes.spines.right":False,"axes.grid":True,"grid.alpha":.15,"lines.linewidth":1.5,
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.dpi":300})
CAPTIONS={}

def load(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def frozen_input(src,dest):
    dest=OUT/"inputs"/dest; dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if src.exists() and dest.read_bytes()!=src.read_bytes(): raise RuntimeError(f"Original input changed: {src}")
    else: shutil.copy2(src,dest)
    return dest
def export(fig,name,caption):
    for ext in ["pdf","svg","png"]: fig.savefig(FIG/f"{name}.{ext}",bbox_inches="tight",facecolor="white")
    plt.close(fig); CAPTIONS[name]=caption
def csv(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(DATA/f"{name}.csv",index=False,float_format="%.12g")
    return df
def finite(v): return v is not None and math.isfinite(float(v))

def collect():
    programs=[]; cells=[]
    def add(result,item,stage,path):
        metric=result.get("metrics",{}); summaries=metric.get("summaries",[])
        norm=[]
        for c in summaries:
            if "lower_bound" not in c: continue
            g=c["normalized_primal_dual_confidence_gap"]
            k=c["confidence_gap"]/g if abs(g)>1e-20 else 1.
            record={"stage":stage,"key":item["key"],"method":item["method"],"seed":item["seed"],
                "instance_id":c["instance_id"],"instance_sha256":c.get("instance_sha256"),
                "payoff_type":c.get("payoff_type"),"dimension":c.get("dimension"),"repeat":c.get("repeat",0),
                "strike":k,"lower":c["lower_bound"],"upper":c["upper_bound"],
                "lower_se":c["lower_bound_standard_error"],"upper_se":c["upper_bound_standard_error"],
                "lower_normalized":c["lower_bound"]/k,"upper_normalized":c["upper_bound"]/k,
                "raw_gap_normalized":c["raw_primal_dual_gap"]/k,"confidence_gap_normalized":g,
                "uncertainty_margin_normalized":(c["confidence_gap"]-c["raw_primal_dual_gap"])/k,
                "dual_verifier":c.get("dual_verifier"),"source_json":str(path.relative_to(OUT))}
            cells.append(record); norm.append(record)
        rec={"stage":stage,"key":item["key"],"method":item["method"],"seed":item["seed"],
            "kind":item["kind"],"operator":item.get("operator"),"status":result["status"],
            "score":result.get("score"),"wall_seconds":result.get("wall_seconds",0),
            "failure_reason":result.get("failure_reason"),"source_digest":result.get("source_digest"),
            "program_path":item.get("program_path"),"n_cells":len(norm),
            "source_json":str(path.relative_to(OUT))}
        for col in ["lower_normalized","upper_normalized","raw_gap_normalized","confidence_gap_normalized","uncertainty_margin_normalized"]:
            rec[col]=float(np.mean([c[col] for c in norm])) if norm else None
        rec["fit_seconds"]=sum(c.get("fit_wall_seconds",0) for c in metric.get("training_cells",[]))
        programs.append(rec)
    cat=load(OUT/"catalog.json")
    for item in cat:
        for stage,folder in [("legacy_audit","legacy_completion"),("calibration","calibration")]:
            path=OUT/"raw"/folder/f"{item['key']}.json"
            if path.exists(): add(load(path),item,stage,path)
        if item.get("is_original_winner"):
            source=OLD/f"audits/{item['method']}-s{item['seed']}.json"
            p=frozen_input(source,Path("original_audits")/source.name)
            add(load(p),item,"legacy_audit",p)
    if (OUT/"holdout_index.json").exists():
        for item in load(OUT/"holdout_index.json"):
            p=OUT/"raw/holdout"/f"{item['key']}.json"
            if p.exists(): add(load(p),item,"holdout",p)
    for seed in SEEDS:
        for arm in ["structured_memory","score_only"]:
            for op in ["whole_program_restart","subsystem_rewrite"]:
                p=ACTIVE/f"s{seed}/{arm}/{op}/record.json"
                if p.exists():
                    rec=load(p)
                    item=dict(key=f"ablation-{arm}-s{seed}-{op}",seed=seed,kind="ablation",method=arm,
                        operator=op,program_path=rec["program_path"])
                    add(rec["result"],item,"calibration",p)
    return csv("program_evaluations",programs),csv("evaluation_cells",cells)

def original_search():
    attempts=[]; trajectories=[]; calls=[]
    cat=load(OUT/"catalog.json")
    for seed in SEEDS:
        refs={}
        for name in ["linear_ridge","mlp","residual_hybrid"]:
            p=frozen_input(OLD/f"references/s{seed}/{name}.json",Path(f"original_references/s{seed}/{name}.json"))
            refs[name]=load(p)
        start_name=max(refs,key=lambda n:refs[n]["score"])
        start_key=f"reference-s{seed}-{start_name}"
        start_gap=load(OUT/"raw/calibration"/f"{start_key}.json")["score"]
        for mode in ["context_proposal","direct_generation"]:
            rp=frozen_input(OLD/f"runs/{mode}-s{seed}/public_complete.json",Path("original_runs")/f"{mode}-s{seed}.json")
            run=load(rp); ctx=[c for c in run["agent_calls"] if c["role"]=="context"]
            for c in run["agent_calls"]:
                calls.append({"experiment":"original","method":mode,"seed":seed,**c})
            best=refs[start_name]["score"]; best_key=start_key; best_gap=start_gap; budget=0.
            trajectories.append(dict(mode=mode,seed=seed,attempt=0,cumulative_model_seconds=0,
                public_incumbent_score=best,calibration_incumbent_gap=best_gap,incumbent_key=best_key,
                best_generated_public_score=None,candidate_public_score=None,candidate_calibration_gap=None,status="seed"))
            best_generated=None
            for i in range(0,len(run["rows"]),2):
                r=run["rows"][i]; ctrl=run["rows"][i+1]; round_idx=r["round"]
                if i%4==0 and mode=="context_proposal": budget+=ctx[round_idx]["wall_seconds"]
                budget+=r["proposal"]["generation_seconds"]
                key=f"legacy-{mode}-s{seed}-{r['candidate_id']}"
                gap_path=OUT/"raw/calibration"/f"{key}.json"
                g=load(gap_path)["score"] if gap_path.exists() else None
                score=r["result"]["score"]
                if finite(score):
                    best_generated=score if best_generated is None else max(best_generated,score)
                    if score>best: best,best_key,best_gap=score,key,g
                attempts.append(dict(mode=mode,seed=seed,attempt=i//2+1,round=round_idx,
                    candidate_id=r["candidate_id"],operator=r["proposal"]["operator"],
                    status=r["result"]["status"],public_score=score,control_score=ctrl["result"]["score"],
                    calibration_gap=g,generation_seconds=r["proposal"]["generation_seconds"],
                    known_tokens=r["proposal"]["tokens_reported"],
                    evaluation_seconds=r["result"]["wall_seconds"],failure_reason=r["result"]["failure_reason"],
                    source_digest=r["result"]["source_digest"]))
                trajectories.append(dict(mode=mode,seed=seed,attempt=i//2+1,cumulative_model_seconds=budget,
                    public_incumbent_score=best,calibration_incumbent_gap=best_gap,incumbent_key=best_key,
                    best_generated_public_score=best_generated,candidate_public_score=score,
                    candidate_calibration_gap=g,status=r["result"]["status"]))
    return csv("original_search_attempts",attempts),csv("original_search_trajectory",trajectories),calls

def ablation_data(programs):
    rows=[]; pairs=[]; calls=[]; selections=[]
    for seed in SEEDS:
        pp=OUT/f"ablation/s{seed}/frozen_parent.json"
        if not pp.exists(): continue
        parent=load(pp); basecal=load(OUT/"raw/calibration"/f"{parent['key']}.json")
        basehold=programs[(programs.stage=="holdout")&(programs.key==parent["key"])]
        basehold=basehold.iloc[0].score if len(basehold) else None
        for op in ["whole_program_restart","subsystem_rewrite"]:
            pair=dict(seed=seed,operator=op,parent_key=parent["key"],
                parent_calibration_gap=basecal["score"],parent_holdout_gap=basehold)
            for arm in ["structured_memory","score_only"]:
                p=ACTIVE/f"s{seed}/{arm}/{op}/record.json"
                if not p.exists(): continue
                rec=load(p); ev=rec["result"]; key=f"ablation-{arm}-s{seed}-{op}"
                hh=programs[(programs.stage=="holdout")&(programs.key==key)]
                holdscore=hh.iloc[0].score if len(hh) and hh.iloc[0].status=="ok" else None
                row=dict(seed=seed,arm=arm,operator=op,key=key,status=ev["status"],
                    calibration_gap=ev["score"],holdout_gap=holdscore,parent_key=parent["key"],
                    failure_reason=ev["failure_reason"],model_seconds=sum(c["wall_seconds"] for c in rec["calls"]),
                    tokens_known=sum(c["tokens_reported"] or 0 for c in rec["calls"]),
                    calls=len(rec["calls"]),missing_usage_calls=sum(c["tokens_reported"] is None for c in rec["calls"]),
                    recovered_pre_generation_failure=rec.get("infrastructure_recovery",{}).get("eligible",False),
                    source_record=str(p.relative_to(OUT)),parent_source_digest=rec["parent_source_digest"],
                    memory_sha256=rec["memory_sha256"],program_path=rec["program_path"])
                rows.append(row)
                for c in rec["calls"]: calls.append(dict(experiment="feedback_ablation",method=arm,seed=seed,operator=op,**c))
                for stage,val,base in [("calibration",ev["score"],basecal["score"]),("holdout",holdscore,basehold)]:
                    # An unavailable proposal retains the shared parent for operational comparison.
                    effective=val if finite(val) else base
                    pair[f"{arm}_{stage}_gap"]=val
                    pair[f"{arm}_{stage}_effective_gap"]=effective
                    pair[f"{arm}_{stage}_reduction"]=base-effective if finite(base) and finite(effective) else None
                pair[f"{arm}_status"]=ev["status"]
            pairs.append(pair)
    if (OUT/"selections_frozen.json").exists():
        for s in load(OUT/"selections_frozen.json")["selections"]:
            hh=programs[(programs.stage=="holdout")&(programs.seed==s["seed"])&(programs.program_path==s["selected_program_path"])]
            if len(hh):
                r=hh.iloc[0].to_dict(); r.update(method=s["arm"],kind="ablation_selected",
                    selected_operator=s["selected_operator"],selection_rule="calibration only, parent included")
                selections.append(r)
    return csv("feedback_attempts",rows),csv("feedback_pairs",pairs),csv("selected_incumbents",selections),calls

def main_table(programs,selected,traj):
    d=programs[(programs.stage=="holdout")&(programs.kind.isin(["reference","legacy_generated"]))].copy()
    original_incumbents=[]
    for (mode,seed),track in traj.groupby(["mode","seed"]):
        chosen=track.sort_values("attempt").iloc[-1].incumbent_key
        source=programs[(programs.stage=="holdout")&(programs.key==chosen)]
        if len(source):
            r=source.iloc[0].to_dict()
            r.update(method="context_incumbent" if mode=="context_proposal" else "direct_incumbent",
                     kind="original_incumbent",selection_rule="original public score only; initial references included")
            original_incumbents.append(r)
    csv("original_selected_incumbents",original_incumbents)
    if original_incumbents: d=pd.concat([d,pd.DataFrame(original_incumbents)],ignore_index=True)
    if len(selected): d=pd.concat([d,selected],ignore_index=True)
    csv("main_results_by_seed",d)
    rows=[]
    for name in ORDER:
        sub=d[d.method==name]
        if not len(sub): continue
        good=sub[sub.status=="ok"]
        r=dict(method=name,label=LABEL[name],successful_seeds=len(good),requested_seeds=3)
        for col in ["lower_normalized","upper_normalized","raw_gap_normalized","confidence_gap_normalized","score","wall_seconds","fit_seconds"]:
            r[col+"_mean"]=good[col].mean(); r[col+"_sd"]=good[col].std(ddof=1)
        rows.append(r)
    summary=csv("main_results_summary",rows)
    if len(summary):
        tex=summary[["label","lower_normalized_mean","upper_normalized_mean","raw_gap_normalized_mean","score_mean","score_sd","wall_seconds_mean"]].rename(
            columns={"label":"Method","lower_normalized_mean":"L / K","upper_normalized_mean":"U / K",
                     "raw_gap_normalized_mean":"(U-L) / K","score_mean":"Gap score","score_sd":"SD (seeds)","wall_seconds_mean":"Eval. seconds"})
        (TABLE/"main_results.tex").write_text(tex.to_latex(index=False,float_format="%.5f",escape=True))
        (TABLE/"main_results.md").write_text(tex.to_markdown(index=False,floatfmt=".5f")+"\n")
    return d,summary

def box(ax,x,y,w,h,text,color="#EAF2F8"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=.02",fc=color,ec="#C7D0D9",lw=.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=7.4,linespacing=1.35)
def arrow(ax,a,b,color="#526273",style="-",rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=1,color=color,
                              linestyle=style,connectionstyle=f"arc3,rad={rad}"))
def architecture():
    design=csv("architecture_comparison",[
        dict(experiment="original",arm="context_proposal",context_agent=True,parent="adaptive between rounds",history="structured experience records",selection="public payoff LCB",test_feedback=False),
        dict(experiment="original",arm="direct_generation",context_agent=False,parent="frozen reference",history="direct generation control",selection="public payoff LCB",test_feedback=False),
        dict(experiment="feedback_ablation",arm="structured_memory",context_agent=True,parent="same fixed reference for both arms",history="same IDs and scores plus bounds, uncertainty and labeled historical behavior/mechanisms",selection="calibration confidence-gap score",test_feedback=False),
        dict(experiment="feedback_ablation",arm="score_only",context_agent=True,parent="same fixed reference for both arms",history="same IDs, source hashes, status and scalar gaps",selection="calibration confidence-gap score",test_feedback=False)])
    fig,ax=plt.subplots(figsize=(7.0,3.8)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    ax.text(0,.99,"A  Search loop and independent gap evaluation",weight="bold",fontsize=10,va="top")
    box(ax,.015,.67,.20,.17,"Context Agent\nstructured history","#E4EFF7")
    box(ax,.30,.67,.19,.17,"Proposal Agent\nprogram edit","#FFF0E4")
    box(ax,.58,.67,.19,.17,"Python algorithm\nfit + predict\nstopping policy","#E5F2EB")
    arrow(ax,(.22,.755),(.295,.755)); arrow(ax,(.495,.755),(.575,.755))
    box(ax,.025,.34,.21,.16,"Experience Bank\ncode, outcome\nand lineage","#E4EFF7")
    box(ax,.32,.34,.42,.16,"Public evaluator\nrollout payoff and uncertainty","#E5F2EB")
    arrow(ax,(.675,.665),(.675,.505)); arrow(ax,(.315,.42),(.242,.42))
    arrow(ax,(.12,.505),(.12,.665))
    ax.text(.49,.55,"original feedback: payoff LCB ↑",ha="center",fontsize=7.2)
    box(ax,.795,.37,.185,.19,"Freeze candidate\nbefore final test","#F1EFF8")
    arrow(ax,(.77,.68),(.86,.565))
    box(ax,.31,.055,.40,.18,"Independent test paths\npolicy return Lπ\ncommon dual upper U","#E5F2EB")
    box(ax,.78,.055,.20,.18,"Gap = U − Lπ\n+ uncertainty","#F1EFF8")
    arrow(ax,(.875,.365),(.71,.18)); arrow(ax,(.715,.145),(.775,.145))
    ax.text(.02,.145,"No hidden-test\nfeedback to search",fontsize=7,color="#6A516D",va="center")
    export(fig,"fig01_pipeline",
      "Executed original pipeline. Public payoff LCB drives search. Candidate selection freezes before independent primal–dual evaluation; hidden gap is not fed back. The upper-bound construction is shared across Python policies.")
    fig,ax=plt.subplots(figsize=(7.0,3.2)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    ax.text(0,.98,"B  Matched feedback ablation",weight="bold",fontsize=9,va="top")
    box(ax,.27,.74,.46,.15,"Shared frozen history + identical parent\n3 seeds × 2 prescribed operators","#ECEFF3")
    box(ax,.03,.42,.40,.22,"Structured memory\nscore + bounds + uncertainty\nhistorical behavior and mechanisms","#E4EFF7")
    box(ax,.57,.42,.40,.22,"Score-only memory\nsame candidate IDs and source hashes\nstatus + scalar gap","#FFF0E4")
    arrow(ax,(.35,.735),(.235,.645)); arrow(ax,(.65,.735),(.765,.645))
    box(ax,.03,.20,.40,.14,"Context → Proposal\nsame model and call/time caps","#E4EFF7")
    box(ax,.57,.20,.40,.14,"Context → Proposal\nsame model and call/time caps","#FFF0E4")
    arrow(ax,(.23,.415),(.23,.345)); arrow(ax,(.77,.415),(.77,.345))
    ax.text(.5,.075,"Calibration gap → freeze selections → separate final holdout",ha="center",fontsize=8)
    arrow(ax,(.23,.195),(.40,.095)); arrow(ax,(.77,.195),(.60,.095))
    export(fig,"fig02_feedback_design",
      "New fixed-history, fixed-parent feedback ablation. Both arms allow one Context and one Proposal call per attempt, the same model, operator, parent and evaluator budget; a failed Context prevents a Proposal call. Only memory contents differ. Ten frozen Context plans continue after a documented pre-generation workspace recovery; two Context timeouts are retained. This isolates feedback representation, not complete architecture or recursive self-improvement.")
    (FIG/"architecture.mmd").write_text(
      "flowchart LR\n  EB[Experience Bank] --> C[Context Agent]\n  C --> P[Proposal Agent]\n  P --> A[Python stopping algorithm]\n  A --> E[Public payoff evaluator]\n  E --> EB\n  A --> F[Freeze selected candidate]\n  F --> H[Independent primal-dual evaluation]\n  U[Common evaluator-owned upper bound] --> H\n")
    return design

def search_plot(t):
    fig,axes=plt.subplots(2,3,figsize=(7.0,4.5),sharex="col")
    for j,seed in enumerate(SEEDS):
        for mode in COL.keys():
            if mode not in ["context_proposal","direct_generation"]: continue
            s=t[(t["seed"]==seed)&(t["mode"]==mode)]
            if len(s)!=5: raise RuntimeError(f"Missing search trajectory: {seed}, {mode}")
            x=s.cumulative_model_seconds/60
            axes[0,j].step(x,s.public_incumbent_score,where="post",color=COL[mode],label="Context → Proposal" if mode=="context_proposal" else "Direct generation")
            axes[0,j].scatter(x.iloc[1:],s.candidate_public_score.iloc[1:],s=18,color=COL[mode],alpha=.55)
            axes[1,j].step(x,s.calibration_incumbent_gap,where="post",color=COL[mode])
            axes[1,j].scatter(x.iloc[1:],s.candidate_calibration_gap.iloc[1:],s=18,color=COL[mode],alpha=.55)
            fail=s[s.status!="ok"]; fail=fail[fail.status!="seed"]
            if len(fail):
                axes[0,j].scatter(fail.cumulative_model_seconds/60,fail.public_incumbent_score,marker="x",s=34,color=COL[mode])
        axes[0,j].set_title(f"Seed {seed}")
        axes[0,j].axhline(0,color="#888",ls=":",lw=.8)
        axes[0,j].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        axes[1,j].set_xlabel("Cumulative model-call time (min)")
        axes[1,j].ticklabel_format(axis="y",style="plain",useOffset=False)
    axes[0,0].set_ylabel("Public payoff-improvement LCB ↑")
    axes[1,0].set_ylabel("Post-hoc calibration gap ↓")
    axes[0,0].legend(loc="best",fontsize=6.5)
    fig.suptitle("Original search: candidates and seed-inclusive incumbents",fontsize=10,y=1.02)
    fig.tight_layout()
    export(fig,"fig04_search_trajectory",
      "Original two-round search, four generation attempts per seed and pipeline. Lines include the initial best reference and follow public-score selection; dots are individual generated candidates, crosses are generation failures. Bottom panels independently re-evaluate those same incumbents on the supplemental calibration panel. These are post-hoc gap diagnostics, not evidence that the original search consumed gap feedback. Model time includes Context calls and timeouts, not parallel elapsed time.")

def result_plot(d,cells):
    if not len(d): return
    names=[n for n in ORDER if n in set(d.method)]
    fig,axes=plt.subplots(1,2,figsize=(8.0,5.3),gridspec_kw={"width_ratios":[1.05,1]})
    y=np.arange(len(names))
    seedcolors=["#0072B2","#D55E00","#009E73"]
    for k,seed in enumerate(SEEDS):
        sub=d[d.seed==seed]; b=sub[sub.method=="linear_ridge"]
        if not len(b): continue
        bscore=b.iloc[0].score
        for i,name in enumerate(names):
            r=sub[sub.method==name]
            if not len(r) or not finite(r.iloc[0].score): continue
            axes[1].scatter(r.iloc[0].score-bscore,i+(k-1)*.16,s=22,color=seedcolors[k],label=f"Seed {seed}" if i==0 else None)
    axes[1].axvline(0,color="#555",lw=.8)
    axes[1].set_yticks(y,[""]*len(names)); axes[1].invert_yaxis()
    axes[1].set_xlabel("Gap-score difference vs Ridge\nnegative = smaller gap")
    axes[1].set_title("B  Paired comparison, fresh holdout")
    axes[1].legend(loc="best",fontsize=6.5)
    for i,name in enumerate(names):
        row=d[(d.seed==17)&(d.method==name)]
        if not len(row): continue
        c=cells[(cells.stage=="holdout")&(cells.key==row.iloc[0]["key"])&(cells.instance_id=="hidden-000")]
        if not len(c): continue
        c=c.iloc[0]; k=c.strike
        axes[0].plot([c.lower_normalized,c.upper_normalized],[i,i],color="#93B1C5",lw=1)
        axes[0].errorbar(c.lower_normalized,i,xerr=1.96*c.lower_se/k,fmt="o",color="#0072B2",ms=3,capsize=2)
        axes[0].errorbar(c.upper_normalized,i,xerr=1.96*c.upper_se/k,fmt="s",color="#CC7945",ms=3,capsize=2)
    axes[0].set_yticks(y,[LABEL[n] for n in names]); axes[0].invert_yaxis()
    axes[0].set_xlabel("Value / strike  (Lπ ●, U ■)")
    axes[0].set_title("A  Bounds on fixed case: seed 17, #0")
    fig.tight_layout()
    export(fig,"fig03_bounds_and_results",
      "Fresh holdout comparison. Left: raw normalized lower and upper estimates with component ±1.96 SE bars on the preselected seed-17 / hidden-000 case. They are conditional Monte Carlo intervals, not population uncertainty. Right: gap-score differences from Ridge under matched seed/instance budgets; each point is one seed, no unsupported significance claim. New ablation incumbents are selected solely by calibration scores with the parent retained as an option.")

def feedback_plot(pairs,selected,programs):
    if len(pairs)!=6 or not len(selected): return
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.85))
    colors=["#0072B2","#D55E00","#009E73"]
    for i,seed in enumerate(SEEDS):
        sub=pairs[pairs.seed==seed]
        a=sub.structured_memory_calibration_reduction.mean()
        b=sub.score_only_calibration_reduction.mean()
        axes[0].plot([0,1],[a,b],"-o",color=colors[i],ms=4,label=f"Seed {seed}")
        pp=load(OUT/f"ablation/s{seed}/frozen_parent.json")
        base=programs[(programs.stage=="holdout")&(programs.key==pp["key"])]
        vals=[]
        for arm in ["structured_memory","score_only"]:
            r=selected[(selected.seed==seed)&(selected.method==arm)]
            vals.append(base.iloc[0].score-r.iloc[0].score if len(base) and len(r) else np.nan)
        offset=(i-1)*.045
        axes[1].plot(np.array([0,1])+offset,vals,"-o",color=colors[i],ms=4)
    for ax in axes:
        ax.axhline(0,color="#555",ls=":",lw=.8)
        ax.set_xticks([0,1],["Structured\nmemory","Score-only\nmemory"])
        ax.set_xlim(-.25,1.25)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
    axes[0].set_title("A  All proposals: calibration")
    axes[0].set_ylabel("Gap reduction vs shared parent ↑")
    axes[1].set_title("B  Selected incumbent: holdout")
    axes[1].text(.5,.90,"Seeds 29 and 41: parent retained",transform=axes[1].transAxes,
                 ha="center",fontsize=6.5,color="#526273")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    export(fig,"fig05_feedback_ablation",
      "Matched feedback ablation with three seed blocks and two prescribed operators. Left: mean proposal gap reduction within seed across both operators; generation failures retain the parent (zero reduction) and remain in the denominator. Right: fresh-holdout reduction of the calibration-selected incumbent, allowing retention of the initial parent; small horizontal offsets distinguish overlapping seeds. Lines pair the two feedback arms within a seed. No claim of statistical superiority from three blocks.")

def behavior_plot():
    points=[]
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.6),sharey=True)
    for mode in ["context_proposal","direct_generation"]:
        run=load(OUT/f"inputs/original_runs/{mode}-s17.json")
        winner=run["winner"]
        cells=winner["result"]["metrics"]["summaries"]
        for j,inst in enumerate(["public-put-atm","public-put-high-vol"]):
            sub=[c for c in cells if c["instance_id"]==inst]
            if not sub: continue
            for repeat,cell in enumerate(sub):
                for actor,field in [("candidate","candidate_exercise_rate_by_time"),("evaluator_baseline","baseline_exercise_rate_by_time")]:
                    for date_index,rate in enumerate(cell[field]):
                        points.append(dict(seed=17,mode=mode,candidate_id=winner["candidate_id"],
                            instance_id=inst,repeat=repeat,actor=actor,exercise_date_index=date_index,stop_fraction=rate))
            arr=np.array([c["candidate_exercise_rate_by_time"] for c in sub]); mean=arr.mean(axis=0)
            axes[j].plot(np.arange(len(mean)),mean,"-o",ms=3,color=COL[mode],
                label="Context → Proposal" if mode=="context_proposal" else "Direct generation")
            if mode=="context_proposal":
                base=np.mean([c["baseline_exercise_rate_by_time"] for c in sub],axis=0)
                axes[j].plot(np.arange(len(base)),base,"--",color="#777",label="Evaluator baseline")
            axes[j].set_title("ATM put" if j==0 else "High-volatility put")
            axes[j].set_xlabel("Exercise-date index (last = maturity)")
    axes[0].set_ylabel("Fraction of paths stopping at date")
    axes[0].legend(fontsize=6.5); fig.tight_layout()
    export(fig,"fig06_stopping_behavior",
      "Observed stopping-time distributions of original frozen winners at seed 17, averaged across two repeats on the original public panel. The last date includes all remaining paths at maturity. This is a behavior illustration, not proof of why a mechanism worked or failed.")
    return csv("stopping_behavior",points)

def export_workbook(frames):
    import openpyxl
    from openpyxl.styles import Font,PatternFill,Alignment
    path=DATA/"paper_data.xlsx"
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        for name,df in frames.items():
            if len(df.columns):
                clean=df.copy()
                for col in clean.select_dtypes(include="object"):
                    clean[col]=clean[col].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x)
                clean.to_excel(writer,sheet_name=name[:31],index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:
                c.fill=PatternFill("solid",fgColor="24465D"); c.font=Font(color="FFFFFF",bold=True)
                c.alignment=Alignment(wrap_text=True,vertical="center")
            ws.row_dimensions[1].height=30
            for col in ws.columns:
                letter=col[0].column_letter
                ws.column_dimensions[letter].width=min(44,max(12,max(len(str(c.value or "")) for c in list(col)[:40])+2))
                for c in list(col)[1:]:
                    if isinstance(c.value,float): c.number_format="0.000000"
    check=openpyxl.load_workbook(path,read_only=True)
    assert check.sheetnames; check.close()

def cost_and_reliability(attempts,feedback,calls):
    rows=[]
    for experiment,frame,column,names in [
        ("original",attempts,"mode",["context_proposal","direct_generation"]),
        ("feedback_ablation",feedback,"arm",["structured_memory","score_only"])]:
        if not len(frame): continue
        for name in names:
            sub=frame[frame[column]==name]
            if not len(sub): continue
            c=calls[(calls.experiment==experiment)&(calls.method==name)]
            tokens=pd.to_numeric(c.tokens_reported,errors="coerce")
            rows.append(dict(experiment=experiment,method=name,attempts=len(sub),
                valid_candidates=int((sub.status=="ok").sum()),failed_attempts=int((sub.status!="ok").sum()),
                context_calls=int((c.role=="context").sum()),proposal_calls=int((c.role=="proposal").sum()),
                model_seconds=float(c.wall_seconds.sum()),reported_tokens_lower_bound=float(tokens.sum()),
                calls_without_token_report=int(tokens.isna().sum()),
                failures_in_denominator=True,wall_time_scope="Sum of model-call duration; not parallel elapsed time"))
    costs=csv("cost_and_reliability",rows)
    if len(costs)!=4: return costs
    names=["Original\nContext", "Original\ndirect", "New\nstructured\nmemory", "New\nscore-only\nmemory"]
    colors=["#0072B2","#D55E00","#0072B2","#D55E00"]
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.2))
    x=np.arange(4)
    for i,r in costs.iterrows():
        axes[0].bar(i,r.valid_candidates,color=colors[i],width=.62)
        axes[0].bar(i,r.failed_attempts,bottom=r.valid_candidates,color="#D9DDE2",width=.62,hatch="///",edgecolor="#9AA2A9",lw=.4)
        axes[0].text(i,r.attempts+.18,f"{r.valid_candidates}/{r.attempts}",ha="center",fontsize=8)
        axes[1].bar(i,r.model_seconds/60,color=colors[i],width=.62)
        axes[1].text(i,r.model_seconds/60+.5,f"{r.model_seconds/60:.1f}",ha="center",fontsize=8)
    for ax in axes:
        ax.set_xticks(x,names,fontsize=7); ax.axvline(1.5,color="#9AA2A9",ls=":",lw=.8)
    axes[0].set_ylabel("Generation attempts (valid / total)")
    axes[0].set_ylim(0,14); axes[0].set_title("A  All attempts retained")
    axes[1].set_ylabel("Cumulative model-call minutes")
    axes[1].set_ylim(0,costs.model_seconds.max()/60*1.15); axes[1].set_title("B  Recorded model cost")
    fig.tight_layout()
    export(fig,"fig07_cost_and_reliability",
      "Actual generation reliability and model-call duration, including failed and timed-out calls. Hatched segments are failures. Original pipelines and the new feedback experiment are separate experiments, not matched against each other. New arms share call/time caps; realized calls, tokens and duration can differ. Context calls from the infrastructure-interrupted run are counted once; recovery issues only previously uncalled Proposals. Missing token reports are explicitly retained in the cost table.")
    return costs

def gallery(summary,attempts,feedback):
    stage="已完成" if (OUT/"experiments_complete.json").exists() else "实验进行中"
    head=f"""# Math-AI 论文图表与数据包\n\n状态：{stage}。图中英文标签用于论文；本页说明数据来源与结论边界。\n\n## 数据分组\n\n- 原始搜索：2026-09-05/06 的真实模型记录，3 seeds、两轮、24 次生成尝试，5 次超时。\n- 原始隐藏审计补全：七类经典基线使用原来的隐藏请求重评；属于回顾性补充。\n- 新校准面板：冻结程序统一计算 gap，供反馈对照使用；不属于最终测试。\n- 新独立测试：四个实例、512 训练路径、2048 定价路径、256 outer × 16 inner；全部选择冻结后才执行。\n- 新反馈对照：相同历史、父程序、模型、操作符与预算上限；结构化反馈 vs 只提供分数。3 seed × 2 操作符，共 6 对。是固定历史的反馈消融，不是新增多轮 RSI。\n\n## 阅读规则\n\n核心数值为上下界间隙。gap score 是归一化置信间隙的均值 + 0.25 × 90 分位数；不是价格相对误差。每个图保留底层 CSV。跨 seed 点/SD 只有 3 个独立区组；路径级置信区间不能解释为跨任务统计显著性。失败保留在尝试与成本表中。\n\n原始搜索过程图上方是当时真实使用的公开收益评分；下方是对同一公开选中 incumbent 的事后 gap 重评。不能将其称作原实验的 gap 反馈曲线。初始种子策略始终作为 incumbent 候选保留。\n\n[Excel 数据总表](data/paper_data.xlsx) · [逐种子结果](data/main_results_by_seed.csv) · [汇总表](tables/main_results.md) · [实验预登记](protocol.json)\n\n"""
    if len(summary):
        head+="## 新独立测试主表\n\n"+summary[["label","lower_normalized_mean","upper_normalized_mean","score_mean","score_sd"]].to_markdown(index=False,floatfmt=".5f")+"\n\n"
    head+="## 实验记录与使用方式\n\n"
    head+="首次运行有 10 个已完成的 Context 计划在候选工作目录准备阶段遇到只读文件错误，另有 2 次 Context 超时。原始记录全部保留；恢复说明在 recovery_amendment.json。恢复只继续尚未发生的 Proposal 调用，没有重新生成 Context、修改候选或更换种子。\n\n"
    head+="主表同时给出原始搜索中最好的已生成算法，以及允许保留初始参考算法的最终 incumbent；后者更接近实际可部署的搜索结果。新反馈组按校准 gap 选取 incumbent，独立测试只做报告。\n\n"
    head+="[指标定义与文献](METHODS.md) · [字段字典](data/DICTIONARY.md) · [成本与失败记录](data/cost_and_reliability.csv) · [恢复说明](recovery_amendment.json) · [核验结果](verification.json)\n\n"
    head+="每幅图均提供 PDF、可编辑 SVG 和 300 dpi PNG。HTML 图册可直接查看；Excel 保留逐方法、逐 seed、逐实例数据。重建图表使用已冻结的 inputs 与 raw 数据，不需要再次调用模型。图中的架构描述以实际运行流程为准；没有将示意图当作实验结果。\n\n"
    if (OUT/"FINDINGS.md").exists(): head+="[本轮实验结论](FINDINGS.md)\n\n"
    head+="## 图表\n\n"
    for name,cap in sorted(CAPTIONS.items()):
        head+=f"### {name}\n\n![{name}]({FIG/name}.png)\n\n{cap}\n\n[PDF](figures/{name}.pdf) · [SVG](figures/{name}.svg)\n\n"
    (OUT/"README.md").write_text(head)
    cards="".join(f'<section><h2>{n}</h2><img src="figures/{n}.svg"><p>{c}</p><a href="figures/{n}.pdf">PDF</a> · <a href="figures/{n}.svg">SVG</a></section>' for n,c in sorted(CAPTIONS.items()))
    html='<!doctype html><html><meta charset="utf-8"><title>OpenHyra paper figures</title><style>body{max-width:1080px;margin:40px auto;font:16px system-ui;color:#183043;padding:0 24px}section{border-top:1px solid #dbe3e9;padding:24px 0}img{max-width:100%;display:block;margin:18px auto}p{line-height:1.6;color:#4b5c68}a{color:#0072b2}h1{font-size:28px}</style>'
    html+=f'<h1>OpenHyra · Math-AI 图表与数据</h1><p>状态：{stage}。统计口径和实验边界见 README。</p><a href="data/paper_data.xlsx">Excel 数据</a> · <a href="tables/main_results.md">结果表</a>'+cards+"</html>"
    (OUT/"index.html").write_text(html)
    save(FIG/"captions.json",CAPTIONS)

def main():
    programs,cells=collect()
    attempts,traj,origcalls=original_search()
    feedback,pairs,selected,newcalls=ablation_data(programs)
    results,summary=main_table(programs,selected,traj)
    calls=csv("model_calls",origcalls+newcalls)
    costs=cost_and_reliability(attempts,feedback,calls)
    infra=[]
    if (OUT/"recovery_amendment.json").exists():
        infra=load(OUT/"recovery_amendment.json")["attempts"]
    infra=csv("infrastructure_recovery",infra)
    design=architecture(); search_plot(traj); behavior=behavior_plot()
    result_plot(results,cells); feedback_plot(pairs,selected,programs)
    export_workbook({"Main summary":summary,"Main by seed":results,"All evaluations":programs,
        "Evaluation cells":cells,"Original attempts":attempts,"Original trajectory":traj,
        "Feedback attempts":feedback,"Feedback pairs":pairs,"Selected incumbents":selected,"Model calls":calls,
        "Cost and reliability":costs,"Infrastructure recovery":infra,"Architecture comparison":design,
        "Stopping behavior":behavior})
    gallery(summary,attempts,feedback)
    save(OUT/"asset_build.json",{"figures":list(CAPTIONS),"n_evaluations":len(programs),
        "n_cells":len(cells),"original_attempts":len(attempts),"feedback_attempts":len(feedback),
        "input_protocol_sha256":hashlib.sha256((OUT/"protocol.json").read_bytes()).hexdigest()})
    save(FIG/"figure_sources.json",{
        "fig01_pipeline":["data/architecture_comparison.csv","protocol.json"],
        "fig02_feedback_design":["data/architecture_comparison.csv","protocol.json","recovery_amendment.json"],
        "fig03_bounds_and_results":["data/main_results_by_seed.csv","data/evaluation_cells.csv"],
        "fig04_search_trajectory":["data/original_search_trajectory.csv","data/original_search_attempts.csv"],
        "fig05_feedback_ablation":["data/feedback_pairs.csv","data/selected_incumbents.csv"],
        "fig06_stopping_behavior":["data/stopping_behavior.csv"],
        "fig07_cost_and_reliability":["data/cost_and_reliability.csv","data/model_calls.csv"]})
    print(f"Built {len(CAPTIONS)} figures; {len(programs)} evaluations; {len(feedback)} feedback attempts")

if __name__=="__main__": main()

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _collect_paths(out_dir: str, filename: str) -> List[str]:
    out_dir = os.path.abspath(out_dir)
    paths = sorted(glob.glob(os.path.join(out_dir, "shard_*", filename)))
    if paths:
        return paths
    p = os.path.join(out_dir, filename)
    if os.path.exists(p):
        return [p]
    return []


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return float(v)
    except Exception:
        pass
    return None


def _algo_key(r: Dict[str, Any]) -> str:
    b = str(r.get("backend_lib") or r.get("optimizer") or "")
    m = str(r.get("method") or "")
    return f"{b}-{m}".strip("-")


def _load_problem_instances(out_dir: str) -> Dict[str, Dict[str, Any]]:
    pi_map: Dict[str, Dict[str, Any]] = {}
    for fp in _collect_paths(out_dir, "problem_instance.jsonl"):
        for row in _read_jsonl(fp):
            iid = str(row.get("instance_id") or "")
            if not iid:
                continue
            pi_map[iid] = row
    return pi_map


def _load_run_summaries(out_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in _collect_paths(out_dir, "run_summary.jsonl"):
        rows.extend(list(_read_jsonl(fp)))
    return rows


def _agg_instance_algo(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in run_rows:
        iid = str(r.get("instance_id") or "")
        if not iid:
            continue
        ak = _algo_key(r)
        if not ak:
            continue
        seed = int(r.get("seed") or 0)
        k = (iid, ak)
        d = by_key.get(k)
        if d is None:
            d = {
                "instance_id": iid,
                "algo": ak,
                "seeds": set(),
                "best_feasible_list": [],
                "walltime_to_best_list": [],
                "anytime_auc_list": [],
                "feasible_list": [],
            }
            by_key[k] = d
        d["seeds"].add(seed)
        bf = _safe_float(r.get("best_feasible_objective"))
        if bf is not None:
            d["best_feasible_list"].append(float(bf))
        wt = _safe_float(r.get("walltime_to_best"))
        if wt is not None:
            d["walltime_to_best_list"].append(float(wt))
        auc = _safe_float(r.get("anytime_auc"))
        if auc is not None:
            d["anytime_auc_list"].append(float(auc))
        d["feasible_list"].append(1.0 if bool(r.get("final_feasible_flag", False)) else 0.0)

    out: List[Dict[str, Any]] = []
    for (_, _), d in by_key.items():
        bf_list = d["best_feasible_list"]
        wt_list = d["walltime_to_best_list"]
        auc_list = d["anytime_auc_list"]
        feas_list = d["feasible_list"]
        out.append(
            {
                "instance_id": d["instance_id"],
                "algo": d["algo"],
                "n_seeds": int(len(d["seeds"])),
                "best_feasible_mean": (sum(bf_list) / float(len(bf_list))) if bf_list else float("inf"),
                "walltime_to_best_mean": (sum(wt_list) / float(len(wt_list))) if wt_list else None,
                "anytime_auc_mean": (sum(auc_list) / float(len(auc_list))) if auc_list else None,
                "feasible_rate": (sum(feas_list) / float(len(feas_list))) if feas_list else 0.0,
            }
        )
    return out


def _family_key(pi_row: Optional[Dict[str, Any]]) -> str:
    if not pi_row:
        return "unknown"
    return str(pi_row.get("problem_type") or "unknown")


def _pde_key(pi_row: Optional[Dict[str, Any]]) -> str:
    if not pi_row:
        return "unknown"
    return str(pi_row.get("pde_type") or "unknown")


def _infer_pde_type_from_problem_type(problem_type: str) -> str:
    s = str(problem_type or "").lower()
    if "advection" in s:
        return "hyperbolic"
    if "blackscholes" in s or "heat_time" in s:
        return "parabolic"
    if "sns" in s or "thermal_fins" in s or "neutron_diffusion" in s or "thmf" in s or "iaea" in s or "heat_v" in s:
        return "elliptic"
    return "unknown"


def _infer_pde_type_from_instance_id(instance_id: str) -> str:
    return _infer_pde_type_from_problem_type(_infer_problem_type_from_instance_id(instance_id))


def _infer_problem_type_from_instance_id(instance_id: str) -> str:
    s = str(instance_id or "")
    if not s.startswith("inst_"):
        return "unknown"
    parts = s.split("_")
    if len(parts) < 3:
        return "unknown"
    return "_".join(parts[1:-1]) or "unknown"


def _count_dataset_instances_from_sweep(out_dir: str) -> Optional[int]:
    cfg_paths = _collect_paths(out_dir, "full_sweep_config.json")
    if not cfg_paths:
        return None
    for fp in cfg_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ds = str(cfg.get("instances_from_dataset_path") or "").strip()
            if not ds or not os.path.exists(ds):
                continue
            n = 0
            with open(ds, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
            return int(n)
        except Exception:
            continue
    return None


def _compute_family_tables(out_dirs: List[str]):
    import pandas as pd

    pi_all: Dict[str, Dict[str, Any]] = {}
    run_all: List[Dict[str, Any]] = []
    for od in out_dirs:
        pi_all.update(_load_problem_instances(od))
        run_all.extend(_load_run_summaries(od))

    ia = _agg_instance_algo(run_all)
    df = pd.DataFrame.from_records(ia)
    if df.empty:
        raise RuntimeError("no run_summary rows found")

    def _family_for_instance(iid: str) -> str:
        pi = pi_all.get(str(iid))
        fam = _family_key(pi)
        if fam != "unknown":
            return fam
        return _infer_problem_type_from_instance_id(str(iid))

    def _pde_for_instance(iid: str) -> str:
        pi = pi_all.get(str(iid))
        p = _pde_key(pi)
        if p != "unknown":
            return p
        if pi is not None:
            pt = str(pi.get("problem_type") or "")
            p2 = _infer_pde_type_from_problem_type(pt)
            if p2 != "unknown":
                return p2
        return _infer_pde_type_from_instance_id(str(iid))

    df["family"] = df["instance_id"].map(lambda x: _family_for_instance(str(x)))
    df["pde_type"] = df["instance_id"].map(lambda x: _pde_for_instance(str(x)))

    df["has_run"] = 1
    df["best_feasible_mean"] = pd.to_numeric(df["best_feasible_mean"], errors="coerce")
    df["walltime_to_best_mean"] = pd.to_numeric(df["walltime_to_best_mean"], errors="coerce")
    df["anytime_auc_mean"] = pd.to_numeric(df["anytime_auc_mean"], errors="coerce")
    df["feasible_rate"] = pd.to_numeric(df["feasible_rate"], errors="coerce").fillna(0.0)

    family_instances = (
        df[["family", "pde_type", "instance_id"]]
        .drop_duplicates(subset=["family", "pde_type", "instance_id"])
        .groupby(["family", "pde_type"], as_index=False)["instance_id"]
        .count()
        .rename(columns={"instance_id": "n_instances_with_pi"})
    )

    import numpy as np

    df_noninf = df.copy()
    df_noninf.loc[~np.isfinite(df_noninf["best_feasible_mean"].astype(float)), "best_feasible_mean"] = float("inf")

    win_idx = (
        df_noninf.sort_values(["family", "instance_id", "best_feasible_mean", "algo"])
        .groupby(["family", "instance_id"], as_index=False)
        .head(1)[["family", "instance_id", "algo"]]
    )
    win_idx["top1"] = 1

    df = df.merge(win_idx, on=["family", "instance_id", "algo"], how="left")
    df["top1"] = df["top1"].fillna(0).astype(int)

    algo_metrics = (
        df.groupby(["family", "pde_type", "algo"], as_index=False)
        .agg(
            n_instances_observed=("instance_id", "nunique"),
            top1_count=("top1", "sum"),
            feasible_rate=("feasible_rate", "mean"),
            best_feasible_median=("best_feasible_mean", "median"),
            walltime_to_best_median=("walltime_to_best_mean", "median"),
            anytime_auc_median=("anytime_auc_mean", "median"),
        )
        .sort_values(["family", "top1_count"], ascending=[True, False])
    )

    fam_progress = df.groupby(["family", "pde_type"], as_index=False).agg(
        n_instances_with_any=("instance_id", "nunique"),
        n_algos=("algo", "nunique"),
        n_seeds=("n_seeds", "max"),
        n_rows=("instance_id", "size"),
    )
    fam_progress = fam_progress.merge(family_instances, on=["family", "pde_type"], how="left")

    totals: Dict[str, int] = {}
    for od in out_dirs:
        n_total = _count_dataset_instances_from_sweep(od)
        if n_total is None:
            continue
        pi_map = _load_problem_instances(od)
        fams = {str(v.get("problem_type") or "") for v in pi_map.values() if isinstance(v, dict)}
        if len(fams) == 1:
            totals[list(fams)[0]] = int(n_total)

    fam_progress["n_instances_total"] = fam_progress["family"].map(lambda f: totals.get(str(f)))
    fam_progress["n_instances_total"] = fam_progress["n_instances_total"].fillna(fam_progress["n_instances_with_pi"])
    fam_progress["n_instances_total"] = fam_progress["n_instances_total"].fillna(fam_progress["n_instances_with_any"])
    fam_progress["expected_rows"] = fam_progress["n_instances_total"] * fam_progress["n_algos"]
    fam_progress["completion_pct"] = 100.0 * (fam_progress["n_rows"] / fam_progress["expected_rows"]).replace([math.inf], 0.0)

    algo_metrics = algo_metrics.merge(fam_progress[["family", "pde_type", "n_instances_with_any"]], on=["family", "pde_type"], how="left")
    algo_metrics["top1_share"] = algo_metrics["top1_count"] / algo_metrics["n_instances_with_any"].clip(lower=1)

    top1_dist = (
        algo_metrics[["family", "pde_type", "algo", "top1_count", "top1_share"]]
        .sort_values(["family", "top1_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return fam_progress, algo_metrics, top1_dist


def _plot_top1_stacked(top1_df, out_png: str, top_k: int = 10):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = top1_df.copy()
    total = df.groupby("algo", as_index=False)["top1_count"].sum().sort_values("top1_count", ascending=False)
    keep = set(total.head(int(top_k))["algo"].tolist())
    df["algo_plot"] = df["algo"].apply(lambda x: x if x in keep else "Other")
    df2 = df.groupby(["family", "pde_type", "algo_plot"], as_index=False).agg(top1_share=("top1_share", "sum"))
    fam_order = (
        df2.groupby(["family", "pde_type"], as_index=False)["top1_share"].sum().sort_values(["pde_type", "family"])
    )
    fam_keys = [f"{r.family}\n({r.pde_type})" for r in fam_order.itertuples(index=False)]
    fam_map = {tuple([r.family, r.pde_type]): i for i, r in enumerate(fam_order.itertuples(index=False))}
    algos = sorted(df2["algo_plot"].unique().tolist(), key=lambda x: (x == "Other", x))

    mat = [[0.0 for _ in fam_keys] for _ in algos]
    for r in df2.itertuples(index=False):
        fi = fam_map.get((r.family, r.pde_type))
        if fi is None:
            continue
        ai = algos.index(r.algo_plot)
        mat[ai][fi] = float(r.top1_share)

    fig_w = max(10, 1.3 * len(fam_keys))
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bottoms = [0.0 for _ in fam_keys]
    colors = plt.cm.tab20.colors
    for i, algo in enumerate(algos):
        vals = mat[i]
        ax.bar(range(len(fam_keys)), vals, bottom=bottoms, label=algo, color=colors[i % len(colors)])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(fam_keys)))
    ax.set_xticklabels(fam_keys, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 Share (by instance)")
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_heatmap(algo_metrics_df, out_png: str, metric_col: str, top_k_algos: int = 12, log1p: bool = False):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = algo_metrics_df.copy()
    df = df[df["algo"].notnull()]
    df = df.replace([math.inf, -math.inf], math.nan)
    total = df.groupby("algo", as_index=False)["top1_count"].sum().sort_values("top1_count", ascending=False)
    keep = total.head(int(top_k_algos))["algo"].tolist()
    df = df[df["algo"].isin(keep)]

    fam_order = (
        df.groupby(["family", "pde_type"], as_index=False)["top1_count"].sum().sort_values(["pde_type", "family"])
    )
    fam_keys = [(r.family, r.pde_type) for r in fam_order.itertuples(index=False)]
    fam_labels = [f"{f}\n({p})" for f, p in fam_keys]
    algo_order = keep

    mat = []
    for a in algo_order:
        row = []
        for f, p in fam_keys:
            sub = df[(df["algo"] == a) & (df["family"] == f) & (df["pde_type"] == p)]
            if sub.empty:
                row.append(math.nan)
            else:
                v = sub.iloc[0].get(metric_col)
                v = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else math.nan
                if log1p and v is not math.nan:
                    v = math.log1p(max(0.0, float(v)))
                row.append(v)
        mat.append(row)

    fig_w = max(10, 1.2 * len(fam_labels))
    fig_h = max(4.5, 0.35 * len(algo_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(range(len(algo_order)))
    ax.set_yticklabels(algo_order)
    ax.set_xticks(range(len(fam_labels)))
    ax.set_xticklabels(fam_labels)
    ax.set_title(metric_col)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--topk_algos", type=int, default=10)
    args = ap.parse_args()

    out_dir = str(args.out_dir).strip()
    if not out_dir:
        import time

        out_dir = os.path.join("/home/ycl/AICO-Intellig/results", f"algo_landscape_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    fam_progress, algo_metrics, top1_dist = _compute_family_tables(args.out_dirs)

    fam_csv = os.path.join(out_dir, "family_progress.csv")
    algo_csv = os.path.join(out_dir, "algo_metrics_by_family.csv")
    top1_csv = os.path.join(out_dir, "top1_distribution.csv")
    fam_progress.to_csv(fam_csv, index=False)
    algo_metrics.to_csv(algo_csv, index=False)
    top1_dist.to_csv(top1_csv, index=False)

    _plot_top1_stacked(top1_dist, os.path.join(out_dir, "fig_top1_share_stacked.png"), top_k=int(args.topk_algos))
    _plot_heatmap(algo_metrics, os.path.join(out_dir, "fig_heatmap_walltime_to_best_median.png"), "walltime_to_best_median", top_k_algos=int(args.topk_algos), log1p=True)
    _plot_heatmap(algo_metrics, os.path.join(out_dir, "fig_heatmap_anytime_auc_median.png"), "anytime_auc_median", top_k_algos=int(args.topk_algos), log1p=True)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

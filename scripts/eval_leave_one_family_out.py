import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _canonical_family_name(name: str) -> str:
    s = str(name or "").strip().lower()
    if s in {"iaea_v1", "neutron_diffusion_family_v1"}:
        return "iaea_v1"
    if s in {"heat_v1", "thermal_fins_family_v1"}:
        return "heat_v1"
    if s in {"advection_v1", "ex_advection2d_family_v1"}:
        return "advection_v1"
    if s in {"blackscholes_v1", "ex_blackscholes2d_family_v1"}:
        return "blackscholes_v1"
    if s in {"heat_time_v1", "ex_heat_time_family_v1"}:
        return "heat_time_v1"
    return str(name or "unknown")


def _family_version(inst: Dict[str, Any], pi_row: Optional[Dict[str, Any]]) -> str:
    typed = {}
    if isinstance(pi_row, dict) and isinstance(pi_row.get("typed_config_json"), dict):
        typed = dict(pi_row.get("typed_config_json") or {})
    fam = str(typed.get("family_version") or "")
    if fam:
        return _canonical_family_name(fam)
    return _canonical_family_name(str(inst.get("problem_type") or "unknown"))


def _pde_type(inst: Dict[str, Any], pi_row: Optional[Dict[str, Any]]) -> str:
    typed = {}
    if isinstance(pi_row, dict) and isinstance(pi_row.get("typed_config_json"), dict):
        typed = dict(pi_row.get("typed_config_json") or {})
    pde = ""
    if pi_row:
        pde = str(pi_row.get("pde_type") or "")
    if not pde:
        pde = str(typed.get("pde_type") or "")
    return pde or "unknown"


def _majority(labels: List[str]) -> Optional[str]:
    labels = [str(x) for x in labels if x]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _accuracy(pred: Dict[str, str], truth: Dict[str, str]) -> float:
    both = [iid for iid in pred.keys() if iid in truth]
    if not both:
        return 0.0
    return float(sum(1 for iid in both if pred[iid] == truth[iid]) / float(len(both)))


def _build_rank_target(
    runs: List[Dict[str, Any]],
    cand_key_set: set,
) -> Dict[Tuple[str, str], float]:
    by_i: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        iid = r.get("instance_id")
        if not iid:
            continue
        ckey = f"{r.get('optimizer')}-{r.get('method')}"
        if ckey not in cand_key_set:
            continue
        if not r.get("success", False):
            continue
        bv = r.get("best_value")
        if bv is None:
            continue
        try:
            by_i[str(iid)][ckey].append(float(bv))
        except Exception:
            continue

    rank_target: Dict[Tuple[str, str], float] = {}
    for iid, mp in by_i.items():
        items = []
        for k, vals in mp.items():
            if vals:
                items.append((k, sum(vals) / float(len(vals))))
        if not items:
            continue
        items.sort(key=lambda x: x[1])
        denom = float(len(items) - 1) if len(items) > 1 else 1.0
        for idx, (k, _) in enumerate(items):
            rank_target[(iid, k)] = float(idx) / denom
    return rank_target


def _make_model(model_type: str, random_state: int):
    model_type = str(model_type).strip().lower()
    if model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=False)),
                (
                    "reg",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        random_state=int(random_state),
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )
    if model_type == "sgd":
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import SGDRegressor
        from sklearn.pipeline import Pipeline

        return Pipeline(
            steps=[
                ("vec", DictVectorizer(sparse=True)),
                ("reg", SGDRegressor(loss="huber", max_iter=5000, tol=1e-4, random_state=int(random_state))),
            ]
        )
    if model_type == "catboost":
        from scripts.train_algo_ranker import CatBoostDictRegressor

        return CatBoostDictRegressor(random_state=int(random_state))
    raise ValueError("unsupported model_type")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="extra_trees")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--warmup_k", type=int, default=0)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(repo_root)
    from scripts import train_algo_ranker as tar

    out_dir = os.path.abspath(args.out_dir)
    runs_paths, dataset_paths = tar._collect_paths(out_dir)
    pi_paths = tar._collect_optional_paths(out_dir, "problem_instance.jsonl")
    ckpt_paths = tar._collect_optional_paths(out_dir, "run_checkpoint.jsonl")
    inst_map = tar._load_instance_map(dataset_paths)
    pi_map = tar._load_problem_instance_map(pi_paths)
    ckpt_map = tar._load_run_checkpoint_map(ckpt_paths, max_ratio=20)
    runs = tar._load_runs(runs_paths)
    candidates = tar._infer_candidates(runs)
    cand_keys = [c.key() for c in candidates]
    cand_key_set = set(cand_keys)

    family_of_instance: Dict[str, str] = {}
    pde_of_instance: Dict[str, str] = {}
    for iid, inst in inst_map.items():
        pi_row = pi_map.get(iid)
        family_of_instance[iid] = _family_version(inst, pi_row)
        pde_of_instance[iid] = _pde_type(inst, pi_row)

    families = sorted(set(family_of_instance.values()))
    family_results: List[Dict[str, Any]] = []

    def _make_model_local():
        model_type = str(args.model).strip().lower()
        if model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.pipeline import Pipeline

            return Pipeline(
                steps=[
                    ("vec", DictVectorizer(sparse=False)),
                    (
                        "reg",
                        ExtraTreesRegressor(
                            n_estimators=int(args.n_estimators),
                            random_state=int(args.random_state),
                            n_jobs=-1,
                            min_samples_leaf=2,
                        ),
                    ),
                ]
            )
        return _make_model(model_type=str(args.model), random_state=int(args.random_state))

    def _write_partial() -> None:
        macro_model = float(sum(x["model_top1_acc"] for x in family_results) / float(len(family_results))) if family_results else 0.0
        macro_global = float(sum(x["global_majority_top1_acc"] for x in family_results) / float(len(family_results))) if family_results else 0.0
        macro_pde = float(sum(x["per_pde_majority_top1_acc"] for x in family_results) / float(len(family_results))) if family_results else 0.0
        payload = {
            "out_dir": out_dir,
            "model": str(args.model),
            "random_state": int(args.random_state),
            "warmup_k": int(args.warmup_k),
            "n_estimators": int(args.n_estimators),
            "families_total": len(families),
            "families_completed": len(family_results),
            "families": family_results,
            "macro_avg": {
                "model_top1_acc": macro_model,
                "global_majority_top1_acc": macro_global,
                "per_pde_majority_top1_acc": macro_pde,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(os.path.abspath(args.out), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    for heldout_family in families:
        train_runs = []
        test_runs = []
        for r in runs:
            iid = r.get("instance_id")
            if not iid:
                continue
            iid = str(iid)
            fam = family_of_instance.get(iid, "unknown")
            if fam == heldout_family:
                test_runs.append(r)
            else:
                train_runs.append(r)

        rank_target_train = _build_rank_target(train_runs, cand_key_set)
        true_winner_train = tar._true_winners_by_instance(train_runs, cand_key_set)
        true_winner_test = tar._true_winners_by_instance(test_runs, cand_key_set)
        test_instances = sorted(true_winner_test.keys())

        X_train: List[Dict[str, Any]] = []
        y_train: List[float] = []
        for r in train_runs:
            iid = r.get("instance_id")
            if not iid:
                continue
            iid = str(iid)
            if not r.get("success", False):
                continue
            ckey = f"{r.get('optimizer')}-{r.get('method')}"
            target = rank_target_train.get((iid, ckey))
            if target is None:
                continue
            inst = inst_map.get(iid)
            if not inst:
                continue
            pi_row = pi_map.get(iid)
            feat = {}
            feat.update(tar._features_from_instance(inst, pi_row))
            feat["optimizer"] = str(r.get("optimizer") or "")
            feat["method"] = str(r.get("method") or "")
            feat["success"] = 1
            feat.update(tar._warm_features(r, int(args.warmup_k)))
            feat.update(tar._checkpoint_features(r, ckpt_map))
            X_train.append(feat)
            y_train.append(float(target))

        model = _make_model_local()
        if X_train and y_train and test_instances:
            model.fit(X_train, y_train)
            pred_model: Dict[str, str] = {}
            warm_pick: Dict[Tuple[str, str], Dict[str, Any]] = {}
            if int(args.warmup_k) > 0:
                by_i_alg: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
                for r in test_runs:
                    iid = str(r.get("instance_id"))
                    k = f"{r.get('optimizer')}-{r.get('method')}"
                    if k in cand_key_set:
                        by_i_alg[(iid, k)].append(r)
                for key, lst in by_i_alg.items():
                    lst.sort(key=lambda x: int(x.get("seed", 10**9)))
                    warm_pick[key] = lst[0]
            for iid in test_instances:
                inst = inst_map.get(iid)
                pi_row = pi_map.get(iid)
                if not inst:
                    continue
                rows = []
                keys = []
                for c in candidates:
                    feat = {}
                    feat.update(tar._features_from_instance(inst, pi_row))
                    feat["optimizer"] = c.optimizer
                    feat["method"] = c.method
                    feat["success"] = 1
                    r0 = warm_pick.get((iid, c.key()))
                    if r0:
                        feat.update(tar._warm_features(r0, int(args.warmup_k)))
                        feat.update(tar._checkpoint_features(r0, ckpt_map))
                    else:
                        feat.update(tar._warm_features({}, int(args.warmup_k)))
                        feat.update(tar._checkpoint_features({}, ckpt_map))
                    rows.append(feat)
                    keys.append(c.key())
                scores = model.predict(rows)
                best_i = min(range(len(keys)), key=lambda i: float(scores[i]))
                pred_model[iid] = keys[best_i]
            model_acc = _accuracy(pred_model, true_winner_test)
        else:
            model_acc = 0.0

        global_majority = _majority(list(true_winner_train.values()))
        pred_global = {iid: global_majority for iid in test_instances if global_majority}
        global_acc = _accuracy(pred_global, true_winner_test)

        by_pde_train: Dict[str, List[str]] = defaultdict(list)
        for iid, winner in true_winner_train.items():
            by_pde_train[pde_of_instance.get(iid, "unknown")].append(winner)
        pde_majority = {k: _majority(v) for k, v in by_pde_train.items()}

        pred_pde: Dict[str, str] = {}
        for iid in test_instances:
            lab = pde_majority.get(pde_of_instance.get(iid, "unknown")) or global_majority
            if lab:
                pred_pde[iid] = lab
        pde_acc = _accuracy(pred_pde, true_winner_test)

        family_results.append(
            {
                "heldout_family": heldout_family,
                "pde_type": _majority([pde_of_instance.get(iid, "unknown") for iid in test_instances]) or "unknown",
                "n_test_instances": int(len(test_instances)),
                "train_instances": int(len(true_winner_train)),
                "model_top1_acc": float(model_acc),
                "global_majority_top1_acc": float(global_acc),
                "per_pde_majority_top1_acc": float(pde_acc),
                "global_majority_label": global_majority,
            }
        )
        print(
            json.dumps(
                {
                    "heldout_family": heldout_family,
                    "model_top1_acc": family_results[-1]["model_top1_acc"],
                    "global_majority_top1_acc": family_results[-1]["global_majority_top1_acc"],
                    "per_pde_majority_top1_acc": family_results[-1]["per_pde_majority_top1_acc"],
                    "completed": len(family_results),
                    "total": len(families),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _write_partial()

    _write_partial()
    with open(os.path.abspath(args.out), "r", encoding="utf-8") as f:
        out = json.load(f)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

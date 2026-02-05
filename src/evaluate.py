import os
import json
import sys
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
from scipy import stats


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args[k] = v
    return args


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(df, run_id, out_dir):
    paths = []
    if "accuracy" not in df.columns:
        return paths
    fig = plt.figure()
    plt.plot(df.index, df["accuracy"], label="accuracy")
    if "proxy_success_rate" in df.columns:
        plt.plot(df.index, df["proxy_success_rate"], label="proxy_success_rate")
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.title(f"{run_id} Learning Curve")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_learning_curve.pdf")
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_confusion_matrix(summary: Dict[str, Any], run_id: str, out_dir: str):
    tp = summary.get("proxy_true_tp", 0)
    fp = summary.get("proxy_true_fp", 0)
    tn = summary.get("proxy_true_tn", 0)
    fn = summary.get("proxy_true_fn", 0)
    mat = np.array([[tp, fp], [fn, tn]])
    fig = plt.figure()
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", xticklabels=["True", "False"], yticklabels=["True", "False"])
    plt.xlabel("Proxy Pass")
    plt.ylabel("True Pass")
    plt.title(f"{run_id} Proxy vs True Confusion")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_id}_confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_bar_comparison(metrics: Dict[str, Dict[str, float]], metric_name: str, out_dir: str):
    fig = plt.figure()
    run_ids = list(metrics[metric_name].keys())
    values = [metrics[metric_name][rid] for rid in run_ids]
    sns.barplot(x=run_ids, y=values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.title(f"Comparison {metric_name}")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    path = os.path.join(out_dir, f"comparison_{metric_name}_bar_chart.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_box_comparison(histories: Dict[str, pd.DataFrame], metric_name: str, out_dir: str):
    rows = []
    for run_id, df in histories.items():
        if metric_name in df.columns:
            for v in df[metric_name].dropna().tolist():
                rows.append({"run_id": run_id, metric_name: v})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    fig = plt.figure()
    sns.boxplot(data=df, x="run_id", y=metric_name)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric_name} Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, f"comparison_{metric_name}_box_plot.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def plot_metric_table(metrics: Dict[str, Dict[str, float]], out_dir: str):
    metric_names = list(metrics.keys())
    run_ids = sorted({rid for m in metrics.values() for rid in m.keys()})
    table = []
    for rid in run_ids:
        row = [rid]
        for m in metric_names:
            row.append(f"{metrics[m].get(rid, float('nan')):.4f}")
        table.append(row)
    fig = plt.figure(figsize=(1 + len(metric_names), 1 + len(run_ids)))
    ax = plt.gca()
    ax.axis("off")
    columns = ["run_id"] + metric_names
    tbl = plt.table(cellText=table, colLabels=columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_metrics_table.pdf")
    fig.savefig(path)
    plt.close(fig)
    return [path]


def main():
    args = parse_kv_args(sys.argv[1:])
    results_dir = args.get("results_dir")
    run_ids_raw = args.get("run_ids")
    if results_dir is None or run_ids_raw is None:
        raise ValueError("Expected results_dir=... and run_ids='[...]'")
    run_ids = json.loads(run_ids_raw)

    with open(os.path.join("config", "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    api = wandb.Api()

    comparison_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    histories: Dict[str, pd.DataFrame] = {}
    all_paths: List[str] = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metrics = {
            "history": history.to_dict(orient="list"),
            "summary": summary,
            "config": config,
        }
        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(metrics_path, metrics)
        all_paths.append(metrics_path)

        histories[run_id] = history

        fig_paths = plot_learning_curve(history, run_id, run_dir)
        all_paths.extend(fig_paths)
        fig_paths = plot_confusion_matrix(summary, run_id, run_dir)
        all_paths.extend(fig_paths)

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                comparison_metrics[k][run_id] = float(v)

    primary_metric = "accuracy"
    best_proposed = {"run_id": None, "value": -1e9}
    best_baseline = {"run_id": None, "value": -1e9}

    for run_id, v in comparison_metrics.get(primary_metric, {}).items():
        if "proposed" in run_id:
            if v > best_proposed["value"]:
                best_proposed = {"run_id": run_id, "value": v}
        if "comparative" in run_id or "baseline" in run_id:
            if v > best_baseline["value"]:
                best_baseline = {"run_id": run_id, "value": v}

    minimize = any(x in primary_metric for x in ["loss", "perplexity", "error"])
    gap = None
    if best_proposed["run_id"] and best_baseline["run_id"]:
        raw_gap = (best_proposed["value"] - best_baseline["value"]) / max(1e-9, best_baseline["value"]) * 100.0
        gap = -raw_gap if minimize else raw_gap

    stat_tests = {}
    run_list = list(histories.keys())
    for i in range(len(run_list)):
        for j in range(i + 1, len(run_list)):
            r1, r2 = run_list[i], run_list[j]
            h1 = histories[r1].get(primary_metric)
            h2 = histories[r2].get(primary_metric)
            if h1 is not None and h2 is not None:
                a = np.array(h1.dropna().tolist())
                b = np.array(h2.dropna().tolist())
                if len(a) > 2 and len(b) > 2:
                    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                    stat_tests[f"{r1}_vs_{r2}"] = {"t_stat": float(t_stat), "p_val": float(p_val)}

    agg = {
        "primary_metric": primary_metric,
        "metrics": comparison_metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "stat_tests": stat_tests,
    }

    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    agg_path = os.path.join(comp_dir, "aggregated_metrics.json")
    save_json(agg_path, agg)
    all_paths.append(agg_path)

    for metric_name in ["accuracy", "proxy_success_rate", "reward_hacking_gap"]:
        if metric_name in comparison_metrics:
            fig_paths = plot_bar_comparison(comparison_metrics, metric_name, comp_dir)
            all_paths.extend(fig_paths)
            fig_paths = plot_box_comparison(histories, metric_name, comp_dir)
            all_paths.extend(fig_paths)

    all_paths.extend(plot_metric_table(comparison_metrics, comp_dir))

    for p in all_paths:
        print(p)


if __name__ == "__main__":
    main()

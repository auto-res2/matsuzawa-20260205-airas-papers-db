import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import hydra
import optuna
import wandb
from omegaconf import OmegaConf

from src.preprocess import load_dataset_split, extract_numbers, normalize_number_str
from src.model import (
    GenModelWrapper,
    ProxyCalibrator,
    direct_generate,
    self_refine_generate,
    scg_sr_proxy,
    accr_refine,
)


@dataclass
class EvalStats:
    accuracy: float
    proxy_success_rate: float
    avg_output_length_tokens: float
    avg_proxy_calls_per_instance: float
    avg_audit_calls_per_instance: float
    avg_refinement_rounds: float
    reward_hacking_gap: float
    calibration_pi_mean: float
    calibration_pi_lcb: float
    calibration_pi_ucb: float
    calibration_nu_mean: float
    calibration_nu_lcb: float
    calibration_nu_ucb: float
    proxy_true_tp: int
    proxy_true_fp: int
    proxy_true_tn: int
    proxy_true_fn: int


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def true_score_strict(pred_text: str, gold: str) -> float:
    nums = extract_numbers(pred_text)
    if len(nums) == 0:
        return 0.0
    return 1.0 if normalize_number_str(nums[0]) == normalize_number_str(gold) else 0.0


def proxy_score_lenient(pred_text: str, gold: str, p_flip: float, rng: random.Random) -> float:
    nums = extract_numbers(pred_text)
    gold_n = normalize_number_str(gold)
    cand = [normalize_number_str(n) for n in nums]
    s = 1.0 if gold_n in cand else 0.0
    if rng.random() < p_flip:
        s = 1.0 - s
    return s


def evaluate_method(
    cfg,
    model_wrapper: GenModelWrapper,
    data: List[Tuple[str, str]],
    method_name: str,
    params: Dict[str, Any],
    log_wandb: bool,
) -> Tuple[EvalStats, Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    cal = ProxyCalibrator()
    total = len(data)
    correct = 0
    proxy_succ = 0
    lengths: List[int] = []
    proxy_calls: List[int] = []
    audit_calls: List[int] = []
    rounds: List[int] = []
    gaps: List[float] = []
    tp = fp = tn = fn = 0

    for idx, (x, g) in enumerate(data):
        if cfg.mode == "trial" and idx >= cfg.trial.max_items:
            break
        if method_name == "Direct":
            y = direct_generate(model_wrapper, x, cfg.generation)
            meta = {"proxy_calls": 0, "audit_calls": 0, "rounds": 0}
        elif method_name == "SELF-REFINE":
            y, meta = self_refine_generate(model_wrapper, x, cfg.generation, T=cfg.method_cfg.T)
        elif method_name == "SCG-SR(Proxy-only)":
            y, meta = scg_sr_proxy(
                model_wrapper,
                x,
                g,
                cfg,
                p_flip=params.get("p_flip", cfg.method_cfg.p_flip),
                lam=params["lam_length_penalty"],
                tau=params["tau_accept_margin"],
                per_round_proxy_budget=params["per_round_proxy_budget"],
            )
        elif method_name == "Audit-Calibrated Confidence Refinement (ACCR)":
            y, meta = accr_refine(
                model_wrapper,
                x,
                g,
                cfg,
                calibrator=cal,
                p_flip=params.get("p_flip", cfg.method_cfg.p_flip),
                lam=params["lam_length_penalty"],
                tau=params["tau_accept_margin"],
                per_round_audit_budget=params["per_round_audit_budget"],
                alpha_cal=params["alpha_calibration"],
            )
        else:
            raise ValueError(f"Unknown method {method_name}")

        t = true_score_strict(y, g)
        p = proxy_score_lenient(y, g, params.get("p_flip", cfg.method_cfg.p_flip), rng)
        correct += t
        proxy_succ += p
        lengths.append(len(model_wrapper.tokenize(y)))
        proxy_calls.append(meta["proxy_calls"])
        audit_calls.append(meta["audit_calls"])
        rounds.append(meta["rounds"])
        gaps.append(p - t)

        if p >= 0.5 and t >= 0.5:
            tp += 1
        elif p >= 0.5 and t < 0.5:
            fp += 1
        elif p < 0.5 and t < 0.5:
            tn += 1
        else:
            fn += 1

        if log_wandb and cfg.wandb.mode != "disabled":
            wandb.log(
                {
                    "step": idx,
                    "accuracy": correct / max(1, idx + 1),
                    "proxy_success_rate": proxy_succ / max(1, idx + 1),
                    "avg_output_length_tokens": sum(lengths) / max(1, len(lengths)),
                    "avg_proxy_calls_per_instance": sum(proxy_calls) / max(1, len(proxy_calls)),
                    "avg_audit_calls_per_instance": sum(audit_calls) / max(1, len(audit_calls)),
                    "avg_refinement_rounds": sum(rounds) / max(1, len(rounds)),
                    "reward_hacking_gap": sum(gaps) / max(1, len(gaps)),
                },
                step=idx,
            )

        if idx == 0:
            assert isinstance(x, str) and isinstance(g, str), "Inputs must be strings"
            assert len(y) > 0, "Model output must be non-empty"

    n = len(lengths)
    l_pi, u_pi, l_nu, u_nu = cal.bounds(alpha=0.05)
    stats = EvalStats(
        accuracy=correct / max(1, n),
        proxy_success_rate=proxy_succ / max(1, n),
        avg_output_length_tokens=sum(lengths) / max(1, n),
        avg_proxy_calls_per_instance=sum(proxy_calls) / max(1, n),
        avg_audit_calls_per_instance=sum(audit_calls) / max(1, n),
        avg_refinement_rounds=sum(rounds) / max(1, n),
        reward_hacking_gap=sum(gaps) / max(1, n),
        calibration_pi_mean=cal.mean_pi(),
        calibration_pi_lcb=l_pi,
        calibration_pi_ucb=u_pi,
        calibration_nu_mean=cal.mean_nu(),
        calibration_nu_lcb=l_nu,
        calibration_nu_ucb=u_nu,
        proxy_true_tp=tp,
        proxy_true_fp=fp,
        proxy_true_tn=tn,
        proxy_true_fn=fn,
    )
    return stats, {"calibrator": cal}


def optuna_objective(trial, cfg, model_wrapper, dev_data, method_name):
    params: Dict[str, Any] = {}
    for ss in cfg.optuna.search_spaces:
        if ss.distribution_type == "uniform":
            params[ss.param_name] = trial.suggest_float(ss.param_name, ss.low, ss.high)
        elif ss.distribution_type == "categorical":
            params[ss.param_name] = trial.suggest_categorical(ss.param_name, ss.choices)
        else:
            raise ValueError(f"Unknown distribution {ss.distribution_type}")
    if method_name == "SCG-SR(Proxy-only)":
        params.setdefault("per_round_proxy_budget", cfg.method_cfg.per_round_proxy_budget)
    if method_name == "Audit-Calibrated Confidence Refinement (ACCR)":
        params.setdefault("per_round_audit_budget", cfg.method_cfg.per_round_audit_budget)
        params.setdefault("alpha_calibration", cfg.method_cfg.alpha_calibration)

    params["p_flip"] = params.get("p_flip_dev", cfg.method_cfg.p_flip)
    stats, _ = evaluate_method(cfg, model_wrapper, dev_data, method_name, params, log_wandb=False)
    return stats.accuracy


def maybe_run_training(cfg, model_wrapper):
    if cfg.training.inference_only or cfg.training.epochs <= 0:
        return
    model_wrapper.model.train()
    optimizer = torch.optim.AdamW(model_wrapper.model.parameters(), lr=cfg.training.learning_rate)
    dummy_inputs = model_wrapper.tokenizer(
        ["Dummy input"], return_tensors="pt", padding=True
    ).to(model_wrapper.model.device)
    dummy_labels = model_wrapper.tokenizer(
        ["0"], return_tensors="pt", padding=True
    ).input_ids.to(model_wrapper.model.device)
    for epoch in range(cfg.training.epochs):
        outputs = model_wrapper.model(**dummy_inputs, labels=dummy_labels)
        loss = outputs.loss
        params = [p for p in model_wrapper.model.parameters() if p.requires_grad]
        aux_grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
        aux_norm = sum(g.abs().sum() for g in aux_grads if g is not None)
        loss.backward()
        if epoch == 0:
            assert dummy_inputs["input_ids"].shape[0] == dummy_labels.shape[0]
        grads = [p.grad for p in model_wrapper.model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), "No gradients found before optimizer step"
        assert any((g is not None and g.abs().sum().item() > 0) for g in grads), "Zero gradients before optimizer step"
        optimizer.step()
        optimizer.zero_grad()
        if cfg.wandb.mode != "disabled":
            wandb.log({"train_loss": float(loss.item()), "aux_grad_norm": float(aux_norm)})
    model_wrapper.model.eval()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.method_cfg.T = min(cfg.method_cfg.T, 1)
        cfg.method_cfg.B = min(cfg.method_cfg.B, 2)
        cfg.method_cfg.per_round_proxy_budget = min(cfg.method_cfg.per_round_proxy_budget, 8)
        cfg.method_cfg.per_round_audit_budget = min(cfg.method_cfg.per_round_audit_budget, 1)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB URL: {wandb.run.url}")

    data = load_dataset_split(cfg.dataset, cfg.dataset.split, limit=cfg.dataset.get("limit", None))
    dev_size = min(cfg.dataset.get("dev_size", 100), len(data))
    test_size = min(cfg.dataset.get("test_size", 500), max(0, len(data) - dev_size))
    rng = random.Random(cfg.seed)
    rng.shuffle(data)
    dev_data = data[:dev_size]
    test_data = data[dev_size : dev_size + test_size]

    if cfg.mode == "trial":
        dev_data = dev_data[: cfg.trial.max_items]
        test_data = test_data[: cfg.trial.max_items]

    model_wrapper = GenModelWrapper(cfg)
    assert model_wrapper.tokenizer.pad_token_id is not None, "Tokenizer pad_token_id must be set"
    assert model_wrapper.model is not None, "Model must be initialized"
    assert model_wrapper.model.config.vocab_size > 0, "Model vocab size must be valid"

    maybe_run_training(cfg, model_wrapper)

    method_name = cfg.run.method
    params = {
        "lam_length_penalty": cfg.method_cfg.lam_length_penalty,
        "tau_accept_margin": cfg.method_cfg.tau_accept_margin,
        "per_round_proxy_budget": cfg.method_cfg.per_round_proxy_budget,
        "per_round_audit_budget": cfg.method_cfg.per_round_audit_budget,
        "alpha_calibration": cfg.method_cfg.alpha_calibration,
        "p_flip": cfg.method_cfg.p_flip,
    }

    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: optuna_objective(t, cfg, model_wrapper, dev_data, method_name),
            n_trials=cfg.optuna.n_trials,
        )
        params.update(study.best_params)

    stats, _ = evaluate_method(cfg, model_wrapper, test_data, method_name, params, log_wandb=True)

    if cfg.wandb.mode != "disabled":
        wandb.summary["accuracy"] = stats.accuracy
        wandb.summary["proxy_success_rate"] = stats.proxy_success_rate
        wandb.summary["avg_output_length_tokens"] = stats.avg_output_length_tokens
        wandb.summary["avg_proxy_calls_per_instance"] = stats.avg_proxy_calls_per_instance
        wandb.summary["avg_audit_calls_per_instance"] = stats.avg_audit_calls_per_instance
        wandb.summary["avg_refinement_rounds"] = stats.avg_refinement_rounds
        wandb.summary["reward_hacking_gap"] = stats.reward_hacking_gap
        wandb.summary["calibration_pi_mean"] = stats.calibration_pi_mean
        wandb.summary["calibration_pi_lcb"] = stats.calibration_pi_lcb
        wandb.summary["calibration_pi_ucb"] = stats.calibration_pi_ucb
        wandb.summary["calibration_nu_mean"] = stats.calibration_nu_mean
        wandb.summary["calibration_nu_lcb"] = stats.calibration_nu_lcb
        wandb.summary["calibration_nu_ucb"] = stats.calibration_nu_ucb
        wandb.summary["proxy_true_tp"] = stats.proxy_true_tp
        wandb.summary["proxy_true_fp"] = stats.proxy_true_fp
        wandb.summary["proxy_true_tn"] = stats.proxy_true_tn
        wandb.summary["proxy_true_fn"] = stats.proxy_true_fn
        wandb.finish()


if __name__ == "__main__":
    main()

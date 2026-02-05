import os
import subprocess
import sys
import hydra


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.method_cfg.T = min(cfg.method_cfg.T, 1)
        cfg.method_cfg.B = min(cfg.method_cfg.B, 2)
        cfg.method_cfg.per_round_proxy_budget = min(cfg.method_cfg.per_round_proxy_budget, 8)
        cfg.method_cfg.per_round_audit_budget = min(cfg.method_cfg.per_round_audit_budget, 1)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    run_id = cfg.run.run_id
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()

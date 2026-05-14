"""
benchmark.py — Sweep adapter configs and build an accuracy-vs-params table.

Each config trains for --epochs epochs (default 30 for a quick sweep) and
logs to a CSV.  Run from the polyadapt root.

Example — rank sweep:
    python benchmark.py --dataset ucf101 --sweep rank \\
        --batch_size 8 --epochs 30 --no_wandb

Example — stage sweep:
    python benchmark.py --dataset ucf101 --sweep stages \\
        --batch_size 8 --epochs 30 --no_wandb

Example — n_lag sweep:
    python benchmark.py --dataset ucf101 --sweep n_lag \\
        --batch_size 8 --epochs 30 --no_wandb

Pass --dry_run to print the configs without training.
"""

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    name:          str
    model:         str   = "r3d_adapted"
    adapter_rank:  int   = 4
    adapter_conv:  str   = "laguerre"
    adapter_stages: list = field(default_factory=lambda: [1, 2, 3, 4])
    n_lag:         int | None = None


# ── Sweep definitions ────────────────────────────────────────────────

SWEEPS: dict[str, list[Config]] = {

    # Baselines (always included)
    "baselines": [
        Config("r3d_frozen",   model="r3d_frozen"),
        Config("r3d_finetune", model="r3d_finetune"),
    ],

    # How much does rank Q matter?
    "rank": [
        Config(f"adapted_q{q}", adapter_rank=q)
        for q in [1, 2, 4, 8]
    ],

    # Which stages benefit from polynomial capacity?
    "stages": [
        Config("adapted_s4",    adapter_stages=[4]),
        Config("adapted_s34",   adapter_stages=[3, 4]),
        Config("adapted_s234",  adapter_stages=[2, 3, 4]),
        Config("adapted_s1234", adapter_stages=[1, 2, 3, 4]),
    ],

    # How many Laguerre temporal orders are needed?
    "n_lag": [
        Config(f"adapted_nlag{n}", n_lag=n)
        for n in [1, 2, 3]   # 3 = full (no compression for T=3 kernel)
    ],

    # Conv type comparison at fixed rank
    "conv_type": [
        Config(f"adapted_{ct}", adapter_conv=ct)
        for ct in ["laguerre", "conv3d"]
    ],
}


def config_to_argv(cfg: Config, shared: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable, "train_par.py",
        "--dataset",   shared.dataset,
        "--model",     cfg.model,
        "--run_name",  f"{shared.dataset}_{cfg.name}",
        "--batch_size", str(shared.batch_size),
        "--lr",        str(shared.lr),
        "--epochs",    str(shared.epochs),
        "--warmup_epochs", str(shared.warmup_epochs),
        "--num_workers", str(shared.num_workers),
    ]
    if shared.no_wandb:
        argv.append("--no_wandb")

    if cfg.model == "r3d_adapted":
        argv += ["--adapter_rank",  str(cfg.adapter_rank)]
        argv += ["--adapter_conv",  cfg.adapter_conv]
        argv += ["--adapter_stages"] + [str(s) for s in cfg.adapter_stages]
        if cfg.n_lag is not None:
            argv += ["--n_lag", str(cfg.n_lag)]

    return argv


def count_trainable(cfg: Config, dataset: str) -> int:
    """Instantiate the model and return trainable param count (no GPU needed)."""
    import torch
    from utils.model_factory import get_model

    class _Args:
        pass

    a = _Args()
    a.model          = cfg.model
    a.dataset        = dataset
    a.num_classes    = {"ucf101": 101, "hmdb51": 51, "ssv2": 174}.get(dataset, 101)
    a.adapter_rank   = cfg.adapter_rank
    a.adapter_conv   = cfg.adapter_conv
    a.adapter_stages = cfg.adapter_stages
    a.n_lag          = cfg.n_lag

    m = get_model(a, torch.device("cpu"))
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_sweep(args):
    configs = list(SWEEPS["baselines"])  # always include baselines
    for sweep_name in args.sweep:
        for c in SWEEPS[sweep_name]:
            if c not in configs:
                configs.append(c)

    csv_path = os.path.join("runs", f"benchmark_{args.dataset}_{'+'.join(args.sweep)}.csv")
    os.makedirs("runs", exist_ok=True)

    print(f"\n{'Config':<20} {'Trainable':>12}  Command")
    print("-" * 80)

    results = []
    for cfg in configs:
        n_tr = count_trainable(cfg, args.dataset)
        argv = config_to_argv(cfg, args)
        print(f"{cfg.name:<20} {n_tr:>12,}  {' '.join(argv[2:5])} ...")

        if args.dry_run:
            continue

        ret = subprocess.run(argv, capture_output=False)

        # Parse best val acc from the run directory (last printed line)
        best_acc = None
        run_dir_prefix = os.path.join("runs", f"{args.dataset}_{cfg.name}")
        for d in sorted(os.listdir("runs"), reverse=True):
            if d.startswith(f"{args.dataset}_{cfg.name}"):
                ckpt = os.path.join("runs", d, "checkpoints", "best_model.pth")
                if os.path.exists(ckpt):
                    import torch
                    c2 = torch.load(ckpt, map_location="cpu", weights_only=True)
                    best_acc = c2.get("best_acc")
                    break

        row = {**asdict(cfg), "trainable_params": n_tr, "best_val_acc": best_acc,
               "returncode": ret.returncode}
        results.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  → val_acc={best_acc}  (saved to {csv_path})")

    if not args.dry_run:
        print(f"\nResults saved to {csv_path}")
        _print_table(results)


def _print_table(results):
    print(f"\n{'Config':<20} {'Trainable':>12}  {'Val Acc':>8}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: -(x["best_val_acc"] or 0)):
        acc = f"{r['best_val_acc']:.2f}%" if r["best_val_acc"] is not None else "  n/a"
        print(f"{r['name']:<20} {r['trainable_params']:>12,}  {acc:>8}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="ucf101")
    p.add_argument("--sweep",      nargs="+",
                   choices=list(SWEEPS.keys()),
                   default=["rank"],
                   help="Which sweep(s) to run")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=4e-4)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--num_workers",   type=int,   default=8)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print configs without training")
    return p.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())

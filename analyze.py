"""
analyze.py — Post-hoc analysis of a trained PolynomialAdapter checkpoint.

Reports per-adapter gate magnitudes and effective rank of the adapter conv weights.

Usage:
    python analyze.py --checkpoint runs/my_run/checkpoints/best_model.pth \\
                      --dataset ucf101 --model r3d_adapted --adapter_rank 4
"""

import argparse
import torch
import numpy as np

from utils.model_factory import get_model


def stable_rank(tensor: torch.Tensor) -> float:
    """Stable rank = ||A||_F^2 / ||A||_2^2.  Lies in [1, min(m,n)]."""
    A = tensor.reshape(tensor.shape[0], -1).float()
    fro_sq = A.norm("fro").item() ** 2
    spec   = torch.linalg.matrix_norm(A, ord=2).item()
    return fro_sq / (spec ** 2 + 1e-12)


def analyze(args):
    device = torch.device("cpu")
    model  = get_model(args, device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Val acc    : {ckpt.get('best_acc', 'n/a')}\n")

    rows = []
    for name, module in model.named_modules():
        from network.adapters.poly_adapter import PolynomialAdapter
        if not isinstance(module, PolynomialAdapter):
            continue

        gate = module.gate.item()
        conv = module.adapter_conv
        w = next(iter(conv.parameters())) if list(conv.parameters()) else None
        eff_rank = stable_rank(w) if w is not None else None

        rows.append({"name": name, "gate": gate, "eff_rank": eff_rank})

    print(f"{'Adapter':<40} {'gate':>8}  {'eff_rank':>9}")
    print("-" * 62)
    for r in rows:
        er = f"{r['eff_rank']:>9.2f}" if r["eff_rank"] is not None else "       —"
        print(f"{r['name']:<40} {r['gate']:>8.4f}  {er}")

    active = [r for r in rows if abs(r["gate"]) > 1e-3]
    print(f"\nActive adapters (|gate| > 1e-3): {len(active)} / {len(rows)}")
    if rows:
        gates = [abs(r["gate"]) for r in rows]
        print(f"Gate stats — min: {min(gates):.4f}  max: {max(gates):.4f}  mean: {np.mean(gates):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset",       default="ucf101")
    p.add_argument("--model",         default="r3d_adapted")
    p.add_argument("--adapter_rank",  type=int, default=4)
    p.add_argument("--adapter_stages", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--adapter_mode",  default="cross_poly")
    args = p.parse_args()
    args.num_classes = {"ucf101": 101, "hmdb51": 51, "ssv2": 174}.get(args.dataset, 101)
    args.adapter_bottleneck_rank = None
    return args


if __name__ == "__main__":
    analyze(parse_args())

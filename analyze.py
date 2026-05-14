"""
analyze.py — Post-hoc analysis of a trained PolynomialAdapter checkpoint.

Answers "how many orthogonal params are actually necessary?" by inspecting:
  1. Gate magnitudes per adapter (which stages are active)
  2. Laguerre order contributions — norm of each coeff[:,:,n] slice,
     relative to total coeff norm.  Drop-off tells you minimum N_lag.
  3. Effective rank of each adapter's coeff tensor — how many dimensions
     are actually used.  Stable rank = ||A||_F^2 / ||A||_2^2.

Usage:
    python analyze.py --checkpoint runs/my_run/checkpoints/best_model.pth \\
                      --dataset ucf101 --model r3d_adapted \\
                      --adapter_rank 4 --adapter_conv laguerre
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


def laguerre_order_contributions(coeff: torch.Tensor) -> list[float]:
    """Energy fraction carried by each Laguerre order (sums to 1).

    coeff shape: [O, I, N_lag, ...]
    Uses squared norms so fractions are proper energy shares and
    cumulative sums are meaningful for n@90% / n@95% thresholds.
    """
    total_sq = coeff.norm().item() ** 2
    if total_sq < 1e-24:
        return [0.0] * coeff.shape[2]
    return [
        (coeff[:, :, n].norm().item() ** 2) / total_sq
        for n in range(coeff.shape[2])
    ]


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

        has_laguerre = hasattr(conv, "coeff")
        if has_laguerre:
            coeff      = conv.coeff.detach()
            N_lag      = coeff.shape[2]
            order_frac = laguerre_order_contributions(coeff)
            eff_rank   = stable_rank(coeff)
            # cumulative fraction: how many orders capture 90% / 95% of norm
            cum = np.cumsum(order_frac)
            n90 = int(np.searchsorted(cum, 0.90)) + 1
            n95 = int(np.searchsorted(cum, 0.95)) + 1
        else:
            N_lag, order_frac, eff_rank, n90, n95 = None, [], None, None, None

        rows.append({
            "name":       name,
            "gate":       gate,
            "N_lag":      N_lag,
            "eff_rank":   eff_rank,
            "n90":        n90,
            "n95":        n95,
            "order_frac": order_frac,
        })

    # ── Table 1: per-adapter summary ──────────────────────────────────
    print(f"{'Adapter':<35} {'gate':>8}  {'eff_rank':>9}  "
          f"{'n@90%':>6}  {'n@95%':>6}  {'N_lag':>5}")
    print("-" * 80)
    for r in rows:
        print(f"{r['name']:<35} {r['gate']:>8.4f}  "
              f"{r['eff_rank']:>9.2f}  "
              f"{str(r['n90']):>6}  {str(r['n95']):>6}  {str(r['N_lag']):>5}")

    # ── Table 2: Laguerre order breakdown ─────────────────────────────
    if any(r["order_frac"] for r in rows):
        print("\nLaguerre order energy fractions (||coeff[:,:,n]||² / ||coeff||²  — sums to 1):")
        max_n = max(len(r["order_frac"]) for r in rows)
        header = f"{'Adapter':<35}" + "".join(f"  L{n}" for n in range(max_n))
        print(header)
        print("-" * len(header))
        for r in rows:
            fracs = "".join(f"  {f:.2f}" for f in r["order_frac"])
            print(f"{r['name']:<35}{fracs}")

    # ── Summary ───────────────────────────────────────────────────────
    active = [r for r in rows if abs(r["gate"]) > 1e-3]
    print(f"\nActive adapters (|gate| > 1e-3): {len(active)} / {len(rows)}")

    if any(r["n90"] for r in rows):
        n90_vals = [r["n90"] for r in rows if r["n90"] is not None]
        print(f"Laguerre orders needed for 90% of norm: max={max(n90_vals)}  "
              f"mean={np.mean(n90_vals):.1f}")
        n95_vals = [r["n95"] for r in rows if r["n95"] is not None]
        print(f"Laguerre orders needed for 95% of norm: max={max(n95_vals)}  "
              f"mean={np.mean(n95_vals):.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset",       default="ucf101")
    p.add_argument("--model",         default="r3d_adapted")
    p.add_argument("--adapter_rank",  type=int,   default=4)
    p.add_argument("--adapter_conv",  default="laguerre")
    p.add_argument("--adapter_stages", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n_lag",         type=int,   default=None)
    args = p.parse_args()
    args.num_classes = {"ucf101": 101, "hmdb51": 51, "ssv2": 174}.get(args.dataset, 101)
    return args


if __name__ == "__main__":
    analyze(parse_args())

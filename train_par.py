"""
train_par.py — Multi-GPU DDP training for polyadapt.

Single-GPU (no torchrun):
    python train_par.py --dataset ucf101 --model r3d_adapted --run_name test --no_wandb

Multi-GPU (4 GPUs):
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \\
        torchrun --nproc_per_node=4 train_par.py \\
        --dataset ucf101 --model r3d_adapted --run_name ucf101_polyadapt_q4 \\
        --batch_size 8 --lr 4e-4 --adapter_rank 4 --adapter_conv laguerre \\
        --warmup_epochs 5 --no_wandb

Effective batch size = batch_size × world_size.  Scale --lr linearly.
"""

import argparse
import atexit
import contextlib
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from utils.data_factory import get_dataloaders
from utils.model_factory import get_model


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- required ---
    p.add_argument("--dataset",     required=True)
    p.add_argument("--model",       required=True)
    p.add_argument("--run_name",    required=True)

    # --- training ---
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--fc_lr_mult",    type=float, default=10.0)
    p.add_argument("--laguerre_lr_mult", type=float, default=3.0)

    # --- data ---
    p.add_argument("--clip_len",     type=int,   default=16)
    p.add_argument("--split",        type=int,   default=1, choices=[1, 2, 3])
    p.add_argument("--num_workers",  type=int,   default=8)

    # --- adapter (Proposal 1) ---
    p.add_argument("--adapter_rank",   type=int,   default=4)
    p.add_argument("--adapter_conv",   default="laguerre", choices=["laguerre", "full", "conv3d"])
    p.add_argument("--adapter_stages", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n_lag",          type=int,   default=None,
                   help="Laguerre orders for adapter conv (None = full, i.e. T)")
    p.add_argument("--adapter_bottleneck_rank", type=int, default=None,
                   help="Bottleneck rank r: compress in_ch→r before expanding to 2*Q*out_ch. "
                        "None = no bottleneck (original). Try 64 for ~10x param reduction.")
    p.add_argument("--unfreeze_after", type=int,   default=None,
                   help="Epoch to unfreeze backbone for full fine-tune (default: never)")

    # --- misc ---
    p.add_argument("--device",     default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--no_amp",     action="store_true")
    p.add_argument("--no_wandb",   action="store_true")
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--resume",     default=None)
    p.add_argument("--test_only",  action="store_true")

    args = p.parse_args()
    args.ucf_split = args.split

    _ds_classes = {"ucf101": 101, "hmdb51": 51, "ssv2": 174, "ucf10": 10, "ucf11": 11}
    args.num_classes = _ds_classes.get(args.dataset.lower())

    return args


# ---------------------------------------------------------------------------
# W&B stub (non-main ranks / --no_wandb)
# ---------------------------------------------------------------------------

class _WandbStub:
    class config:
        @staticmethod
        def update(*a, **kw): pass

    class summary(dict):
        def update(self, d=None, **kw): pass  # type: ignore[override]

    def __init__(self): self.summary = {}  # type: ignore[assignment]
    def log(self, *a, **kw): pass
    def finish(self, **kw): pass
    def init(self, **kw): pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_ddp(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.ddp        = local_rank >= 0
        self.local_rank = max(local_rank, 0)
        if self.ddp:
            dist.init_process_group(backend="nccl")
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
            atexit.register(dist.destroy_process_group)
        else:
            self.rank, self.world_size = 0, 1
        self.is_main = (self.rank == 0)

    def _setup_device(self, pref):
        if self.ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        elif pref == "auto":
            if   torch.cuda.is_available():             self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():     self.device = torch.device("mps")
            else:                                       self.device = torch.device("cpu")
        else:
            self.device = torch.device(pref)
        if self.is_main:
            print(f"==> Device: {self.device}  World size: {self.world_size}")

    def _setup_logging(self, args):
        ts = datetime.now().strftime("%m%d-%H%M")
        self.out_dir = os.path.join("runs", f"{args.run_name}_{ts}")

        if self.is_main:
            os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)

        if self.is_main and not args.no_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                name=args.run_name, dir=self.out_dir, config=vars(args),
                group=args.wandb_group,
            )
            atexit.register(lambda: self.wandb.finish())
        else:
            self.wandb = _WandbStub()

    def _setup_data(self, args):
        loaders = get_dataloaders(args)
        if not self.ddp:
            return loaders

        use_workers = args.num_workers > 0
        new = {}
        for split, loader in loaders.items():
            sampler = DistributedSampler(
                loader.dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=(split == "train"), drop_last=(split == "train"),
            )
            new[split] = DataLoader(
                loader.dataset, batch_size=args.batch_size, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=use_workers,
                prefetch_factor=2 if use_workers else None,
            )
        return new

    def _setup_optimizer(self, args):
        model = self.raw_model
        laguerre_lr = args.lr * args.laguerre_lr_mult
        fc_lr       = args.lr * args.fc_lr_mult

        # Collect trainable params into three groups:
        #   1. coeff (Laguerre Tucker params) — higher LR to offset Tucker grad attenuation
        #   2. head  (fc layer)              — higher LR, fresh random init
        #   3. rest  (adapter gate, BN, etc.)
        coeff_p, head_p, other_p = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fc."):
                head_p.append(p)
            elif name.endswith(".coeff"):
                coeff_p.append(p)
            else:
                other_p.append(p)

        param_groups = [
            {"params": other_p, "lr": args.lr,    "weight_decay": args.weight_decay},
            {"params": head_p,  "lr": fc_lr,       "weight_decay": args.weight_decay},
        ]
        if coeff_p:
            param_groups.append(
                {"params": coeff_p, "lr": laguerre_lr, "weight_decay": args.weight_decay}
            )

        self.optimizer = optim.AdamW(param_groups, lr=args.lr,
                                     weight_decay=args.weight_decay)

        cosine = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=1e-6,
        )
        if args.warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=args.warmup_epochs,
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        else:
            self.scheduler = cosine

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, args):
        self.args       = args
        self.start_epoch = 0
        self.best_acc    = 0.0

        self._setup_ddp()
        self._setup_device(args.device)
        self._setup_logging(args)

        self.loaders   = self._setup_data(args)
        self.model     = get_model(args, self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(self.device)
        self._setup_optimizer(args)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.amp_enabled = (self.device.type == "cuda") and not args.no_amp
        self.scaler      = GradScaler("cuda") if self.amp_enabled else None

        self.skipped_stats    = {"total": 0, "input": 0, "output": 0, "loss": 0}
        self.finite_dbg_count = 0

        n_total    = sum(p.numel() for p in self.raw_model.parameters())
        n_trainable = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
        if self.is_main:
            print(f"==> Params: {n_total:,} total | {n_trainable:,} trainable "
                  f"({100 * n_trainable / n_total:.1f}%)")
        self.wandb.config.update({"total_params": n_total, "world_size": self.world_size})

        if args.resume and os.path.isfile(args.resume):
            if self.is_main:
                print(f"==> Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=self.device)
            self.start_epoch = ckpt["epoch"]
            self.best_acc    = ckpt.get("best_acc", 0.0)
            self.raw_model.load_state_dict(ckpt["state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_finite(self, tensor, name, epoch, batch_idx, mode):
        tensors = tensor if isinstance(tensor, (list, tuple)) else [tensor]
        if all(torch.isfinite(t).all() for t in tensors):
            return True
        self.skipped_stats["total"] += 1
        self.skipped_stats[name]    += 1
        if self.finite_dbg_count < 3 and self.is_main:
            print(f"[{mode.upper()}][Ep {epoch+1}][B {batch_idx}] skipping: non-finite {name}")
            self.finite_dbg_count += 1
        return False

    def _gate_stats(self):
        stats, idx = {}, 0
        for m in self.raw_model.modules():
            if hasattr(m, "gate"):
                stats[f"gates/b{idx}"] = m.gate.abs().mean().item()
                idx += 1
        return stats

    def _maybe_unfreeze(self, epoch):
        ua = getattr(self.args, "unfreeze_after", None)
        if ua is not None and epoch == ua:
            n = 0
            for name, p in self.raw_model.named_parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    n += 1
            if self.is_main and n:
                print(f"==> Epoch {epoch+1}: unfreezing {n} backbone params")

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch, mode="train"):
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[mode]

        if is_train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        stats = {"loss": 0.0, "correct": 0, "total": 0, "batches": 0,
                 "grad_norm": 0.0, "grad_batches": 0, "nan_grad": 0}
        self.finite_dbg_count = 0
        if is_train:
            self.skipped_stats = {"total": 0, "input": 0, "output": 0, "loss": 0}

        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"Ep {epoch+1} [{mode.upper()}]", disable=not self.is_main)

        for bi, (inputs, targets) in pbar:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device, dtype=torch.long).view(-1)

            if not self._check_finite(inputs, "input", epoch, bi, mode):
                continue

            if is_train:
                self.optimizer.zero_grad()

            ctx_amp = autocast(device_type=self.device.type) if self.amp_enabled \
                      else contextlib.nullcontext()
            ctx_grad = contextlib.nullcontext() if is_train else torch.no_grad()

            with ctx_amp, ctx_grad:
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, targets)

            if not self._check_finite(outputs, "output", epoch, bi, mode) or \
               not self._check_finite(loss,    "loss",   epoch, bi, mode):
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            if is_train:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                bad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.raw_model.parameters()
                )
                if bad:
                    stats["nan_grad"] += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler:
                        self.scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), 1.0
                    ).item()
                    stats["grad_norm"]    += gn
                    stats["grad_batches"] += 1
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            stats["loss"]    += loss.item()
            stats["batches"] += 1
            stats["total"]   += targets.size(0)
            stats["correct"] += outputs.max(1)[1].eq(targets).sum().item()

            if self.is_main:
                pf = {
                    "L": f"{stats['loss']/stats['batches']:.3f}",
                    "A": f"{100.*stats['correct']/stats['total']:.1f}%",
                }
                if is_train:
                    pf["gn"] = f"{stats['grad_norm']/max(1, stats['grad_batches']):.2f}"
                pbar.set_postfix(pf)

        if self.ddp:
            agg = torch.tensor(
                [stats["loss"], float(stats["correct"]),
                 float(stats["total"]), float(stats["batches"])],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            stats["loss"], stats["correct"] = agg[0].item(), int(agg[1].item())
            stats["total"], stats["batches"] = int(agg[2].item()), int(agg[3].item())

        n = stats["batches"]
        res = {
            "loss": stats["loss"] / n if n else 0.0,
            "acc":  100. * stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        if is_train:
            res["grad_norm"] = stats["grad_norm"] / max(1, stats["grad_batches"])
            res["nan_grad"]  = stats["nan_grad"]
        return res

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _save(self, epoch, tag="best"):
        path = os.path.join(self.out_dir, "checkpoints", f"{tag}_model.pth")
        torch.save({
            "epoch":      epoch + 1,
            "state_dict": self.raw_model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "best_acc":   self.best_acc,
        }, path)

    def run(self):
        if self.args.test_only:
            if self.is_main:
                print("==> Test only …")
            ts = self._run_epoch(0, "test")
            if self.is_main:
                print(f"Test | Loss: {ts['loss']:.3f} | Acc: {ts['acc']:.2f}%")
            return ts["acc"]

        t0 = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self._maybe_unfreeze(epoch)
            tr = self._run_epoch(epoch, "train")
            vl = self._run_epoch(epoch, "val")
            self.scheduler.step()

            if self.is_main:
                print(
                    f"Ep {epoch+1:3d} | "
                    f"T {tr['loss']:.3f}/{tr['acc']:.1f}% | "
                    f"V {vl['loss']:.3f}/{vl['acc']:.1f}% | "
                    f"GN {tr['grad_norm']:.3f} | "
                    f"NaN_g {tr['nan_grad']} Skip {self.skipped_stats['total']}"
                )

                log = {f"train/{k}": v for k, v in tr.items()}
                log.update({f"val/{k}": v for k, v in vl.items()})
                log.update({
                    "epoch": epoch + 1,
                    "lr":    self.optimizer.param_groups[0]["lr"],
                    **self._gate_stats(),
                })
                if self.scaler:
                    log["amp/scale"] = self.scaler.get_scale()
                self.wandb.log(log)

                if vl["acc"] > self.best_acc:
                    self.best_acc = vl["acc"]
                    self._save(epoch, "best")

                if epoch + 1 == self.args.epochs:
                    self._save(epoch, "last")

        if self.ddp:
            dist.barrier()

        if self.is_main:
            runtime = time.time() - t0
            print(f"==> Training done. {runtime/60:.1f} min")
            best = os.path.join(self.out_dir, "checkpoints", "best_model.pth")
            if os.path.exists(best):
                ckpt = torch.load(best, map_location=self.device)
                self.raw_model.load_state_dict(ckpt["state_dict"])

        ts = self._run_epoch(self.args.epochs - 1, "test")
        if self.is_main:
            print(f"Test | Loss: {ts['loss']:.3f} | Acc: {ts['acc']:.2f}%")
            self.wandb.summary.update({f"test/{k}": v for k, v in ts.items()})
            self.wandb.finish()
        return ts["acc"]


if __name__ == "__main__":
    Trainer(parse_args()).run()

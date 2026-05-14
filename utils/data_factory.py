"""Data factory for polyadapt (video datasets only)."""

import torch
from torch.utils.data import DataLoader

from dataloaders.dataset import VideoDataset


def get_dataloaders(args):
    print(f"==> Preparing data: {args.dataset}")
    clip_len  = getattr(args, "clip_len",  16)
    ucf_split = getattr(args, "ucf_split",  1)
    n_workers = getattr(args, "num_workers", 8)

    train_ds = VideoDataset(args.dataset, split="train", clip_len=clip_len,
                            augment=True,  ucf_split=ucf_split)
    val_ds   = VideoDataset(args.dataset, split="val",   clip_len=clip_len,
                            augment=False, ucf_split=ucf_split)
    test_ds  = VideoDataset(args.dataset, split="test",  clip_len=clip_len,
                            augment=False, ucf_split=ucf_split)

    pin   = torch.cuda.is_available()
    wkw   = dict(persistent_workers=True, prefetch_factor=2) if n_workers > 0 else {}

    return {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=n_workers, pin_memory=pin, **wkw),
        "val":   DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=pin, **wkw),
        "test":  DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=pin, **wkw),
    }

"""Model factory for polyadapt."""

from network.backbones.r3d_adapted import R3DAdapted
from network.backbones.r3d_baseline import R3DBaseline


_DS_CLASSES = {
    "ucf101": 101,
    "hmdb51": 51,
    "ssv2":   174,
    "ucf10":  10,
    "ucf11":  11,
}


def get_model(args, device):
    print(f"==> Building model: {args.model}")

    num_classes = getattr(args, "num_classes", None) or _DS_CLASSES.get(
        (args.dataset or "").lower()
    )
    if num_classes is None:
        raise ValueError(
            f"Unknown dataset {args.dataset!r} — set args.num_classes explicitly."
        )

    if args.model == "r3d_adapted":
        net = R3DAdapted(
            num_classes=num_classes,
            Q=getattr(args, "adapter_rank", 4),
            conv_type=getattr(args, "adapter_conv", "laguerre"),
            N_lag=getattr(args, "n_lag", None),
            adapter_stages=tuple(getattr(args, "adapter_stages", [1, 2, 3, 4])),
        )
    elif args.model == "r3d_frozen":
        net = R3DBaseline(num_classes=num_classes, freeze_backbone=True)
    elif args.model == "r3d_finetune":
        net = R3DBaseline(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model: {args.model!r}")

    return net.to(device)

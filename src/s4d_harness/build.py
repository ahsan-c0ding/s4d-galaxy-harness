"""
build_model(spec) -- turn one registry.py entry into an instantiated
nn.Module, regardless of which of the four architecture families it
belongs to. This is the one place that needs to know about all four; every
other part of the harness (engine.py, tests/, scripts/run_experiment.py)
just calls build_model and doesn't care which family it got.
"""
from .production_model import GalaxyClassifierS4DFast
from .richer_grid_models import make_original_grid_model
from model.gclassifier import GalaxyClassifierS4D
from model.gclassifier_cnn_only import GalaxyClassifierCNNOnly
from model.gclassifier_hybrid import GalaxyClassifierCNNS4D


def build_model(spec):
    family = spec["family"]

    if family == "production":
        return GalaxyClassifierS4DFast(
            s4_state=spec["s4_state"], d_model=spec["d_model"], num_classes=4,
            colored=spec.get("colored", True), num_layers=spec["num_layers"],
            patch_size=spec.get("patch_size", 4), pooling=spec.get("pooling", "last"),
            use_norm=spec.get("use_norm", False), use_residual=spec.get("use_residual", False),
            dropout=spec.get("dropout", 0.2), patch_embed=spec["patch_embed"],
        )

    if family == "richer_hybrid":
        return GalaxyClassifierCNNS4D(
            s4_state=spec["s4_state"], d_model=spec["d_model"], num_classes=4,
            colored=spec.get("colored", True), stem_reduction=spec.get("stem_reduction", 16),
            mid_channels=spec["mid_channels"], stem_dropout=spec.get("stem_dropout", 0.1),
            head_dropout=spec.get("head_dropout", 0.2), num_s4_layers=spec["num_s4_layers"],
        )

    if family == "richer_cnn_only":
        return GalaxyClassifierCNNOnly(
            num_classes=4, colored=spec.get("colored", True),
            stem_reduction=spec.get("stem_reduction", 16), mid_channels=spec["mid_channels"],
            d_model=spec["d_model"], stem_dropout=spec.get("stem_dropout", 0.1),
            head_dropout=spec.get("head_dropout", 0.2), use_refine_conv=spec["use_refine_conv"],
        )

    if family == "richer_s4d_only":
        return GalaxyClassifierS4D(
            s4_state=spec["s4_state"], d_model=spec["d_model"], num_classes=4,
            colored=spec.get("colored", True),
        )

    if family == "richer_grid":
        in_channels = 3  # every richer-grid run used RGB input
        return make_original_grid_model(spec["stem_depth"], spec["num_s4_layers"], in_channels)

    raise ValueError(f"Unknown family {family!r} in spec {spec.get('id')}")

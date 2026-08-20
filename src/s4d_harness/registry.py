"""
Single source of truth for every architecture configuration this project has
ever trained. Each entry carries the exact constructor kwargs needed to
reproduce that architecture's parameter count, plus (where known) the
reported test-set metrics from the LaTeX report, so tests can assert the
harness reproduces the report's own numbers before anyone trusts it to run
something new.

Every ``expected_params`` value below was cross-checked against real,
extracted notebook code -- not re-derived from the report text alone. Where
the exact constructor kwargs weren't visible in an extracted cell (the
scale-sweep hybrid configs), they were recovered by solving the model's
parameter-count formula and then confirmed against the actual training
script (see scripts/legacy/train_hybrid_scale.py, train_cnn_only.py). That
provenance is noted per-entry via the "source" field:

    "source": "extracted"   -> constructor kwargs copied verbatim from a
                                notebook cell (ground truth)
    "source": "solved"      -> kwargs recovered by solving the architecture's
                                closed-form parameter count against the
                                report's published figure, then confirmed
                                against a legacy script comment
"""

PRODUCTION_SPECS = [
    # ---- Master Results Table (Table master) ----
    dict(id="prod_best_s4d_d256", label="Best S4D (d=256)", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=528004, reported_acc=0.8680, reported_f1=0.8688),
    dict(id="prod_2layer_d256", label="2-Layer S4D (d=256)", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=2, pooling="last", patch_size=4,
         colored=True, expected_params=396420, reported_acc=0.8640, reported_f1=0.8647),
    dict(id="prod_64dim_rerun", label="64-dim S4D (3L, re-run)", source="extracted", family="production",
         patch_embed="conv", d_model=64, s4_state=64, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=58948, reported_acc=0.8420, reported_f1=0.8425),
    dict(id="prod_conv_no_s4d", label="Conv Embed + No S4D", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=0, pooling="mean", patch_size=4,
         colored=True, expected_params=133252, reported_acc=0.5150, reported_f1=0.5156),
    dict(id="prod_linear_s4d_d256_last", label="Linear Embed + S4D (d=256)", source="extracted", family="production",
         patch_embed="linear", d_model=256, s4_state=256, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=408324, reported_acc=0.8285, reported_f1=0.8290),
    dict(id="prod_linear_s4d_d256_mean", label="Linear Embed + S4D, mean pool", source="extracted", family="production",
         patch_embed="linear", d_model=256, s4_state=256, num_layers=3, pooling="mean", patch_size=4,
         colored=True, expected_params=408324, reported_acc=0.8220, reported_f1=0.8210),
    dict(id="prod_linear_no_s4d", label="Linear Embed + No S4D", source="extracted", family="production",
         patch_embed="linear", d_model=256, s4_state=256, num_layers=0, pooling="mean", patch_size=4,
         colored=True, expected_params=13572, reported_acc=0.3585, reported_f1=0.3528),
    dict(id="prod_pooling_confound", label="Pooling Confound (conv+S4D, mean)", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=3, pooling="mean", patch_size=4,
         colored=True, expected_params=528004, reported_acc=0.8545, reported_f1=0.8544),
    dict(id="prod_small_conv_d72", label="Small Conv + S4D (d=72)", source="extracted", family="production",
         patch_embed="conv", d_model=72, s4_state=72, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=69660, reported_acc=0.8515, reported_f1=0.8522),
    dict(id="prod_linear_d72", label="Linear + S4D (d=72)", source="extracted", family="production",
         patch_embed="linear", d_model=72, s4_state=72, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=35356, reported_acc=0.8020, reported_f1=0.8028),
    dict(id="prod_linear_d90", label="Linear + S4D (d=90)", source="extracted", family="production",
         patch_embed="linear", d_model=90, s4_state=90, num_layers=3, pooling="last", patch_size=4,
         colored=True, expected_params=53914, reported_acc=0.8010, reported_f1=0.8016),
    dict(id="prod_linear_d108_seed1", label="Linear + S4D (d~108, seed 30485)", source="extracted", family="production",
         patch_embed="linear", d_model=108, s4_state=108, num_layers=3, pooling="last", patch_size=4,
         colored=True, seed=30485, expected_params=76360, reported_acc=0.7895, reported_f1=0.7902),
    dict(id="prod_linear_d108_seed2", label="Linear + S4D (d~108, seed 8842)", source="extracted", family="production",
         patch_embed="linear", d_model=108, s4_state=108, num_layers=3, pooling="last", patch_size=4,
         colored=True, seed=8842, expected_params=76360, reported_acc=0.8220, reported_f1=0.8220),

    # ---- Controlled recipe-crossing study reruns (Table recipe-crossing / metric-recovery) ----
    # Same three architectures as prod_best_s4d_d256 / prod_2layer_d256 / prod_small_conv_d72,
    # retrained under both recipes to test recipe as a confound; kept as separate ids because
    # they are separate runs (different seed/split/session) with their own reported numbers.
    dict(id="crossed_best_528k_short", label="Best S4D (528K) under short recipe", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=3, pooling="last", patch_size=4,
         colored=True, recipe="short", expected_params=528004, reported_acc=0.7760, reported_f1=0.7763),
    dict(id="crossed_2layer_396k_short", label="2-Layer S4D (396K) under short recipe", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=2, pooling="last", patch_size=4,
         colored=True, recipe="short", expected_params=396420, reported_acc=0.7835, reported_f1=0.7840),
    dict(id="crossed_small_70k_short", label="Small Conv+S4D (69.7K) under short recipe", source="extracted", family="production",
         patch_embed="conv", d_model=72, s4_state=72, num_layers=3, pooling="last", patch_size=4,
         colored=True, recipe="short", expected_params=69660, reported_acc=0.7530, reported_f1=0.7531),
    dict(id="metric_recovery_best_528k", label="Best S4D (528K) metric-recovery retrain", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=3, pooling="last", patch_size=4,
         colored=True, recipe="main", expected_params=528004, reported_acc=0.8530, reported_f1=0.8536),
    dict(id="metric_recovery_2layer_396k", label="2-Layer S4D (396K) metric-recovery retrain", source="extracted", family="production",
         patch_embed="conv", d_model=256, s4_state=256, num_layers=2, pooling="last", patch_size=4,
         colored=True, recipe="main", expected_params=396420, reported_acc=0.8555, reported_f1=0.8560),
]

RICHER_HYBRID_SPECS = [
    # ---- Table cnn-only-both: CNN-only baselines + hybrid, richer-stem family ----
    dict(id="richer_cnn_only_tiny", label="CNN-only Tiny (~10K)", source="solved", family="richer_cnn_only",
         d_model=32, mid_channels=16, use_refine_conv=False, stem_reduction=16, colored=True,
         expected_params=10020, reported_acc_short=0.8155, reported_acc_main=0.8770),
    dict(id="richer_cnn_only_small", label="CNN-only Small (~43K)", source="solved", family="richer_cnn_only",
         d_model=64, mid_channels=32, use_refine_conv=True, stem_reduction=16, colored=True,
         expected_params=42756, reported_acc_short=0.8410, reported_acc_main=0.8900),
    dict(id="richer_cnn_only_large", label="CNN-only Large (~61K)", source="solved", family="richer_cnn_only",
         d_model=70, mid_channels=40, use_refine_conv=True, stem_reduction=16, colored=True,
         expected_params=61044, reported_acc_short=0.8460, reported_acc_main=0.8925),
    dict(id="richer_s4d_only_baseline", label="S4D-only baseline (no CNN, 17K)", source="extracted", family="richer_s4d_only",
         d_model=64, s4_state=64, colored=True,
         expected_params=17156, reported_acc_short=0.7050, reported_acc_main=0.7845),
    dict(id="richer_hybrid_55k", label="CNN+S4D hybrid (55.1K, 2L)", source="extracted", family="richer_hybrid",
         d_model=64, s4_state=64, mid_channels=32, num_s4_layers=2, stem_reduction=16, colored=True,
         expected_params=55108, reported_acc_short=0.8735, reported_acc_main=0.8975),

    # ---- Table scale-sweep: hybrid at ~100K/300K/700K params (grayscale, short recipe) ----
    dict(id="richer_hybrid_100k", label="hybrid_100k", source="extracted", family="richer_hybrid",
         d_model=120, s4_state=60, mid_channels=32, num_s4_layers=3, stem_reduction=16, colored=False,
         expected_params=98332, reported_acc_short=0.8750),
    dict(id="richer_hybrid_300k", label="hybrid_300k", source="extracted", family="richer_hybrid",
         d_model=112, s4_state=224, mid_channels=80, num_s4_layers=2, stem_reduction=16, colored=False,
         expected_params=298868, reported_acc_short=0.8715),
    dict(id="richer_hybrid_700k", label="hybrid_700k", source="extracted", family="richer_hybrid",
         d_model=192, s4_state=384, mid_channels=80, num_s4_layers=3, stem_reduction=16, colored=False,
         expected_params=699748, reported_acc_short=0.8750),

    # ---- Table recipe-crossing: same three hybrids, crossed under the main recipe ----
    dict(id="crossed_hybrid_55k_main", label="Hybrid (55.1K) under main recipe", source="extracted", family="richer_hybrid",
         d_model=64, s4_state=64, mid_channels=32, num_s4_layers=2, stem_reduction=16, colored=True,
         recipe="main", expected_params=55108, reported_acc=0.8975, reported_f1=0.8979),
    dict(id="crossed_hybrid_98k_main", label="Hybrid (98.3K) under main recipe", source="extracted", family="richer_hybrid",
         d_model=120, s4_state=60, mid_channels=32, num_s4_layers=3, stem_reduction=16, colored=False,
         recipe="main", expected_params=98332, reported_acc=0.9005, reported_f1=0.9011),
    dict(id="crossed_hybrid_700k_main", label="Hybrid (699.7K) under main recipe", source="extracted", family="richer_hybrid",
         d_model=192, s4_state=384, mid_channels=80, num_s4_layers=3, stem_reduction=16, colored=False,
         recipe="main", expected_params=699748, reported_acc=0.8975, reported_f1=0.8982),
]

# ---- Table richer-grid / followup-grid: stem-depth x S4D-layer grid ----
# Two implementations (see richer_grid_models.py docstring); each spec below
# is tagged with which implementation actually produced its reported numbers.
RICHER_GRID_SPECS = []
_GRID_SHORT_RECIPE_ACC = {
    (4, 0): 0.8210, (4, 1): 0.8275, (4, 2): 0.8375,
    (3, 0): 0.7975, (3, 1): 0.8110, (3, 2): 0.8190,
    (2, 0): 0.6510, (2, 1): 0.7830, (2, 2): 0.8030,
    (1, 0): 0.4800, (1, 1): 0.6990, (1, 2): 0.7405,
}
_GRID_MAIN_RECIPE_ACC = {
    (4, 0): 0.8925, (4, 1): 0.8865, (4, 2): 0.9015,
    (3, 0): 0.8760, (3, 1): 0.8595, (3, 2): 0.8755,
    (2, 0): 0.8375, (2, 1): 0.8575, (2, 2): 0.8485,
    (1, 0): 0.5970, (1, 1): 0.8060, (1, 2): None,  # not run before the GPU quota ran out
}
for depth in (4, 3, 2, 1):
    for n_s4 in (0, 1, 2):
        params = {(4, 0): 38468, (4, 1): 46788, (4, 2): 55108,
                  (3, 0): 29156, (3, 1): 37476, (3, 2): 45796,
                  (2, 0): 19844, (2, 1): 28164, (2, 2): 36484,
                  (1, 0): 2180, (1, 1): 10500, (1, 2): 18820}[(depth, n_s4)]
        RICHER_GRID_SPECS.append(dict(
            id=f"richer_grid_stem{depth}_s4x{n_s4}", family="richer_grid",
            label=f"stem depth {depth}, {n_s4} S4D layers", source="extracted",
            stem_depth=depth, num_s4_layers=n_s4, expected_params=params,
            reported_acc_short=_GRID_SHORT_RECIPE_ACC[(depth, n_s4)],
            reported_acc_main=_GRID_MAIN_RECIPE_ACC[(depth, n_s4)],
        ))

ALL_SPECS = PRODUCTION_SPECS + RICHER_HYBRID_SPECS + RICHER_GRID_SPECS


def get_spec(spec_id):
    for spec in ALL_SPECS:
        if spec["id"] == spec_id:
            return spec
    raise KeyError(f"No spec registered with id={spec_id!r}")

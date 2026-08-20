"""
Galaxy Classification Model Package
"""

from .gclassifier import GalaxyClassifierS4D
from .gclassifier_hybrid import GalaxyClassifierCNNS4D
from .gclassifier_cnn_only import GalaxyClassifierCNNOnly
from .cnn_stem import CNNStem
from . import functions
from .interface import ModelInterface

# NOTE (Kaggle notebook): the interactive pygame GUI is unrelated to
# training and isn't used by any of the four training scripts this
# notebook runs. The import is wrapped defensively so that an unrelated
# GUI/display dependency hiccup in a headless Kaggle kernel can never
# block model training (pygame does import cleanly headless in testing,
# so this is a belt-and-suspenders guard, not a workaround for a known
# failure).
try:
    from .gui import GalaxyExplorerGUI
except Exception as _gui_exc:  # pragma: no cover
    GalaxyExplorerGUI = None
    print(f"[model/__init__] GalaxyExplorerGUI unavailable ({_gui_exc}); "
          f"not needed for training, continuing.")

__all__ = [
    'GalaxyClassifierS4D',
    'GalaxyClassifierCNNS4D',
    'GalaxyClassifierCNNOnly',
    'CNNStem',
    'functions',
    'ModelInterface',
    'GalaxyExplorerGUI',
]

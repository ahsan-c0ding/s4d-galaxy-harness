"""
s4d_harness -- unified training/testing harness for the S4D Galaxy
Classifier project (Team 0x43).

Consolidates two independently-developed model codebases:
  - production_model.py   : the ConvPatchStem/S4DConv architecture behind
                             the project's headline 86.80% result
  - richer_grid_models.py : the CNNStem-based richer-stem family
  - model/ (repo root)    : the shared richer-stem package used by
                             GalaxyClassifierS4D / CNNS4D / CNNOnly

...behind one registry (registry.py), one model dispatcher (build.py), one
set of training recipes (recipes.py), and one train/eval loop (engine.py),
so that the training-recipe confound documented in the project's LaTeX
report can't quietly reappear.
"""

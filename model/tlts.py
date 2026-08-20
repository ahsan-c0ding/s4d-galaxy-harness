import torch
import torch.nn as nn

class TakeLastTimestep(nn.Module):
    """
    Module that extracts the last timestep from a sequence.

    This layer is used to summarize sequence outputs from recurrent 
    or sequence models by taking only the final timestep as a feature vector.

    Parameters
    ----------
    None

    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, L, D), where
        B : batch size,
        L : sequence length,
        D : feature dimension.

    Returns
    -------
    out : torch.Tensor
        Output tensor of shape (B, D), corresponding to the last timestep
        of each sequence in the batch.
    """
    def forward(self, x):
        # x: (B, L, D)
        # FIX (Kaggle notebook, applied on top of the uploaded zip): this
        # method was `return x.mean(dim=1)` -- mean-pooling, not
        # last-timestep pooling. That contradicted this class's own name
        # and docstring, its own commented-out self-test below (which
        # asserts the output equals x[:, -1, :]), and the reference
        # `TakeLastTimestep` in notebook-best-s4d-model_1_.ipynb, which
        # correctly implements last-timestep pooling. Every S4D model in
        # this repo (GalaxyClassifierS4D, GalaxyClassifierCNNS4D, and the
        # grid model in train_ablation.py) uses this class unconditionally,
        # so as shipped every "last pooling" run was silently mean-pooling.
        # Restored to genuine last-timestep pooling to match both the
        # documented intent and the reference notebook.
        return x[:, -1, :]

"""
if __name__ == "__main__":
    print("Testing tlts code")

    layer = TakeLastTimestep()
    print("Layer established")

    x = torch.randn(3, 6, 2)
    print(f"Input shape: {x.shape}")

    output = layer(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected (3, 2): {output.shape == (3, 2)}")
    print(f"Match? {torch.allclose(x[0, -1, :], output[0, :])}")

    print("Successful")
"""
"""
Explaination:
The TakeLastTimeStep layer transforms an input tensor of shape (B, L, D) 
into an output tensor of shape (B, D) by indexing the last position.
The hidden state at position L has been updated by all L previous inputs
and by the time model reaches position L-1, which is the last time step, 
the state has came across and collected information from all inputs u(0) 
through u(L-1) and therefore the (B, D) tensor at the final position
serves as a compressed summary of the entire (B, L, D) sequence.
This is identical to how RNNs, LSTMs, and GRUs use their final hidden state
for classification tasks — the last timestep naturally accumulates the 
history of the whole sequence through the recurrent processes.
"""

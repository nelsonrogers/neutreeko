"""
Export a trained PyTorch DQN checkpoint to a numpy .npz file
that can be loaded by the pure-numpy inference agent (Pygbag compatible).

Usage:
    python -m training.export_weights training/checkpoints/dqn_final.pth assets/weights.npz
"""
import sys
import os
import numpy as np

def export_weights(pth_path, npz_path):
    """Load a .pth checkpoint and save weights as .npz with keys W1,b1,...,W3,b3."""
    import torch
    ckpt = torch.load(pth_path, map_location="cpu")
    # Support both new format (dict with 'policy_state') and old format (plain state_dict)
    state = ckpt["policy_state"] if isinstance(ckpt, dict) and "policy_state" in ckpt else ckpt
    # Expected keys: net.0.weight, net.0.bias, net.2.weight, net.2.bias, net.4.weight, net.4.bias
    np.savez(
        npz_path,
        W1 = state["net.0.weight"].numpy(),
        b1 = state["net.0.bias"].numpy(),
        W2 = state["net.2.weight"].numpy(),
        b2 = state["net.2.bias"].numpy(),
        W3 = state["net.4.weight"].numpy(),
        b3 = state["net.4.bias"].numpy(),
    )
    print(f"Exported weights: {pth_path} → {npz_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m training.export_weights <input.pth> <output.npz>")
        sys.exit(1)
    export_weights(sys.argv[1], sys.argv[2])

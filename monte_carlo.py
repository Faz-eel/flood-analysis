"""
Monte Carlo flood-risk mapping.

Instead of asking "what floods in THIS one scenario", this asks: across
many plausible flood events - different locations along the channel,
different severities - how often does each cell flood? That per-cell
frequency, taken over a large number of random draws, is a flood
probability map: the thing an actual risk/resilience decision (where to
harden infrastructure, where to place a sensor or warning system) would
be based on, rather than any single simulation run.

The surrogate is what makes running thousands of draws practical here:
each one is a full CNN forward pass rather than a from-scratch BFS
search, and all of them can be batched into the network at once.
"""

import numpy as np

from flood_sim import find_channel_cells
from surrogate import build_input


def sample_scenarios(elevation, n_samples, severity_range=(0.3, 1.5), seed=0):
    """
    Draws n_samples random (source_point, severity) pairs the same way
    generate_flood_scenarios does - uniform over channel cells, uniform
    severity in severity_range - so the Monte Carlo draws reflect the
    same "plausible event" assumptions the surrogate was trained on.
    """
    rng = np.random.default_rng(seed)
    channel_cells = find_channel_cells(elevation)

    sources, severities = [], []
    for _ in range(n_samples):
        sources.append(channel_cells[rng.integers(len(channel_cells))])
        severities.append(rng.uniform(*severity_range))

    return sources, severities


def run_monte_carlo(model, elevation, n_samples=2000, severity_range=(0.3, 1.5),
                     decay_length=10, max_severity=1.5, seed=0, batch_size=256):
    """
    Draws n_samples random scenarios, runs them all through the surrogate
    in batches, and returns:
        probability_map: (rows, cols) - fraction of draws where each cell
            was predicted flooded (thresholded at 0.5 per draw, then
            averaged - this counts "how often does this cell flood",
            not "how confident is any one prediction")
        mean_prob_map: (rows, cols) - the surrogate's raw average
            predicted probability per cell across draws (a softer,
            un-thresholded version of the same idea)
    """
    rows, cols = elevation.shape
    sources, severities = sample_scenarios(elevation, n_samples, severity_range, seed)

    flood_count = np.zeros((rows, cols), dtype=np.float64)
    prob_sum = np.zeros((rows, cols), dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        batch_sources = sources[start:start + batch_size]
        batch_severities = severities[start:start + batch_size]

        X_batch = np.stack([
            build_input(elevation, s, sev, decay_length, max_severity)
            for s, sev in zip(batch_sources, batch_severities)
        ], axis=0)

        probs = model.predict(X_batch, verbose=0)[..., 0]  # (batch, rows, cols)
        flood_count += (probs >= 0.5).sum(axis=0)
        prob_sum += probs.sum(axis=0)

    probability_map = flood_count / n_samples
    mean_prob_map = prob_sum / n_samples
    return probability_map, mean_prob_map


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from terrain import generate_watershed

    elevation = generate_watershed()

    model_path = "flood_surrogate_model.keras"
    if not os.path.exists(model_path):
        raise SystemExit(
            f"No trained model found at {model_path}. Run `python main.py` first "
            "to train and save the surrogate, then re-run this script."
        )

    print(f"Loading trained surrogate from {model_path}...")
    from tensorflow import keras
    from surrogate import bce_dice_loss
    model = keras.models.load_model(model_path, custom_objects={"bce_dice_loss": bce_dice_loss})

    print("Running Monte Carlo sweep (2000 random scenarios)...")
    probability_map, mean_prob_map = run_monte_carlo(model, elevation, n_samples=2000)

    print(f"\nHighest-risk cell: {probability_map.max()*100:.1f}% of draws flooded it")
    print(f"Median non-zero risk cell: "
          f"{100*np.median(probability_map[probability_map > 0]):.1f}%")
    print(f"Cells with >10% flood probability: {(probability_map > 0.10).sum()} "
          f"of {elevation.size} ({100*(probability_map > 0.10).sum()/elevation.size:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(elevation, cmap="terrain", alpha=0.6)
    im0 = axes[0].imshow(np.ma.masked_where(probability_map == 0, probability_map),
                          cmap="Reds", alpha=0.85, vmin=0, vmax=probability_map.max())
    axes[0].set_title("Flood probability map (2000 random scenarios)")
    plt.colorbar(im0, ax=axes[0], label="P(cell floods)")

    axes[1].imshow(elevation, cmap="terrain", alpha=0.6)
    im1 = axes[1].imshow(np.ma.masked_where(mean_prob_map < 0.01, mean_prob_map),
                          cmap="Reds", alpha=0.85, vmin=0, vmax=mean_prob_map.max())
    axes[1].set_title("Mean predicted flood probability (unthresholded)")
    plt.colorbar(im1, ax=axes[1], label="mean predicted P")

    plt.tight_layout()
    plt.savefig("flood_risk_map.png", dpi=120, bbox_inches="tight")
    print("Saved flood_risk_map.png")

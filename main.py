"""
Runs the flood surrogate project end to end:

  1. Build the watershed terrain (fixed, same for every scenario)
  2. Generate training and test flood scenarios with the real physics
     simulation (simulate_local_flood) - this is the ground truth
  3. Train the surrogate CNN to predict the same flood masks directly
     from scenario parameters (source point, severity)
  4. Evaluate how accurately it reproduces the physics on held-out test
     scenarios (IoU/Dice)
  5. Save the trained model for monte_carlo.py to load

This project is not about the surrogate being faster than the physics -
on this simplified simulator the physics is already cheap, so it isn't.
The point is generalisation: once trained, the surrogate can be queried
across many more hypothetical scenarios than would be practical to run
one at a time, which is what a probabilistic flood-risk map (a later
step) is actually built from.
"""

import numpy as np

from terrain import generate_watershed
from flood_sim import generate_flood_scenarios
from surrogate import train_surrogate
from evaluate import evaluate_surrogate


def main():
    print("Building watershed terrain...")
    elevation = generate_watershed()  # default 80 rows x 120 cols

    print("Generating training scenarios...")
    train_scenarios = generate_flood_scenarios(elevation, n_scenarios=300, seed=10)
    print("Generating test scenarios...")
    test_scenarios = generate_flood_scenarios(elevation, n_scenarios=80, seed=99)

    print(f"\n{len(train_scenarios)} training scenarios, {len(test_scenarios)} test scenarios\n")

    print("Training surrogate CNN...")
    # Seed Python/NumPy/TensorFlow and force deterministic ops, so the
    # numbers reported in the README can actually be reproduced run to run
    from tensorflow import keras
    import tensorflow as tf
    keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    model = train_surrogate(elevation, train_scenarios, epochs=120, batch_size=16, verbose=2)

    print("\nEvaluating surrogate against the real physics simulation...")
    results = evaluate_surrogate(model, elevation, test_scenarios)

    ious = results["iou_scores"]
    dices = results["dice_scores"]

    print("\n" + "=" * 70)
    print(f"Accuracy over {len(test_scenarios)} held-out test scenarios")
    print("=" * 70)
    print(f"Mean IoU:   {ious.mean():.3f}  (median {np.median(ious):.3f})")
    print(f"Mean Dice:  {dices.mean():.3f}  (median {np.median(dices):.3f})")
    print(f"Worst-case IoU: {ious.min():.3f}   Best-case IoU: {ious.max():.3f}")

    model_path = "flood_surrogate_model.keras"
    model.save(model_path)
    print(f"\nSaved trained surrogate to {model_path}")


if __name__ == "__main__":
    main()

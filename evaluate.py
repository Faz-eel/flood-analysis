"""
Scores the surrogate model against the real physics simulation
(simulate_local_flood) on held-out test scenarios, using IoU and Dice -
region-overlap metrics that ignore correctly predicted dry cells (see
"Results" in the README).
"""

import numpy as np

from surrogate import predict_surrogate


def iou_score(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask).sum()
    union = np.logical_or(predicted_mask, true_mask).sum()
    if union == 0:
        # no flooding at all, predicted or true - count as a perfect match
        return 1.0
    return intersection / union


def dice_score(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask).sum()
    total = predicted_mask.sum() + true_mask.sum()
    if total == 0:
        return 1.0
    return 2 * intersection / total


def evaluate_surrogate(model, elevation, test_scenarios, decay_length=10, max_severity=1.5,
                        threshold=0.5):
    """
    Runs the surrogate on every test scenario, compares each prediction
    against the true flooded_grid already stored in the scenario, and
    returns per-scenario IoU/Dice.
    """
    ious, dices = [], []

    for source_point, severity, true_mask in test_scenarios:
        predicted_mask, _ = predict_surrogate(
            model, elevation, source_point, severity, decay_length, max_severity, threshold)
        ious.append(iou_score(predicted_mask, true_mask))
        dices.append(dice_score(predicted_mask, true_mask))

    return {
        "iou_scores": np.array(ious),
        "dice_scores": np.array(dices),
    }

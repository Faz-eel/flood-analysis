"""
Scores the surrogate model against the real physics simulation
(simulate_local_flood), on the same held-out test scenarios, and times
both so the actual speed-up can be reported honestly rather than assumed.

Because flooding is a whole region rather than a single point, "distance
error" (the metric the DTS project used) doesn't apply here. The right
measure is how well the predicted flooded area overlaps the true flooded
area:

    IoU (intersection-over-union) = (predicted AND true) / (predicted OR true)

IoU is harsh in a way plain per-cell accuracy is not: since most of an
80x120 grid is dry in any one scenario, a model that predicts "nothing
floods" would still get very high per-cell accuracy while being useless.
IoU only rewards actually getting the flooded region right.
"""

import time

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
    returns per-scenario IoU/Dice plus the surrogate's own prediction time.
    """
    ious, dices = [], []

    start = time.perf_counter()
    for source_point, severity, true_mask in test_scenarios:
        predicted_mask, _ = predict_surrogate(
            model, elevation, source_point, severity, decay_length, max_severity, threshold)
        ious.append(iou_score(predicted_mask, true_mask))
        dices.append(dice_score(predicted_mask, true_mask))
    surrogate_seconds = time.perf_counter() - start

    return {
        "iou_scores": np.array(ious),
        "dice_scores": np.array(dices),
        "surrogate_seconds": surrogate_seconds,
    }


def time_physics_simulation(elevation, test_scenarios, decay_length=10, min_rise=0.05):
    """
    Re-runs the real simulate_local_flood for every test scenario, purely
    to time it under the same conditions as evaluate_surrogate - the
    honest baseline the surrogate is being compared against.
    """
    from flood_sim import simulate_local_flood

    start = time.perf_counter()
    for source_point, severity, _ in test_scenarios:
        simulate_local_flood(elevation, source_point, severity, decay_length, min_rise)
    physics_seconds = time.perf_counter() - start

    return physics_seconds

"""
Train a fully-convolutional neural network to predict
flooded/not-flooded directly from three per-cell inputs:

  1. elevation           - the (fixed) terrain, normalised to [0, 1]
  2. distance-from-source - how many grid-steps from the flood's origin,
                            capped at max_distance and normalised to [0, 1]
  3. severity             - the scenario's severity, broadcast to every
                            cell, normalised to [0, 1]
"""

import numpy as np


def build_input(elevation, source_point, severity, decay_length=10, max_severity=1.5):
    """
    Builds the 3-channel (rows, cols, 3) input tensor for one scenario:
    [elevation, distance-from-source, severity], all normalised to [0, 1].
    """
    rows, cols = elevation.shape
    max_distance = 5 * decay_length

    # apply min-max elevation scaling so the lowest elevation gets 0 and highest gets 1
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    # calculate the Manhattan distance of cells to source point, capped at max_distance
    rr, cc = np.indices((rows, cols))
    step_dist = np.abs(rr - source_point[0]) + np.abs(cc - source_point[1])
    dist_norm = np.minimum(step_dist, max_distance) / max_distance

    # apply the severity scale to each cell, scaled at max_severity
    severity_channel = np.full((rows, cols), severity / max_severity)

    return np.stack([elev_norm, dist_norm, severity_channel], axis=-1)


def build_dataset(elevation, scenarios, decay_length=10, max_severity=1.5):
    """
    Turns a list of (source_point, severity, flooded_grid) scenarios (as
    returned by generate_flood_scenarios) into stacked arrays ready for
    training/evaluation:
        X: (n_scenarios, rows, cols, 3)
        y: (n_scenarios, rows, cols, 1)
    """
    X, y = [], []
    for source_point, severity, flooded_grid in scenarios:

        # x takes the generated input to our model for each scenario (training set)
        X.append(build_input(elevation, source_point, severity, decay_length, max_severity))

        # y takes the true outcome of those scenarios for evaluation with the model's output
        y.append(flooded_grid.astype(np.float32))

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)[..., np.newaxis]
    return X, y


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    1 - Dice score, computed over the whole batch at once (all cells of
    all scenarios flattened together). Only scores overlap between the
    predicted and true flooded areas, so the dry majority can't hide a
    poor flood prediction - see "Why Dice loss" in the README.

    `smooth` avoids a divide-by-zero when nothing is flooded.
    """
    import tensorflow as tf

    # y_true & y_pred are n batches of (row, col) arrays.
    # flatten both arrays into one long 1D list of numbers
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return 1.0 - dice


def bce_dice_loss(y_true, y_pred):
    """
    Combines the binary crossentropy with dice loss
    """
    import tensorflow as tf
    from tensorflow.keras.losses import binary_crossentropy

    bce = tf.reduce_mean(binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def build_surrogate_model(rows, cols):
    """
    A small fully-convolutional network: every layer preserves the
    (rows, cols) grid shape. The final layer is a 1x1 convolution that
    collapses the feature channels down to one flood-probability score
    per cell, via a sigmoid. Each cell gets its own independent
    probability, since a flood is a region rather than a single location.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(rows, cols, 3))
    x = layers.Conv2D(16, 5, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(16, 5, padding="same", activation="relu")(x)
    x = layers.Conv2D(16, 5, padding="same", activation="relu")(x)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=bce_dice_loss, metrics=["accuracy"])
    return model


def train_surrogate(elevation, train_scenarios, decay_length=10, max_severity=1.5,
                     epochs=25, batch_size=16, verbose=1):
    rows, cols = elevation.shape
    X_train, y_train = build_dataset(elevation, train_scenarios, decay_length, max_severity)

    model = build_surrogate_model(rows, cols)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def predict_surrogate(model, elevation, source_point, severity,
                       decay_length=10, max_severity=1.5, threshold=0.5):
    """
    Runs the surrogate on a single scenario and returns a boolean flooded
    mask the same shape as elevation, thresholded at `threshold`.
    """
    x = build_input(elevation, source_point, severity, decay_length, max_severity)
    prob = model.predict(x[np.newaxis, ...], verbose=0)[0, ..., 0]
    return prob >= threshold, prob

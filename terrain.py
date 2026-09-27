"""
Generates a synthetic watershed-like elevation surface.

Rather than using a flat random grid, this builds something that behaves
like real terrain: a general slope (so water has somewhere to drain to),
a meandering river channel cut into that slope, and some rolling noise
so the surface isn't perfectly smooth. This gives the flood-fill logic
something realistic to work with - a low-lying channel that floods first,
and higher banks that only flood under larger events.
"""

import numpy as np


def channel_row_positions(rows=80, cols=120):
    """
    Returns the exact row position of the channel centreline at every
    column - the same formula used to carve the channel into the terrain.
    Exposed separately so other modules (e.g. flood_sim) can identify
    genuine channel cells directly, rather than re-deriving the channel's
    location indirectly from elevation thresholds.
    """
    row_centre = rows / 2
    return row_centre + 8 * np.sin(np.linspace(0, 4 * np.pi, cols))


def generate_watershed(rows=80, cols=120, seed=42):
    """
    Build a synthetic elevation grid (a DEM - digital elevation model).

    Returns a 2D numpy array of elevations in metres.
    """
    rng = np.random.default_rng(seed)

    # Base slope: high on the left (upstream), low on the right (downstream/outlet)
    x = np.linspace(20, 0, cols)   # elevation drops left -> right
    base = np.tile(x, (rows, 1))

    # A meandering river channel: a sinusoidal path down the middle of the grid,
    # carved lower than its surroundings
    channel_path = channel_row_positions(rows, cols)

    elevation = base.copy()
    for c in range(cols):
        channel_row = channel_path[c]
        for r in range(rows):
            dist_from_channel = abs(r - channel_row)
            # the channel itself sits ~4m lower than the surrounding land,
            # tapering back up to the base elevation within ~10 cells
            depth = 4.0 * np.exp(-(dist_from_channel ** 2) / (2 * 4.0 ** 2))
            elevation[r, c] -= depth

    # Add rolling terrain noise (smoothed random field) so it doesn't look
    # artificially clean
    noise = rng.normal(0, 1.0, size=(rows, cols))
    # simple smoothing by averaging neighbourhoods
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, sigma=3)
    elevation += noise

    return elevation


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    elev = generate_watershed()
    plt.figure(figsize=(10, 6))
    plt.imshow(elev, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("Synthetic watershed terrain")
    plt.savefig("terrain_preview.png", dpi=120, bbox_inches="tight")
    print("Saved terrain_preview.png")
    print(f"Elevation range: {elev.min():.2f} m to {elev.max():.2f} m")

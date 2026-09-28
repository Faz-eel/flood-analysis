"""
Simulates flood extent for a localized flood event, and generates a set
of scenarios varying in both location and severity.

Core idea (a localized "bathtub" flood model):
    A cell floods if it is below the local flood water level AND it is
    hydraulically connected (via a connected path of low-lying cells) to
    the source point of that specific event.
"""

from collections import deque

import numpy as np
from scipy.ndimage import label


def find_channel_cells(elevation, spacing=3):
    """
    Identify cells lying on the actual river channel centreline, spaced
    out along its length - used as candidate source points for flood
    events. Uses the true channel path from terrain.py directly, rather
    than an elevation threshold, which would incorrectly include the
    entire low-lying downstream floodplain rather than just the channel.
    """
    from terrain import channel_row_positions

    rows, cols = elevation.shape
    channel_path = channel_row_positions(rows, cols)

    channel_cells = []
    for c in range(0, cols, spacing):
        r = int(round(channel_path[c]))
        r = max(0, min(rows - 1, r))
        channel_cells.append((r, c))

    return channel_cells


def simulate_local_flood(elevation, source_point, severity, decay_length=10, min_rise=0.05):
    """
    Simulates a flood originating at source_point, with a given severity
    (metres of water-level rise AT the source point itself).

    A cell floods if it can be reached from the source by stepping only
    onto neighbouring cells that are below the LOCAL water level at that
    point in the search - and the local water level is not constant. It
    decays the farther a cell is (in grid steps) from the source:

        rise_at(steps) = severity * exp(-steps / decay_length)
        local_water_level = elevation[source_point] + rise_at(steps)

    Once the decayed rise at a node falls below min_rise (a negligible
    amount of water), expansion from that node stops.
    """
    rows, cols = elevation.shape
    source_elevation = elevation[source_point]

    flooded = np.zeros_like(elevation, dtype=bool)
    visited_steps = {source_point: 0}
    queue = deque([source_point])

    if severity < min_rise:
        return flooded

    flooded[source_point] = True

    while queue:
        r, c = queue.popleft()
        steps = visited_steps[(r, c)]
        rise = severity * np.exp(-steps / decay_length)
        if rise < min_rise:
            continue  # negligible flood left this far from the source

        local_water_level = source_elevation + rise

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited_steps:
                continue
            if elevation[nr, nc] < local_water_level:
                visited_steps[(nr, nc)] = steps + 1
                flooded[nr, nc] = True
                queue.append((nr, nc))

    return flooded


def generate_flood_scenarios(elevation, n_scenarios=60, seed=1):
    """
    Generates flood scenarios that vary in BOTH location (where along the
    channel the event originates) and severity (how much the local water
    level rises).

    Returns a list of (source_point, severity, flooded_grid) tuples.
    """
    rng = np.random.default_rng(seed)
    channel_cells = find_channel_cells(elevation)

    scenarios = []
    for _ in range(n_scenarios):
        source_point = channel_cells[rng.integers(len(channel_cells))]
        # Kept deliberately modest: with severities this small, most events
        # stay fairly local to their source rather than spilling into the
        # entire downstream floodplain regardless of where they start
        severity = rng.uniform(0.3, 1.5)  # metres of local rise
        flooded = simulate_local_flood(elevation, source_point, severity)
        scenarios.append((source_point, severity, flooded))

    return scenarios


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from terrain import generate_watershed

    elevation = generate_watershed()
    channel_cells = find_channel_cells(elevation)
    print(f"{len(channel_cells)} channel cells identified as candidate flood sources")

    # Show three example events at DIFFERENT locations along the channel,
    # to demonstrate that flooding stays localized to its source
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    example_sources = [channel_cells[i] for i in
                        rng.choice(len(channel_cells), 3, replace=False)]

    for ax, source in zip(axes, example_sources):
        severity = 3.0
        flooded = simulate_local_flood(elevation, source, severity)
        ax.imshow(elevation, cmap="terrain", alpha=0.6)
        ax.imshow(np.ma.masked_where(~flooded, flooded), cmap="Blues", alpha=0.8)
        ax.plot(source[1], source[0], "r*", markersize=15)
        ax.set_title(f"Event at {source}\n"
                     f"(severity {severity:.1f} m, {flooded.sum()} cells flooded)")

    plt.tight_layout()
    plt.savefig("flood_scenarios_preview.png", dpi=120, bbox_inches="tight")
    print("Saved flood_scenarios_preview.png")

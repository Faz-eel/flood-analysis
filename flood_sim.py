"""
Simulates flood extent for a localized flood event, and generates a set
of scenarios varying in both location and severity.

"""

from collections import deque

import numpy as np
from scipy.ndimage import label


def find_channel_cells(elevation, spacing=3):
    """
    Identify cells lying on the actual river channel centreline, spaced
    out along its length - used as candidate source points for flood
    events. Uses the true channel path from terrain.py directly.
    """
    from terrain import channel_row_positions

    rows, cols = elevation.shape
    channel_path = channel_row_positions(rows, cols)

    channel_cells = []
    for c in range(0, cols, spacing):
        r = int(round(channel_path[c]))

        # min ensures it doesn't exceed the last row and max ensures it's not below row 0
        # (only matters on small grids, where the ±8 row channel swing would run off the edge)
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

    # Setup: nothing flooded yet, the source is 0 steps from itself, and the
    # queue of cells still to spread from starts with just the source
    flooded = np.zeros_like(elevation, dtype=bool)
    visited_steps = {source_point: 0}
    queue = deque([source_point])

    # a flood too small to count never gets started
    if severity < min_rise:
        return flooded

    flooded[source_point] = True

    while queue:
        # take the oldest cell in the queue and look up its distance from the source
        row, col = queue.popleft()
        steps = visited_steps[(row, col)]

        # the rise shrinks with distance; stop spreading once it's negligible
        rise = severity * np.exp(-steps / decay_length)
        if rise < min_rise:
            continue

        local_water_level = source_elevation + rise

        # try to spread into each of the 4 neighbours: up, down, left, right
        for row_offset, col_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset

            # off the edge of the grid
            if not (0 <= neighbour_row < rows and 0 <= neighbour_col < cols):
                continue
            # already flooded (reached earlier by a route at least as short)
            if (neighbour_row, neighbour_col) in visited_steps:
                continue

            # below the water level here -> it floods, and joins the queue
            # so the flood can carry on spreading from it
            if elevation[neighbour_row, neighbour_col] < local_water_level:
                visited_steps[(neighbour_row, neighbour_col)] = steps + 1
                flooded[neighbour_row, neighbour_col] = True
                queue.append((neighbour_row, neighbour_col))

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

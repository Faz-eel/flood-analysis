"""
Simulates flood extent for a localized flood event, and generates a set
of scenarios varying in both location and severity - the flood equivalent
of the leak scenarios in the water distribution script.

Important design choice: each scenario represents heavy, localized rainfall
or a channel overtop occurring at ONE point along the river (analogous to
"a leak occurs at this junction"), rather than a single global water level
applied everywhere. This matters because a global water level always floods
outward from the same lowest point in the watershed, which would make one
sensor placed there trivially detect every scenario - an unrealistic and
uninteresting test, in the same way a leak detection script would be
uninteresting if every leak happened at the same pipe.

With a localized source, whether a given sensor location detects a given
event depends on how close it is - both in distance and in elevation - to
where that event actually originated, which is what makes sensor placement
a genuinely nontrivial problem, just as it is for leak detection.

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
    events, analogous to "pipe junctions" in the water distribution
    script. Uses the true channel path from terrain.py directly, rather
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

    This replaces an earlier version of this function that used a hard
    max_reach step cutoff instead. That cutoff turned out to be hiding a
    real gap, not fixing one: on this watershed's smooth, unbroken
    downhill slope, there is no ridge or barrier anywhere to make
    "connected and below a CONSTANT water level" behave sensibly - an
    upstream source's fixed local_water_level (its own elevation plus
    severity) is still higher than nearly the entire downstream half of
    the map, so an unbounded search connects almost the whole watershed
    for any severity at all, upstream or not. A hard step cutoff patched
    the symptom without addressing why: a single flood event does not
    really raise the water level by the same fixed amount arbitrarily far
    from where it started, because real floodwater loses height as it
    spreads out and travels. Making the rise itself decay with distance
    fixes that directly - connectivity is left to do its real job (a
    genuine barrier still blocks a path, if one exists on the terrain),
    while distance in the open, unblocked case naturally limits spread
    because there is less and less rise left to work with the farther out
    you go.

    The search still needs a way to know when to stop: once the decayed
    rise at a node falls below min_rise (a negligible amount of water),
    expansion from that node stops - not because of an arbitrary step
    count, but because there is essentially no flood left to give.
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
    level rises), analogous to varying both leak location and leak size
    in the water distribution script.

    Returns a list of (source_point, severity, flooded_grid) tuples.
    """
    rng = np.random.default_rng(seed)
    channel_cells = find_channel_cells(elevation)

    scenarios = []
    for _ in range(n_scenarios):
        source_point = channel_cells[rng.integers(len(channel_cells))]
        # Kept deliberately modest: with severities this small, most events
        # stay fairly local to their source rather than spilling into the
        # entire downstream floodplain regardless of where they start -
        # the flood equivalent of using realistic, not catastrophic, leak
        # sizes in the water distribution script
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
    # to demonstrate that flooding is now localized rather than always
    # spreading from one fixed global point
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

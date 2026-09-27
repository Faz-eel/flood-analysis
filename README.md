# Localized Flood Extent Prediction and Risk Mapping

![Flood probability map](flood_risk_map.png)

A small research-style project simulating localized flooding in a
synthetic watershed, and training a surrogate model to predict flood
extent from a scenario's location and severity alone. The trained
surrogate is then used to sweep across thousands of hypothetical
scenarios and build a probabilistic flood-risk map — not "what floods in
this one event," but "how often does each part of the watershed flood,
across everything that could plausibly happen."

Nothing here uses a real watershed or real rainfall data. The terrain,
the physics, and the severities are all synthetic, generated the same
way throughout so results are directly comparable to one another. The
point of the project is the modeling approach — how to represent a
spreading physical event as network input, how to train on a rare-event
target without the model just learning to ignore it, and how to turn a
single-scenario predictor into a risk map — not a claim about any real
place.

## Running it

```
pip install -r requirements.txt
python main.py          # trains and evaluates the surrogate, saves the model
python monte_carlo.py   # runs the Monte Carlo sweep using the saved model
```

`main.py` takes a few minutes (120 training epochs) and saves the trained
model to `flood_surrogate_model.keras`. `monte_carlo.py` loads that saved
model — it does not retrain — and writes `flood_risk_map.png`.

Training is seeded (`keras.utils.set_random_seed(0)` plus TensorFlow's
deterministic-ops mode), so re-running `main.py` on the same machine
reproduces the numbers below exactly. `requirements.txt` pins the exact
library versions they were produced with; other versions (TensorFlow in
particular) may shift the results slightly.

## Files

| File | What it does |
|---|---|
| `terrain.py` | Builds a synthetic watershed: a general downhill slope, a meandering channel carved into it, and smoothed random noise. Returns a 2D elevation grid. |
| `flood_sim.py` | The physics: a localized flood originates at one point along the channel and spreads outward, but the water-level rise decays with distance from the source rather than staying constant, so the search naturally stops once there is nothing meaningful left to give. Also generates random (location, severity) scenarios and their true flooded masks. |
| `surrogate.py` | A fully convolutional neural network that predicts the same flooded/dry mask directly from three inputs per cell: elevation, distance from the source, and severity (broadcast to every cell). Trained with a combined crossentropy + Dice loss so the rare flooded cells actually drive learning. |
| `evaluate.py` | Scores the surrogate against the real physics on held-out scenarios using IoU and Dice (region-overlap metrics, not simple per-cell accuracy). |
| `main.py` | Builds the terrain, generates training/test scenarios, trains the surrogate, evaluates it, and saves the trained model. |
| `monte_carlo.py` | Loads the saved surrogate and runs it on thousands of random scenarios, aggregating the results into a per-cell flood-probability map. |

## The physics: a localized, decaying flood

A flood in this project starts at one point along the channel (a stand-in
for a culvert constriction, a tributary joining, or a burst of localized
rain) and spreads to neighbouring cells that are both connected to the
source and below the local water level.

The water-level rise at the source decays with distance from it
(`rise = severity × exp(-steps / decay_length)`), rather than holding the
same level indefinitely far away, because real floodwater loses height as
it spreads and travels. The search stops on its own once that decayed
rise becomes negligible, so nothing needs an arbitrary step limit for the
spread to stay bounded. Connectivity still does real work within that
range — a genuine barrier in the terrain blocks a path, if one exists —
so a cell floods only when it is both reachable and still below the
decayed local water level at that distance.

## The surrogate: predicting flood extent directly

The surrogate takes three per-cell channels — elevation, distance from
the source, and the scenario's severity broadcast to every cell — and
outputs a flood probability per cell, using only convolutional layers (no
flattening), so a cell's prediction only ever depends on information
actually near it.

Only about 2.2% of any grid is actually flooded in a given scenario, so
the network is trained on a combined crossentropy + Dice loss rather than
plain crossentropy. Dice measures overlap between the predicted and true
flooded area directly and gives no credit for correctly predicting the
dry majority, which keeps the rare flooded cells — the ones that actually
matter — driving the training signal instead of being outweighed by them.

Evaluated on 80 held-out test scenarios (scored with IoU and Dice, not
plain per-cell accuracy, since the dry majority would make accuracy
misleadingly high on its own):

| | Value |
|---|---|
| Mean IoU | 0.869 |
| Mean Dice | 0.929 |
| Worst-case IoU | 0.736 |
| Best-case IoU | 0.953 |

These are from the seeded run. Training is sensitive to the random seed:
across three unseeded runs, mean IoU ranged 0.834–0.856, mean Dice
0.908–0.921, and worst-case IoU swung widely, 0.571–0.675 — so the seeded
run above is on the good end, and the worst-case figure in particular
should not be read as a stable property of the model.


## From single predictions to a risk map

A single simulation, physics or surrogate, only ever answers "what floods
in this one event." `monte_carlo.py` asks a different question: draw a
large number of random plausible events — random location along the
channel, random severity — run each one through the surrogate, and for
every cell, count what fraction of those events actually flooded it. That
per-cell fraction is a flood probability, and together they form a
hazard/risk map rather than a single scenario's outcome.

Over 2,000 random draws:

- The highest-risk single cell flooded in 24.1% of draws.
- 9.9% of the grid (948 of 9,600 cells) has more than a 1-in-10 chance of
  flooding across plausible events.
- Risk is highest along the channel itself and fades smoothly toward the
  banks, matching the terrain rather than looking arbitrary.

## Honest limitations

Everything here is synthetic — the terrain, the severities, and the
resulting risk map are not calibrated to any real place or real rainfall
data, and cannot be used to say whether this watershed is more or less
"safe" than a real one. The risk map also only measures the *hazard*
(how often water shows up somewhere), not exposure or vulnerability —
whether anything of value (a road, a building, people) is actually there
to be affected, which real flood-risk assessment also requires and which
this project does not model.

The surrogate's remaining errors are not uniform: the worst held-out
scenario still only reaches an IoU of 0.736 in the seeded run (and as
low as 0.571 in unseeded runs), and the thresholded Monte Carlo map shows
a visibly ragged, speckled probability estimate along the channel banks
around columns 70–80, plus a small detached cluster of predicted-flooded
cells well off the channel (near row 61, column 90) — signs that some
scenarios remain harder for the model than others, not yet diagnosed
further. The model was also only trained and evaluated on one fixed
terrain; whether it generalises to a differently shaped watershed without
retraining has not been tested.

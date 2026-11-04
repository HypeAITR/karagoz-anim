---
name: puppet
description: The layered puppet engine (src/karagoz/engine/puppet/) - building a puppet set from reference views, fitting it to key drawings, the rig/split/ret decision per interval, rendering in-betweens, and how the draw step uses it. Read before building or changing a puppet set, before touching engine/puppet/, when a puppet in-between looks wrong, or when the draw log says split or ret.
---

# Puppet

The image model draws every **key pose**. Between two keys, a character with a puppet set is not redrawn: the
engine cuts the character into layered parts once, fits that puppet to each key and turns the parts. The result is
smooth in-betweens with no identity drift, at no image cost.

## Pipeline

1. **Build** a set from reference views of one character (front, back, 3/4 left and right, profile):
   `karagoz puppet build-set <preset>/set.yaml <raw_out> [--no-sam] [--max-image-calls N]`, then
   `karagoz puppet repivot <raw_out> <set_out>`.
   - The preset (`engine/puppet/presets/<character>/`) names the parts, their parents, z order, seed points and
     optional rotation `limits` per view.
   - The build splits the reference into complete parts and fills each part under its covers. Pivots, joint limits
     and slack are then measured from the shapes.
   - Output per view: `views/<view>/puppet.json` and `parts/*.png`.
2. **Fit** a key drawing: `karagoz puppet fit <set> <drawing.png> <out>`.
   - A cheap root-only round ranks the views, then the best two views get a full chain fit.
   - Writes `pose.json` (angles, offsets, scale) and a `fit_sheet.png`.
3. **Chain**: `karagoz puppet chain <set> a.png b.png c.png ... <out>`. Every key is fitted once; then every
   consecutive interval gets a transition. Members locked in one interval carry into the next.
4. **Transition** per interval writes `transition.json` with a `decision`:
   - `rig`: the puppet can do it, and K-2 interpolates X→Y.
   - `split`: a clean but large move (more than 27° per segment); add keys in between.
   - `ret`: the puppet cannot do it. Either a changed chain cannot reach Y, or a key's fit leaves more than 3% of its
     drawing unexplained (`FIT_RESIDUAL_MAX`).
5. **In-betweens**: `karagoz puppet inbetween <set> <transition.json> <out> --frames N`.
   - Renders numbered frames; the holds come from the puppet too, so there is no jump at the keys.
   - Writes `inbetween.json` with per-frame measures: locked change, joint paper gap, bbox jump.

In production none of this is called by hand. `studio/puppet_route.py` (`draw_shot`) runs it inside the draw step
for every character with `puppet: <set dir>` in `characters.yaml`:
- `rig` intervals are drawn by the puppet: `<a>-<b>.<k>.png` plus `<pose>.puppet.png` holds, which compose prefers;
- everything else falls back to the image model;
- each interval logs a `draw.puppet` event;
- `karagoz draw --no-puppet` forces today's path.

## What the puppet can and cannot do

- **Can:** rotate every part about its joint within its limits, slide a part a little under a dome cap (`slack_px`),
  move and scale the root, and switch the z order of two parts mid-interval when the drawing shows it.
- **Cannot:**
  - bend at a joint the preset does not have (the robot has no knee: no crouch);
  - foreshorten a limb toward the camera;
  - change view inside an interval;
  - lift the body without its shadow following.
- Measured 4 Nov on the robot:
  - poses it seats leave 0.3–0.9% of the drawing unexplained;
  - knee bends, arms toward the viewer and a lifted body leave 6–12%.
  - The 3% gate sits in that gap.

## Speed

- **Measured** (12-thread machine, 4 Nov):
  - one view fit takes about 110 s;
  - the K-1 acceptance (8 fits and 3 transitions) takes about 400 s;
  - a 5-key chain takes about 4.5 min.
- **What made it fast:**
  - The score kernel `_exclusive_core` is compiled with numba; `USE_COMPILED=False` runs the numpy reference.
    Results are bit-identical, checked against the numpy path.
  - Chains fit only the two best-ranked views.
  - Transitions skip a from-scratch refit that repeats `pose_Y`.
  - Worker count follows the machine: CPUs − 1, capped by free memory at 700 MB per worker; override with
    `KARAGOZ_WORKERS`.
- **Before you change search code:**
  1. Run `_wp/kabul/kabul.py`-style acceptance on a set.
  2. Compare poses and renders pixel by pixel with the old code.
- A speed change that moves a pose is a quality change. The greedy beam (`_greedy_beam`) costs about 25% but finds
  the better arm on two acceptance poses: keep it.

## When something looks wrong

Find the cause before you fix anything: measure the drawing change d_c and the puppet change d_k per limb
(`limb_measure.py`, `limb_exam.json`).
- If the drawing changed, it is a drawing difference.
- If the drawing is the same and the puppet differs, it is an engine fault.

Common causes:
- **A part slides off its parent.** A slack value it should not have; slack belongs only under a real dome. Check
  `slack_px` in `puppet.json`.
- **A limb at an impossible angle.** Missing `limits` in the preset.
- **A patch or ragged outline where a limb meets a cap.** The completion fill under a cover; see the cap-mouth rules
  in `complete.py`.
- **A part drawn behind something the reference shows in front.** Preset z disagrees with the reference.
  `report.json["z_conflicts"]` flags it.
- **Puppet frames jump at a key.** Holds taken from the key drawing instead of the puppet (`hold_from_puppet=False`).

Never patch one case: write the general rule, add a failing test, rebuild the set, rerun the acceptance, and look at
1:1 crops yourself.

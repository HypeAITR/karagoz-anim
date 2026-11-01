"""Rig motor: the inbetween frames between two key poses of one character, made from the key drawings themselves.

Ported from the rig experiment (X->Y method, `docs/rig-motoru/PLAN.md` in the research repo). The motor writes the
same inbetween drawings the draw department would (`<pA>-<pB>.<k>.png` in the shot's character folder), in key A's
canvas, so placement, patches, camera and render stay unchanged. Modules:

- `masks`, `measure`, `interval`: SAM part masks and their reconciliation, shape measures, one interval's context.
- `joints`, `pivot`, `plan`: joints of both keys, shoulder pivot, the per-frame plan.
- `deform`, `changed`, `match`, `bend`, `jump`: mesh/MLS deformer, changed region, matched points, body bending,
  jump measure.
- `canonical`, `head`, `layers`, `frames`, `fill`: canonical sources, rigid head, layer compositing, the frames,
  gap fill.
"""

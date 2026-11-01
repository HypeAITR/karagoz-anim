"""WP-42: single-source ARAP rig (T-82). Synthetic checks; acceptance on real drawings is done by the reviewer."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from scipy import ndimage as ni
from scipy.spatial import cKDTree

from karagoz.engine import arap, rig

ENGINE = Path(arap.__file__).resolve().parent


# AnimatedDrawings `tests/test_arap.py` expectations (float32 origin, atol 1e-3)
def test_arap_single_triangle():
    v = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 2.0]])
    a = arap.ARAP(np.array([[2.0, 2.0], [4.0, 2.0]]), np.array([[0, 1, 2]]), v)
    assert np.isclose(a.solve(np.array([[-5.0, 0.0], [5.0, 0.0]])), [[-5, 0], [0, 1], [5, 0]], atol=1e-3).all()


def test_arap_two_triangles():
    v = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]])
    a = arap.ARAP(np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([[0, 1, 2], [0, 2, 3]]), v)
    b = a.solve(np.array([[1.0, 0.0], [1.7, 0.7]]))
    expected = [[1, 0], [2.91471856e-01, 7.05685418e-01], [9.97157285e-01, 1.41137085], [1.7, 0.7]]
    assert np.isclose(b, expected, atol=1e-3).all()


def test_arap_four_squares():
    v = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 1], [2, 0], [0, 2], [1, 2], [2, 2]], float)
    t = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5], [1, 6, 7], [1, 7, 2], [2, 7, 8], [2, 8, 4]])
    a = arap.ARAP(np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]), t, v)
    b = a.solve(np.array([[0.0, 0.0], [0.0, 3.0], [6.0, 0.0]]))
    expected = [
        [0, 0],
        [6.78428061e-01, 1.37166545],
        [2.14606263, 1.19790398],
        [2.81917351, 1.12790606e-02],
        [3.95163838, 1.34725960],
        [6, 0],
        [0, 3],
        [1.46633111, 2.60720416],
        [2.82413859, 2.62209072],
    ]
    assert np.isclose(b, expected, atol=1e-3).all()


def test_arap_keeps_meta_licence_notice():
    text = (ENGINE / "arap.py").read_text(encoding="utf-8")
    assert "AnimatedDrawings" in text and "MIT" in text and "Meta Platforms" in text


def test_rig_module_is_standalone():
    tree = ast.parse((ENGINE / "rig.py").read_text(encoding="utf-8"))
    roots = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            roots.add("." if n.level else (n.module or "").split(".")[0])
    assert roots <= {"__future__", "dataclasses", "numpy", "scipy", "karagoz", "."}, roots
    assert "API" in (ast.get_docstring(tree) or "")


def _arm():
    """Outlined rectangle 'arm'; pins on its axis."""
    h, w = 400, 500
    rgb = np.full((h, w, 3), 255, np.uint8)
    x0, x1, y0, y1 = 100, 400, 170, 230
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    rgb[m] = (235, 180, 140)
    rgb[m & ~ni.binary_erosion(m, iterations=3)] = (20, 20, 20)
    pins = np.array([[x0 + 5.0, 200.0], [175.0, 200.0], [250.0, 200.0], [325.0, 200.0], [x1 - 5.0, 200.0]])
    return rgb, m, pins


def _thickness(g: np.ndarray) -> float:
    """Median ink line thickness on the ink ridge."""
    ink = g < 128
    d = ni.distance_transform_edt(ink)
    ridge = ink & (d == ni.maximum_filter(d, 3))
    return float(np.median(2 * d[ridge] - 1))


def _rotate(p: np.ndarray, centre: np.ndarray, deg: float) -> np.ndarray:
    r = np.radians(deg)
    return centre + (p - centre) @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])


def test_identity_frame_is_the_source():
    rgb, m, pins = _arm()
    f = rig.Rig([rig.Layer("arm", rgb, m.astype(float), pins)]).frame({"arm": pins})
    assert np.abs(f.rgb.astype(int) - rgb.astype(int)).max() <= 2


def test_rigid_30_degrees_keeps_outline_and_pins():
    rgb, m, pins = _arm()
    r = rig.Rig([rig.Layer("arm", rgb, m.astype(float), pins)])
    target = _rotate(pins, pins[0], 30.0)
    f = r.frame({"arm": target})
    assert np.hypot(*(r.carry("arm", f.vertices["arm"], pins) - target).T).max() < 1.0
    assert abs(_thickness(rgb.min(axis=2)) - _thickness(f.rgb.min(axis=2))) <= 0.7
    assert f.ratio["arm"].min() > 0.9 and f.ratio["arm"].max() < 1.1


def test_bend_in_the_middle_keeps_outline_without_folding():
    rgb, m, pins = _arm()
    target = pins.copy()
    target[3:] = _rotate(pins[3:], pins[2], 30.0)
    f = rig.Rig([rig.Layer("arm", rgb, m.astype(float), pins)]).frame({"arm": target})
    assert abs(_thickness(rgb.min(axis=2)) - _thickness(f.rgb.min(axis=2))) <= 0.7
    # a 60 px thick bar broken by 30 degrees really narrows on the inner side (measured 0.65): checks no fold
    assert f.ratio["arm"].min() > 0.5 and f.ratio["arm"].max() < 1.5


def test_mesh_density_and_coverage():
    _, m, _ = _arm()
    mesh = rig.build_mesh(m)
    edge = np.argwhere(m ^ ni.binary_erosion(m))[:, ::-1]
    assert cKDTree(mesh.vertices).query(edge)[0].max() <= 8.0
    u, _ = rig.pixel_map(mesh.vertices, mesh.vertices, mesh.triangles, m.shape)
    assert np.isfinite(u[m]).all()


def test_rigid_layer_and_draw_order():
    rgb, m, pins = _arm()
    top = np.zeros_like(rgb)
    top[:] = (0, 0, 255)
    top_alpha = np.zeros(m.shape, np.float32)
    top_alpha[180:220, 200:260] = 1.0
    layers = [rig.Layer("arm", rgb, m.astype(float), pins), rig.Layer("patch", top, top_alpha)]
    shift = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    f = rig.Rig(layers).frame({"arm": pins, "patch": shift})
    assert tuple(f.rgb[200, 235]) == (0, 0, 255)  # top layer drawn last, moved 10 px right
    assert tuple(f.rgb[200, 205]) != (0, 0, 255)


def test_revealed_texture_only_inside_region():
    rgb, m, pins = _arm()
    mesh = rig.build_mesh(m)
    other = np.full_like(rgb, 7)
    region = np.zeros(m.shape, bool)
    region[190:210, 240:260] = True
    out = rig.revealed_texture(rgb, mesh, mesh.vertices, other, region)
    assert (out[region & m] == 7).all()
    assert (out[~region] == rgb[~region]).all()

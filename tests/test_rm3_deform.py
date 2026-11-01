"""RM-3: mesh/MLS deformer matches ag_deformer, plus synthetic checks."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest
import rig_golden

from karagoz.engine.rig import deform


def _experiment():
    root = rig_golden.fixture()
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import ag_deformer

    return ag_deformer


@pytest.fixture(scope="module")
def exp():
    return _experiment()


def _blob_mask() -> np.ndarray:
    m = np.zeros((200, 200), bool)
    m[40:150, 30:140] = True
    yy, xx = np.ogrid[:200, :200]
    m |= (xx - 160) ** 2 + (yy - 160) ** 2 <= 25**2
    return m


def _blob_image() -> np.ndarray:
    rgb = np.full((200, 200, 3), 255, np.uint8)
    rgb[40:150, 30:140] = (240, 200, 160)
    rgb[40:42, 30:140] = (0, 0, 0)
    rgb[148:150, 30:140] = (0, 0, 0)
    rgb[40:150, 30:32] = (0, 0, 0)
    rgb[40:150, 138:140] = (0, 0, 0)
    return rgb


@pytest.mark.rig_golden
def test_premultiplied_matches_experiment_opaque_and_alpha(exp) -> None:
    rgb = _blob_image()
    assert np.array_equal(deform.premultiplied(rgb), exp.pm_kur(rgb))
    alpha = np.linspace(0, 1, 200 * 200, dtype=np.float32).reshape(200, 200)
    assert np.array_equal(deform.premultiplied(rgb, alpha), exp.pm_kur(rgb, alpha))


@pytest.mark.rig_golden
def test_build_mesh_matches_experiment_with_dense_and_extra(exp) -> None:
    mask = _blob_mask()
    dense = np.zeros_like(mask)
    dense[60:130, 50:120] = True
    extra = np.array([[50.0, 60.0], [120.0, 130.0], [500.0, 500.0]])
    got = deform.build_mesh(mask, dense=dense, extra=extra)
    want = exp.ag_kur(mask, yogun=dense, ek=extra)
    assert np.array_equal(got.rest, want.kose)
    assert np.array_equal(got.uv, want.uv)
    assert np.array_equal(got.tri, want.ucgen)


@pytest.mark.rig_golden
def test_flow_map_matches_experiment_on_shifted_mesh(exp) -> None:
    mask = _blob_mask()
    mesh = deform.build_mesh(mask)
    target = mesh.rest + np.array([6.5, -3.25])
    got = deform.flow_map(mesh, target, mask.shape)
    want = exp.harita(exp.Ag(kose=mesh.rest, uv=mesh.uv, ucgen=mesh.tri), target, mask.shape)
    assert np.array_equal(got[0], want[0])
    assert np.array_equal(got[1], want[1])


@pytest.mark.rig_golden
def test_warp_matches_experiment_on_shifted_mesh(exp) -> None:
    rgb = _blob_image()
    mask = _blob_mask()
    mesh = deform.build_mesh(mask)
    target = mesh.rest + np.array([6.5, -3.25])
    pm = deform.premultiplied(rgb)
    got = deform.warp(pm, mesh, target)
    want = exp.bas(pm, exp.Ag(kose=mesh.rest, uv=mesh.uv, ucgen=mesh.tri), target)
    assert np.array_equal(got, want)


@pytest.mark.rig_golden
def test_unpremultiply_matches_experiment_and_keeps_zero_transparent(exp) -> None:
    rgb = _blob_image()
    pm = deform.premultiplied(rgb)
    pm[0, 0] = (0, 0, 0, 0)
    got_rgb, got_a = deform.unpremultiply(pm)
    want_rgb, want_a = exp.coz(pm)
    assert np.array_equal(got_rgb, want_rgb)
    assert np.array_equal(got_a, want_a)
    assert (got_rgb[0, 0] == 0).all()


@pytest.mark.rig_golden
def test_area_ratio_matches_experiment_scaled_and_folded(exp) -> None:
    mask = _blob_mask()
    mesh = deform.build_mesh(mask)
    exp_mesh = exp.Ag(kose=mesh.rest, uv=mesh.uv, ucgen=mesh.tri)
    assert np.array_equal(deform.area_ratio(mesh, mesh.rest * 2), exp.alan_orani(exp_mesh, mesh.rest * 2))
    folded = mesh.rest.copy()
    folded[mesh.tri[0, 2]] = mesh.rest[mesh.tri[0, 0]]  # collapse one triangle past its edge
    folded[mesh.tri[0, 2]] += mesh.rest[mesh.tri[0, 0]] - mesh.rest[mesh.tri[0, 1]]
    got = deform.area_ratio(mesh, folded)
    assert np.array_equal(got, exp.alan_orani(exp_mesh, folded))
    assert (got < 0).any()


@pytest.mark.rig_golden
def test_mls_rigid_matches_experiment_on_random_controls(exp) -> None:
    rng = np.random.default_rng(7)
    vertices = rng.uniform(0, 200, (50, 2))
    p = np.array([[40.0, 40.0], [160.0, 40.0], [100.0, 160.0]])
    q = p + np.array([[5.0, -3.0], [-4.0, 6.0], [2.0, 2.0]])
    assert np.array_equal(deform.mls_rigid(vertices, p, q), exp.mls_sert(vertices, p, q))


@pytest.mark.rig_golden
def test_identity_source_matches_experiment(exp) -> None:
    rest = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(deform.identity_source(3, rest), exp.kimlik_kaynak(3, rest))


@pytest.mark.rig_golden
def test_lbs_source_matches_experiment_with_affine_and_pre(exp) -> None:
    def matrix(frame: int) -> np.ndarray:
        return np.array([[1.0, 0.0, 2.0 * frame], [0.0, 1.0, -1.0 * frame]])

    def pre(rest: np.ndarray) -> np.ndarray:
        return rest * 2.0

    rest = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert np.array_equal(deform.lbs_source(matrix)(2, rest), exp.lbs_kaynak(matrix)(2, rest))
    assert np.array_equal(deform.lbs_source(matrix, pre)(2, rest), exp.lbs_kaynak(matrix, pre)(2, rest))


def test_mls_identity_leaves_vertices_in_place() -> None:
    rng = np.random.default_rng(3)
    vertices = rng.uniform(0, 200, (30, 2))
    p = np.array([[40.0, 40.0], [160.0, 40.0], [100.0, 160.0]])
    # Not bit-identical: the (c, s) rotation normalisation rounds at ~1e-12.
    np.testing.assert_allclose(deform.mls_rigid(vertices, p, p.copy()), vertices, rtol=0, atol=1e-9)


def test_mls_control_point_lands_exactly_on_its_target() -> None:
    p = np.array([[40.0, 40.0], [160.0, 40.0], [100.0, 160.0]])
    q = p + np.array([[5.0, -3.0], [-4.0, 6.0], [2.0, 2.0]])
    # Within ~1e-9, not bit-exact: the inverse-distance weights are capped by the 1e-6 floor.
    np.testing.assert_allclose(deform.mls_rigid(p.copy(), p, q), q, rtol=0, atol=1e-8)


def test_identity_warp_reproduces_the_source_pixels() -> None:
    rgb = _blob_image()
    mesh = deform.build_mesh(_blob_mask())
    out, _ = deform.unpremultiply(deform.warp(deform.premultiplied(rgb), mesh, deform.identity_source(0, mesh.rest)))
    assert np.array_equal(out[45:145, 35:135], rgb[45:145, 35:135])


def test_bent_rectangle_keeps_its_outline() -> None:
    rgb = _blob_image()
    mask = np.zeros((200, 200), bool)
    mask[40:150, 30:140] = True
    mesh = deform.build_mesh(mask, dense=mask)
    p = np.array([[30.0, 40.0], [140.0, 40.0], [30.0, 150.0], [140.0, 150.0]])
    q = p + np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 0.0], [8.0, 0.0]])
    bent = deform.mls_rigid(mesh.rest, p, q)
    out, alpha = deform.unpremultiply(deform.warp(deform.premultiplied(rgb), mesh, bent))
    # Outside-mesh pixels are black with zero alpha by design; ink counts only where the layer is drawn.
    ink = (out.astype(np.int16).sum(axis=2) < 3 * 48) & (alpha > 0.5)
    ink_ref = rgb.astype(np.int16).sum(axis=2) < 3 * 48
    # Remap resampling changes edge pixels; the outline itself must survive whole.
    assert 0.85 * ink_ref.sum() < ink.sum() < 1.15 * ink_ref.sum()
    n, _ = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    assert n == 2  # background + one unbroken outline loop

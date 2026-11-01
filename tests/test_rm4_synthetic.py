"""RM-4: synthetic checks without the fixture (identity, z-order, paper tone, fill cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from karagoz.engine.rig import canonical, fill, frames, head, layers, masks
from karagoz.engine.rig.interval import IntervalContext


def _ctx(tmp_path: Path) -> IntervalContext:
    return IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
    )


def test_identity_transform_leaves_layer_unchanged(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ground = np.full((60, 60, 3), 255, np.uint8)
    ground[10:50, 10:50] = (240, 240, 240)
    piece_rgb = np.full((60, 60, 3), 255, np.uint8)
    piece_rgb[20:40, 20:40] = (200, 120, 110)
    piece_rgb[20:22, 20:40] = (15, 15, 18)
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    identity = ((30.0, 30.0), (30.0, 30.0), 0.0, 1.0, (1.0, 0.0))
    assert layers.is_identity_tf(identity)
    result = layers.composite_arm_frame(
        ctx,
        ground,
        [{"ad": "hand_L", "rgb": piece_rgb, "mask": mask, "tf": identity}],
        [],
        skin=np.array([200.0, 120.0, 110.0]),
    )
    # The piece's inside arrives as drawn; far outside the piece the ground is untouched.
    inner = np.zeros((60, 60), bool)
    inner[24:36, 24:36] = True
    assert np.array_equal(result["render"][inner], piece_rgb[inner])
    far = ~masks.dilate(mask, 4)
    assert np.array_equal(result["render"][far], ground[far])


def test_z_order_puts_arm_over_body(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ground = np.full((60, 60, 3), 255, np.uint8)
    ground[:, :] = (235, 235, 235)
    lower = np.zeros((60, 60), bool)
    lower[20:40, 10:34] = True
    upper = np.zeros((60, 60), bool)
    upper[20:40, 26:50] = True
    rgb_lower = np.full((60, 60, 3), 255, np.uint8)
    rgb_lower[lower] = (0, 0, 255)
    rgb_upper = np.full((60, 60, 3), 255, np.uint8)
    rgb_upper[upper] = (255, 0, 0)
    identity = ((0.0, 0.0), (0.0, 0.0), 0.0, 1.0, (1.0, 0.0))
    result = layers.composite_arm_frame(
        ctx,
        ground,
        [
            {"ad": "fore_L", "rgb": rgb_lower, "mask": lower, "tf": identity},
            {"ad": "hand_L", "rgb": rgb_upper, "mask": upper, "tf": identity},
        ],
        [],
    )
    inner = np.zeros((60, 60), bool)
    inner[24:36, 29:31] = True  # inside the overlap, away from soft edges
    assert (result["render"][inner] == (255, 0, 0)).all()
    only_lower = np.zeros((60, 60), bool)
    only_lower[24:36, 14:22] = True
    assert (result["render"][only_lower] == (0, 0, 255)).all()


def test_paper_tone_threshold_parts_paper_from_garment(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    rgb = np.full((40, 40, 3), 255, np.uint8)
    rgb[5:15, 5:15] = (245, 245, 245)  # light garment, edge-connected through white
    rgb[20:30, 20:30] = (250, 250, 250)  # paper grain, enclosed by a line
    rgb[19:31, 19:20] = (10, 10, 10)
    rgb[19:31, 30:31] = (10, 10, 10)
    rgb[19:20, 19:31] = (10, 10, 10)
    rgb[30:31, 19:31] = (10, 10, 10)
    paper = layers.canonical_paper(ctx, rgb)
    assert paper[0, 0]
    assert not paper[10, 10]  # 245 < 251: garment, not paper
    assert not paper[25, 25]  # enclosed grain: unreachable from the edge


def test_background_walks_only_through_light_pixels() -> None:
    rgb = np.full((40, 40, 3), 255, np.uint8)
    rgb[10:30, 10:30] = (200, 200, 200)
    rgb[10:30, 10:12] = (10, 10, 10)
    assert canonical.background(rgb)[0, 0]
    assert not canonical.background(rgb)[20, 20]


def test_fill_cache_hit_never_calls_model(tmp_path: Path) -> None:
    drawing = tmp_path / "a.png"
    Image.fromarray(np.full((30, 30, 3), 200, np.uint8)).save(drawing)
    cached = fill.cache_path(tmp_path, "A", drawing, "prompt")
    assert cached.name.startswith("muse_kolsuz_A_") and cached.suffix == ".png"
    noisy = np.random.default_rng(3).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(noisy).save(cached)
    assert cached.stat().st_size > fill.CACHE_MIN_BYTES

    def _boom(_d: Path, _s: tuple[int, int], _p: str):
        raise AssertionError("model called on a cache hit")

    got = fill.armless_fill(drawing, (200, 200), "prompt", cached, _boom)
    assert np.array_equal(got, noisy)


def test_fill_cache_miss_without_client_raises(tmp_path: Path) -> None:
    drawing = tmp_path / "a.png"
    Image.fromarray(np.full((30, 30, 3), 200, np.uint8)).save(drawing)
    with pytest.raises(RuntimeError, match="no Fill client"):
        fill.armless_fill(drawing, (30, 30), "prompt", tmp_path / "muse_kolsuz_A_x.png", None)


def test_fill_cache_miss_calls_injected_client_and_saves(tmp_path: Path) -> None:
    drawing = tmp_path / "a.png"
    Image.fromarray(np.full((30, 30, 3), 200, np.uint8)).save(drawing)
    seen = {}

    def _double(d: Path, s: tuple[int, int], p: str):
        seen["args"] = (d, s, p)
        return np.full((30, 30, 3), 77, np.uint8)

    target = tmp_path / "sub" / "muse_kolsuz_A_x.png"
    got = fill.armless_fill(drawing, (30, 30), "prompt", target, _double)
    assert seen["args"] == (drawing, (30, 30), "prompt")
    assert (got == 77).all()
    assert target.is_file()
    assert (np.array(Image.open(target).convert("RGB")) == 77).all()


@pytest.mark.live
def test_live_fill_returns_sized_rgb(tmp_path: Path) -> None:
    drawing = tmp_path / "a.png"
    Image.fromarray(np.full((60, 40, 3), 200, np.uint8)).save(drawing)
    got = fill.live_fill(drawing, (40, 60), "test prompt, ignore")
    assert got.shape == (60, 40, 3) and got.dtype == np.uint8


def test_head_frame_matrix_is_identity_at_rest() -> None:
    data = {"boyun_a": [10.0, 20.0], "oteleme": [5.0, -3.0], "derece": 4.0}
    np.testing.assert_allclose(head.frame_matrix(data, 0.0, False), [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    back = head.frame_matrix(data, 1.0, True)
    np.testing.assert_allclose(back, [[1, 0, 0], [0, 1, 0]], atol=1e-12)


def test_head_layer_draw_identity_is_bit_identical(tmp_path: Path) -> None:
    rgb = np.full((50, 50, 3), 255, np.uint8)
    rgb[10:30, 10:30] = (200, 120, 110)
    Image.fromarray(rgb).save(tmp_path / "a_kafa.png")
    alpha = np.zeros((50, 50), np.uint8)
    alpha[10:30, 10:30] = 255
    Image.fromarray(alpha).save(tmp_path / "a_kafa_alfa.png")
    layer = head.Layer(tmp_path, "a", {})
    ground = np.full((50, 50, 3), 230, np.uint8)
    got = layer.draw(ground, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert np.array_equal(got[10:30, 10:30], rgb[10:30, 10:30])
    assert np.array_equal(got[0:5, 0:5], ground[0:5, 0:5])


def test_time_curve_eases_zero_to_one() -> None:
    plan = {"kareler": [{"kare": i} for i in range(8)]}
    curve = frames.time_curve(plan)
    assert curve(0) == 0.0
    assert curve(7) == 1.0
    assert all(0.0 <= curve(i) <= 1.0 for i in range(8))
    assert curve(3) < 0.5 < curve(4)
    assert abs(curve(3) - (1 - curve(4))) < 1e-9


def test_apply_cut_moves_frames_to_next_source() -> None:
    plan = {
        "olcek_siniri": 0.15,
        "kaynaklar": {
            "A": {"kare": 0, "eklemler": {}, "boy": {"upper_L": 10.0, "fore_L": 10.0, "hand_L": 10.0}},
            "B": {"kare": 3, "eklemler": {}, "boy": {"upper_L": 10.0, "fore_L": 10.0, "hand_L": 10.0}},
        },
        "kareler": [
            {
                "kare": i,
                "kaynak": "A",
                "hedef_eklemler": {
                    "shoulder_L": [0.0, 0.0],
                    "elbow_L": [10.0, 0.0],
                    "wrist_L": [20.0, 0.0],
                    "hand_center_L": [30.0, 0.0],
                },
                "vekil_boy": {"upper_L": 10.0, "fore_L": 10.0, "hand_L": 10.0},
            }
            for i in range(4)
        ],
    }
    cut = frames.apply_cut(plan, [2])
    assert [k["kaynak"] for k in cut["kareler"]] == ["A", "A", "B", "B"]
    assert cut["degisim_kareleri"] == [2]


def test_fill_mode_picks_majority_tone() -> None:
    px = np.zeros((100, 3), np.uint8)
    px[:80] = (240, 240, 240)
    px[80:] = (180, 180, 180)
    assert (canonical.fill_mode(px) == (240, 240, 240)).all()

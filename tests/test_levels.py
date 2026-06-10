"""Tests for the columnar Level primitive."""

import pytest
import torch

from torchsx.levels import ColumnSpec, Level


def _obs_level(n=5):
    lv = Level()
    lv["hkl"] = torch.arange(n * 3, dtype=torch.int16).reshape(n, 3)
    lv["I"] = torch.arange(n, dtype=torch.float32)
    lv["crystal_id"] = torch.zeros(n, dtype=torch.int32)
    return lv


def test_shared_n_invariant_enforced():
    lv = Level()
    lv["I"] = torch.arange(5, dtype=torch.float32)
    assert lv.n == 5
    with pytest.raises(ValueError, match="rows"):
        lv["bad"] = torch.arange(4, dtype=torch.float32)


def test_arbitrary_trailing_dims():
    lv = Level()
    lv["scalar"] = torch.zeros(3)
    lv["vec"] = torch.zeros(3, 3)
    lv["patch"] = torch.zeros(3, 21, 21)
    assert lv.schema["scalar"].element_shape == ()
    assert lv.schema["vec"].element_shape == (3,)
    assert lv.schema["patch"].element_shape == (21, 21)


def test_extensible_setitem():
    lv = _obs_level()
    lv["peak_snr"] = torch.ones(5)  # add a column, no class edit
    assert "peak_snr" in lv
    assert set(lv.keys()) == {"hkl", "I", "crystal_id", "peak_snr"}


def test_select_bool_and_index():
    lv = _obs_level()
    sub = lv.select(torch.tensor([True, False, True, False, True]))
    assert sub.n == 3
    assert torch.equal(sub["I"], torch.tensor([0.0, 2.0, 4.0]))
    sub2 = lv.gather(torch.tensor([4, 0]))
    assert torch.equal(sub2["I"], torch.tensor([4.0, 0.0]))
    assert sub2["hkl"].dtype == torch.int16


def test_select_drops_derived():
    lv = _obs_level()
    lv["hkl_id"] = torch.arange(5, dtype=torch.int64)
    lv.schema["hkl_id"] = ColumnSpec(
        dtype=torch.int64, role="derived", derived_from=("hkl",)
    )
    sub = lv.select(torch.tensor([0, 1]))
    assert "hkl_id" not in sub
    assert "I" in sub


def test_concat_basic():
    a, b = _obs_level(3), _obs_level(2)
    c = Level.concat([a, b])
    assert c.n == 5
    assert c["I"].shape == (5,)
    assert c["hkl"].dtype == torch.int16


def test_concat_rejects_ragged_trailing():
    a, b = Level(), Level()
    a["patch"] = torch.zeros(2, 21, 21)
    b["patch"] = torch.zeros(2, 15, 15)
    with pytest.raises(ValueError, match="element_shape"):
        Level.concat([a, b])


def test_categorical_roundtrip_and_concat():
    a, b = Level(), Level()
    a.set_categorical("fname", ["x.h5", "x.h5", "y.h5"])
    b.set_categorical("fname", ["y.h5", "z.h5"])
    assert a.get_categorical("fname") == ["x.h5", "x.h5", "y.h5"]
    c = Level.concat([a, b])
    # unified table remaps indices correctly
    assert c.get_categorical("fname") == ["x.h5", "x.h5", "y.h5", "y.h5", "z.h5"]


def test_to_device_cpu_noop():
    lv = _obs_level()
    out = lv.to("cpu")
    assert out.n == 5
    assert out["I"].device.type == "cpu"


def test_state_dict_roundtrip_restores_int16():
    lv = _obs_level()
    tensors, schema = lv.state_dict("obs")
    # int16 hkl upcast to int32 on save
    assert tensors["obs/hkl"].dtype == torch.int32
    restored = Level.from_state("obs", tensors, schema)
    assert restored["hkl"].dtype == torch.int16
    assert torch.equal(restored["hkl"].int(), lv["hkl"].int())
    assert torch.equal(restored["I"], lv["I"])


def test_state_dict_roundtrip_categorical():
    lv = Level()
    lv.set_categorical("fname", ["a.h5", "b.h5", "a.h5"])
    tensors, schema = lv.state_dict("image")
    restored = Level.from_state("image", tensors, schema)
    assert restored.get_categorical("fname") == ["a.h5", "b.h5", "a.h5"]


def test_empty_level_roundtrip():
    lv = Level(n=0)
    lv["I"] = torch.zeros(0, dtype=torch.float32)
    lv["vec"] = torch.zeros(0, 3, dtype=torch.float32)
    assert lv.n == 0
    assert lv.schema["vec"].element_shape == (3,)
    tensors, schema = lv.state_dict("obs")
    restored = Level.from_state("obs", tensors, schema)
    assert restored.n == 0
    assert restored.schema["vec"].element_shape == (3,)

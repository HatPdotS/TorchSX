"""Invalid raw measurements must not reach calibration or crystal pooling."""

import pytest
import torch

from torchsx import SFXStream
from torchsx.io._metadata import StreamMetadata
from torchsx.levels import Level
from torchsx.merging.config import MergeConfig
from torchsx.merging.hkl.noise import PoissonNoiseModel
from torchsx.merging.observations import prepare_observations
from torchsx.nn.obs_features import ObsFeatureSpec


@pytest.mark.parametrize("column,value", [
    ("sigma", float("nan")), ("sigma", float("inf")),
    ("sigma", 0.0), ("sigma", -1.0),
    ("I", float("nan")), ("I", float("inf")),
    ("background", float("nan")), ("background", float("-inf")),
])
def test_filter_before_calibration_preserves_alignment(sample_core, monkeypatch, capsys, column, value):
    intensity = sample_core.I.clone()
    intensity[0] = -1.0  # Valid negative signal must survive.
    core = sample_core.replace(I=intensity)
    values = getattr(core, column).clone()
    values[1] = value
    core = core.replace(**{column: values})
    partiality = torch.arange(5, dtype=torch.float32) / 10 + 0.1
    weights = torch.tensor([0.25, 0.75])
    keep = torch.tensor([True, False, True, True, True])
    calls = []

    def calibrate(self, I, bg, ewald):
        assert torch.equal(I, intensity[keep])
        assert torch.isfinite(bg).all()
        calls.append(True)
        return 2.0, 0.0, 0.0

    monkeypatch.setattr(PoissonNoiseModel, "calibrate_sigma2_of_bg", calibrate)
    actual = prepare_observations(SFXStream(core), MergeConfig(gain="calibrate"),
                                  partiality=partiality, crystal_weights=weights)
    assert calls == [True]
    assert f"{column}=1" in capsys.readouterr().out
    expected = prepare_observations(SFXStream(core.select_cascade({"obs": keep})),
                                    MergeConfig(gain=2.0), partiality=partiality[keep],
                                    crystal_weights=weights)
    for name in ("I_obs", "bg", "hkl_id", "partiality", "obs_weights"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    for name in ("crystal_id", "sigma", "peak", "fs", "ss", "q_dir", "ewald_offset"):
        torch.testing.assert_close(getattr(actual.obs_info, name), getattr(expected.obs_info, name))
    assert actual.n_obs_total == 4
    assert (actual.I_obs < 0).sum() == 1
    assert core.n_observations == 5  # Source stream stays untouched.
    features = ObsFeatureSpec(actual)
    assert all(torch.isfinite(t).all() for t in features.build(actual.obs_info))


def test_filter_preserves_empty_crystal_ids_and_timepoints(sample_core):
    core = sample_core.replace(sigma=torch.tensor([1., 2., 3., float("nan"), 0.]),
                               metadata=StreamMetadata(image={"timepoint": [0., 1.]}))
    prep = prepare_observations(SFXStream(core), MergeConfig(gain=1.0),
                                crystal_weights=torch.tensor([0.25, 0.75]))
    assert prep.n_obs_total == 3
    assert prep.obs_info.n_crystals == 2
    assert prep.obs_info.crystal_feat.shape[0] == 2
    assert prep.obs_info.crystal_id.tolist() == [0, 0, 0]
    assert prep.obs_info.delay_bin_idx.tolist() == [0, 1]
    assert (prep.obs_weights == 0.25).all()


def test_missing_sigma_retains_noise_fallback(sample_core):
    obs = sample_core.levels["obs"]
    without_sigma = Level({k: v for k, v in obs.columns.items() if k != "sigma"}, obs.schema)
    prep = prepare_observations(SFXStream(sample_core.replace(obs=without_sigma)), MergeConfig(gain=1.0))
    assert prep.n_obs_total == 5
    assert prep.obs_info.sigma is None


def test_all_invalid_fails_before_calibration(sample_core, monkeypatch):
    def no_calibration(*args):
        pytest.fail("empty input reached calibration")
    monkeypatch.setattr(PoissonNoiseModel, "calibrate_sigma2_of_bg", no_calibration)
    core = sample_core.replace(sigma=torch.full_like(sample_core.sigma, float("nan")))
    with pytest.raises(ValueError, match="No valid observations"):
        prepare_observations(SFXStream(core), MergeConfig(gain="calibrate"))

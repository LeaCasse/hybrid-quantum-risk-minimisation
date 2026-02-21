import numpy as np

from risk_pipeline.scaling import StandardScaler, MinMaxPM1Scaler, fit_transform_inverse_check


def test_standard_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    scaler = StandardScaler()
    _, ok = fit_transform_inverse_check(scaler, x)
    assert ok


def test_standard_constant_feature_no_nan():
    x = np.c_[np.linspace(0, 1, 50), np.ones(50)]
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    assert np.isfinite(z).all()
    x_hat = scaler.inverse_transform(z)
    assert np.allclose(x, x_hat)


def test_minmax_pm1_roundtrip_and_bounds():
    rng = np.random.default_rng(1)
    x = rng.uniform(-5, 7, size=(200, 2))
    scaler = MinMaxPM1Scaler().fit(x)
    z = scaler.transform(x)
    assert z.min() >= -1.0 - 1e-12
    assert z.max() <= 1.0 + 1e-12
    x_hat = scaler.inverse_transform(z)
    assert np.allclose(x, x_hat)

"""
XRD / XAS / Raman / photoelectron / UV_vis モジュールの
Quantity/QArray 移行後の動作テスト(DataSeriese サブクラスの構築と往復)。

実行方法:
    python test_spectroscopy_modules.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import pandas as pd

import BFC_libs.XRD as xrd
import BFC_libs.Raman as rmn
import BFC_libs.UV_vis as uv
import BFC_libs.photoelectron as pe
import BFC_libs.XAS as xas
from BFC_libs.quantity_array import QArray


def test_xrd_aliases_and_roundtrip() -> None:
    # エイリアスは本物のクラスで、QArray[Theta] と同一
    assert xrd.ThetaArray is QArray[xrd.Theta]
    theta_array = xrd.ThetaArray([10.0, 20.0, 30.0])
    assert type(theta_array[0]) is xrd.Theta
    pattern = xrd.XRDPattern(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_xrd",
        _two_theta=xrd.ThetaArray([10.0, 20.0, 30.0]),
        _intensity=xrd.DiffractionIntensityArray([100.0, 500.0, 200.0]),
    )
    assert type(pattern.x) is xrd.ThetaArray
    assert type(pattern.y[1]) is xrd.DiffractionIntensity
    # DataFrame 往復
    df = pattern.to_data_frame()
    restored = xrd.XRDPattern.from_data_frame(df)
    assert type(restored.two_theta) is xrd.ThetaArray
    assert math.isclose(float(restored.two_theta[2]), 30.0)
    assert math.isclose(float(restored.intensity[1]), 500.0)


def test_raman_roundtrip() -> None:
    assert rmn.WavenumberArray is QArray[rmn.Wavenumber]
    spectrum = rmn.RamanSpectrum(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_raman",
        _wavenumber=rmn.WavenumberArray([500.0, 1000.0, 1500.0]),
        _intensity=rmn.RammanIntensityArray([1.0, 5.0, 2.0]),
    )
    assert type(spectrum.x[0]) is rmn.Wavenumber
    df = spectrum.to_data_frame()
    restored = rmn.RamanSpectrum.from_data_frame(df)
    assert math.isclose(float(restored.wavenumber[1]), 1000.0)
    assert math.isclose(float(restored.intensity[2]), 2.0)


def test_uv_vis_construction() -> None:
    assert uv.WavelengthArray is QArray[uv.Wavelength]
    spectrum = uv.UV_VisSpectrum(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_uv",
        _wavelength=uv.WavelengthArray([400.0, 500.0, 600.0]),
        _absorption=uv.AbsorptionArray([0.1, 0.8, 0.3]),
    )
    assert type(spectrum.x) is uv.WavelengthArray
    assert type(spectrum.y[1]) is uv.Absorption
    df = spectrum.to_data_frame()
    assert list(df.columns) == ["wavelength", "absorption"]
    assert math.isclose(df["absorption"][1], 0.8)


def test_photoelectron_construction() -> None:
    assert pe.PhotoelectronEnergyArray is QArray[pe.PhotoelectronEnergy]
    # 旧実装では Intensity 配列の要素型が Energy になっていた不整合を、
    # エイリアス化で正しく Intensity に修正済み
    assert pe.PhotoelectronIntensityArray is QArray[pe.PhotoelectronIntensity]
    spectrum = pe.PhotoelectronSpectrum(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_pe",
        _photoelectron_energy=pe.PhotoelectronEnergyArray([280.0, 285.0, 290.0]),
        _photoelectron_intensity=pe.PhotoelectronIntensityArray(
            [10.0, 50.0, 20.0]
        ),
    )
    assert type(spectrum.x[0]) is pe.PhotoelectronEnergy
    df = spectrum.to_data_frame()
    assert math.isclose(df["photoelectron energy"][2], 290.0)
    # 定数も Quantity として生きている
    assert type(pe.HELIUM_UV_ENERGY) is pe.PhotoelectronEnergy
    assert math.isclose(float(pe.HELIUM_UV_ENERGY), 21.22)


def test_xas_aliases() -> None:
    assert xas.DistanceArray is QArray[xas.Distance]
    assert xas.ChiR_MagArray is QArray[xas.ChiR_Mag]
    distance_array = xas.DistanceArray([1.0, 2.0, 3.0])
    assert type(distance_array[1]) is xas.Distance
    magnitude_array = xas.ChiR_MagArray([0.5, 1.5, 0.8])
    assert type(magnitude_array.max()) is xas.ChiR_Mag
    assert math.isclose(float(magnitude_array.max()), 1.5)


if __name__ == "__main__":
    test_functions = [
        obj
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed_names: list[str] = []
    for test_function in test_functions:
        try:
            test_function()
            print("PASS: {}".format(test_function.__name__))
        except AssertionError as error:
            failed_names.append(test_function.__name__)
            print("FAIL: {} ({})".format(test_function.__name__, error))
    print("-" * 40)
    print(
        "{} passed, {} failed / {} total".format(
            len(test_functions) - len(failed_names),
            len(failed_names),
            len(test_functions),
        )
    )
    if failed_names:
        raise SystemExit(1)

"""
electrochemistry モジュールの Quantity/QArray 移行後の動作テスト。

実行方法:
    python test_electrochemistry.py
(パッケージ import が必要なため、親ディレクトリを sys.path に追加している)
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

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
from BFC_libs.quantity_array import QArray


def _assert_raises(expected_error: type[BaseException], func) -> None:
    """func() が expected_error を投げることを検証するヘルパー。"""
    try:
        func()
    except expected_error:
        return
    raise AssertionError(
        "{} が送出されなかった".format(expected_error.__name__)
    )


def test_value_objects() -> None:
    # オーム則: E / I -> Resistance (明示的な特殊化)
    resistance = ec.Potential(1.0) / ec.Current(0.5)
    assert type(resistance) is ec.Resistance
    assert float(resistance) == 2.0
    # 通常の規則も生きている: スカラー除算は型を保ち、同型比は float
    assert type(ec.Potential(1.0) / 2.0) is ec.Potential
    assert type(ec.Potential(1.0) / ec.Potential(2.0)) is float
    # Current のメソッド
    log_current = ec.Current(1e-3).log10()
    assert type(log_current) is ec.LogCurrent
    assert math.isclose(float(log_current), -3.0)
    ir_drop = ec.Current(0.5).iR_correction(ec.Resistance(2.0))
    assert type(ir_drop) is ec.Potential
    assert float(ir_drop) == 1.0
    # Resistance -> Impedance
    impedance = ec.Resistance(3.0).to_impedance()
    assert type(impedance) is ec.Impedance
    assert complex(impedance) == 3 + 0j
    # Impedance の real/imag/abs は Resistance
    z = ec.Impedance(3 + 4j)
    assert type(z.real) is ec.Resistance
    assert type(abs(z)) is ec.Resistance
    assert float(abs(z)) == 5.0


def test_arrays() -> None:
    current_array = ec.CurrentArray([1e-3, 1e-4])
    # 継承クラスなのでスカラー除算後も CurrentArray のまま(mA -> A 変換の形)
    assert type(current_array / 1000) is ec.CurrentArray
    log_array = current_array.log10()
    assert type(log_array) is ec.LogCurrentArray
    assert math.isclose(float(log_array[0]), -3.0)
    ir_array = current_array.iR_correction(ec.Resistance(2.0))
    assert type(ir_array) is ec.PotentialArray
    assert math.isclose(float(ir_array[0]), 2e-3)
    # ResistanceArray -> ImpedanceArray
    resistance_array = ec.ResistanceArray([1.0, 2.0])
    impedance_array = resistance_array.to_impedance()
    assert type(impedance_array) is ec.ImpedanceArray
    assert impedance_array.dtype == np.complex128
    # ImpedanceArray の real は QArray[Resistance](エイリアス同士で一致)
    assert type(impedance_array.real) is QArray[ec.Resistance]


def _make_voltammogram() -> ec.Voltammogram:
    return ec.Voltammogram(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_CV",
        _potential=ec.PotentialArray([0.0, 0.1, 0.2, 0.3]),
        _current=ec.CurrentArray([1e-3, 2e-3, 3e-3, 4e-3]),
        _RE=ec.SHE,
        _time=cmn.TimeArray([0.0, 1.0, 2.0, 3.0]),
        _others_data=pd.DataFrame(),
    )


def test_voltammogram() -> None:
    voltammogram = _make_voltammogram()
    assert type(voltammogram.potential) is ec.PotentialArray
    assert type(voltammogram.x[0]) is ec.Potential
    # iR 補正: E' = E - I*R
    corrected = voltammogram.IR_correction(ec.Resistance(10.0))
    assert type(corrected.potential) is ec.PotentialArray
    assert math.isclose(float(corrected.potential[1]), 0.1 - 2e-3 * 10.0)
    # 元のオブジェクトは変更されない
    assert math.isclose(float(voltammogram.potential[1]), 0.1)
    # 参照電極変換: E' = E - (RE_after - RE_before)
    converted = voltammogram.convert_potential_reference(ec.SHE, ec.SSCE)
    assert math.isclose(float(converted.potential[0]), -0.2)
    assert converted.reference_electrode == ec.SSCE
    # DataFrame 変換
    df = voltammogram.to_data_frame()
    assert list(df.columns[:3]) == ["potential", "current", "time"]
    assert math.isclose(df["current"][3], 4e-3)


def test_eis() -> None:
    eis = ec.EIS(
        _comment=["c"],
        _condition=["cond"],
        _original_file_path="",
        _data_name="test_EIS",
        _real_Z=ec.ResistanceArray([1.0, 2.0, 3.0]),
        _imaginary_Z=ec.ResistanceArray([2.0, 0.5, -1.0]),
        _frequency=ec.FrequencyArray([100.0, 10.0, 1.0]),
        _other_data=pd.DataFrame(),
        _applied_potential=None,
    )
    # |Z| と位相
    abs_z = eis.abs_Z
    assert type(abs_z) is ec.ResistanceArray
    assert math.isclose(float(abs_z[0]), math.sqrt(5.0))
    assert type(eis.phase) is np.ndarray
    # 複素インピーダンス化
    impedance = eis.impedance
    assert type(impedance) is ec.ImpedanceArray
    assert complex(impedance[0]) == 1 + 2j
    # 実軸交点の抵抗(虚部符号反転は index 1->2 の間 -> index 2 の実部)
    solution_resistance = eis.get_resistance()
    assert type(solution_resistance) is ec.Resistance
    assert float(solution_resistance) == 3.0
    # DataFrame 往復
    df = eis.to_data_frame()
    restored = ec.EIS.from_data_frame(df)
    assert len(restored.real_Z) == 3
    assert math.isclose(float(restored.real_Z[2]), 3.0)
    assert math.isclose(float(restored.frequency[0]), 100.0)


def test_chronoamperogram_slice() -> None:
    amperogram = ec.ChronoAmperogram(
        _comment=[],
        _condition=[],
        _original_file_path="",
        _data_name="test_CA",
        _time=cmn.TimeArray([0.0, 1.0, 2.0, 3.0, 4.0]),
        _potential=ec.PotentialArray([0.5, 0.5, 0.5, 0.5, 0.5]),
        _current=ec.CurrentArray([5e-3, 4e-3, 3e-3, 2e-3, 1e-3]),
        _other_data=pd.DataFrame(),
    )
    # DataSeriese.slice が QArray.find 経由で動く(x = time)
    sliced = amperogram.slice(cmn.Time(1.0), cmn.Time(3.0))
    assert type(sliced) is ec.ChronoAmperogram
    assert len(sliced.time) == 2
    assert math.isclose(float(sliced.time[0]), 1.0)
    assert math.isclose(float(sliced.current[1]), 3e-3)
    # iR 補正も動く
    corrected = amperogram.IR_correction(ec.Resistance(1.0))
    assert math.isclose(float(corrected.potential[0]), 0.5 - 5e-3)


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

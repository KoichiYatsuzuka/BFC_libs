"""
QArray(とその特殊化 electrochemistry.FrequencyArray 等)を
matplotlib.pyplot に渡したときの互換性テスト。

背景:
    ax.scatter は内部で np.ma.ravel -> cbook._combine_masks ->
    np.ma.masked_where を通り、マスク配列の生成時に「元配列のクラス
    (_baseclass)で dtype=bool の view」を作る。QArray は __array_finalize__
    で dtype の不変条件を検証するため、ここで ValueError になっていた
    (ax.plot はこの経路を通らないので動いていた)。
    QArray._baseclass = np.ndarray により numpy.ma が素の ndarray を
    基底として扱うようにして解消した。本テストはその回帰テスト。

既知の未対応:
    2 次元 QArray を ax.plot に渡すと、matplotlib が内部で ravel して
    1 次元の view を作る段階で __array_finalize__ の次元検証に掛かる
    (numpy.ma とは別経路。QArray.ravel の実装で対応予定)。

実行方法:
    python test_matplotlib_compat.py
    (pytest があれば `pytest test_matplotlib_compat.py` でも可)
(パッケージ import が必要なため、親ディレクトリを sys.path に追加している)
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import matplotlib

matplotlib.use("Agg")  # 画面を開かずに描画まで通す(pyplot の import より前に指定)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

import BFC_libs.electrochemistry as ec
from BFC_libs.quantity_array import QArray


def _make_dummy_eis_arrays() -> tuple[
    ec.FrequencyArray, ec.ResistanceArray, ec.ResistanceArray
]:
    """
    Bode / Nyquist プロット用のダミーデータを返す。

    返り値
        (frequency_array, real_z_array, imaginary_z_array)
        frequency_array: 1 kHz から 0.1 Hz までの対数等間隔 5 点 [Hz]
        real_z_array, imaginary_z_array: R_s = 10 Ohm, R_ct = 100 Ohm,
            C_dl = 1e-4 F の Randles 回路(Warburg なし)のインピーダンス。
            imaginary_z_array は Nyquist 表示の慣習どおり -Im(Z) を入れる。

    例
        frequency_array, real_z_array, imaginary_z_array = _make_dummy_eis_arrays()
        ax.scatter(real_z_array, imaginary_z_array)
    """
    frequency_values = np.logspace(3, -1, 5)
    angular_frequency_values = 2.0 * np.pi * frequency_values
    solution_resistance = 10.0
    charge_transfer_resistance = 100.0
    double_layer_capacitance = 1e-4
    impedance_values = solution_resistance + charge_transfer_resistance / (
        1.0
        + 1j
        * angular_frequency_values
        * charge_transfer_resistance
        * double_layer_capacitance
    )
    return (
        ec.FrequencyArray(frequency_values),
        ec.ResistanceArray(impedance_values.real),
        ec.ResistanceArray(-impedance_values.imag),
    )


def _draw_and_get_offsets(
    x_array: np.ndarray, y_array: np.ndarray, **scatter_kwargs: object
) -> np.ma.MaskedArray:
    """
    x_array, y_array を pyplot で scatter し、描画まで通したうえで
    点の座標(PathCollection のオフセット, shape (N, 2))をマスク配列で返す。

    引数
        x_array, y_array: X, Y 軸のデータ(QArray でも ndarray でも良い)。
        scatter_kwargs: ax.scatter にそのまま渡す追加引数(c=, s= など)。
    返り値
        全点の座標配列。NaN を含む点は削除されず、行ごとマスクされる
        (素の ndarray を渡したときの matplotlib の挙動と同じ)。

    例
        offsets = _draw_and_get_offsets(frequency_array, real_z_array)
    """
    figure, axes = plt.subplots()
    try:
        collection = axes.scatter(x_array, y_array, **scatter_kwargs)  # type: ignore[arg-type]
        assert isinstance(collection, PathCollection)
        figure.canvas.draw()
        offsets: np.ma.MaskedArray = np.ma.asarray(collection.get_offsets())
    finally:
        plt.close(figure)
    return offsets


def test_plot_accepts_qarray() -> None:
    # 対照実験: ax.plot はマスク処理を通らないので現状でも動くこと
    frequency_array, real_z_array, _ = _make_dummy_eis_arrays()
    figure, axes = plt.subplots()
    try:
        (line,) = axes.plot(frequency_array, real_z_array)
        figure.canvas.draw()
        np.testing.assert_allclose(
            np.asarray(line.get_xdata()), frequency_array.float_array()
        )
    finally:
        plt.close(figure)


def test_scatter_bode_frequency_vs_resistance() -> None:
    # Bode 風: X = FrequencyArray, Y = ResistanceArray
    frequency_array, real_z_array, _ = _make_dummy_eis_arrays()
    offsets = _draw_and_get_offsets(frequency_array, real_z_array)
    np.testing.assert_allclose(offsets[:, 0], frequency_array.float_array())
    np.testing.assert_allclose(offsets[:, 1], real_z_array.float_array())


def test_scatter_nyquist_resistance_vs_resistance() -> None:
    # Nyquist 風: X, Y ともに ResistanceArray、色を FrequencyArray で付ける
    frequency_array, real_z_array, imaginary_z_array = _make_dummy_eis_arrays()
    offsets = _draw_and_get_offsets(
        real_z_array, imaginary_z_array, c=frequency_array
    )
    np.testing.assert_allclose(offsets[:, 0], real_z_array.float_array())
    np.testing.assert_allclose(offsets[:, 1], imaginary_z_array.float_array())


def test_scatter_masks_nan_in_qarray() -> None:
    # 実測データの欠損(NaN)を含む点は、素の ndarray を渡したときと
    # 同じくマスクされること(点は削除されず、行ごとマスクされる)
    frequency_array = ec.FrequencyArray([100.0, np.nan, 1.0])
    real_z_array = ec.ResistanceArray([1.0, 2.0, 3.0])
    offsets_qarray = _draw_and_get_offsets(frequency_array, real_z_array)
    offsets_plain = _draw_and_get_offsets(
        frequency_array.float_array(), real_z_array.float_array()
    )
    expected_mask = [[False, False], [True, True], [False, False]]
    assert np.ma.getmaskarray(offsets_qarray).tolist() == expected_mask
    assert np.ma.getmaskarray(offsets_plain).tolist() == expected_mask
    np.testing.assert_allclose(
        offsets_qarray.compressed(), offsets_plain.compressed()
    )


def test_scatter_complex_y_same_as_plain_ndarray() -> None:
    # complex の QArray も素の complex ndarray と同じ扱い(実部が使われ、
    # matplotlib 側の ComplexWarning が出る)になり、ValueError にならないこと
    _, real_z_array, imaginary_z_array = _make_dummy_eis_arrays()
    impedance_array = ec.ImpedanceArray(
        real_z_array.float_array() + 1j * imaginary_z_array.float_array()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        offsets = _draw_and_get_offsets(real_z_array, impedance_array)
    np.testing.assert_allclose(offsets[:, 1], real_z_array.float_array())


def test_pcolormesh_accepts_2d_qarray() -> None:
    # 2 次元 QArray でも numpy.ma 経由のマスク処理が通ること
    potential_matrix = QArray[QArray[ec.Potential]]([[1.0, 2.0], [3.0, np.nan]])
    figure, axes = plt.subplots()
    try:
        mesh = axes.pcolormesh(potential_matrix)
        figure.canvas.draw()
        mesh_mask = np.ma.getmaskarray(mesh.get_array())
        assert mesh_mask.ravel().tolist() == [False, False, False, True]
    finally:
        plt.close(figure)


def test_masked_where_on_raveled_qarray() -> None:
    # scatter 内部で起きていたことの最小再現:
    # 1. Axes.scatter が np.ma.ravel(x) で MaskedArray を作る
    # 2. cbook.safe_masked_invalid が np.ma.masked_where(~np.isfinite(x), x) を呼ぶ
    # 3. 条件(bool の MaskedArray)を filled する際に
    #    ndarray.view(cond, _baseclass) が dtype=bool で呼ばれる
    #    (_baseclass が QArray[Frequency] だと ValueError になっていた)
    frequency_array = ec.FrequencyArray([100.0, np.nan, 1.0])
    raveled_frequency = np.ma.ravel(frequency_array)
    assert raveled_frequency._baseclass is np.ndarray
    masked_frequency = np.ma.masked_where(
        ~np.isfinite(raveled_frequency), raveled_frequency, copy=False
    )
    assert list(np.ma.getmaskarray(masked_frequency)) == [False, True, False]


def test_masked_where_on_plain_qarray() -> None:
    # ravel を経ない(素の QArray を直接渡す)場合も動くこと。
    # ~np.isfinite の結果は素の bool ndarray に降格される
    frequency_array = ec.FrequencyArray([100.0, np.nan, 1.0])
    masked_frequency = np.ma.masked_where(
        ~np.isfinite(frequency_array), frequency_array, copy=False
    )
    assert list(np.ma.getmaskarray(masked_frequency)) == [False, True, False]


def test_numpy_ma_degrades_qarray_but_keeps_original() -> None:
    # numpy.ma の結果の中身は素の ndarray に降格する
    # (ufunc 以外の numpy API は降格する、という QArray の方針と同じ)。
    # 元の QArray は型・値とも変わらないこと
    frequency_array = ec.FrequencyArray([100.0, np.nan, 1.0])
    masked_frequency = np.ma.masked_invalid(frequency_array)
    assert type(masked_frequency.data) is np.ndarray
    assert type(frequency_array) is ec.FrequencyArray
    assert type(frequency_array[0]) is ec.Frequency
    assert type(frequency_array + frequency_array) is ec.FrequencyArray


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
        except Exception as error:
            failed_names.append(test_function.__name__)
            print(
                "FAIL: {} ({}: {})".format(
                    test_function.__name__, type(error).__name__, error
                )
            )
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

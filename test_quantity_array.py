"""
quantity_array.py (QArray) のテスト。

実行方法:
    python test_quantity_array.py
pytest がインストールされていれば `pytest test_quantity_array.py` でも実行できる。
"""
from __future__ import annotations

import math

import numpy as np

from quantity import ComplexQuantity, Quantity
from quantity_array import QArray


# テスト用の物理量クラス(利用側の宣言方法と同じ形)
class Potential(Quantity):
    pass


class Current(Quantity):
    pass


class Resistance(Quantity):
    pass


class Impedance(ComplexQuantity[Resistance]):
    _real_type = Resistance


class CurrentC(ComplexQuantity):
    # _real_type 未指定の複素物理量
    pass


# 後方互換エイリアスの宣言方法(実行時にも本物のクラス)
PotentialArray = QArray[Potential]
CurrentArray = QArray[Current]
ImpedanceArray = QArray[Impedance]
PotentialMatrix = QArray[QArray[Potential]]


def _assert_raises(expected_error: type[BaseException], func) -> None:
    """func() が expected_error を投げることを検証するヘルパー。"""
    try:
        func()
    except expected_error:
        return
    raise AssertionError(
        "{} が送出されなかった".format(expected_error.__name__)
    )


def test_specialization_identity_and_alias() -> None:
    # メモ化により同じ添字は常に同一クラス(厳密一致の型判定が機能する前提)
    assert QArray[Potential] is QArray[Potential]
    assert QArray[Potential] is not QArray[Current]
    assert PotentialArray is QArray[Potential]
    assert issubclass(PotentialArray, QArray)
    assert PotentialArray.__name__ == "QArray[Potential]"
    p_array = PotentialArray([1.2, 1.5])
    assert isinstance(p_array, PotentialArray)
    assert isinstance(p_array, QArray)
    assert isinstance(p_array, np.ndarray)


def test_construction_and_validation() -> None:
    # list / ndarray / 物理量リストから構築でき、実体は float64
    from_list = PotentialArray([1.0, 2.0])
    assert from_list.dtype == np.float64
    from_quantities = PotentialArray([Potential(1.0), Potential(2.0)])
    assert float(from_quantities[1]) == 2.0
    from_ndarray = PotentialArray(np.array([1, 2, 3]))
    assert len(from_ndarray) == 3
    # 不正な入力は早期にエラー(入力検証なので消してはいけないエラー)
    _assert_raises(TypeError, lambda: QArray([1.0, 2.0]))  # 要素型未指定
    _assert_raises(TypeError, lambda: QArray[int])  # type: ignore[type-var]
    _assert_raises(ValueError, lambda: PotentialArray([[1.0, 2.0]]))  # 次元不一致


def test_construction_from_ndarray() -> None:
    a_1d = np.array([1, 2, 3])
    a_2d = np.array([[1, 2, 3], [4, 5, 6]])
    # 型の入れ子の深さと ndarray の次元数が一致する場合のみ構築できる
    a = QArray[Potential](a_1d)
    assert type(a) is QArray[Potential]
    assert a.dtype == np.float64  # int64 からの変換
    _assert_raises(ValueError, lambda: QArray[Potential](a_2d))
    _assert_raises(ValueError, lambda: QArray[QArray[Potential]](a_1d))
    d = QArray[QArray[Potential]](a_2d)
    assert type(d) is QArray[QArray[Potential]]
    # スカラー(0次元)も次元不一致
    _assert_raises(ValueError, lambda: QArray[Potential](5.0))
    # 変換不能な入力(入力検証なので消してはいけないエラー)
    _assert_raises(ValueError, lambda: QArray[Potential]([[1.0], [2.0, 3.0]]))
    _assert_raises(TypeError, lambda: QArray[Potential]([1 + 2j]))
    # 期待次元の問い合わせ
    assert QArray[Potential].expected_ndim() == 1
    assert QArray[QArray[Potential]].expected_ndim() == 2
    _assert_raises(TypeError, lambda: QArray.expected_ndim())


def test_construction_copies_input() -> None:
    source = np.array([1.0, 2.0, 3.0])
    p_array = PotentialArray(source)
    # コンストラクタは常に防御的コピーを保持する(外部からの変更が波及しない)
    assert not np.shares_memory(p_array, source)
    source[0] = 99.0
    assert float(p_array[0]) == 1.0
    # ゼロコピーが必要な場合は view を使う(不変条件は検証される)
    shared = source.view(PotentialArray)
    assert np.shares_memory(shared, source)


def test_scalar_access_and_slice() -> None:
    p_array = PotentialArray([1.2, 1.5, 1.8])
    # int 添字 -> 葉の物理量型
    assert type(p_array[0]) is Potential
    assert float(p_array[0]) == 1.2
    assert type(p_array[-1]) is Potential
    # slice -> 同じ特殊化クラス(ゼロコピー view)
    sliced = p_array[0:2]
    assert type(sliced) is PotentialArray
    assert np.shares_memory(sliced, p_array)
    # ブールマスク・ファンシーインデックス -> 同じ特殊化クラス(コピー)
    masked = p_array[p_array > 1.3]
    assert type(masked) is PotentialArray
    assert len(masked) == 2
    fancy = p_array[np.array([0, 2])]
    assert type(fancy) is PotentialArray
    assert float(fancy[1]) == 1.8


def test_iteration() -> None:
    p_array = PotentialArray([1.2, 1.5])
    elements = list(p_array)
    assert all(type(element) is Potential for element in elements)
    assert float(elements[1]) == 1.5


def test_2d_nested() -> None:
    matrix = PotentialMatrix([[1.0, 2.0], [3.0, 4.0]])
    assert matrix.dtype == np.float64
    assert matrix.ndim == 2
    # 添字一回で行(1次元特殊化クラス)、二回で葉のスカラー
    row = matrix[0]
    assert type(row) is PotentialArray
    assert np.shares_memory(row, matrix)  # 行はゼロコピー view
    assert type(matrix[0][1]) is Potential
    assert float(matrix[0][1]) == 2.0
    # タプル添字・列アクセスも実行時は正しい型になる
    assert type(matrix[1, 0]) is Potential
    column = matrix[:, 0]
    assert type(column) is PotentialArray
    assert float(column[1]) == 3.0
    # 行スライスは2次元クラスのまま
    assert type(matrix[0:1]) is PotentialMatrix
    # 1次元入力は次元不一致でエラー
    _assert_raises(ValueError, lambda: PotentialMatrix([1.0, 2.0]))


def test_ufunc_add_sub() -> None:
    p1 = PotentialArray([1.0, 2.0])
    p2 = PotentialArray([0.5, 0.5])
    c = CurrentArray([1.0, 1.0])
    added = p1 + p2
    assert type(added) is PotentialArray
    assert float(added[0]) == 1.5
    assert type(p1 - p2) is PotentialArray
    # 同じ葉のスカラーとの加減算も型を保つ(両方向)
    assert type(p1 + Potential(1.0)) is PotentialArray
    assert type(Potential(1.0) + p1) is PotentialArray
    assert type(p1 - Potential(1.0)) is PotentialArray
    # 異なる物理量・素のスカラーとは降格
    assert type(p1 + c) is np.ndarray
    assert type(p1 + 0.5) is np.ndarray
    assert type(0.5 + p1) is np.ndarray


def test_ufunc_scalar_mul_div() -> None:
    p_array = PotentialArray([1.0, 2.0])
    for result in (p_array * 2, 2 * p_array, p_array * 2.0, 2.0 * p_array):
        assert type(result) is PotentialArray
        assert float(result[1]) == 4.0
    # 素の ndarray は「無次元スカラーの配列」として型を保つ
    scaled = p_array * np.array([2.0, 3.0])
    assert type(scaled) is PotentialArray
    assert float(scaled[1]) == 6.0
    halved = p_array / 2.0
    assert type(halved) is PotentialArray
    assert float(halved[0]) == 0.5
    assert type(p_array / np.array([2.0, 4.0])) is PotentialArray


def test_ufunc_degrade() -> None:
    p1 = PotentialArray([1.0, 2.0])
    p2 = PotentialArray([0.5, 0.5])
    c = CurrentArray([1.0, 1.0])
    # 物理量同士の * / は次元が変わるので降格
    assert type(p1 * p2) is np.ndarray
    assert type(p1 / p2) is np.ndarray
    assert type(p1 * c) is np.ndarray
    assert type(p1 / c) is np.ndarray
    # 同じ葉のスカラーとの * / も次元が変わるので降格
    assert type(p1 * Potential(2.0)) is np.ndarray
    assert type(Potential(2.0) * p1) is np.ndarray
    # スカラー / 配列 は次元が反転するので降格
    assert type(2.0 / p1) is np.ndarray
    # float 配列への complex スカラー倍は dtype が変わるため降格
    assert type(p1 * 1j) is np.ndarray
    # 次元が変わる ufunc は降格
    assert type(np.sqrt(p1)) is np.ndarray
    assert type(np.log10(p1)) is np.ndarray
    # 比較は素の bool 配列
    comparison = p1 > p2
    assert type(comparison) is np.ndarray
    assert comparison.dtype == np.bool_


def test_unary_keeps_type() -> None:
    p_array = PotentialArray([1.0, -2.0])
    negated = -p_array
    assert type(negated) is PotentialArray
    assert float(negated[1]) == 2.0
    absolute = abs(p_array)
    assert type(absolute) is PotentialArray
    assert float(absolute[1]) == 2.0


def test_reductions() -> None:
    p_array = PotentialArray([1.0, 2.0, 3.0])
    assert type(p_array.min()) is Potential
    assert float(p_array.min()) == 1.0
    assert type(p_array.max()) is Potential
    assert type(p_array.sum()) is Potential
    assert float(p_array.sum()) == 6.0
    mean_value = p_array.mean()
    assert type(mean_value) is Potential
    assert math.isclose(float(mean_value), 2.0)
    # 次元が変わる集約(prod)は降格
    assert not isinstance(p_array.prod(), Potential)
    # 2次元の axis 集約は対応する次元のクラスになる
    matrix = PotentialMatrix([[1.0, 2.0], [3.0, 4.0]])
    row_sum = matrix.sum(axis=0)
    assert type(row_sum) is PotentialArray
    assert float(row_sum[1]) == 6.0
    assert type(matrix.sum()) is Potential
    assert float(matrix.sum()) == 10.0


def test_numpy_functions_degrade() -> None:
    p_array = PotentialArray([1.0, 2.0])
    # ufunc 以外の numpy API は一律、素の ndarray に降格する
    joined = np.concatenate([p_array, p_array])
    assert type(joined) is np.ndarray
    assert len(joined) == 4
    grid_x, grid_y = np.meshgrid(p_array, p_array)
    assert type(grid_x) is np.ndarray
    assert grid_x.ndim == 2
    # np.sum() は __array_function__ 経由なので降格(arr.sum() は型を保つ)
    assert not isinstance(np.sum(p_array), Potential)


def test_invariant_guard() -> None:
    p_array = PotentialArray([1.0, 2.0, 3.0, 4.0])
    # 次元を変える view 操作は QArray のまま行えない(入力検証エラー)
    _assert_raises(ValueError, lambda: p_array.reshape(2, 2))
    # dtype が違うバッファの view も拒否する
    _assert_raises(
        ValueError,
        lambda: np.array([1.0, 2.0], dtype=np.float32).view(PotentialArray),
    )


def test_complex_array() -> None:
    z_array = ImpedanceArray([3 + 4j, 1 - 2j])
    assert z_array.dtype == np.complex128
    assert type(z_array[0]) is Impedance
    assert complex(z_array[0]) == 3 + 4j
    # 同じ型の加減算・スカラー倍(複素含む)は型を保つ
    assert type(z_array + z_array) is ImpedanceArray
    assert type(z_array * 2.0) is ImpedanceArray
    rotated = z_array * 1j
    assert type(rotated) is ImpedanceArray
    assert complex(rotated[0]) == -4 + 3j
    # 物理量同士の乗除は降格
    assert type(z_array * z_array) is np.ndarray
    # real / imag / abs は対の実数型配列(ゼロコピー view)
    real_part = z_array.real
    assert type(real_part) is QArray[Resistance]
    assert float(real_part[0]) == 3.0
    assert np.shares_memory(real_part, z_array)
    imag_part = z_array.imag
    assert type(imag_part) is QArray[Resistance]
    assert float(imag_part[1]) == -2.0
    magnitude = abs(z_array)
    assert type(magnitude) is QArray[Resistance]
    assert float(magnitude[0]) == 5.0
    # 共役は次元不変なので型を保つ
    conjugated = np.conjugate(z_array)
    assert type(conjugated) is ImpedanceArray
    assert complex(conjugated[0]) == 3 - 4j
    # 集約
    total = z_array.sum()
    assert type(total) is Impedance
    assert complex(total) == 4 + 2j
    # _real_type 未指定の複素物理量は real が素の ndarray に落ちる
    cc_array = QArray[CurrentC]([1j, 2j])
    assert type(cc_array.real) is np.ndarray


def test_float_array_compat() -> None:
    p_array = PotentialArray([1.0, 2.0])
    plain = p_array.float_array()
    assert type(plain) is np.ndarray
    assert plain.dtype == np.float64
    assert not np.shares_memory(plain, p_array)  # コピーであること


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

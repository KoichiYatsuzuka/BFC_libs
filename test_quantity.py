"""
quantity.py (QuantityBase / Quantity) のテスト。

実行方法:
    python test_quantity.py
pytest がインストールされていれば `pytest test_quantity.py` でも実行できる。
"""
from __future__ import annotations

import math

import numpy as np

from quantity import ComplexQuantity, Quantity, QuantityBase


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
    # _real_type 未指定の複素物理量(real/imag/abs は素の float に落ちる)
    pass


def _assert_raises(expected_error: type[BaseException], func) -> None:
    """func() が expected_error を投げることを検証するヘルパー。"""
    try:
        func()
    except expected_error:
        return
    raise AssertionError(
        "{} が送出されなかった".format(expected_error.__name__)
    )


def test_construction() -> None:
    p = Potential(1.2)
    assert type(p) is Potential
    assert float(p) == 1.2
    # int / np.float64 / 同型からの構築
    assert float(Potential(2)) == 2.0
    assert float(Potential(np.float64(2.5))) == 2.5
    assert float(Potential(Potential(0.7))) == 0.7
    # 不正な入力は早期にエラー(入力検証なので消してはいけないエラー)
    _assert_raises(TypeError, lambda: Potential(1 + 2j))  # type: ignore[arg-type]
    _assert_raises(TypeError, lambda: Potential("1.2"))  # type: ignore[arg-type]


def test_value_property() -> None:
    p = Potential(1.2)
    assert type(p.value) is float
    assert p.value == 1.2


def test_add_sub_same_type() -> None:
    p1 = Potential(1.2)
    p2 = Potential(0.3)
    added = p1 + p2
    assert type(added) is Potential
    assert math.isclose(float(added), 1.5)
    subed = p1 - p2
    assert type(subed) is Potential
    assert math.isclose(float(subed), 0.9)


def test_add_sub_degrade() -> None:
    p = Potential(1.2)
    c = Current(0.5)
    # 異なる物理量 -> 素の float
    assert type(p + c) is float
    assert type(p - c) is float
    # 素の float / int -> 素の float
    assert type(p + 0.5) is float
    assert type(0.5 + p) is float
    assert type(p - 1) is float
    # complex -> complex
    assert type(p + 1j) is complex
    assert type(1j - p) is complex


def test_mul_div_scalar_keeps_type() -> None:
    p = Potential(1.2)
    for result in (p * 2, 2 * p, p * 2.0, 2.0 * p, p * np.float64(2.0)):
        assert type(result) is Potential
        assert math.isclose(float(result), 2.4)
    quotient = p / 2.0
    assert type(quotient) is Potential
    assert math.isclose(float(quotient), 0.6)


def test_mul_div_degrade() -> None:
    p1 = Potential(1.2)
    p2 = Potential(0.3)
    c = Current(0.5)
    # 同じ型同士の * / は次元が変わるので float
    assert type(p1 * p2) is float
    assert type(p1 / p2) is float
    assert math.isclose(p1 / p2, 4.0)
    # 異なる物理量 -> float
    assert type(p1 * c) is float
    assert type(p1 / c) is float
    # スカラー / Quantity は次元が反転するので float
    assert type(2.4 / p1) is float
    assert math.isclose(2.4 / p1, 2.0)
    # complex -> complex
    assert type(p1 * 1j) is complex
    assert type(1j * p1) is complex
    assert type(1j / p1) is complex


def test_pow() -> None:
    p = Potential(1.2)
    powered = p**2
    assert type(powered) is float
    assert math.isclose(powered, 1.44)
    assert type(2.0**p) is float


def test_unary_keeps_type() -> None:
    p = Potential(-1.2)
    assert type(-p) is Potential
    assert float(-p) == 1.2
    assert type(+p) is Potential
    assert type(abs(p)) is Potential
    assert float(abs(p)) == 1.2


def test_comparison_same_type_and_primitive() -> None:
    p1 = Potential(1.2)
    p2 = Potential(0.3)
    assert p1 > p2
    assert p2 < p1
    assert p1 >= p2
    assert p2 <= p1
    assert p1 == Potential(1.2)
    assert not (p1 == p2)
    assert p1 != p2
    # プリミティブとの比較は許可
    assert p1 > 1.0
    assert p1 == 1.2
    assert p1 <= 2
    # 数値以外との == は False に落ちる(TypeError にしない標準挙動)
    assert not (p1 == "1.2")


def test_comparison_different_quantity_forbidden() -> None:
    p = Potential(1.2)
    c = Current(1.2)
    # 入力検証としての TypeError(消してはいけないエラー)
    _assert_raises(TypeError, lambda: p < c)  # type: ignore[operator]
    _assert_raises(TypeError, lambda: p >= c)  # type: ignore[operator]
    _assert_raises(TypeError, lambda: p == c)


def test_hash_follows_float() -> None:
    p = Potential(1.2)
    assert hash(p) == hash(1.2)
    assert p in {Potential(1.2)}


def test_isinstance_relations() -> None:
    p = Potential(1.2)
    assert isinstance(p, float)
    assert isinstance(p, Quantity)
    assert isinstance(p, QuantityBase)
    assert not isinstance(1.2, QuantityBase)


def test_numpy_interoperability() -> None:
    p1 = Potential(1.2)
    p2 = Potential(0.3)
    array = np.array([p1, p2])
    # ブランド付き float はそのまま float64 配列になる(object 配列にならない)
    assert array.dtype == np.float64
    assert math.isclose(float(array.sum()), 1.5)
    # ufunc に単体で渡すと numpy スカラーに降格する(次元が変わる演算相当)
    assert isinstance(np.sqrt(Potential(4.0)), np.float64)


def test_complex_construction() -> None:
    z = Impedance(3 + 4j)
    assert type(z) is Impedance
    assert complex(z) == 3 + 4j
    assert float(Impedance(2.5).real) == 2.5
    assert complex(Impedance(np.complex128(1 + 2j))) == 1 + 2j
    _assert_raises(TypeError, lambda: Impedance("1+2j"))  # type: ignore[arg-type]


def test_complex_value_property() -> None:
    z = Impedance(3 + 4j)
    assert type(z.value) is complex
    assert z.value == 3 + 4j


def test_complex_add_sub() -> None:
    z1 = Impedance(3 + 4j)
    z2 = Impedance(1 - 2j)
    added = z1 + z2
    assert type(added) is Impedance
    assert complex(added) == 4 + 2j
    subed = z1 - z2
    assert type(subed) is Impedance
    assert complex(subed) == 2 + 6j
    # 異なる物理量・素のスカラーとの加減算は complex に降格
    assert type(z1 + CurrentC(1j)) is complex
    assert type(z1 + 1j) is complex
    assert type(1j + z1) is complex
    assert type(z1 - 2.0) is complex
    assert type(z1 + Resistance(1.0)) is complex
    assert type(Resistance(1.0) + z1) is complex


def test_complex_scalar_mul_div_keeps_type() -> None:
    z = Impedance(3 + 4j)
    for result in (z * 2, 2 * z, z * 2.0, 2.0 * z):
        assert type(result) is Impedance
        assert complex(result) == 6 + 8j
    # 複素スカラー倍も次元不変なので型を保つ(Quantity と異なる規則)
    rotated = z * 1j
    assert type(rotated) is Impedance
    assert complex(rotated) == -4 + 3j
    assert type(1j * z) is Impedance
    assert type(z * np.complex128(1j)) is Impedance
    halved = z / 2
    assert type(halved) is Impedance
    assert complex(halved) == 1.5 + 2j
    assert type(z / 1j) is Impedance


def test_complex_mul_div_degrade() -> None:
    z1 = Impedance(3 + 4j)
    z2 = Impedance(1 - 2j)
    r = Resistance(2.0)
    # 同じ型同士の * / は次元が変わるので complex
    assert type(z1 * z2) is complex
    assert type(z1 / z1) is complex
    assert abs(z1 / z1 - 1.0) < 1e-12
    # 実数物理量との乗除も complex(Quantity 側の降格分岐の検証を含む)
    assert type(z1 * r) is complex
    assert type(r * z1) is complex
    assert type(z1 / r) is complex
    assert type(r / z1) is complex
    # スカラー / ComplexQuantity は次元が反転するので complex
    assert type(2.0 / z1) is complex
    assert type(1j / z1) is complex


def test_complex_real_imag_abs() -> None:
    z = Impedance(3 + 4j)
    assert type(z.real) is Resistance
    assert float(z.real) == 3.0
    assert type(z.imag) is Resistance
    assert float(z.imag) == 4.0
    assert type(abs(z)) is Resistance
    assert float(abs(z)) == 5.0
    # _real_type 未指定のサブクラスは素の float に落ちる
    cc = CurrentC(3 + 4j)
    assert type(cc.real) is float
    assert type(abs(cc)) is float


def test_complex_conjugate_neg() -> None:
    z = Impedance(3 + 4j)
    conj = z.conjugate()
    assert type(conj) is Impedance
    assert complex(conj) == 3 - 4j
    negated = -z
    assert type(negated) is Impedance
    assert complex(negated) == -3 - 4j
    assert type(+z) is Impedance


def test_complex_pow() -> None:
    z = Impedance(3 + 4j)
    powered = z**2
    assert type(powered) is complex
    assert powered == -7 + 24j


def test_complex_eq_hash() -> None:
    z = Impedance(3 + 4j)
    assert z == Impedance(3 + 4j)
    assert not (z == Impedance(1 - 2j))
    # プリミティブ complex との比較は許可
    assert z == 3 + 4j
    # 異なる物理量同士の比較は禁止(入力検証としての TypeError)
    _assert_raises(TypeError, lambda: z == CurrentC(3 + 4j))
    _assert_raises(TypeError, lambda: z == Resistance(3.0))
    _assert_raises(TypeError, lambda: Resistance(3.0) == z)
    assert hash(z) == hash(3 + 4j)


def test_complex_isinstance_relations() -> None:
    z = Impedance(3 + 4j)
    assert isinstance(z, complex)
    assert isinstance(z, ComplexQuantity)
    assert isinstance(z, QuantityBase)
    assert not isinstance(z, float)
    assert not isinstance(3 + 4j, QuantityBase)


def test_complex_numpy_interoperability() -> None:
    z1 = Impedance(3 + 4j)
    z2 = Impedance(1 - 2j)
    array = np.array([z1, z2])
    assert array.dtype == np.complex128
    assert array.sum() == 4 + 2j


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

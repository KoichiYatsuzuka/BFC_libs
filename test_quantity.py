"""
quantity.py (QuantityBase / Quantity) のテスト。

実行方法:
    python test_quantity.py
pytest がインストールされていれば `pytest test_quantity.py` でも実行できる。
"""
from __future__ import annotations

import math

import numpy as np

from quantity import Quantity, QuantityBase


# テスト用の物理量クラス(利用側の宣言方法と同じ形)
class Potential(Quantity):
    pass


class Current(Quantity):
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

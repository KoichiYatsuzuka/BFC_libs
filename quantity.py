"""
### BFC_libs.quantity

物理量を「ブランド付き float」として表すための基盤モジュール。
旧 common.ValueObjectBase 系の置き換え(段階的移行中)。

設計方針:
- Quantity は float を直接継承する。実体は本物の float なので、
  numpy / matplotlib / pandas へキャストなしで渡せる。
- 型名(Potential, Current など)で物理量を判別する。
- 演算の型伝播規則:
    同じ型同士の + -               -> 同じ型を保つ
    int/float スカラーとの * /     -> 同じ型を保つ(無次元スカラー倍)
    同じ型同士の * /               -> float に落とす(次元が変わるため)
    異なる物理量が絡む演算         -> float に落とす
    complex が絡む演算             -> complex(Python の数値プロトコルに委譲)
    スカラー / Quantity            -> float(次元が反転するため型を保たない)
    ** など次元が変わる演算        -> float(または complex)
    上記以外(// や % など)       -> float の既定動作のまま(素の値に落ちる)
- 比較演算子は「同じ型同士」と「int/float プリミティブとの比較」のみ許可し、
  異なる物理量同士の比較は TypeError を投げる。
- ダンダーメソッドは静的型検査(overload + Self)を効かせるため、
  ループ生成せずすべて明示的に定義する。
"""
from __future__ import annotations

import abc
from typing import Self, SupportsFloat, TypeGuard, overload


class QuantityBase(abc.ABC):
    """
    物理量スカラー(Quantity / ComplexQuantity)の共通基底。

    状態を持たない mixin であり、以下の2つの役割だけを担う:
    1. isinstance(x, QuantityBase) による「何らかの物理量である」ことの一括判定
    2. 演算・比較の型伝播規則で使う判定ヘルパーの共通実装

    float と complex はメモリレイアウトが衝突するため、値の実体を持つ
    共通基底は作れない。具象クラスは Quantity(float 系統)と
    ComplexQuantity(complex 系統、未実装)の2系統に分かれる。

    メンバ一覧:
        _is_plain_scalar(other) -> TypeGuard[float] (staticmethod)
            other が「物理量ではない無次元スカラー(int/float)」かを判定する。
        _is_same_quantity(other) -> TypeGuard[Self]
            other が self と厳密に同じ物理量クラスかを判定する。
    """
    __slots__ = ()

    @staticmethod
    def _is_plain_scalar(other: object) -> TypeGuard[float]:
        """
        other が「物理量ではない無次元スカラー」かどうかを返す。

        引数
            other: 判定対象。演算・比較の右辺(または左辺)に来た値。
        返り値
            True なら int/float 系のスカラー(np.float64 は float のサブクラス
            なのでスカラー扱い)。np.int64 等の numpy 整数は int のサブクラス
            ではないため False(その場合の演算は numpy 側に委譲される)。
        """
        return isinstance(other, (int, float)) and not isinstance(other, QuantityBase)

    def _is_same_quantity(self, other: object) -> TypeGuard[Self]:
        """
        other が self と厳密に同じ物理量クラスかどうかを返す。

        引数
            other: 判定対象。
        返り値
            型が厳密一致(サブクラスも不一致扱い)のときのみ True。
        """
        return type(other) is type(self)


class Quantity(float, QuantityBase):
    """
    物理量を表す float のサブクラス(ブランド付き float)。
    物理量ごとにこのクラスを継承した空クラスを宣言して使う。

    使用例:
        class Potential(Quantity):
            pass

        p1 = Potential(1.2)
        p2 = Potential(0.3)
        p1 + p2          # -> Potential(1.5)   同じ型は型を保つ
        p1 * 2.0         # -> Potential(2.4)   スカラー倍も型を保つ
        p1 / p2          # -> float(4.0)       同型比は無次元
        p1 * p2          # -> float            次元が変わる
        p1 + 0.5         # -> float            素の float との加算は降格
        p1 + 1j          # -> complex          complex に委譲
        2.0 / p1         # -> float            次元が反転する
        p1 < p2          # -> bool             同型・プリミティブとの比較のみ可
        np.array([p1])   # -> dtype=float64 の ndarray

    メンバ一覧:
        value: float (property)
            旧 ValueObject の .value 互換アクセサ。素の float を返す。
        __new__(value: SupportsFloat)
            構築。complex や str は TypeError(不正入力の早期検出)。
        __add__ __radd__ __sub__ __rsub__
            加減算。同じ型なら型を保ち、それ以外は素の値に落とす。
        __mul__ __rmul__ __truediv__
            乗除算。無次元スカラー(int/float)なら型を保ち、
            物理量が絡むと素の値に落とす。
        __rtruediv__
            スカラー / self。次元が反転するため常に素の値。
        __pow__ __rpow__
            冪乗。次元が変わるため素の値(負の底の非整数冪は complex)。
        __neg__ __pos__ __abs__
            符号反転・絶対値。次元不変なので型を保つ。
        __lt__ __le__ __gt__ __ge__ __eq__
            比較。同じ型と int/float のみ許可。異なる物理量は TypeError。
        __hash__
            float と同じ(__eq__ 定義に伴う明示的な再設定)。
    """
    __slots__ = ()

    def __new__(cls, value: SupportsFloat) -> Self:
        """
        物理量インスタンスを構築する。

        引数
            value: float に変換可能な数値(int, float, np.float64 など)。
                   complex は float.__new__ が、str はこのメソッドが拒否する。
        返り値
            cls 型のインスタンス。
        """
        if isinstance(value, str):
            raise TypeError(
                "{} は文字列からは構築できない。数値を渡すこと。".format(cls.__name__)
            )
        return float.__new__(cls, value)

    @property
    def value(self) -> float:
        """旧 ValueObject の .value 互換アクセサ。素の float を返す。"""
        return float(self)

    # ---------------- 加減算: 同じ型のみ型を保つ ----------------

    @overload
    def __add__(self, other: Self) -> Self: ...
    @overload
    def __add__(self, other: Quantity) -> float: ...
    @overload
    def __add__(self, other: float) -> float: ...
    @overload
    def __add__(self, other: complex) -> complex: ...
    def __add__(self, other: complex) -> Self | float | complex:
        if self._is_same_quantity(other):
            return type(self)(float(self) + other)
        return float(self) + other

    @overload
    def __radd__(self, other: float) -> float: ...
    @overload
    def __radd__(self, other: complex) -> complex: ...
    def __radd__(self, other: complex) -> Self | float | complex:
        if self._is_same_quantity(other):
            return type(self)(other + float(self))
        return other + float(self)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: Quantity) -> float: ...
    @overload
    def __sub__(self, other: float) -> float: ...
    @overload
    def __sub__(self, other: complex) -> complex: ...
    def __sub__(self, other: complex) -> Self | float | complex:
        if self._is_same_quantity(other):
            return type(self)(float(self) - other)
        return float(self) - other

    @overload
    def __rsub__(self, other: float) -> float: ...
    @overload
    def __rsub__(self, other: complex) -> complex: ...
    def __rsub__(self, other: complex) -> Self | float | complex:
        if self._is_same_quantity(other):
            return type(self)(other - float(self))
        return other - float(self)

    # -------- 乗除算: 無次元スカラーのみ型を保つ(物理量同士は降格) --------

    @overload
    def __mul__(self, other: Quantity) -> float: ...
    @overload
    def __mul__(self, other: float) -> Self: ...
    @overload
    def __mul__(self, other: complex) -> complex: ...
    def __mul__(self, other: complex) -> Self | float | complex:
        if self._is_plain_scalar(other):
            return type(self)(float(self) * other)
        # 物理量が相手のときは両辺を素の値に落とす。自分側だけ落とすと、
        # 相手側の __rmul__ が「素のスカラー×物理量」と解釈して相手の型に
        # ラップしてしまう(ComplexQuantity 実装時は complex の降格分岐を追加)。
        if isinstance(other, float) and isinstance(other, QuantityBase):
            return float(self) * float(other)
        return float(self) * other

    @overload
    def __rmul__(self, other: Quantity) -> float: ...
    @overload
    def __rmul__(self, other: float) -> Self: ...
    @overload
    def __rmul__(self, other: complex) -> complex: ...
    def __rmul__(self, other: complex) -> Self | float | complex:
        if self._is_plain_scalar(other):
            return type(self)(other * float(self))
        if isinstance(other, float) and isinstance(other, QuantityBase):
            return float(other) * float(self)
        return other * float(self)

    @overload
    def __truediv__(self, other: Quantity) -> float: ...
    @overload
    def __truediv__(self, other: float) -> Self: ...
    @overload
    def __truediv__(self, other: complex) -> complex: ...
    def __truediv__(self, other: complex) -> Self | float | complex:
        if self._is_plain_scalar(other):
            return type(self)(float(self) / other)
        if isinstance(other, float) and isinstance(other, QuantityBase):
            return float(self) / float(other)
        return float(self) / other

    @overload
    def __rtruediv__(self, other: float) -> float: ...
    @overload
    def __rtruediv__(self, other: complex) -> complex: ...
    def __rtruediv__(self, other: complex) -> float | complex:
        # スカラー / self は次元が反転する(例: 1/秒)ため、型は保たない。
        if isinstance(other, float) and isinstance(other, QuantityBase):
            return float(other) / float(self)
        return other / float(self)

    # ---------------- 冪乗: 次元が変わるため常に降格 ----------------

    def __pow__(self, other: float) -> float | complex:
        # 負の底の非整数冪は Python の仕様で complex が返る。
        return float(self) ** other

    def __rpow__(self, other: float) -> float | complex:
        return other ** float(self)

    # ---------------- 単項演算: 次元不変なので型を保つ ----------------

    def __neg__(self) -> Self:
        return type(self)(-float(self))

    def __pos__(self) -> Self:
        return type(self)(float(self))

    def __abs__(self) -> Self:
        return type(self)(abs(float(self)))

    # ------- 比較: 同じ型とプリミティブのみ許可(異なる物理量は禁止) -------

    def _comparison_value(self, other: object) -> float | None:
        """
        比較演算の相手を検証し、比較に使う float 値を返す。

        引数
            other: 比較演算子の相手側の値。
        返り値
            同じ型または int/float プリミティブならその float 値。
            比較を numpy 等に委譲すべき型なら None(呼び出し側は
            NotImplemented を返す)。
        エラー
            異なる物理量クラス同士の比較は TypeError。
        """
        if self._is_same_quantity(other):
            return float(other)
        if isinstance(other, QuantityBase):
            raise TypeError(
                "異なる物理量同士の比較は禁止されている: {} と {}".format(
                    type(self).__name__, type(other).__name__
                )
            )
        if self._is_plain_scalar(other):
            return float(other)
        return None

    def __lt__(self, other: Self | float) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return float(self) < other_value

    def __le__(self, other: Self | float) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return float(self) <= other_value

    def __gt__(self, other: Self | float) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return float(self) > other_value

    def __ge__(self, other: Self | float) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return float(self) >= other_value

    def __eq__(self, other: object) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return float(self) == other_value

    # __eq__ を定義すると __hash__ が None になるため明示的に戻す。
    __hash__ = float.__hash__

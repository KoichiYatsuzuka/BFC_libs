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
- ComplexQuantity(complex 継承)では無次元スカラーに complex も含める
  (複素スカラー倍は次元を変えないため型を保つ)。実部・虚部・絶対値は
  対になる実数型(_real_type)でラップして返す。
- 比較演算子は「同じ型同士」と「int/float プリミティブとの比較」のみ許可し、
  異なる物理量同士の比較は TypeError を投げる。
- ダンダーメソッドは静的型検査(overload + Self)を効かせるため、
  ループ生成せずすべて明示的に定義する。
"""
from __future__ import annotations

import abc
from typing import ClassVar, Generic, Self, SupportsComplex, SupportsFloat
from typing import TypeGuard, TypeVar, cast, overload


class QuantityBase(abc.ABC):
    """
    物理量スカラー(Quantity / ComplexQuantity)の共通基底。

    状態を持たない mixin であり、以下の2つの役割だけを担う:
    1. isinstance(x, QuantityBase) による「何らかの物理量である」ことの一括判定
    2. 演算・比較の型伝播規則で使う判定ヘルパーの共通実装

    float と complex はメモリレイアウトが衝突するため、値の実体を持つ
    共通基底は作れない。具象クラスは Quantity(float 系統)と
    ComplexQuantity(complex 系統)の2系統に分かれる。

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
        # ラップしてしまうため。
        if isinstance(other, QuantityBase):
            if isinstance(other, float):
                return float(self) * float(other)
            return float(self) * complex(other)
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
        if isinstance(other, QuantityBase):
            if isinstance(other, float):
                return float(other) * float(self)
            return complex(other) * float(self)
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
        if isinstance(other, QuantityBase):
            if isinstance(other, float):
                return float(self) / float(other)
            return float(self) / complex(other)
        return float(self) / other

    @overload
    def __rtruediv__(self, other: float) -> float: ...
    @overload
    def __rtruediv__(self, other: complex) -> complex: ...
    def __rtruediv__(self, other: complex) -> float | complex:
        # スカラー / self は次元が反転する(例: 1/秒)ため、型は保たない。
        if isinstance(other, QuantityBase):
            if isinstance(other, float):
                return float(other) / float(self)
            return complex(other) / float(self)
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


RealT = TypeVar("RealT", bound=Quantity)


class ComplexQuantity(complex, QuantityBase, Generic[RealT]):
    """
    複素数の物理量を表す complex のサブクラス(ブランド付き complex)。
    インピーダンスなど、実体が複素数の物理量ごとに継承して使う。
    型引数 RealT には実部・虚部・絶対値が返すべき「対になる実数型」を指定し、
    実行時用に同じ型を ClassVar _real_type にも設定する(二重指定)。

    使用例:
        class Resistance(Quantity):
            pass

        class Impedance(ComplexQuantity[Resistance]):
            _real_type = Resistance

        Z1 = Impedance(3 + 4j)
        Z2 = Impedance(1 - 2j)
        Z1 + Z2          # -> Impedance          同じ型は型を保つ
        Z1 * 2           # -> Impedance          スカラー倍は型を保つ
        Z1 * 1j          # -> Impedance          複素スカラー倍も次元不変
        Z1 * Z2          # -> complex             次元が変わる
        Z1 / Z2          # -> complex             無次元比
        2.0 / Z1         # -> complex             次元が反転する
        Z1.real          # -> Resistance(3.0)
        Z1.imag          # -> Resistance(4.0)
        abs(Z1)          # -> Resistance(5.0)     |Z| は同じ次元の実数
        Z1.conjugate()   # -> Impedance           共役も次元不変
        np.array([Z1])   # -> dtype=complex128 の ndarray

    メンバ一覧:
        _real_type: ClassVar[type[Quantity]]
            実部・虚部・絶対値をラップする実数型。サブクラスで指定する。
            未指定の場合、real/imag/abs は素の float を返す。
        value: complex (property)
            旧 ValueObjectComplex の .value 互換アクセサ。素の complex を返す。
        real, imag: RealT (property)
            実部・虚部を _real_type でラップして返す。
        __new__(value: SupportsComplex | SupportsFloat)
            構築。str は TypeError(不正入力の早期検出)。
        __add__ __radd__ __sub__ __rsub__
            加減算。同じ型なら型を保ち、それ以外は素の complex に落とす。
        __mul__ __rmul__ __truediv__
            乗除算。無次元スカラー(int/float/complex)なら型を保ち、
            物理量が絡むと素の complex に落とす。
        __rtruediv__
            スカラー / self。次元が反転するため常に素の complex。
        __pow__ __rpow__
            冪乗。次元が変わるため素の complex。
        __neg__ __pos__
            符号反転。次元不変なので型を保つ。
        __abs__ -> RealT
            絶対値。次元不変の実数なので _real_type でラップする。
        conjugate() -> Self
            複素共役。次元不変なので型を保つ。
        __eq__
            同じ型と int/float/complex のみ許可。異なる物理量は TypeError。
            complex に順序比較はないため <, <= 等はもともと存在しない。
        __hash__
            complex と同じ(__eq__ 定義に伴う明示的な再設定)。
    """
    __slots__ = ()

    _real_type: ClassVar[type[Quantity]]

    def __new__(cls, value: SupportsComplex | SupportsFloat) -> Self:
        """
        複素物理量インスタンスを構築する。

        引数
            value: complex に変換可能な数値(int, float, complex,
                   np.complex128 など)。str はこのメソッドが拒否する。
        返り値
            cls 型のインスタンス。
        """
        if isinstance(value, str):
            raise TypeError(
                "{} は文字列からは構築できない。数値を渡すこと。".format(cls.__name__)
            )
        return complex.__new__(cls, value)

    @staticmethod
    def _is_plain_complex_scalar(other: object) -> TypeGuard[complex]:
        """
        other が「物理量ではない無次元スカラー(int/float/complex)」かを返す。

        引数
            other: 判定対象。演算の相手側の値。
        返り値
            True ならスカラー扱い。複素スカラー倍は次元を変えないため、
            ComplexQuantity では complex もスカラーに含まれる
            (float 実体の Quantity とは規則が異なる点に注意)。
        """
        return isinstance(other, (int, float, complex)) and not isinstance(
            other, QuantityBase
        )

    @property
    def value(self) -> complex:
        """旧 ValueObjectComplex の .value 互換アクセサ。素の complex を返す。"""
        return complex(self)

    def _wrap_real(self, value: float) -> RealT:
        """
        実部・虚部・絶対値を対になる実数型(_real_type)でラップして返す。

        引数
            value: ラップする素の float 値。
        返り値
            _real_type のインスタンス。_real_type 未指定のサブクラスでは
            素の float を返す(そのとき静的注釈 RealT は近似になる)。
        """
        real_type: type[Quantity] | None = getattr(type(self), "_real_type", None)
        if real_type is None:
            return cast("RealT", float(value))
        return cast("RealT", real_type(value))

    @property
    def real(self) -> RealT:
        return self._wrap_real(complex(self).real)

    @property
    def imag(self) -> RealT:
        return self._wrap_real(complex(self).imag)

    # ---------------- 加減算: 同じ型のみ型を保つ ----------------

    @overload
    def __add__(self, other: Self) -> Self: ...
    @overload
    def __add__(self, other: complex) -> complex: ...
    def __add__(self, other: complex) -> Self | complex:
        if self._is_same_quantity(other):
            return type(self)(complex(self) + other)
        return complex(self) + other

    def __radd__(self, other: complex) -> complex:
        return other + complex(self)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: complex) -> complex: ...
    def __sub__(self, other: complex) -> Self | complex:
        if self._is_same_quantity(other):
            return type(self)(complex(self) - other)
        return complex(self) - other

    def __rsub__(self, other: complex) -> complex:
        return other - complex(self)

    # -------- 乗除算: 無次元スカラーのみ型を保つ(物理量同士は降格) --------

    @overload
    def __mul__(self, other: ComplexQuantity) -> complex: ...
    @overload
    def __mul__(self, other: Quantity) -> complex: ...
    @overload
    def __mul__(self, other: complex) -> Self: ...
    def __mul__(self, other: complex) -> Self | complex:
        if self._is_plain_complex_scalar(other):
            return type(self)(complex(self) * other)
        # 物理量が相手のときは両辺を素の値に落とす(Quantity 側と同じ理由)。
        if isinstance(other, QuantityBase):
            return complex(self) * complex(other)
        return complex(self) * other

    @overload
    def __rmul__(self, other: ComplexQuantity) -> complex: ...
    @overload
    def __rmul__(self, other: Quantity) -> complex: ...
    @overload
    def __rmul__(self, other: complex) -> Self: ...
    def __rmul__(self, other: complex) -> Self | complex:
        if self._is_plain_complex_scalar(other):
            return type(self)(other * complex(self))
        if isinstance(other, QuantityBase):
            return complex(other) * complex(self)
        return other * complex(self)

    @overload
    def __truediv__(self, other: ComplexQuantity) -> complex: ...
    @overload
    def __truediv__(self, other: Quantity) -> complex: ...
    @overload
    def __truediv__(self, other: complex) -> Self: ...
    def __truediv__(self, other: complex) -> Self | complex:
        if self._is_plain_complex_scalar(other):
            return type(self)(complex(self) / other)
        if isinstance(other, QuantityBase):
            return complex(self) / complex(other)
        return complex(self) / other

    def __rtruediv__(self, other: complex) -> complex:
        # スカラー / self は次元が反転するため、型は保たない。
        if isinstance(other, QuantityBase):
            return complex(other) / complex(self)
        return other / complex(self)

    # ---------------- 冪乗: 次元が変わるため常に降格 ----------------

    def __pow__(self, other: complex) -> complex:
        return complex(self) ** other

    def __rpow__(self, other: complex) -> complex:
        return other ** complex(self)

    # ---------------- 単項演算: 次元不変なので型を保つ ----------------

    def __neg__(self) -> Self:
        return type(self)(-complex(self))

    def __pos__(self) -> Self:
        return type(self)(complex(self))

    def __abs__(self) -> RealT:
        # |Z| は同じ次元を持つ実数なので、対になる実数型でラップする。
        return self._wrap_real(abs(complex(self)))

    def conjugate(self) -> Self:
        """複素共役を返す。次元不変なので型を保つ(EIS の -Z'' 処理などで使用)。"""
        return type(self)(complex(self).conjugate())

    # ------- 等価比較: 同じ型とプリミティブのみ許可(異なる物理量は禁止) -------

    def _comparison_value(self, other: object) -> complex | None:
        """
        比較演算の相手を検証し、比較に使う complex 値を返す。

        引数
            other: 比較演算子の相手側の値。
        返り値
            同じ型または int/float/complex プリミティブならその complex 値。
            比較を委譲すべき型なら None(呼び出し側は NotImplemented を返す)。
        エラー
            異なる物理量クラス同士の比較は TypeError。
        """
        if self._is_same_quantity(other):
            return complex(other)
        if isinstance(other, QuantityBase):
            raise TypeError(
                "異なる物理量同士の比較は禁止されている: {} と {}".format(
                    type(self).__name__, type(other).__name__
                )
            )
        if self._is_plain_complex_scalar(other):
            return complex(other)
        return None

    def __eq__(self, other: object) -> bool:
        other_value = self._comparison_value(other)
        if other_value is None:
            return NotImplemented
        return complex(self) == other_value

    # __eq__ を定義すると __hash__ が None になるため明示的に戻す。
    __hash__ = complex.__hash__

"""
### BFC_libs.quantity_array

物理量配列 QArray の実装。旧 common.ValueObjectArray の置き換え(段階的移行中)。

設計方針:
- 実体は平坦な dtype=float64(葉が ComplexQuantity なら complex128)の
  ndarray サブクラス。ベクトル演算は常に C 速度で走る。
- `QArray[Potential]` と添字した瞬間に、__class_getitem__ が本物の
  特殊化クラスを動的生成・メモ化して返す(C++ テンプレート方式)。
  `PotentialArray = QArray[Potential]` のようなエイリアスは実行時にも
  本物のクラスなので、isinstance / view / 継承すべてに使える。
- 多次元は「入れ子の深さ ≡ ndim」で表現する:
  `QArray[QArray[Potential]]` は 2 次元。実体は常に 1 個の連続バッファで、
  行アクセス(m[0])はゼロコピーの view。
- 要素は境界(__getitem__ / __iter__)で葉の物理量型にラップして返す。
- ufunc(__array_ufunc__)の型伝播規則はスカラー(quantity.py)と相似形:
    同じ型同士(+ 同型スカラー)の加減算       -> 型を保つ
    無次元スカラー・素の ndarray との * /      -> 型を保つ(除算は被除数側のみ)
    物理量同士の * / 、その他の ufunc          -> 素の ndarray に降格
    negative / positive / conjugate            -> 型を保つ(次元不変)
    absolute                                   -> float 系は型を保ち、
                                                  complex 系は対の実数型配列
    reduce は add / minimum / maximum のみ型を保つ(sum, min, max, mean が対応)
- ufunc 以外の numpy API(np.meshgrid, np.concatenate 等)は
  __array_function__ で一律、素の ndarray に降格する。
- view / reshape 等で不変条件(次元・dtype)が壊れる操作は
  __array_finalize__ が検出して ValueError を投げる(入力検証)。
- 動的生成クラスはディスクへの pickle 不可(copy.deepcopy は可)。
  必要になった時点で __reduce__ を追加する。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Generic, Self, SupportsFloat, SupportsIndex
from typing import TypeVar, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from .quantity import ComplexQuantity, Quantity, QuantityBase
except ImportError:
    # パッケージ外(リポジトリ直下のスクリプト等)から使う場合
    from quantity import ComplexQuantity, Quantity, QuantityBase


ElemT = TypeVar("ElemT", bound="Quantity | ComplexQuantity | QArray[Any]")
LeafT = TypeVar("LeafT", bound=Quantity)


class QArray(np.ndarray, Generic[ElemT]):
    """
    物理量の配列。実体は平坦な float64 / complex128 の ndarray。

    使用例:
        class Potential(Quantity):
            pass

        PotentialArray = QArray[Potential]        # 実行時にも本物のクラス

        p_array = PotentialArray([1.2, 1.5, 1.8])
        p_array[0]           # -> Potential(1.2)   スカラーは葉の型でラップ
        p_array[0:2]         # -> QArray[Potential] (ゼロコピー view)
        p_array + p_array    # -> QArray[Potential] 同じ型は型を保つ
        p_array * 2.0        # -> QArray[Potential] スカラー倍も型を保つ
        p_array * p_array    # -> 素の ndarray      次元が変わる
        np.sqrt(p_array)     # -> 素の ndarray      次元が変わる
        p_array.min()        # -> Potential         集約ホワイトリスト
        np.meshgrid(...)     # -> 素の ndarray      ufunc 以外は一律降格

        matrix = QArray[QArray[Potential]]([[1.0, 2.0], [3.0, 4.0]])
        matrix[0]            # -> QArray[Potential] (行のゼロコピー view)
        matrix[0][1]         # -> Potential(2.0)

    型引数:
        ElemT: 添字一回で取り出される要素の型。葉の物理量
               (Quantity / ComplexQuantity のサブクラス)か、
               入れ子の QArray 特殊化クラス。

    メンバ一覧(クラス属性は特殊化クラス生成時に自動設定される):
        _element_type: ClassVar[type]
            添字一回で取り出される要素の型。
        _leaf_type: ClassVar[type]
            入れ子を葉まで辿った物理量型。スカラーのラップに使う。
        _expected_ndim: ClassVar[int]
            この特殊化クラスの次元数(= 入れ子の深さ)。構築時に検証される。
        _dtype: ClassVar[np.dtype]
            実体の dtype。葉が ComplexQuantity なら complex128、他は float64。
        _specializations: ClassVar[dict]
            特殊化クラスのメモ化キャッシュ。`QArray[X] is QArray[X]` を保証する。
        real, imag: property
            実部・虚部。葉が ComplexQuantity のとき、対の実数型
            (_real_type)の QArray を返す(未指定なら素の ndarray)。
        float_array(), complex_array()
            旧 ValueObjectArray 互換。素の ndarray コピーを返す。
        __getitem__ / __iter__
            結果の次元数に応じて葉の型・対応する特殊化クラスにラップする。
        __array_ufunc__ / __array_function__
            型伝播規則の実装(モジュール docstring 参照)。
    """

    _element_type: ClassVar[type]
    _leaf_type: ClassVar[type]
    _expected_ndim: ClassVar[int]
    _dtype: ClassVar["np.dtype[Any]"]
    _specializations: ClassVar[dict[tuple[type, type], type["QArray[Any]"]]] = {}

    # ---------------- 特殊化クラスの生成(テンプレート機構) ----------------

    def __class_getitem__(cls, item: object) -> object:
        """
        `QArray[Potential]` の形で本物の特殊化クラスを生成・メモ化して返す。

        引数
            item: 要素型。Quantity / ComplexQuantity のサブクラス、または
                  QArray の特殊化クラス(入れ子 = 多次元)。
                  TypeVar 等の「型でない添字」はジェネリック関数の注釈用に
                  通常の静的型検査用エイリアスへフォールバックする。
        返り値
            特殊化された本物のクラス(同じ item には常に同一のクラス)。
        エラー
            物理量型でも QArray でもない型を渡すと TypeError。
        """
        if not isinstance(item, type):
            # TypeVar / Any など: 静的型検査用の通常のエイリアスに委ねる
            return super().__class_getitem__(item)  # type: ignore[misc]

        if issubclass(item, QArray):
            if not hasattr(item, "_element_type"):
                raise TypeError(
                    "特殊化されていない QArray は要素型に使えない。"
                    "QArray[QArray[Potential]] のように葉まで指定すること。"
                )
            leaf_type: type = item._leaf_type
            ndim = item._expected_ndim + 1
        elif issubclass(item, (Quantity, ComplexQuantity)):
            leaf_type = item
            ndim = 1
        else:
            raise TypeError(
                "QArray の要素型は Quantity / ComplexQuantity のサブクラス"
                "または QArray の特殊化クラスに限る: {}".format(item)
            )

        memo_key = (cls, item)
        if memo_key not in QArray._specializations:
            if issubclass(leaf_type, ComplexQuantity):
                dtype = np.dtype(np.complex128)
            else:
                dtype = np.dtype(np.float64)
            specialized = type(
                "{}[{}]".format(cls.__name__, item.__name__),
                (cls,),
                {
                    "_element_type": item,
                    "_leaf_type": leaf_type,
                    "_expected_ndim": ndim,
                    "_dtype": dtype,
                },
            )
            QArray._specializations[memo_key] = cast(
                "type[QArray[Any]]", specialized
            )
        return QArray._specializations[memo_key]

    # ---------------- 構築と不変条件 ----------------

    def __new__(cls, values: ArrayLike) -> Self:
        """
        物理量配列を構築する。

        引数
            values: 配列に変換可能な値(list, ndarray, 物理量のリスト等)。
                    次元数はクラスの期待次元(入れ子の深さ)と一致すること。
                    常に防御的コピーを保持する(入力配列とバッファを共有
                    しない)。ゼロコピーが必要な場合は
                    ndarray.view(QArray[X]) を使う(不変条件は
                    __array_finalize__ が検証する)。
        返り値
            cls 型のインスタンス(実体は _dtype の ndarray)。
        エラー
            要素型未指定(裸の QArray)は TypeError。次元数不一致は ValueError。
            変換不能な入力は numpy 由来のエラーをそのまま通す
            (不揃いなリスト・文字列は ValueError、complex -> float64 の
            ような型として不可能な変換は TypeError)。
        """
        if not hasattr(cls, "_element_type"):
            raise TypeError(
                "QArray は要素型を指定して使う(例: QArray[Potential]([...]))"
            )
        array_data = np.array(values, dtype=cls._dtype)
        if array_data.ndim != cls._expected_ndim:
            raise ValueError(
                "{} は {} 次元の配列を要求するが、{} 次元の入力が渡された".format(
                    cls.__name__, cls._expected_ndim, array_data.ndim
                )
            )
        return array_data.view(cls)

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        """
        view / copy 経由で生成された個体の不変条件(次元・dtype)を検証する。
        reshape 等で次元が変わる操作は QArray のまま行えない
        (素の ndarray に落としてから行うこと)。
        """
        if obj is None:
            # ndarray.__new__ 直接呼び出し(deepcopy の内部など)。
            # この後 __setstate__ 等で正しい状態が入るため検証しない。
            return
        cls = type(self)
        if not hasattr(cls, "_element_type"):
            raise TypeError("特殊化されていない QArray の view は作れない")
        if self.ndim != cls._expected_ndim or self.dtype != cls._dtype:
            raise ValueError(
                "{} の不変条件に反する view: ndim={} (期待 {}), dtype={} (期待 {})".format(
                    cls.__name__, self.ndim, cls._expected_ndim,
                    self.dtype, cls._dtype,
                )
            )

    # ---------------- 次元と特殊化クラスの対応 ----------------

    @classmethod
    def expected_ndim(cls) -> int:
        """
        このクラスが受け付ける配列の次元数(= 型の入れ子の深さ)を返す。

        例
            QArray[Potential].expected_ndim()          # -> 1
            QArray[QArray[Potential]].expected_ndim()  # -> 2

        入れ子の再帰的な解決は __class_getitem__ が特殊化クラスの生成時に
        済ませている(内側のクラスの値 + 1)ため、ここでは定数を返すだけ。
        エラー
            特殊化されていない裸の QArray に対して呼ぶと TypeError。
        """
        if not hasattr(cls, "_element_type"):
            raise TypeError("特殊化されていない QArray に次元数はない")
        return cls._expected_ndim

    @classmethod
    def _class_for_ndim(cls, ndim: int) -> type[np.ndarray]:
        """
        結果の次元数に対応する特殊化クラスを返す。

        引数
            ndim: 演算・添字の結果の次元数。
        返り値
            ndim 段の入れ子に対応するクラス。newaxis 等で次元が増えた場合は
            表現できないため素の np.ndarray を返す(降格)。
        """
        if ndim < 1 or ndim > cls._expected_ndim:
            return np.ndarray
        current: type[QArray[Any]] = cls
        for _ in range(cls._expected_ndim - ndim):
            current = cast("type[QArray[Any]]", current._element_type)
        return current

    @classmethod
    def _with_leaf(cls, new_leaf: type[Quantity]) -> type[QArray[Any]]:
        """
        入れ子構造(次元数)を保ったまま、葉の型だけ差し替えた
        特殊化クラスを返す(例: 2次元 Impedance 配列 -> 2次元 Resistance 配列)。
        """
        element = cls._element_type
        if isinstance(element, type) and issubclass(element, QArray):
            inner = cast("type[QArray[Any]]", element)._with_leaf(new_leaf)
            return cast("type[QArray[Any]]", QArray.__class_getitem__(inner))
        return cast("type[QArray[Any]]", QArray.__class_getitem__(new_leaf))

    @classmethod
    def _paired_real_class(cls) -> type[np.ndarray]:
        """
        real / imag / absolute の結果を包むべきクラスを返す。
        葉が ComplexQuantity なら対の実数型(_real_type)の配列クラス
        (_real_type 未指定なら素の ndarray)、実数の葉なら自分自身。
        """
        leaf = cls._leaf_type
        if issubclass(leaf, ComplexQuantity):
            real_type: type[Quantity] | None = getattr(leaf, "_real_type", None)
            if real_type is None:
                return np.ndarray
            return cls._with_leaf(real_type)
        return cls

    @staticmethod
    def _view_result(raw: object, target: type) -> object:
        """
        raw が target(QArray 特殊化クラス)の不変条件を満たすときだけ
        view して返し、満たさなければ素の値のまま返す(降格)。
        例えば float 配列に complex スカラーを掛けた結果は dtype が
        complex128 になるため、ここで自動的に素の配列へ降格される。
        """
        if not isinstance(raw, np.ndarray):
            return raw
        if not (isinstance(target, type) and issubclass(target, QArray)):
            return raw
        if not hasattr(target, "_element_type"):
            return raw
        if raw.ndim != target._expected_ndim or raw.dtype != target._dtype:
            return raw
        return raw.view(target)

    # ---------------- 要素アクセス: 境界でのラップ ----------------

    @overload
    def __getitem__(self, key: SupportsIndex) -> ElemT: ...
    @overload
    def __getitem__(
        self,
        key: (
            slice
            | NDArray[np.bool_]
            | NDArray[np.integer[Any]]
            | list[int]
            | list[bool]
        ),
    ) -> Self: ...
    def __getitem__(self, key: object) -> ElemT | Self:
        """
        添字アクセス。結果の次元数で返す型が決まる:
        スカラーまで落ちたら葉の物理量型、配列なら対応する特殊化クラスの view。
        タプル添字(m[i, j])も実行時は正しく動くが静的には型付けされない
        (m[i][j] の連鎖を推奨)。ブールマスクを2次元以上に使うと結果は
        1次元になる(numpy の仕様)ため、静的型は Self だが実行時は
        1次元クラスが返る点に注意。
        """
        raw = self.view(np.ndarray)[key]  # type: ignore[index]
        if not isinstance(raw, np.ndarray):
            return cast("ElemT", self._leaf_type(raw))
        if raw.ndim == 0:
            return cast("ElemT", self._leaf_type(raw.item()))
        return cast("Self", raw.view(self._class_for_ndim(raw.ndim)))

    def __iter__(self) -> Iterator[ElemT]:
        for index in range(len(self)):
            yield self[index]

    # ---------------- ufunc の型伝播 ----------------

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        """
        ufunc の実行と結果の型決定。入力の QArray を素の ndarray に降格して
        C 速度で計算した後、モジュール docstring の規則で型を付け直す。
        """
        degraded_inputs = tuple(
            x.view(np.ndarray) if isinstance(x, QArray) else x for x in inputs
        )
        out = kwargs.get("out")
        if isinstance(out, tuple):
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, QArray) else x for x in out
            )
        raw = getattr(ufunc, method)(*degraded_inputs, **kwargs)
        if raw is NotImplemented:
            return NotImplemented

        cls = type(self)

        if method == "reduce":
            # 次元を保つ集約(sum / min / max、mean は sum 経由)のみ型を保つ
            if ufunc in (np.add, np.minimum, np.maximum):
                if not isinstance(raw, np.ndarray):
                    return cls._leaf_type(raw)
                return self._view_result(raw, cls._class_for_ndim(raw.ndim))
            return raw

        if method != "__call__":
            # accumulate / outer / at 等は降格
            return raw

        if ufunc in (np.negative, np.positive, np.conjugate):
            # 次元不変の単項演算
            return self._view_result(raw, cls)

        if ufunc is np.absolute:
            # |Z| は次元不変の実数。complex 系は対の実数型配列になる。
            return self._view_result(raw, cls._paired_real_class())

        if ufunc in (np.add, np.subtract):
            # 同じ型の配列・同じ葉のスカラーだけで構成されていれば型を保つ
            if all(
                type(x) is cls or type(x) is cls._leaf_type for x in inputs
            ):
                return self._view_result(raw, cls)
            return raw

        if ufunc is np.multiply:
            # 無次元スカラー・素の ndarray との積のみ型を保つ
            branded = [
                x for x in inputs if isinstance(x, (QuantityBase, QArray))
            ]
            if len(branded) == 1 and type(branded[0]) is cls:
                return self._view_result(raw, cls)
            return raw

        if ufunc in (np.true_divide, np.divide):
            # 被除数が自分の型で、除数が無次元のときのみ型を保つ
            # (スカラー / 配列 は次元が反転するため常に降格)
            if (
                len(inputs) == 2
                and type(inputs[0]) is cls
                and not isinstance(inputs[1], (QuantityBase, QArray))
            ):
                return self._view_result(raw, cls)
            return raw

        # その他の ufunc(sqrt, exp, 比較演算など)は素の値に降格
        return raw

    # ---------------- ufunc 以外の numpy API: 一括降格 ----------------

    def __array_function__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """
        np.meshgrid や np.concatenate 等の numpy 関数は一律、素の ndarray に
        降格して実行する。view 経由で不変条件が壊れた個体が生まれるのを防ぐ。
        型を保ちたい関数が出てきたら、ここでホワイトリスト分岐を足す。
        """
        degraded_args = cast(tuple, _degrade_nested(args))
        degraded_kwargs = cast(dict, _degrade_nested(kwargs))
        return cast(Any, func)(*degraded_args, **degraded_kwargs)

    # ---------------- 実部・虚部と互換メソッド ----------------

    @property
    def real(self: QArray[ComplexQuantity[LeafT]]) -> QArray[LeafT]:
        """
        実部。葉が ComplexQuantity のとき対の実数型配列を返す
        (_real_type 未指定なら素の ndarray)。ゼロコピーの view。
        """
        raw = self.view(np.ndarray).real
        return cast(
            "QArray[LeafT]",
            QArray._view_result(raw, type(self)._paired_real_class()),
        )

    @property
    def imag(self: QArray[ComplexQuantity[LeafT]]) -> QArray[LeafT]:
        """虚部。real と同じ規則でラップする。"""
        raw = self.view(np.ndarray).imag
        return cast(
            "QArray[LeafT]",
            QArray._view_result(raw, type(self)._paired_real_class()),
        )

    def float_array(self) -> NDArray[np.float64]:
        """旧 ValueObjectArray 互換。素の float64 ndarray のコピーを返す。"""
        return np.array(self.view(np.ndarray), dtype=np.float64)

    def complex_array(self) -> NDArray[np.complex128]:
        """旧 ValueObjectArray 互換。素の complex128 ndarray のコピーを返す。"""
        return np.array(self.view(np.ndarray), dtype=np.complex128)

    # ------- データ系列向けユーティリティ(旧 ValueObjectArray 互換) -------

    def find(
        self,
        target_value: SupportsFloat,
        begin_index: int = 0,
        end_index: int | None = None,
    ) -> NDArray[np.intp]:
        """
        系列が target_value と交差する位置のインデックス配列を返す
        (隣接要素間で符号が反転する箇所と、値が完全一致する箇所)。
        DataSeriese.slice() などが x 軸上の値から添字を引くために使う。

        引数
            target_value: 探す値。葉の物理量でも素の float でも良い。
            begin_index, end_index: 探索範囲 [begin, end)。省略時は全範囲。
        返り値
            交差点の絶対インデックス(配列全体基準)の昇順配列。
            見つからなければ空配列。
        エラー
            範囲指定が不正なら IndexError、float 系列以外(complex・多次元)
            は TypeError(いずれも入力検証)。

        例
            PotentialArray([0.0, 1.0, 2.0, 1.0]).find(1.5)  # -> [1, 2]
        """
        if self.ndim != 1 or self.dtype != np.float64:
            raise TypeError(
                "find は 1 次元の float 系列専用: {} (ndim={})".format(
                    type(self).__name__, self.ndim
                )
            )
        _end_index = len(self) if end_index is None else end_index
        if begin_index < 0 or begin_index > _end_index or _end_index > len(self):
            raise IndexError(
                "不正な探索範囲: begin={}, end={}, length={}".format(
                    begin_index, _end_index, len(self)
                )
            )
        raw = self.view(np.ndarray)[begin_index:_end_index]
        difference = raw - float(target_value)
        # 符号が反転する箇所(左側要素のインデックス)と完全一致の箇所
        crossing_indexes = np.where(difference[1:] * difference[:-1] < 0.0)[0]
        exact_indexes = np.where(difference == 0.0)[0]
        merged_indexes = np.sort(np.append(crossing_indexes, exact_indexes))
        return merged_indexes + begin_index

    def normalize(
        self,
        begin_index: int | None = None,
        end_index: int | None = None,
    ) -> Self:
        """
        範囲 [begin, end) 内の最小値・最大値を基準に、配列全体を 0-1 に
        規格化した新しい配列を返す。自身は変更しない。

        引数
            begin_index, end_index: 基準となる最小・最大を探す範囲。
                                    省略時は全範囲。
        返り値
            規格化された配列。値は無次元になるが、旧 ValueObjectArray との
            互換のため型は保つ(規格化後も find やプロットを型付きで
            続行できるようにする意図的な例外)。
        """
        _begin_index = 0 if begin_index is None else begin_index
        _end_index = len(self) if end_index is None else end_index
        raw = self.view(np.ndarray)
        reference_window = raw[_begin_index:_end_index]
        min_value = reference_window.min()
        max_value = reference_window.max()
        return type(self)((raw - min_value) / (max_value - min_value))

    def join(self, another: Self) -> Self:
        """
        同じ型の配列を末尾に連結した新しい配列を返す。自身は変更しない。
        (2 次元では行方向に連結する)

        エラー
            型が違えば TypeError(入力検証)。
        """
        if type(another) is not type(self):
            raise TypeError(
                "join は同じ型同士に限る: {} と {}".format(
                    type(self).__name__, type(another).__name__
                )
            )
        joined = np.concatenate(
            [self.view(np.ndarray), another.view(np.ndarray)]
        )
        return type(self)(joined)

    def __and__(self, another: Self) -> Self:  # type: ignore[override]
        """旧 ValueObjectArray 互換の連結演算子(join と同じ)。"""
        return self.join(another)


def _degrade_nested(value: object) -> object:
    """
    引数の入れ子構造(tuple / list / dict)を辿り、QArray を素の ndarray に
    降格したコピーを返す。__array_function__ の引数処理に使う
    (np.concatenate([a, b]) のように配列がリストに入って渡るため再帰が必要)。
    """
    if isinstance(value, QArray):
        return value.view(np.ndarray)
    if isinstance(value, tuple):
        return tuple(_degrade_nested(item) for item in value)
    if isinstance(value, list):
        return [_degrade_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _degrade_nested(item) for key, item in value.items()}
    return value

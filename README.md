# BFC_libs

Libraries for the Biofunctional Catalyst research team (Nakamura Lab.)

化学測定データを **値オブジェクト(物理量)** として直観的かつ型安全に扱うための
Python ライブラリです。電位・電流・インピーダンスといった物理量を「ブランド付きの
数値」として表現し、それらの配列（`QArray`）・データ系列（`DataSeriese`）・
データファイル（`DataFile`）へと積み上げていくことで、測定データの読み込みから
処理・プロットまでを一貫した型の上で扱えるようにします。

**言語 / Language:** [日本語](#日本語) ・ [English](#english)

> このドキュメントは、値オブジェクト（`Quantity` / `ComplexQuantity`）、
> 物理量配列 `QArray`、そのエイリアス、および `DataSeriese` / `DataArray` /
> `DataFile` を中心とした機能群を対象としています。

---

## 日本語

### 目次

- [基本理念](#基本理念)
- [データ階層](#データ階層)
- [動作環境とインストール](#動作環境とインストール)
- [物理量スカラー: Quantity / ComplexQuantity](#物理量スカラー-quantity--complexquantity)
- [物理量配列: QArray](#物理量配列-qarray)
- [エイリアスとカスタムメソッドクラス](#エイリアスとカスタムメソッドクラス)
- [データ系列とファイル: DataSeriese / DataArray / DataFile](#データ系列とファイル-dataseriese--dataarray--datafile)
- [すぐ使える物理量（electrochemistry モジュール）](#すぐ使える物理量electrochemistry-モジュール)
- [旧 ValueObject 系に関する注記](#旧-valueobject-系に関する注記)
- [テスト](#テスト)

### 基本理念

- **直観的なデータの読み込み・処理**（`pandas.read_csv()` とカラム処理の自動化を土台にする）
- **プリミティブ型を極力使わない** — 値は物理量オブジェクトで表す（`float` ではなく `Potential`）
- **値の書き換えを許可しない** — 主要オブジェクトは不変（frozen）として扱う
- **型ヒントに `Any` を出さない** — 静的型検査で意味のある型を維持する
- 上記を満たしたうえで、**`numpy` によりベクトル演算を C 速度で走らせる**

### データ階層

物理量は次のように積み上がります。下ほど粒度が小さく、上ほど大きな単位です。

```
DataFile                         1 ファイル分のデータ（測定条件・コメント込み）
  └─ DataArray[T]                データ系列の配列（T は DataSeriese のサブクラス）
       └─ DataSeriese            1 本のデータ系列（ボルタモグラム, スペクトル 等）
            ├─ QArray  (x 系列)  物理量の配列（例: PotentialArray）
            └─ QArray  (y 系列)  物理量の配列（例: CurrentArray）
                 └─ Quantity / ComplexQuantity   物理量スカラー（例: Potential）
```

- **`Quantity` / `ComplexQuantity`**（`quantity.py`）… 値オブジェクトの最小単位。
  実体は `float` / `complex` を継承した「ブランド付き数値」。
- **`QArray`**（`quantity_array.py`）… 物理量の配列。実体は平坦な
  `float64` / `complex128` の `numpy.ndarray` サブクラス。
- **`DataSeriese` / `DataArray` / `DataFile`**（`common.py`）… 1 本の系列・
  系列の配列・1 ファイル分のデータを表す上位クラス。

### 動作環境とインストール

- **Python 3.12+**（`match` 構文と `typing.Self` を使用）
- 依存: `numpy`, `pandas`, `matplotlib`, `scipy`
- 一部モジュールは将来 Cython/Rust への移植を予定（`.pyx` のビルド定義あり）

現状は pip パッケージ化されていないため、リポジトリを取得してインポートパスに置いて使います。

```python
# パッケージとしてインポートする場合
from BFC_libs import common as cmn
from BFC_libs import electrochemistry as ec
from BFC_libs.quantity import Quantity, ComplexQuantity
from BFC_libs.quantity_array import QArray
```

`quantity.py` / `quantity_array.py` / `common.py` はリポジトリ直下からの直接実行にも
対応しています（相対 import 失敗時に絶対 import へフォールバックする）。

### 物理量スカラー: Quantity / ComplexQuantity

#### Quantity — ブランド付き float

`Quantity` は `float` を直接継承したクラスです。物理量ごとに、これを継承した
**空のクラス**を宣言して使います。実体は本物の `float` なので、`numpy` /
`matplotlib` / `pandas` へキャストなしで渡せます。

```python
from BFC_libs.quantity import Quantity

class Potential(Quantity):   # 電位 [V]
    pass

p1 = Potential(1.2)
p2 = Potential(0.3)

p1 + p2          # -> Potential(1.5)   同じ型同士は型を保つ
p1 * 2.0         # -> Potential(2.4)   無次元スカラー倍も型を保つ
p1 / p2          # -> float(4.0)       同型同士の比は無次元
p1 * p2          # -> float            次元が変わる
p1 + 0.5         # -> float            素の float との加算は降格
2.0 / p1         # -> float            次元が反転する
p1 < p2          # -> bool             同型・プリミティブとの比較のみ可
float(p1)        # -> 1.2              素の float
p1.value         # -> 1.2              旧 ValueObject 互換アクセサ
```

演算の型伝播規則（`Quantity`）:

| 演算 | 相手 | 結果 |
|------|------|------|
| `+` `-` | 同じ型 | 同じ型を保つ |
| `+` `-` | 異なる物理量 / 素の `float` | `float` に降格 |
| `+` `-` | `complex` | `complex` に降格 |
| `*` `/` | `int` / `float`（無次元スカラー） | 同じ型を保つ |
| `*` `/` | 同じ型・異なる物理量 | `float`（次元が変わる） |
| `scalar / Quantity` | — | `float`（次元が反転する） |
| `**` | — | `float`（負の底の非整数冪は `complex`） |
| `-x` `+x` `abs(x)`（単項） | — | 同じ型を保つ |
| `<` `<=` `>` `>=` `==` | 同じ型 / `int` / `float` | `bool` |
| `<` `<=` `>` `>=` `==` | 異なる物理量 | **`TypeError`** |

> **設計意図:** 「同じ物理量同士の加減算は同じ物理量」「無次元スカラー倍は次元を
> 変えない」といった、次元解析として正しい規則だけを型に反映しています。異なる
> 物理量の比較を `TypeError` にするのは、単位取り違えを実行時に早期検出するためです。

#### ComplexQuantity — ブランド付き complex

インピーダンスのように実体が複素数の物理量には `ComplexQuantity`（`complex`
継承）を使います。型引数と `_real_type` に、実部・虚部・絶対値が返すべき
「対になる実数型」を指定します（静的型と実行時の二重指定）。

```python
from BFC_libs.quantity import Quantity, ComplexQuantity

class Resistance(Quantity):                      # 抵抗 [Ohm]
    pass

class Impedance(ComplexQuantity[Resistance]):    # 複素インピーダンス [Ohm]
    _real_type = Resistance

Z1 = Impedance(3 + 4j)
Z2 = Impedance(1 - 2j)

Z1 + Z2          # -> Impedance          同じ型は型を保つ
Z1 * 2           # -> Impedance          スカラー倍は型を保つ
Z1 * 1j          # -> Impedance          複素スカラー倍も次元不変
Z1 * Z2          # -> complex            次元が変わる
Z1.real          # -> Resistance(3.0)    対の実数型でラップ
Z1.imag          # -> Resistance(4.0)
abs(Z1)          # -> Resistance(5.0)    |Z| は同じ次元の実数
Z1.conjugate()   # -> Impedance          共役も次元不変（EIS の -Z'' 処理などで使用）
```

`Quantity` との違いは、**複素スカラー（`complex`）も無次元スカラー扱い**になる点です
（複素スカラー倍は次元を変えないため）。順序比較（`<` など）は `complex` に存在しない
ため定義されません。

いずれのダンダーも静的型検査（overload + `Self`）が効くよう明示的に定義されており、
型の厳密一致（`type(other) is type(self)`）で判定します。

### 物理量配列: QArray

`QArray` は物理量の配列です。実体は平坦な `float64`（葉が `ComplexQuantity` なら
`complex128`）の `numpy.ndarray` サブクラスで、ベクトル演算は常に C 速度で走ります。

#### テンプレート機構

`QArray[Potential]` と添字した瞬間に、`__class_getitem__` が **本物の特殊化クラス**を
動的生成・メモ化して返します（C++ テンプレートに近い方式）。同じ添字は常に同一クラスに
なるため、`isinstance` / view / 継承のすべてに使えます。

```python
from BFC_libs.quantity_array import QArray

PotentialArray = QArray[Potential]        # 実行時にも本物のクラス

assert QArray[Potential] is QArray[Potential]     # メモ化により常に同一
assert PotentialArray is QArray[Potential]
assert issubclass(PotentialArray, QArray)

p_array = PotentialArray([1.2, 1.5, 1.8])
p_array[0]           # -> Potential(1.2)          スカラーは葉の型でラップ
p_array[0:2]         # -> QArray[Potential]        ゼロコピー view
p_array + p_array    # -> QArray[Potential]        同じ型は型を保つ
p_array * 2.0        # -> QArray[Potential]        スカラー倍も型を保つ
p_array * p_array    # -> 素の ndarray             次元が変わる
np.sqrt(p_array)     # -> 素の ndarray             次元が変わる
p_array.min()        # -> Potential                集約は型を保つ
```

裸の `QArray([...])`（要素型を指定しない）は `TypeError` です。

#### 多次元 — 入れ子の深さ = 次元数

多次元配列は型の**入れ子の深さ**で表現します。`QArray[QArray[Potential]]` は 2 次元。
実体は常に 1 個の連続バッファで、行アクセスはゼロコピーの view です。

```python
matrix = QArray[QArray[Potential]]([[1.0, 2.0], [3.0, 4.0]])
matrix[0]            # -> QArray[Potential]   行のゼロコピー view
matrix[0][1]         # -> Potential(2.0)
matrix.expected_ndim()   # -> 2
```

> タプル添字 `matrix[i, j]` も実行時は正しく動きますが、静的型としては不完全です。
> 型付きで扱うなら `matrix[i][j]` の連鎖を推奨します。

#### 型伝播規則（QArray）

スカラー（`Quantity`）と相似形です。演算は一度素の `ndarray` に降格して C 速度で
計算した後、規則に従って型を付け直します。

| 演算 | 型を保つ条件 |
|------|--------------|
| `+` `-` | 同じ型の配列・同じ葉のスカラーだけで構成されるとき |
| `*` | 無次元スカラー・素の `ndarray` との積のとき |
| `/` | 被除数が自分の型で、除数が無次元のとき（`scalar / 配列` は降格） |
| `negative` / `positive` / `conjugate` | 常に（次元不変） |
| `absolute` | `float` 系は型を保ち、`complex` 系は対の実数型配列になる |
| 集約 (`sum` / `min` / `max` / `mean`) | `add` / `minimum` / `maximum` の reduce のみ |
| その他の ufunc (`sqrt`, `exp`, 比較 等) | 保たない（素の `ndarray` に降格） |
| ufunc 以外の numpy API (`np.meshgrid`, `np.concatenate` 等) | 一律降格 |

#### ユーティリティメソッド

`QArray` はデータ系列処理向けのメソッドを備えます（いずれも旧 `ValueObjectArray` 互換）。

```python
p = PotentialArray([0.0, 1.0, 2.0, 1.0])

p.find(1.5)          # -> [1, 2]   値と交差する位置のインデックス配列
p.normalize()        # -> 0-1 に規格化した新しい配列（自身は不変）
p.join(other)        # -> 同じ型の配列を末尾に連結（p & other も同義）
p.float_array()      # -> 素の float64 ndarray のコピー
p.complex_array()    # -> 素の complex128 ndarray のコピー
# 複素配列では .real / .imag が対の実数型配列を返す
```

### エイリアスとカスタムメソッドクラス

物理量配列の宣言には **2 通り**あり、用途で使い分けます。

```python
# (1) エイリアス: メモ化により QArray[Potential] と同一クラス
PotentialArray = QArray[Potential]

# (2) 継承: カスタムメソッドを持たせたいとき
class CurrentArray(QArray[Current]):
    def log10(self) -> "LogCurrentArray":
        return LogCurrentArray(np.log10(np.abs(self.float_array())))
```

> **注意:** エイリアス `QArray[Current]` と 継承クラス `class CurrentArray(QArray[Current])`
> は **別のクラス**です。`isinstance` や厳密な型一致で不整合が起きないよう、
> **物理量ごとにどちらか一方へ統一**してください（カスタムメソッドが必要なら継承、
> 不要ならエイリアス）。

### データ系列とファイル: DataSeriese / DataArray / DataFile

上位 3 クラスは `common.py` に定義された、測定データの構造を表す抽象基底です。

#### DataSeriese — 1 本のデータ系列

ボルタモグラムやスペクトルなど、1 本の測定系列を表す **frozen dataclass** の抽象基底。
継承時に x / y 系列の型を指定し（`DataSeriese[PotentialArray, CurrentArray]`）、
`x` / `y` / `to_data_frame` / `from_data_frame` をオーバーライドして使います。

- **メンバ**: `_comment`, `_condition`, `_original_file_path`, `_data_name`
  （それぞれ読み取り専用プロパティ経由でアクセス）
- **提供メソッド**: `slice(x_min, x_max)`（**インデックスではなく x の値**で切り出し）、
  `plot(fig, ax)`（簡易プロット、`fig`/`ax` は再利用可）、`to_csv(path)`

```python
from dataclasses import dataclass
from BFC_libs import common as cmn
from BFC_libs import electrochemistry as ec

@dataclass(frozen=True, repr=False)
class Voltammogram(cmn.DataSeriese[ec.PotentialArray, ec.CurrentArray]):
    _potential: ec.PotentialArray
    _current: ec.CurrentArray
    # ... x, y, to_data_frame, from_data_frame をオーバーライド ...
```

#### DataArray — データ系列の配列

`DataSeriese` サブクラスのインスタンスを要素に持つ配列（`numpy.ndarray` サブクラス、
`dtype=object`）。Python の仕様上、型を **2 度**指定する必要があります。

```python
# CV のリストから配列を作る例
data_list = cmn.DataArray[ec.Voltammogram](cv_list, ec.Voltammogram)

data_list[0]                    # -> Voltammogram
for cv in data_list: ...        # 要素は Voltammogram
data_list.map(some_function)    # 各要素へ関数を適用し、新しい DataArray を返す
data_list.join(another)         # 同じ型の DataArray を連結
```

#### DataFile — 1 ファイル分のデータ

1 つの測定ファイルに対応する **frozen dataclass** の抽象基底。`DataArray[T]` を内包し、
測定条件・コメントを保持します。継承時に `@dataclass(frozen=True)` を付けて使います。

- **メンバ**: `_data`（`DataArray[T]`）, `_comment`, `_condition`, `_file_path`, `_data_name`

```python
@dataclass(frozen=True)
class BioLogicVoltammogramData(cmn.DataFile[ec.Voltammogram]):
    _data: cmn.DataArray[ec.Voltammogram]
    # ...
```

### すぐ使える物理量（electrochemistry モジュール）

`electrochemistry` モジュールには、電気化学測定で使う物理量とその配列が定義済みです。
自分で `Quantity` を継承しなくても、そのまま使えます。

| 物理量 | 種別 | 配列 | 備考 |
|--------|------|------|------|
| `Potential` [V] | `Quantity` | `PotentialArray` | `/ Current` でオームの法則 → `Resistance` |
| `Current` [A] | `Quantity` | `CurrentArray` | `log10()`, `iR_correction()` を持つ |
| `Resistance` [Ohm] | `Quantity` | `ResistanceArray` | `to_impedance()` を持つ |
| `Impedance` [Ohm] | `ComplexQuantity[Resistance]` | `ImpedanceArray` | `real`/`imag`/`abs` は `Resistance` |
| `Frequency` [Hz] | `Quantity` | `FrequencyArray` | |
| `LogCurrent` | `Quantity` | `LogCurrentArray` | log10(電流)、無次元だがブランドを保持 |
| `Time` [s]（`common`） | `Quantity` | `TimeArray` | |

参照電極の変換ヘルパー（`SHE`, `NHE`, `SSCE`, `RHE(pH)`）も用意されています。

### 旧 ValueObject 系に関する注記

`common.py` の `ValueObject` / `ValueObjectComplex` / `ValueObjectArray` は
**旧実装**であり、現在は上記の `Quantity` / `ComplexQuantity` / `QArray` へ
段階的に移行中です。後方互換のため当面は残されますが、**新規コードでは
`Quantity` / `QArray` 系を使ってください**。`Quantity.value` / `QArray.float_array()`
などの互換アクセサは、旧 API からの移行を容易にするために用意されています。

### テスト

物理量・物理量配列にはテストが付属します。

```bash
python test_quantity.py          # Quantity / ComplexQuantity
python test_quantity_array.py    # QArray
python test_electrochemistry.py  # electrochemistry モジュール
# pytest が入っていれば pytest <file> でも実行可
```

---

## English

### Table of Contents

- [Philosophy](#philosophy)
- [Data hierarchy](#data-hierarchy)
- [Requirements & installation](#requirements--installation)
- [Scalar quantities: Quantity / ComplexQuantity](#scalar-quantities-quantity--complexquantity)
- [Quantity arrays: QArray](#quantity-arrays-qarray)
- [Aliases vs. custom-method subclasses](#aliases-vs-custom-method-subclasses)
- [Data series & files: DataSeriese / DataArray / DataFile](#data-series--files-dataseriese--dataarray--datafile)
- [Ready-made quantities (electrochemistry module)](#ready-made-quantities-electrochemistry-module)
- [A note on the legacy ValueObject family](#a-note-on-the-legacy-valueobject-family)
- [Tests](#tests)

### Philosophy

- **Intuitive data loading and processing** (built on `pandas.read_csv()` with automated column handling)
- **Avoid primitive types** — values are represented by quantity objects (`Potential`, not `float`)
- **Values are not editable** — the main objects are treated as immutable (frozen)
- **No `Any` in type hints** — keep meaningful types under static type checking
- On top of the above, **run vectorized math at C speed via `numpy`**

### Data hierarchy

Quantities stack up as follows — finer-grained at the bottom, larger units at the top.

```
DataFile                         Data of a single file (with conditions & comments)
  └─ DataArray[T]                Array of data series (T is a DataSeriese subclass)
       └─ DataSeriese            A single data series (voltammogram, spectrum, ...)
            ├─ QArray  (x axis)  Array of a quantity (e.g. PotentialArray)
            └─ QArray  (y axis)  Array of a quantity (e.g. CurrentArray)
                 └─ Quantity / ComplexQuantity   Scalar quantity (e.g. Potential)
```

- **`Quantity` / `ComplexQuantity`** (`quantity.py`) — the atomic value object;
  a "branded number" subclassing `float` / `complex`.
- **`QArray`** (`quantity_array.py`) — an array of quantities; internally a flat
  `float64` / `complex128` subclass of `numpy.ndarray`.
- **`DataSeriese` / `DataArray` / `DataFile`** (`common.py`) — a single series,
  an array of series, and one file's worth of data, respectively.

### Requirements & installation

- **Python 3.12+** (uses the `match` statement and `typing.Self`)
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`
- Partial migration to Cython/Rust is planned for some modules (`.pyx` build defs exist)

There is no pip package yet, so clone the repository and put it on your import path.

```python
# Importing as a package
from BFC_libs import common as cmn
from BFC_libs import electrochemistry as ec
from BFC_libs.quantity import Quantity, ComplexQuantity
from BFC_libs.quantity_array import QArray
```

`quantity.py` / `quantity_array.py` / `common.py` also work when run directly from
the repository root (they fall back to absolute imports if relative imports fail).

### Scalar quantities: Quantity / ComplexQuantity

#### Quantity — a branded float

`Quantity` subclasses `float` directly. For each physical quantity you declare an
**empty subclass**. Because an instance is a genuine `float`, it can be passed to
`numpy` / `matplotlib` / `pandas` without any cast.

```python
from BFC_libs.quantity import Quantity

class Potential(Quantity):   # potential [V]
    pass

p1 = Potential(1.2)
p2 = Potential(0.3)

p1 + p2          # -> Potential(1.5)   same type is preserved
p1 * 2.0         # -> Potential(2.4)   dimensionless scalar keeps the type
p1 / p2          # -> float(4.0)       ratio of the same type is dimensionless
p1 * p2          # -> float            dimension changes
p1 + 0.5         # -> float            adding a bare float degrades
2.0 / p1         # -> float            dimension inverts
p1 < p2          # -> bool             only same type / primitive comparisons
float(p1)        # -> 1.2              the raw float
p1.value         # -> 1.2              legacy ValueObject-compatible accessor
```

Type-propagation rules (`Quantity`):

| Operation | Other operand | Result |
|-----------|---------------|--------|
| `+` `-` | same type | keeps the type |
| `+` `-` | different quantity / bare `float` | degrades to `float` |
| `+` `-` | `complex` | degrades to `complex` |
| `*` `/` | `int` / `float` (dimensionless scalar) | keeps the type |
| `*` `/` | same or different quantity | `float` (dimension changes) |
| `scalar / Quantity` | — | `float` (dimension inverts) |
| `**` | — | `float` (`complex` for non-integer powers of a negative base) |
| `-x` `+x` `abs(x)` (unary) | — | keeps the type |
| `<` `<=` `>` `>=` `==` | same type / `int` / `float` | `bool` |
| `<` `<=` `>` `>=` `==` | different quantity | **`TypeError`** |

> **Design intent:** only rules that are correct under dimensional analysis are
> reflected in the type — "sum of the same quantity is that quantity", "a
> dimensionless scalar multiple keeps the dimension", etc. Comparing different
> quantities raises `TypeError` so that unit mix-ups are caught early at runtime.

#### ComplexQuantity — a branded complex

For quantities that are inherently complex (e.g. impedance), use `ComplexQuantity`
(subclassing `complex`). Specify, both as a type argument and via `_real_type`, the
"paired real type" that `.real` / `.imag` / `abs()` should return.

```python
from BFC_libs.quantity import Quantity, ComplexQuantity

class Resistance(Quantity):                      # resistance [Ohm]
    pass

class Impedance(ComplexQuantity[Resistance]):    # complex impedance [Ohm]
    _real_type = Resistance

Z1 = Impedance(3 + 4j)
Z2 = Impedance(1 - 2j)

Z1 + Z2          # -> Impedance          same type is preserved
Z1 * 2           # -> Impedance          scalar multiple keeps the type
Z1 * 1j          # -> Impedance          a complex scalar multiple is dimension-preserving
Z1 * Z2          # -> complex            dimension changes
Z1.real          # -> Resistance(3.0)    wrapped in the paired real type
Z1.imag          # -> Resistance(4.0)
abs(Z1)          # -> Resistance(5.0)    |Z| is a real of the same dimension
Z1.conjugate()   # -> Impedance          conjugate is dimension-preserving (used for -Z'' in EIS)
```

The difference from `Quantity` is that **a `complex` scalar also counts as a
dimensionless scalar** (a complex scalar multiple does not change the dimension).
Ordering comparisons (`<`, etc.) do not exist because `complex` has none.

All dunder methods are defined explicitly (with `overload` + `Self`) so that static
type checking works, and they dispatch on strict type identity (`type(other) is type(self)`).

### Quantity arrays: QArray

`QArray` is an array of quantities. Internally it is a flat `float64` (or
`complex128` when the leaf is a `ComplexQuantity`) subclass of `numpy.ndarray`, so
vector operations always run at C speed.

#### Template mechanism

The moment you subscript `QArray[Potential]`, `__class_getitem__` generates and
memoizes a **real specialized class** (much like a C++ template). The same subscript
always yields the same class, so it works for `isinstance`, views, and inheritance.

```python
from BFC_libs.quantity_array import QArray

PotentialArray = QArray[Potential]        # a real class at runtime too

assert QArray[Potential] is QArray[Potential]     # always identical (memoized)
assert PotentialArray is QArray[Potential]
assert issubclass(PotentialArray, QArray)

p_array = PotentialArray([1.2, 1.5, 1.8])
p_array[0]           # -> Potential(1.2)          scalar is wrapped in the leaf type
p_array[0:2]         # -> QArray[Potential]        zero-copy view
p_array + p_array    # -> QArray[Potential]        same type is preserved
p_array * 2.0        # -> QArray[Potential]        scalar multiple keeps the type
p_array * p_array    # -> raw ndarray              dimension changes
np.sqrt(p_array)     # -> raw ndarray              dimension changes
p_array.min()        # -> Potential                aggregation keeps the type
```

A bare `QArray([...])` (with no element type) raises `TypeError`.

#### Multidimensional — nesting depth = number of dimensions

Multidimensional arrays are expressed by the **nesting depth** of the type.
`QArray[QArray[Potential]]` is 2-D. The backing store is always one contiguous
buffer, and row access is a zero-copy view.

```python
matrix = QArray[QArray[Potential]]([[1.0, 2.0], [3.0, 4.0]])
matrix[0]            # -> QArray[Potential]   zero-copy view of the row
matrix[0][1]         # -> Potential(2.0)
matrix.expected_ndim()   # -> 2
```

> Tuple indexing `matrix[i, j]` also works at runtime but is incompletely typed
> statically. For typed access, prefer chaining `matrix[i][j]`.

#### Type-propagation rules (QArray)

These mirror the scalar rules. An operation first degrades inputs to raw `ndarray`
and computes at C speed, then re-attaches the type according to the rules.

| Operation | Keeps the type when |
|-----------|---------------------|
| `+` `-` | composed only of same-type arrays and same-leaf scalars |
| `*` | multiplied by a dimensionless scalar or a raw `ndarray` |
| `/` | dividend is the array's own type and the divisor is dimensionless (`scalar / array` degrades) |
| `negative` / `positive` / `conjugate` | always (dimension-preserving) |
| `absolute` | `float` leaves keep the type; `complex` leaves become the paired real-type array |
| aggregation (`sum` / `min` / `max` / `mean`) | only `add` / `minimum` / `maximum` reduce |
| other ufuncs (`sqrt`, `exp`, comparisons, ...) | not kept (degrades to raw `ndarray`) |
| non-ufunc numpy APIs (`np.meshgrid`, `np.concatenate`, ...) | uniformly degraded |

#### Utility methods

`QArray` ships with methods for data-series processing (all compatible with the
legacy `ValueObjectArray`).

```python
p = PotentialArray([0.0, 1.0, 2.0, 1.0])

p.find(1.5)          # -> [1, 2]   indices where the series crosses the value
p.normalize()        # -> a new array normalized to 0-1 (self is unchanged)
p.join(other)        # -> concatenates a same-type array (p & other is equivalent)
p.float_array()      # -> a copy as a raw float64 ndarray
p.complex_array()    # -> a copy as a raw complex128 ndarray
# For complex arrays, .real / .imag return the paired real-type array
```

### Aliases vs. custom-method subclasses

There are **two ways** to declare a quantity array; choose by intent.

```python
# (1) Alias: identical class to QArray[Potential] thanks to memoization
PotentialArray = QArray[Potential]

# (2) Inheritance: when you want to attach custom methods
class CurrentArray(QArray[Current]):
    def log10(self) -> "LogCurrentArray":
        return LogCurrentArray(np.log10(np.abs(self.float_array())))
```

> **Caution:** the alias `QArray[Current]` and the subclass
> `class CurrentArray(QArray[Current])` are **different classes**. To avoid
> mismatches under `isinstance` or strict type identity, **standardize on one of
> them per quantity** (inheritance if you need custom methods, an alias otherwise).

### Data series & files: DataSeriese / DataArray / DataFile

These three higher-level classes, defined in `common.py`, are abstract bases that
describe the structure of measured data.

#### DataSeriese — a single data series

A **frozen-dataclass** abstract base for one measured series such as a voltammogram
or spectrum. When subclassing, specify the x / y series types
(`DataSeriese[PotentialArray, CurrentArray]`) and override `x` / `y` /
`to_data_frame` / `from_data_frame`.

- **Members**: `_comment`, `_condition`, `_original_file_path`, `_data_name`
  (each exposed via a read-only property)
- **Provided methods**: `slice(x_min, x_max)` (slices **by x value, not index**),
  `plot(fig, ax)` (quick plot; `fig`/`ax` are reusable), `to_csv(path)`

```python
from dataclasses import dataclass
from BFC_libs import common as cmn
from BFC_libs import electrochemistry as ec

@dataclass(frozen=True, repr=False)
class Voltammogram(cmn.DataSeriese[ec.PotentialArray, ec.CurrentArray]):
    _potential: ec.PotentialArray
    _current: ec.CurrentArray
    # ... override x, y, to_data_frame, from_data_frame ...
```

#### DataArray — an array of data series

An array whose elements are instances of a `DataSeriese` subclass (a
`numpy.ndarray` subclass with `dtype=object`). Because of Python's limitations, the
type must be specified **twice**.

```python
# Building an array from a list of CVs
data_list = cmn.DataArray[ec.Voltammogram](cv_list, ec.Voltammogram)

data_list[0]                    # -> Voltammogram
for cv in data_list: ...        # elements are Voltammogram
data_list.map(some_function)    # applies a function to each element, returns a new DataArray
data_list.join(another)         # concatenates a same-type DataArray
```

#### DataFile — one file's worth of data

A **frozen-dataclass** abstract base corresponding to a single measurement file. It
holds a `DataArray[T]` and keeps the measurement conditions and comments. Add
`@dataclass(frozen=True)` when subclassing.

- **Members**: `_data` (`DataArray[T]`), `_comment`, `_condition`, `_file_path`, `_data_name`

```python
@dataclass(frozen=True)
class BioLogicVoltammogramData(cmn.DataFile[ec.Voltammogram]):
    _data: cmn.DataArray[ec.Voltammogram]
    # ...
```

### Ready-made quantities (electrochemistry module)

The `electrochemistry` module defines the quantities and arrays used in
electrochemical measurements, ready to use without subclassing `Quantity` yourself.

| Quantity | Kind | Array | Notes |
|----------|------|-------|-------|
| `Potential` [V] | `Quantity` | `PotentialArray` | `/ Current` gives Ohm's law → `Resistance` |
| `Current` [A] | `Quantity` | `CurrentArray` | has `log10()`, `iR_correction()` |
| `Resistance` [Ohm] | `Quantity` | `ResistanceArray` | has `to_impedance()` |
| `Impedance` [Ohm] | `ComplexQuantity[Resistance]` | `ImpedanceArray` | `real`/`imag`/`abs` are `Resistance` |
| `Frequency` [Hz] | `Quantity` | `FrequencyArray` | |
| `LogCurrent` | `Quantity` | `LogCurrentArray` | log10(current); dimensionless but branded |
| `Time` [s] (in `common`) | `Quantity` | `TimeArray` | |

Reference-electrode conversion helpers (`SHE`, `NHE`, `SSCE`, `RHE(pH)`) are also provided.

### A note on the legacy ValueObject family

`ValueObject` / `ValueObjectComplex` / `ValueObjectArray` in `common.py` are the
**legacy implementation** and are being migrated, stepwise, to the `Quantity` /
`ComplexQuantity` / `QArray` classes described above. They remain for backward
compatibility for now, but **new code should use the `Quantity` / `QArray`
family**. Compatibility accessors such as `Quantity.value` and
`QArray.float_array()` exist to ease migration from the old API.

### Tests

Tests accompany the quantities and quantity arrays.

```bash
python test_quantity.py          # Quantity / ComplexQuantity
python test_quantity_array.py    # QArray
python test_electrochemistry.py  # electrochemistry module
# If pytest is installed, `pytest <file>` also works
```

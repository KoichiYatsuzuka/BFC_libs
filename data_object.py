"""
### BFC_libs.data_object

データ階層の抽象クラス群を定義する。common.py の汎用関数・定数から分離した
「データオブジェクト層」。

データ階層:
    DataFile
        :- DataArray[T]          (T は DataSeriese のサブクラス)
            :- DataSeriese       (x/y 系列に QArray を持つ)

DataSeriese の x/y 系列は QArray(quantity_array.py)である。旧 ValueObjectArray
との互換(bound=np.ndarray)は廃止し、QArray に限定している。
"""
from __future__ import annotations

import abc
from copy import deepcopy as copy
from dataclasses import dataclass
from typing import Any, Generator, Generic, Iterator, Optional
from typing import Self, SupportsIndex, TypeVar, Union, overload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import ufunc
from numpy._typing import NDArray

try:
    from .common import immutator
    from .quantity_array import QArray
except ImportError:
    # リポジトリ直下から直接 import する場合(パッケージ外実行)
    from common import immutator
    from quantity_array import QArray


# DataSeriese の x/y 系列の型。QArray(およびその特殊化・サブクラス)に限定する。
X = TypeVar('X', bound=QArray)
Y = TypeVar('Y', bound=QArray)


@dataclass(frozen=True, repr=False)
class DataSeriese(Generic[X, Y], metaclass=abc.ABCMeta):
    """
    An abstracted class for DataSereise, for example, voltammogram, spectrum, and so on.
    Instances of DataFile class have instance(s) of this type.
    Some methods must be overrided to fully use the member methods.
    When inheriting, type of x series and y series (i.g. Voltammogram = DataSeries[PotentialArray, CurrentArray]).
    These are used for type hints of some methods.

    データ系列を表すための抽象クラス（例: ボルタモグラム、スペクトル、など）
    DataFileクラスのインスタンスはこのクラスのインスタンスを内包する。
    いくつかのメンバメソッドをオーバーライドしなければ使えないメソッドがある。
    継承時にx, y系列の型を指定する（例: Voltammogram = DataSeries[PotentialArray, CurrentArray]）。
    これらはいくつかのメソッドの型ヒントに使われる。
    継承時に@dataclass(frozen=True, repr=False)を使う。

    ## virtual methods
    x(), y(), to_data_frame(), from_data_frame()\n
    They throw AttributeError when called without override.

    ## members
    x,y: QArray (property getter)
        Have to be overrided.
        Refferes data corresponding to x and y in usual figures.

        i.g.
        Voltammograms
            x: potential y: current
        XRD
            x: 2theta y: diffraction intensity

    comment: list[str] (property getter)
        This includes history of instance modifications. All methods to generate modified instances must log the modifications.
        Users can track the log of modification of the instance.

    condition: list[str] (property getter)
        Automatically substracted from meta data zone, if possible.

    original_file_path: list[str] (property getter)
        As the name means

    plot()
        Can be overrided to add axes labels.
        Method to roughly plot the data.
        The axes are reusable.

    slice()
        Slicing data using the lower and upper values of x series, not with index.

    to_data_frame()
        Have to be overrided
        Method to convert the content to pd.DataFrame.
        Some imformation (i.g. original file name) is missed.

    ## class method
    from_data_frame()
        Have to be overrided.
        Method to generate an instance from pd.DataFrame.
    """
    _comment: list[str]
    _condition: list[str]
    _original_file_path: str
    _data_name : str

    @property
    def x(self)->X:
        raise AttributeError("x getter of this class has not been overrided. Now this class is calling a vertual method in the abstracted parent class.")
        pass

    @property
    def y(self)->Y:
        raise AttributeError("y getter of this class has not been overrided. Now this class is calling a vertual method in the abstracted parent class")
        pass

    @property
    def data_name(self):
        return self._data_name

    @property
    def condition(self):
        return self._condition

    @property
    def comment(self):
        return self._comment

    @property
    def original_file_path(self):
        return self._original_file_path

    def to_data_frame(self)->pd.DataFrame:
        raise AttributeError("This method must be overrided")

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        comment: list[str] = [],
        condition: list[str] = [],
        original_file_path: str = "",
        )->Self:
        raise AttributeError("This method must be overrided")

    def to_csv(self, file_path: str):
        self.to_data_frame().to_csv(path_or_buf=file_path, encoding="UTF-8", index=False)
        return

    @immutator
    def slice(self, x_min: Optional[X] = None, x_max: Optional[X] = None)->Self:
        if x_min is None:
            x_min = self.x[0]

        if x_max is None:
            x_max = self.x[-1]

        try:
            if x_min < self.x.min() or x_max > self.x.max():
                raise ValueError(
                    f"x_min and x_max must be more and less than the mimimum and mazimum of x, respectively.\n\
                    x_min: {x_min}, minimum x: {self.x.min()}, x_max: {x_max}, maximum x: {self.x.max()}"
                    )
        except TypeError:
            raise TypeError(
                f"TypeError was raised during a comparison.\n\
                x_min: {type(x_min)}, x_max: {type(x_max)}, \n\
                type of x element: {type(self.x[0])} "
            )

        x_min_index = self.x.find(x_min)[0]
        x_max_index = self.x.find(x_max)[-1]
        tmp_df = self.to_data_frame()

        sliced = self.from_data_frame(
            df = tmp_df.iloc[x_min_index:x_max_index],
            comment = self.comment + [f"sliced: x_min = {x_min}, x_max = {x_max}"],
            condition = self.condition,
            original_file_path = self.original_file_path
            )

        return sliced

        pass

    def __repr__(self):
        return self.data_name+": "+str(type(self))+"\n"+self.to_data_frame().__repr__()

    def plot(
        self,
        fig: Optional[Figure]=None,
        ax: Optional[Axes]=None,
        **kargs
        )->tuple[Figure, Axes]:
        """
        データの簡易プロット用クラスメソッド
        figとaxは書き換える。
        """
        match fig:
            case None:
                _fig = plt.figure(figsize = (4,3))
            case _:
                _fig = fig

        match ax:
            case None:
                _ax = _fig.add_axes((0.2,0.2,0.7,0.7))

            case _:
                _ax = ax

        _ax.plot(self.x,self.y, **kargs)
        return (_fig, _ax)


T = TypeVar('T', bound=DataSeriese)
class DataArray(np.ndarray, Generic[T]):
    """Array for chemical data, such as spectra or voltammograms.
    The items are instances of a DataSeries subclass.


    スペクトルやボルタモグラムと言った、測定データや計算結果の配列。
    要素はDataSeriesのサブクラスのインスタンス。

    How to declare & use
    ----------
    This is a template class. Because of the fucking specification of Python, the data type must be specified twice.
    ```
    # In the case of cyclic voltammograms.

    CV_list: list[ec.CyclicVoltammetry] = CV_list_generation_function(data_file_path)
    data_list = DataArray[ec.CyclicVoltammetry](CV_list, ec.CyclicVoltammetry) # Initialization with a list of DataSeries subclass
    ```
    Parameters
    ----------
    obj : array-like
        Input data to be converted to the array.
    dtype : np.dtype
        Desired data type of the array. This must be identical to T.
    meta : str, optional
        Metadata label attached to the array.

    Attributes
    ----------
    meta : str
        Metadata string describing the array's origin or content.

    Type Parameters
    ---------------
    T
        Element type returned by indexing (``__getitem__``, ``__iter__``).
    """
    #array : np.ndarray

    #def __init__(self, array):
        #self.array = np.array(array)

    def __new__(cls, obj, dtype = np.object_, meta: Optional[str] = None):
        self = np.asarray(obj, dtype=dtype).view(cls)

        match meta:
            case None:
                self.meta=""

            case _:
                self.meta = meta

        return self

    def __array_finalize__(self, obj: Optional[NDArray[Any]]):
        #おそらく動いていないが、必要になったら改変
        if obj is None:
            return None
        self.meta = getattr(obj, "meta", None) or ""

    def __array_ufunc__(self, ufunc: ufunc, method, *args: Any, **kwargs: Any):
        metalist = [] # メタ情報のリスト
        args_ = [] # 入力引数のリスト
        for arg in args:
            # 可能ならメタ情報をリストに追加
            if isinstance(arg, self.__class__) and hasattr(arg, "meta"):
                metalist.append(arg.meta)
            # MetaArrayはndarrayに直す
            arg = arg.view(np.ndarray) if isinstance(arg, Self) else arg
            args_.append(arg)
        # 関数を呼び出す
        out_raw = getattr(ufunc, method)(*args_, **kwargs)

        # なんか必要らしい
        if out_raw is NotImplemented:
            return NotImplemented

        # 型を戻す。このとき、スカラー(np.float64など)は変化しない。
        out = out_raw.view(self.__class__) if isinstance(out_raw, np.ndarray) else out_raw

        # メタ情報を引き継ぐ。このとき、入力したメタ情報を連結する。
        if isinstance(out, self.__class__):
            #print(metalist)
            #print(ufunc.__name__)
            out.meta = ','.join(metalist)+"_"+ufunc.__name__

        return out



    @overload
    def __getitem__(self, suffix: SupportsIndex)->T:
        ...
        #return np.ndarray.__getitem__(self, position)

    @overload
    def __getitem__(self, suffix: slice)->T:
        ...
        #return np.ndarray.__getitem__(self, slice)

    def __getitem__(self, suffix: Union[SupportsIndex, slice])->T: # type: ignore
        return np.ndarray.__getitem__(self, suffix) # type: ignore

    def __iter__(self)->Generator[T, None, None]:
        return np.ndarray.__iter__(self)

    def map(self, function: function, *args, **kargs):
        """
        The first parameter of function will be elements of this self.
        """
        tmp_list:list[T] = []

        for data in self:
            tmp = function(data, *args, **kargs)

            if not isinstance(tmp, type(self[0])):
                raise TypeError("The returned value of mapped function, {}, is not {}, but {}.".format(
                    function,
                    type(self[0]),
                    type(tmp)
                ))

            tmp_list.append(copy(tmp))

        return DataArray[T](tmp_list)

    @immutator
    def join(self, another: Self)->Self:
        if type(another) != type(self):
            raise TypeError

        joined_ndarray = np.append(self, another)
        return type(self)(joined_ndarray)


@dataclass(frozen=True)
class DataFile(Generic[T], metaclass=abc.ABCMeta):
    """
    T: type of data (DataArray[T])
    use @dataclass(frozen=True) when succeeding this class

    example:
    class BiologicDataFile(DataFile[voltammogram]):

    """
    _data: DataArray[T]
    _comment: list[str]
    _condition: list[str]
    _file_path: str
    _data_name: str

    @property
    def condition(self):
        return self._condition

    @property
    def comment(self):
        return self._comment

    @property
    def data(self):
        return self._data

    @property
    def file_name(self):
        return self._file_path

    @overload
    def __getitem__(self, key: SupportsIndex)->T:
        ...

    @overload
    def __getitem__(self, key: slice)->DataArray[T]:
        ...

    @overload
    def __getitem__(self, key: str)->T:
        ...

    def __getitem__(self, key: Union[SupportsIndex, slice, str])->Union[T, DataArray[T]]:
        match key:
            case str():
                for data in self._data:
                    if isinstance(data, DataSeriese):
                        if data.data_name == key:
                            return data

                raise KeyError("Key: {}".format(key))
            case SupportsIndex():
                return self._data[key]
            case slice():
                return self._data[key]
            case _:
                print(type(key))
                return self._data[key]

    def map(self, function: function, *args, **kwargs)->Self:
        members = vars(self)
        members["_data"] = self._data.map(function, *args, **kwargs)
        tmp = copy(self._comment)
        tmp.append(
            ["mapped with {}".format(function),
            "params: {}".format(args, kwargs)]
            )
        members["_comment"] = tmp

        return type(self)(**members)

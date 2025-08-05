"""
### BFC_libs.common
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy import ufunc
from numpy._typing import NDArray
from typing import Any, Literal, Optional, SupportsIndex
from typing import Union, NewType, Self
from typing import Generic, TypeVar, TypeAlias, Final, cast
from typing import overload, Type, ClassVar, Generator
from dataclasses import dataclass
import abc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import deepcopy as copy
from functools import wraps
import codecs
#import inspect
#from numba import jit


#------------------------------------------------------
#------------------decorators--------------------------
#------------------------------------------------------

def immutator(func):
    """
    This decorator passes deepcopied aruments list.
    It is guaranteeed that the all original arguments will not be overwritten.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_copy = tuple(copy(arg) for arg in args)
        kwargs_copy = {
            key: copy(value) for key, value in kwargs.items()
        }
        return func(*args_copy, **kwargs_copy)
    return wrapper

def self_mutator(func):
    """
    This decorator passes deepcopied aruments list other than itself.
    It is guaranteeed that the all original arguments will not be overwritten.
    """
    """
        TO DO: 可能であれば、第一引数がselfかどうかのチェックをしたい。
        現在の問題点として、第一引数のオブジェクトが有するメソッドど同名のグローバル関数にこのデコレータを付けても問題なく動いてしまう。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        """

        # Checking wheatehr the first arg is self
        
        # Check the length of the arguments and get the firstr argument
        if len(args)==1:
            first_arg = args
        elif len(args)==0:
            raise InvalidDecorator("This decorator is used in class method with a self argument.")
        else:
            first_arg = args[0]
        
        # extract the name of the method which is calling this decorator
        func_name = func.__name__


        # try to get the id of class method
        try:
            first_arg.__getattribute__(func_name)

        except AttributeError:
            # Meaning that the object does not have the method whose name is tha same as the method calling this decorator.
            raise InvalidDecorator("This decorator must be used in class method.")
        
    
        if len(args)==1:
            args_copy = args
        else:
            args_copy = tuple([args[0]]) + \
                tuple(copy(arg) for arg in args[1:])
        kwargs_copy = {
            key: copy(value) for key, value in kwargs.items()
        }
        return func(*args_copy, **kwargs_copy)
    return wrapper


#------------------------------------------------------
#----------classes and relative variables--------------
#------------------------------------------------------

NOT_ALLOWED_ERROR_STR: Final[str] = "An invalid object was substituted.\nAllowd type is {},\n but {} was used"
OPERATION_ALLOWED_TYPES: Final[list[type]] = [
    float,
    int,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    complex,
    np.complex64,
    np.complex128,
]

def typeerror_other_type(self, another):
    if type(another)!=type(self):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(another))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)

class ValueObject:
    """
    値オブジェクト用のクラス。
    このライブラリは基本的にこのクラスを継承したクラスを値として扱う。
    不正な演算（すなわち、意図しない演算）を防ぐため、このクラスの演算を許可する型を限定しており、float型への自動キャストを行わない。
    明示的なfloatへのキャストは行える。

    メンバ
        _value
            protected属性。値をnp.float64型で保持。
        
        value
            property。値をnp.float64型で返す。
        
        __add__, __sub__
            右辺で足す、引く。同じ型でないとTypeErrorを投げる。

        __mul__, __truedev__
            右辺でかける、割る。
            同じ型か一般的な数値型（intやnp.float64など）でなければTypeErrorを投げる。
            同じValueObject型でも、異なるサブクラス同士の演算は基本許可しない。
            サブクラスに別にメンバメソッドを用意する

        比較演算子系特殊メソッド
            同じ型でなければTypeErrorを投げる。

        __str__, __repr__
            self._value.__str__()を返す。

        __abs__
            self._value.__abs__()を返す。

        __float__
            _valueをfloatにキャストして返す。			

    クラスメソッド
        _cast_to_this()
            値をこのクラスにキャストして返す。
            floatなどの値型でなかった場合、TypeErrorを投げる。
    """
    _value: np.float64|complex

    def __init__(self, value: float|np.float64|complex):
        if not(type(value) in OPERATION_ALLOWED_TYPES) and type(value) != type(self):
            error_report = NOT_ALLOWED_ERROR_STR.format(OPERATION_ALLOWED_TYPES, str(type(value))).replace("<", "").replace(">", "")
            raise TypeError(error_report)

        if isinstance(value, ValueObject):
            self._value = value.value
        else:
            self._value = value

    @classmethod
    def _cast_to_this(cls, value: Union[int, float, np.float64, np.int64])->Self:
        return cls(value)
        
    @property
    def value(self)->np.float64|complex:
        return self._value
    
    def __neg__(self):
        """
        負の値を返す。
        """
        cls_type = type(self)
        return cls_type(-self.value)
    
    @immutator
    def __add__(self, added_value):
        # error
        if type(self)!=type(added_value):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(added_value))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)
        
        # normal process
        cls_type=type(self)
        sum: cls_type = cls_type(self.value+added_value.value)
        return sum
    
    @immutator
    def __sub__(self, subed_value):
        # error
        if type(self)!=type(subed_value):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(subed_value))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)
        
        # normal process
        cls_type=type(self)
        diff: cls_type = cls_type(self.value-subed_value.value)
        return diff
    
    @immutator
    def __mul__(self, muled_value: Union[int, float, Self]):
        # error
        if not(type(muled_value) in OPERATION_ALLOWED_TYPES) and type(muled_value)!= type(self):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(OPERATION_ALLOWED_TYPES), str(type(muled_value))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)
        
        # normal process
        cls_type=type(self)
        if isinstance(muled_value, (complex, np.complex64, np.complex128)):
            product = cls_type(self.value*muled_value)
        else:
            product: cls_type = cls_type(self.value*float(muled_value))
        return product
    
    @immutator
    def __rmul__(self, muld_value: Union[int, float, Self]):
        return self.__mul__(muld_value)
    
    @immutator
    def __truediv__(self, dived_value: Union[int, float, Self]):
        # error
        if not(type(dived_value) in OPERATION_ALLOWED_TYPES) and type(dived_value)!= type(self):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(str(OPERATION_ALLOWED_TYPES)), str(type(dived_value))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)
        
        cls_type=type(self)
        quotient: cls_type = cls_type(self.value / np.float64(dived_value))
        return quotient
    
    @immutator
    def __rtruediv__(self, dived_value: Union[int, float, Self]):
        # error
        if type(dived_value)!=int and type(dived_value)!= float and type(dived_value)!= type(self):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(Union[float, int, type(self)]), str(type(dived_value))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)
        
        cls_type=type(self)
        quotient: cls_type = cls_type(np.float64(dived_value)/self.value)
        
        return quotient
    
    @immutator
    def __lt__(self, another):
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value<another.value

    @immutator
    def __le__(self, another):
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value<=another.value
    
    @immutator
    def __gt__(self, another: Self):
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value>another.value
    
    @immutator
    def __ge__(self, another: Self):
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value>=another.value
    
    @immutator
    def __eq__(self, another: Self):
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value==another.value
    
    @overload
    def __pow__(self, power: int|float) -> float:
        ...
    @overload
    def __pow__(self, power: complex) -> complex:
        ...

    @immutator
    def __pow__(self, power: Union[int, float, complex])->float|complex:
        return self.value**power

    @immutator
    def __str__(self):
        
        return str(self.value)
    
    @immutator
    def __repr__(self)->str:
        
        return str(self.value)
    
    @immutator
    def __float__(self)->float:
        
        return float(self.value)
    
    @immutator
    def __abs__(self):
        return np.abs(self._value)


ValObj = TypeVar('ValObj', bound=ValueObject)
class ValueObjectArray(np.ndarray, Generic[ValObj]):
    """
    値オブジェクト用のnp.ndarray。
    np.ndarrayの関数は基本的に使えるが、ValueObjectが許可しない演算（floatの加算など）を許可しない。
    
    継承時の注意
        ·一部の型ヒントを有効するため、以下のように継承する。
            class Sub(ValueObjectArray[cls]):	
    
        ·以下のように__new__をoverrideする
            def __new__(cls, obj, dtype=cls, meta: Optional[str] = None):
                return super().__new__(cls, obj, dtype, meta)
        
    ## メンバ
    np.ndarrayのメンバに加えて以下のメソッドを定義している。
        normalize()
            配列全体を0-1の値に規格化した配列を返す。自身を変更しない。
        float_array()
            配列全体をflaot型にして返す。自身は変更しない。
        find()
            指定された値に一番近い値を格納している要素へのindexの配列を返す。
    
    Succeeds numpy.ndarray
    additional method
        normalize()
        float_array()
        find()
    """
    
    """
    自分用メモ
    何かあればこのQiita
    https://qiita.com/Hanjin_Liu/items/02b9880d055390e11c8e
    """
    #data_type: ClassVar[Type[ValObj]]

    def __new__(cls, obj, dtype, meta: Optional[str] = None):
        #self = np.asarray(list(map(dtype, obj)), dtype=np.object_).view(cls)
        self = np.asarray(list(np.vectorize(dtype)(obj)), dtype=np.object_).view(cls)
        
        self.data_type = dtype
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
        self.meta = getattr(obj, "meta", None)

    def __array_ufunc__(self, ufunc: ufunc, method, *args: Any, **kwargs: Any):
        metalist = [] # メタ情報のリスト
        args_ = [] # 入力引数のリスト
        for arg in args:
            # 可能ならメタ情報をリストに追加
            if isinstance(arg, self.__class__) and hasattr(arg, "meta"):
                metalist.append(arg.meta)
            # MetaArrayはndarrayに直す
            arg = arg.view(np.ndarray) if isinstance(arg, ValueObjectArray) else arg
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

    #@immutator
    def normalize(self, begin_index=None, end_index=None)->Self:
        """"""
        match begin_index:
            case None:
                _begin_index = 0
            case _:
                _begin_index = begin_index
        
        match end_index:
            case None:
                _end_index = len(self)-1
            case _:
                _end_index = end_index
        
        sliced_array = self[_begin_index:_end_index]
        max_value = sliced_array.max()
        min_value = sliced_array.min()

        normd_value: ValueObjectArray = (self-min_value)/(max_value-min_value)
        normd_value.meta = normd_value.meta.removesuffix("_subtract_divide")+"_normalized"
        return normd_value
    
    #@immutator
    def float_array(self)->np.ndarray:
        return np.array(self, dtype=np.float64)

    #@immutator
    def find(self, target_value, begin_index: int = 0, end_index: int = None)->Optional[list[int]]:
        if end_index == None:
            _end_index = len(self)
        
        if begin_index < 0 or begin_index > _end_index or _end_index > len(self):
            raise IndexError("Invalid index. begin: {}, end: {} length: {}".format(begin_index, _end_index, len(self)))
        
        tmp_ary = self[begin_index:_end_index]
        explored_ary = tmp_ary-type(self[0])(target_value)
        #print(explored_ary)
        #最近接値: 一個ずらした配列の各要素の積が負になった箇所
        index_list_neighber = np.where(np.delete(explored_ary, [0])*np.delete(explored_ary, [-1]) < type(self[0])(0))[0]
        
        #完全に同値の要素（上記方法に等号入れると、0乗算の要素が二つあるので二つの要素が返ってきてしまう）
        index_list_equall = np.where(explored_ary == type(self[0])(0.0))[0] 

        index_list_edge = []
        """#端っこの判定
        if len(tmp_ary)>2:
            print(tmp_ary[0], tmp_ary[1], tmp_ary[-2], tmp_ary[-1])
            #値が0番目と1番目の間
            if (tmp_ary[0] < target_value and target_value < tmp_ary[1]) or\
            (tmp_ary[1] < target_value and target_value < tmp_ary[0]):
                #1番目より0番目の方が値が近い
                if abs(tmp_ary[0] - target_value) < abs(tmp_ary[1] - target_value):
                    index_list_edge.append(0)
                
            #値が-2番目と-1番目の間
            if (tmp_ary[-2] < target_value and target_value < tmp_ary[-1]) or\
            (tmp_ary[-2] < target_value and target_value < tmp_ary[-1]):
                #-2番目より-1番目の方が値が近い
                if abs(tmp_ary[-1] - target_value) < abs(tmp_ary[-2] - target_value):
                    index_list_edge.append(len(self)-1)"""
                
        tmp = np.append(index_list_neighber, index_list_equall)
        tmp2 = np.sort(np.append(tmp, np.array(index_list_edge, dtype=int)))
        #print(tmp2)
        return tmp2
    
        
    @immutator
    def join(self, another: Self)->Self:
        if type(another) != type(self):
            raise TypeError
        self_ndarray = self.float_array()
        another_ndarray = another.float_array()
        joined_ndarray = np.append(self_ndarray, another_ndarray)
        return type(self)(joined_ndarray)
    

    def __and__(self, another: Self)->Self:
        return self.join(another)
    
    
    
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Self:
        ...

    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | tuple[None | slice | ellipsis | SupportsIndex, ...]
    )) -> Self:
        ...

    @overload
    def __getitem__(self, key: (
        NDArray[np.integer[Any]]
        | NDArray[np.bool_]
        | list[bool]
        | tuple[NDArray[np.integer[Any]] | NDArray[np.bool_], ...]
    )) -> Self:
        ...

    def __getitem__(self, key: Any)->Union[Self, ValObj]:
        #print("array")
        return np.ndarray.__getitem__(self, key)

    # @overload
    # def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Any: ...
    # @overload
    # def __getitem__(self: NDArray[void], key: str) -> NDArray[Any]: ...
    # @overload
    # def __getitem__(self: NDArray[void], key: list[str]) -> ndarray[_ShapeType, _dtype[void]]: ...
    
    def __sub__(self, subed_value: Union[Self, ValObj])->Self:
        """
        引き算のオーバーロード
        引き算はValueObjectArray同士でしか行えない。
        """
        """if type(subed_value) is not Self and type(subed_value) is not ValObj:
            raise TypeError(
                "ValueObjectArray can only be subtracted with the same type.\n\
                This is {} but {}".format(type(self), type(subed_value))
            )"""
        
        if isinstance(subed_value, ValueObjectArray):
            return type(self)(self.float_array() - subed_value.float_array()) # type: ignore

        if isinstance(subed_value, ValueObject):
            return type(self)(self.float_array() - subed_value.value) # type: ignore

        raise TypeError(
            "ValueObjectArray can only be subtracted with the same type.\n\
            This is {} but {}".format(type(self), type(subed_value))
        )


    def __iter__(self)->Self:
        
        return np.ndarray.__iter__(self)
    

    def __next__(self)->ValObj:
        
        return np.ndarray.__next__(self)


T = TypeVar('T')
class DataArray(np.ndarray, Generic[T]):
    """
    """

    """
    ValueObjectArrayを参考にしながら作成
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
        self.meta = getattr(obj, "meta", None)

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
    def __getitem__(self, index: SupportsIndex)->T:
        pass
        return self._data[index]
    
    @overload
    def __getitem__(self, slice: slice)->DataArray[T]:
        pass
        return self._data[slice]
    
    @overload
    def __getitem__(self, data_name: str)->T:
        pass

    def __getitem__(self, key: Union[SupportsIndex, slice, str]):
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

X = TypeVar('X', ValueObjectArray, np.ndarray)
Y = TypeVar('Y', ValueObjectArray, np.ndarray)

@dataclass(frozen=True, repr=False)
class DataSeriese(Generic[X, Y], metaclass=abc.ABCMeta):
    """
    An abstracted class for DataSereise, for example, voltammogram, spectrum, and so on.
    Instances of DataFile class have instance(s) of this type.
    Some methods must be overrided to fully use the member methods.
    When inheriting, type of x series and y series (i.g. Voltammogram = DataSeries[Potential, Current]).
    These are used for type hints of some methods.

    データ系列を表すための抽象クラス（例: ボルタモグラム、スペクトル、など）
    DataFileクラスのインスタンスはこのクラスのインスタンスを内包する。
    いくつかのメンバメソッドをオーバーライドしなければ使えないメソッドがある。
    継承時にx, y系列の型を指定する（例: Voltammogram = DataSeries[Potential, Current]）。
    これらはいくつかのメソッドの型ヒントに使われる。
    継承時に@dataclass(frozen=True, repr=False)を使う。

    ## virtual methods
    x(), y(), to_data_frame(), from_data_frame()\n
    They throw AttributeError when called without override.

    ## members
    x,y: ValueObjectArray (property getter)
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
    def x(self)->ValueObjectArray:
        raise AttributeError("x getter of this class has not been overrided. Now this class is calling a vertual method in the abstracted parent class.")
        pass

    @property
    def y(self)->ValueObjectArray:
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


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_tupple(self):
        return (self.x, self.y)


#------------------------------------------------------
#---------------common object values-------------------
#------------------------------------------------------	
        
class Time(ValueObject):
    pass
"""
Time_Array = NewType("Time_Array", ValueObjectArray[Time])
"""
class TimeArray(ValueObjectArray):
    def __new__(cls, obj, dtype=Time, meta: Optional[str] = None):
        
        return super().__new__(cls, obj, dtype, meta)
    pass

#------------------------------------------------------
#-------------------functions--------------------------
#------------------------------------------------------

"""@immutator
def to_value_object_array(list:list, type: type, meta: Optional[str] = None)->ValueObjectArray[type]:
    
    converted_array= np.array([])
    for value in list:
        converted_array = np.append(converted_array, type(value))
    
    return copy(ValueObjectArray(converted_array, dtype=type, meta=meta))"""


def set_matpltlib_rcParameters()->None:
    """
    Set parameters list:
    "font.size" = 10
    'axes.linewidth' = 1.5
    "xtick.top" = True
    "xtick.bottom" = True
    "ytick.left" = True
    "ytick.right" = True
    'xtick.direction' = 'in'
    'ytick.direction' = 'in'
    "xtick.major.size" =6.0
    "ytick.major.size" = 6.0
    "xtick.major.width" = 1.5
    "ytick.major.width" = 1.5
    "xtick.minor.size" =4.0
    "ytick.minor.size" = 4.0
    "xtick.minor.width" = 1.5
    "ytick.minor.width" = 1.5
    plt.rc('legend', fontsize=7)
    'lines.markersize' =3\n
    Any others? Do by yourselves.
    ## Returns
        nothing
    """
    plt.rcParams["font.size"] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["xtick.top"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["xtick.major.size"] =6.0
    plt.rcParams["ytick.major.size"] = 6.0
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.size"] =4.0
    plt.rcParams["ytick.minor.size"] = 4.0
    plt.rcParams["xtick.minor.width"] = 1.5
    plt.rcParams["ytick.minor.width"] = 1.5
    plt.rc('legend', fontsize=7)
    plt.rcParams['lines.markersize'] =6
    return

def create_standard_matplt_canvas()->tuple[Figure, Axes]:
    """
    ## Returns
    Returns tuple of usual Figure and Axes instances.
    fig = plt.figure(figsize = (4,3)),\n
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    """
    fig=plt.figure(figsize = (4,3))
    ax = fig.add_axes(rect=(0.2,0.2,0.7,0.7))
    return (fig, ax)

def extract_extension(file_path: str)->Optional[str]:
    """
    ### Extract the extension from a file name.
    ig. test.txt -> txt
        /data/data.mpt -> mpt
        spec_0.5V.spc -> spc
    ## parameter
    file_path: 
        Targetted file name.
        Relative path or absolute path is also acceptable.
        This can contain period other than extension.

    ## Return Value
    The extracted extension in str type.
    If the parameter does not inculde period, this function returns None.
        
    ## Error
    TypeError
        The parameter accepts only str. Any other types cause TypeError.

    """
    try:
        splitted_str = file_path.split(sep=".")
    except(AttributeError):
        raise(TypeError(
            "Invalid file path: Parameter is not str. \nThe type is {}.".format(type(file_path))))
    if len(splitted_str) <2:
        return None
    return splitted_str[len(splitted_str)-1]

def extract_filename(file_path: str)->str:
    splitted_str_slash = file_path.split(sep="/")
    splitted_str_backslash = splitted_str_slash[len(splitted_str_slash)-1].split(sep="\\")
    return splitted_str_backslash[len(splitted_str_backslash)-1]

def find_line_with_key(
        file_path: str, 
        key_word: str,
        encoding: str = 'UTF-8'
        )->Optional[int]:
    """
    ### Count the number of lines firstly include the key word.
    ig.\n
    file contents=
        date: 2001/2/3 <- skip\n
        method: CV <- skip\n
        time, potential, current <- column name \n
        0, 0, 0.1 <- data row \n
        1, 0.005, 0.2 \n
        .......
    key_word = "potential"\n
    In this case, this function will return 3. 
    Read as UTF-8. If file is written in Shift-JIS, it may cause an error.

    ## Parameters
    file_path: the path to the file to read.
            Relative or absolute patha is acceptable.
    key_word: the word to find.

    ## Return value
    The position of the line firstly including the key word
    None: the key word was not found.
    """
    # Reading each line with finding the key word
    file = codecs.open(file_path, 'r', encoding=encoding, errors='ignore')
    lines = file.readlines()
    file.close()

    i_line_count :int = 0
    for line in lines:
        if line.find(key_word) != -1:
            # now the key word was found. 
            # current i values is (the number of lines read) - 1
            break
        i_line_count += 1

    if i_line_count == len(lines):
        # the key word was not found
        return None

    # succesfully finished
    return i_line_count+1

def convert_relative_pos_to_pos_in_axes_label(pos: Point, ax: plt.Axes)->Point: # type: ignore
    
    rel_x, rel_y = pos.x, pos.y
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_in_label = x_min + (x_max - x_min) * rel_x
    y_in_label = y_min + (y_max - y_min) * rel_y

    return Point(x_in_label, y_in_label)
#------------------------------------------------------
#------------------exceptions--------------------------
#------------------------------------------------------
class InvalidDecorator(Exception):
    """
    If a decorator is not used as expected, this error must be raised. 
    """
    pass

class FileContentError(Exception):
    """
    If the loaded file content is different from expected, this error wil be raised
    """
    pass

#------------------------------------------------------
#------------------executions--------------------------
#------------------------------------------------------
set_matpltlib_rcParameters()
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
    np.ndarray
]

def typeerror_other_type(self, another):
    if type(another)!=type(self):
            error_report = \
                NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(another))).\
                replace("<", "").replace(">", "")
            raise TypeError(error_report)

ValType = TypeVar('ValType', bound=float|complex)
class ValueObjectBase(Generic[ValType]):
    _value: ValType

    @classmethod
    def _cast_to_this(cls, value: Union[int, float, np.float64, np.int64, complex])->Self:
        return cls(value)
    
    @property
    def value(self)->ValType:
        ...
    
    def __neg__(self)->Self:
        """
        負の値を返す。
        """
        cls_type = type(self)
        return cls_type(-self.value)
    
    @property
    def real(self):
        clstype = type(self)
        return clstype(self.value.real)
    
    @property
    def imag(self):
        clstype = type(self)
        return clstype(self.value.imag)
    
    @immutator
    def __add__(self, added_value:Self):
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
    def __sub__(self, subed_value:Self):
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
        elif isinstance(muled_value, np.ndarray):
            product = muled_value*self.value
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
    def __lt__(self, another)->bool:
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value<another.value

    @immutator
    def __le__(self, another:Self)->bool:
        # error
        try:
            typeerror_other_type(self, another)
        except ValueError as error_report:
            raise ValueError(error_report)
        
        return self.value<=another.value
    
    @immutator
    def __gt__(self, another: Self)->bool:
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

    def __abs__(self)->float:
            return abs(self._value)


    @immutator
    def __str__(self):
        return str(self.value)
    
    @immutator
    def __repr__(self)->str:
        return str(self.value)
    
    @immutator
    def __complex__(self)->complex:
        return complex(self.value)
    
    def sqrt(self)->ValType:
        return np.sqrt(self._value)
    
    def exp(self)->ValType:
        return np.exp(self._value)
    
    def sin(self)->ValType:
        return np.sin(self._value)
    



class ValueObject(ValueObjectBase[float]):
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
    #_value: np.float64|complex

    def __init__(self, value: float|np.float64|complex):
        if not(type(value) in OPERATION_ALLOWED_TYPES) and type(value) != type(self):
            error_report = NOT_ALLOWED_ERROR_STR.format(OPERATION_ALLOWED_TYPES, str(type(value))).replace("<", "").replace(">", "")
            raise TypeError(error_report)

        if isinstance(value, type(self)):
            self._value = value.value
        else:
            self._value = value

    # @classmethod
    # def _cast_to_this(cls, value: Union[int, float, np.float64, np.int64])->Self:
    #     return cls(value)
        
    @property
    def value(self)->float:
        return self._value
    
    # def __neg__(self):
    #     """
    #     負の値を返す。
    #     """
    #     cls_type = type(self)
    #     return cls_type(-self.value)
    
    # @property
    # def real(self):
    #     clstype = type(self)
    #     return clstype(self.value.real)
    
    # @property
    # def imag(self):
    #     clstype = type(self)
    #     return clstype(self.value.imag)
    
    # @immutator
    # def __add__(self, added_value):
    #     # error
    #     if type(self)!=type(added_value):
    #         error_report = \
    #             NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(added_value))).\
    #             replace("<", "").replace(">", "")
    #         raise TypeError(error_report)
        
    #     # normal process
    #     cls_type=type(self)
    #     sum: cls_type = cls_type(self.value+added_value.value)
    #     return sum
    
    # @immutator
    # def __sub__(self, subed_value):
    #     # error
    #     if type(self)!=type(subed_value):
    #         error_report = \
    #             NOT_ALLOWED_ERROR_STR.format(str(type(self)), str(type(subed_value))).\
    #             replace("<", "").replace(">", "")
    #         raise TypeError(error_report)
        
    #     # normal process
    #     cls_type=type(self)
    #     diff: cls_type = cls_type(self.value-subed_value.value)
    #     return diff
    
    # @immutator
    # def __mul__(self, muled_value: Union[int, float, Self]):
    #     # error
    #     if not(type(muled_value) in OPERATION_ALLOWED_TYPES) and type(muled_value)!= type(self):
    #         error_report = \
    #             NOT_ALLOWED_ERROR_STR.format(str(OPERATION_ALLOWED_TYPES), str(type(muled_value))).\
    #             replace("<", "").replace(">", "")
    #         raise TypeError(error_report)
        
    #     # normal process
    #     cls_type=type(self)
    #     if isinstance(muled_value, (complex, np.complex64, np.complex128)):
    #         product = cls_type(self.value*muled_value)
    #     elif isinstance(muled_value, np.ndarray):
    #         product = muled_value*self.value
    #     else:
    #         product: cls_type = cls_type(self.value*float(muled_value))
    #     return product
    
    # @immutator
    # def __rmul__(self, muld_value: Union[int, float, Self]):
    #     return self.__mul__(muld_value)
    
    # @immutator
    # def __truediv__(self, dived_value: Union[int, float, Self]):
    #     # error
    #     if not(type(dived_value) in OPERATION_ALLOWED_TYPES) and type(dived_value)!= type(self):
    #         error_report = \
    #             NOT_ALLOWED_ERROR_STR.format(str(str(OPERATION_ALLOWED_TYPES)), str(type(dived_value))).\
    #             replace("<", "").replace(">", "")
    #         raise TypeError(error_report)
        
    #     cls_type=type(self)
    #     quotient: cls_type = cls_type(self.value / np.float64(dived_value))
    #     return quotient
    
    # @immutator
    # def __rtruediv__(self, dived_value: Union[int, float, Self]):
    #     # error
    #     if type(dived_value)!=int and type(dived_value)!= float and type(dived_value)!= type(self):
    #         error_report = \
    #             NOT_ALLOWED_ERROR_STR.format(str(Union[float, int, type(self)]), str(type(dived_value))).\
    #             replace("<", "").replace(">", "")
    #         raise TypeError(error_report)
        
    #     cls_type=type(self)
    #     quotient: cls_type = cls_type(np.float64(dived_value)/self.value)
        
    #     return quotient
    
    # @immutator
    # def __lt__(self, another):
    #     # error
    #     try:
    #         typeerror_other_type(self, another)
    #     except ValueError as error_report:
    #         raise ValueError(error_report)
        
    #     return self.value<another.value

    # @immutator
    # def __le__(self, another):
    #     # error
    #     try:
    #         typeerror_other_type(self, another)
    #     except ValueError as error_report:
    #         raise ValueError(error_report)
        
    #     return self.value<=another.value
    
    # @immutator
    # def __gt__(self, another: Self):
    #     # error
    #     try:
    #         typeerror_other_type(self, another)
    #     except ValueError as error_report:
    #         raise ValueError(error_report)
        
    #     return self.value>another.value
    
    # @immutator
    # def __ge__(self, another: Self):
    #     # error
    #     try:
    #         typeerror_other_type(self, another)
    #     except ValueError as error_report:
    #         raise ValueError(error_report)
        
    #     return self.value>=another.value
    
    # @immutator
    # def __eq__(self, another: Self):
    #     # error
    #     try:
    #         typeerror_other_type(self, another)
    #     except ValueError as error_report:
    #         raise ValueError(error_report)
        
    #     return self.value==another.value
    
    # @overload
    # def __pow__(self, power: int|float) -> float:
    #     ...
    # @overload
    # def __pow__(self, power: complex) -> complex:
    #     ...

    # @immutator
    # def __pow__(self, power: Union[int, float, complex])->float|complex:
    #     return self.value**power

    # @immutator
    # def __str__(self):
        
    #     return str(self.value)
    
    # @immutator
    # def __repr__(self)->str:
        
    #     return str(self.value)
    
    #@immutator
    def __float__(self)->float:
        return float(self.value)
    
    # @immutator
    # def __complex__(self)->complex:
    #     return complex(self.value)
    
    # @immutator
    # def __abs__(self):
    #     return np.abs(self._value)

RealNumberType = TypeVar('RealNumberType', bound=ValueObjectBase)
class ValueObjectComplex(ValueObjectBase[complex], Generic[RealNumberType]):
    """
    仮想クラス
    """

    def __init__(self, value: float|np.float64|complex):
        if not(type(value) in OPERATION_ALLOWED_TYPES) and type(value) != type(self):
            error_report = NOT_ALLOWED_ERROR_STR.format(OPERATION_ALLOWED_TYPES, str(type(value))).replace("<", "").replace(">", "")
            raise TypeError(error_report)

        if isinstance(value, type(self)):
            self._value = complex(value.value)
        else:
            self._value = complex(value)
    
    @property
    def real(self)->RealNumberType:
        ...

    @property
    def imag(self)->RealNumberType:
        ...

    @property
    def value(self)->complex:
        return self._value

ValObj = TypeVar('ValObj', bound=ValueObjectBase, covariant=True)
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

    def __new__(cls, obj, dtype=None, meta: Optional[str] = None):
        if dtype is None:
            dtype = type(obj)
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
        self.meta = getattr(obj, "meta", None) or ""

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
        try:
            out_raw = getattr(ufunc, method)(*args_, **kwargs)
        except(TypeError) as e:
            print("shape: {}".format(self.shape))
            print("dtype: {}".format(type(self[0])))
            raise e

        
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

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        # self.view(np.ndarray) で基底クラスの object 配列を取得（再帰防止）
        # matplotlib など数値ライブラリとの互換性のため float64 を返す
        base = self.view(np.ndarray)
        result = base.astype(np.float64)
        if dtype is not None and dtype != np.float64:
            result = result.astype(dtype)
        return result

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
    
    def complex_array(self)->np.ndarray:
        return np.array(self, dtype=complex)

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
    
    @property
    @override
    def real(self):
        self_type = type(self)
        return self_type(self.complex_array().real)
    
    @property
    @override
    def imag(self):
        self_type = type(self)
        return self_type(self.complex_array().imag)

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


    def __iter__(self)->Iterator[Self|ValObj]:
        
        return np.ndarray.__iter__(self)



def is_voarray(self: Union[ValObj,ValueObjectArray[ValObj]])->TypeGuard[ValueObjectArray[ValObj]]:
    return isinstance(self, ValueObjectArray)

def is_vo(self: Union[ValObj,ValueObjectArray[ValObj]])->TypeGuard[ValObj]:
    return isinstance(self, ValueObject)

    

"""    def __next__(self)->ValObj:
        
        return np.ndarray.__next__(self)"""


"""class VO_NDArray[T](np.ndarray):


    #data_type: ClassVar[Type[ValObj]]

    def __new__(cls, obj: list[T], dtype=None, meta: Optional[str] = None):
        
        if dtype is None:
            dtype = type(obj)
        
        #self = np.asarray(list(map(dtype, obj)), dtype=np.object_).view(cls)
        #self = np.asarray(list(np.vectorize(dtype)(obj)), dtype=np.object_).view(cls)
        
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
        self.meta = getattr(obj, "meta", None) or ""

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

        return out"""
    


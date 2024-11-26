"""
フィッテイングのための操作を簡素にするためのライブラリ
特定の関数クラスを用いることで、パラメータの混同を防ぐ。
また、scipyのフィッティング関数の操作を簡略化する。

## classes
### FittingFunctionElement
関数を定義するための抽象クラス。
継承すべきもの:
    def params(): パラメータのtupleを返す\n

    @property\n
    def para_len(): パラメータの数を返す\n

    def calc(): その関数の計算結果

追加すべきメンバ:
    def calc()に必要な、関数のパラメータ（ピーク幅、傾き、寿命など）


"""
import numpy as np
import scipy.optimize as opt
from copy import deepcopy as copy
from dataclasses import dataclass

from typing import Union,NewType, Optional, Self, TypeVar, Generic
import abc

import BFC_libs.common as cmn

#float_like = Union[int, float, np.float64, cmn.ValueObject]
ndarray_like = Union[list[float], np.ndarray, cmn.ValueObjectArray]
#params = list[float]

class ValueWithRange:
    "a virtual class"
    value: float

    def __float__(self):
        return self.value

    def print(self):
        raise(TypeError)
        pass

class ValueWithBounds(ValueWithRange):
    least: Optional[float]
    most: Optional[float]

    def __init__(self, _value: float, _least: Optional[float]= None, _most: Optional[float]=None):
        
        if _least is not None and _least > _value:
            raise ValueError("least bound ({}) is more than x0({})".format(_least, _value)) 
        
        if _most is not None and _most < _value:
            raise ValueError("most bound ({}) is less than x0({})".format(_most, _value))

        self.value = _value
        self.least = _least
        self.most = _most
        
        if _value is None:
            raise ValueError("None is substituted for the main fitting parameter")
    def print(self):
        print(("{:.3e} - ".format(self.least) if self.least is not None else "No min bound - ")+\
            "{:.3e} - ".format(self.value)+\
            ("{:.3e}".format(self.most) if self.most is not None else "No max bound")
        )

class ValueWithErrror(ValueWithRange):
    def __init__(self, _value: float, _error: float):
        if _value is None:
            raise ValueError("None is substituted for the main fitting parameter")
        
        self.value = _value
        self.error = _error
        
    def print(self):
        print("{:.3e} ± {:.3e}".format(self.value, self.error))

ValType = TypeVar("T", ValueWithBounds, ValueWithErrror)
Param = ValueWithBounds

@dataclass(frozen=True)
class FittingFunctionElement(Generic[ValType], metaclass=abc.ABCMeta):

    name: Optional[str]

    @classmethod
    def func_name(cls)->str:
        return cls.__name__
        pass
        
    def params(self)->Union[ValType, tuple[ValType]]:
        pass

    @classmethod
    def para_names(self)->list[str]:
        pass

    @property
    def para_len(self)->int:
        params = self.params()
        if type(params) == tuple:
            return len(params)
        elif isinstance(params, ValueWithRange):
            return 1
        else:
            raise TypeError("The type of parameter is invalid: {}".format(type(params)))

    
    def calc(self, x: ndarray_like)->ndarray_like:
        pass

    def __call__(self, x: ndarray_like)->ndarray_like:
        return self.calc(x)
    
    """def show_with_err(self, error: Self)->None:
        pass"""

    """def _print_params_and_errs(
        self, 
        name: str, 
        para_names: list[str],
        errs: Self)->None:
        
        print("function: "+name)
        try:
            for i, para in enumerate(self.params()):
                print("{}: {:.3e}".format(para_names[i], para)+" ± " + "{:.2e}".format(errs.params()[i]))
        except(TypeError): #paramsが一個の時、返り値がiterableでない
            print("{}: {:.3e}".format(para_names[0],self.params())+" ± " + "{:.2e}".format(errs.params()))

        print("--")
        return"""
    
    def print_params(self):
        if self.name == "":
            print("function: " + self.func_name())
        else:
            print("function: " + self.name)
        try:
            for i, para in enumerate(self.params()):
                print(self.para_names()[i]+": ", end="")
                para.print()
        except(TypeError): #paramsが一個の時、返り値がiterableでない
            para = self.params()
            print(self.para_names()[0]+": ", end="")
            para.print()
        print("--")


    
    def to_result_type(self):
        return Self[ValueWithErrror]

def to_float_tuples(funcs: list[FittingFunctionElement])->list[tuple[ValType]]:
    params: list[tuple[ValType]] = []
    for func in funcs:
        if not hasattr(func.params(), "__iter__"):
            params.append(func.params())
        else:
            for param in func.params():
                params.append(param)
    return params



class TotalFunc(Generic[ValType]):
    func_components: list[FittingFunctionElement[ValType]]
    #minimum_bounds: Optional[list[FittingFunctionElement]]
    #maximum_bounds: Optional[list[FittingFunctionElement]]

    def __init__(
            self, 
            func_components: list[FittingFunctionElement],
            #minimum_bounds: Optional[list[FittingFunctionElement]] = None,
            #maximum_bounds: Optional[list[FittingFunctionElement]] = None
    ):
        self.func_components = func_components
        #self.minimum_bounds = minimum_bounds
        #self.maximum_bounds = maximum_bounds


    def calc(self, x: np.ndarray):
        sum = 0.0
        for func in self.func_components:
            sum += func(x)
        return sum
    
    def __call__(self, x: ndarray_like)->ndarray_like:
        return self.calc(x)
    
    
    def parameter_to_tuples(self)->list[tuple[float]]:
        params: list[tuple[ValType]] = []
        for func in self.func_components:
            if not hasattr(func.params(), "__iter__"):
                params.append(func.params().value)
            else:
                for param in func.params():
                    params.append(param.value)
        return params
        pass

    def print(self):
        for func in self.func_components:
            func.print_params()

        

class FittingParameters(TotalFunc[ValueWithBounds]):
    def fit_func(self):

        def func(x, *params):
            
            sum :np.ndarray =0.0
            para_index = 0
            #params = to_float_list(self.fitted_funcs)

            for i, func in enumerate(self.func_components):
                sliced_params = []
                for j in range(0, func.para_len):
                    sliced_params.append(ValueWithBounds(params[para_index]))
                    para_index+=1
                #print(x)        
                sum += type(func)(func.name, *sliced_params)(np.array(x))
            return sum
    
        return func

    def minimum_bounds_to_tuples(self)->list[tuple[Optional[float]]]:
            params: list[tuple[ValueWithBounds]] = []
            for func in self.func_components:
                if not hasattr(func.params(), "__iter__"):
                    params.append(func.params().least)
                else:
                    for param in func.params():
                        
                        params.append(param.least)
            return params
    
    def maximum_bounds_to_tuples(self)->list[tuple[Optional[float]]]:
            params: list[tuple[ValueWithBounds]] = []
            for func in self.func_components:
                if not hasattr(func.params(), "__iter__"):
                    params.append(func.params().most)
                else:
                    for param in func.params():
                        
                        params.append(param.most)
            return params

    """minimum_bounds: Optional[list[FittingFunctionElement]]
    maximum_bounds: Optional[list[FittingFunctionElement]]

    def __init__(
            self, 
            func_components: list[FittingFunctionElement],
            minimum_bounds: Optional[list[FittingFunctionElement]] = None,
            maximum_bounds: Optional[list[FittingFunctionElement]] = None
    ):
        self.func_components = func_components
        self.minimum_bounds = minimum_bounds
        self.maximum_bounds = maximum_bounds"""

class FittingResult(TotalFunc[ValueWithErrror]):

    """error: list[FittingFunctionElement]

    def __init__(self,
        func_components: list[FittingFunctionElement],
        error: list[FittingFunctionElement]):

        self.func_components = func_components
        self.error = error
        pass"""
    

def fitting(
        x, 
        y, 
        func_and_initial_params: FittingParameters,
        fitting_method = "lm",
        sigma = 1,
        )->FittingResult:
    """
    
    """
    
    params: list[tuple[float]] = []
    #func_list = list[Callable]
    
    for func in func_and_initial_params.func_components:
        if not hasattr(func.params(), "__iter__"):
            params.append(func.params())
        else:
            for param in func.params():
                params.append(param)
        #func_list.append(func.calc)
    
    """params = to_float_tuples(func_and_initial_params.func_components)
    if func_and_initial_params.minimum_bounds is None:
        minimums = None
    else:
        minimums = to_float_tuples(func_and_initial_params.minimum_bounds)
    
    if func_and_initial_params.maximum_bounds is None:
        maximums = None
    else:
        maximums = to_float_tuples(func_and_initial_params.maximum_bounds)"""
    
    params = func_and_initial_params.parameter_to_tuples()

    max_bounds = func_and_initial_params.maximum_bounds_to_tuples()
    min_bounds = func_and_initial_params.minimum_bounds_to_tuples()
    #print(max_bounds, min_bounds)
    """def total_fit_func(x, *params):
        #print(params)
        sum :np.ndarray =0.0
        para_index = 0
        for i, func in enumerate(func_and_initial_params.fitted_funcs):
            #print(params)
            sliced_params = []
            for j in range(0, func.para_len):
                sliced_params.append(params[para_index])
                para_index+=1
            #print(x)        
            sum += type(func)(*sliced_params)(np.array(x))
        return sum"""
    
    #print(params)
    
    #ここでフィッティング
    opt_para, covariance, infodict, mesg, ier = opt.curve_fit(
        func_and_initial_params.fit_func(),
        x, 
        y, 
        bounds=(min_bounds, max_bounds),
        p0 = params, 
        maxfev = 100000,
        method=fitting_method,
        full_output=True,
        sigma=sigma
        )
    err = np.sqrt(np.diag(covariance)) #謎の行列から標準誤差取得（公式ドキュメントのいうがまま）

    opt_func_list = []
    #err_list = []
    para_index = 0
    opt_res: list[ValueWithErrror] = []

    for i, _ in enumerate(opt_para):
        opt_res.append(ValueWithErrror(opt_para[i], err[i]))
    
    for i, func in enumerate(func_and_initial_params.func_components):
        func_type = type(func)[ValueWithErrror]
        
        """opt_para_sliced = opt_para[para_index:para_index+func.para_len]
        err_sliced = err[para_index:para_index+func.para_len]"""

        opt_res_sliced = opt_res[para_index:para_index+func.para_len]

        #print(opt_para_sliced)
        """opt_func_list.append(func_type(*opt_para_sliced))
        err_list.append(func_type(*err_sliced))"""
        opt_func_list.append(func_type(func.name, *opt_res_sliced))

        para_index += func.para_len
    
    fit_res = FittingResult(opt_func_list)
    #print(fit_res)

    print("nev = {}".format(infodict["nfev"]))
    print("fvec = {}".format(infodict["fvec"][-1]))
          
    return fit_res

#--------------------------------peak functions--------------------------------

@dataclass(frozen=True)
class PeakFunction(FittingFunctionElement[ValType], metaclass=abc.ABCMeta):
    _center: ValType
    @property
    def center(self):
        return self._center
    
    _width: ValType
    @property
    def width(self):
        return self._width
    
    _height: ValType
    @property
    def height(self):
        return self._height
    
    
    def params(self):
        return (self._center, self.width, self._height)

    def para_names(self):
        return ["peak center", "FWHM", "height"]

    
    def calc(self, x: ndarray_like)->ndarray_like:
        pass
    

    pass

class LorentzPeak(PeakFunction[ValType]):
    """def show_with_err(self, error: Self) -> None:
        func_name = "Lorentzian peak"
        para_names = ["center", "width", "amplitude"]


        self._print_params_and_errs(func_name, para_names, error)
        return"""
        
    def calc(self, x: ndarray_like)->ndarray_like:
        return 1/np.pi * self._width.value**2/((x-self._center.value)**2 + self._width.value**2) * self.height.value
    
class GaussianPeak(PeakFunction[ValType]):
    """def show_with_err(self, error: Self) -> None:
        func_name = "Gaussian peak"
        para_names = ["center", "width", "amplitude"]


        self._print_params_and_errs(func_name, para_names, error)"""
    def calc(self, x: ndarray_like)->ndarray_like:
        
        sigma = self._width.value /2  / np.sqrt(2*np.log(2))
        ampletude = self.height.value 

        #return np.exp(-(x - self._center.value)**2 / 2/sigma**2)/ np.sqrt(2*np.pi) / sigma * self._ampletude.value
        return np.exp(-(x - self._center.value)**2 / 2/sigma**2)*ampletude
@dataclass(frozen=True)
class GL_MixedPeak(PeakFunction[ValType]):

    _GL_ratio: ValType
    @property
    def GL_ratio(self):
        return self._GL_ratio

    def params(self):
        return (self._center, self._width, self._ampletude, self._GL_ratio)

    def para_names(self):
        return ["peak center", "FWHM", "area", "Gaussian-Lorentzian ratio"]

    def calc(self, x: ndarray_like):
        return self._GL_ratio.value*GaussianPeak("", self._center, self._width, self._ampletude).calc(x) + \
            (1-self._GL_ratio.value)*LorentzPeak("", self._center, self._width, self._ampletude).calc(x)


#--------------------------------linear and poly--------------------------------

@dataclass(frozen=True)
class Constant(FittingFunctionElement[ValType]):
    value: ValType

    """def show_with_err(self, error: Self) -> None:
        func_name = "Constant"
        para_names = ["constant"]


        self._print_params_and_errs(func_name, para_names, error)
        return"""
    
    def params(self):
        return (self.value)
    
    def para_names(self):
        return ["constant"]

    def calc(self, x: cmn.ValueObjectArray) -> cmn.ValueObjectArray:
        return self.value.value
    

@dataclass(frozen=True)
class Linear(FittingFunctionElement[ValType]):
    slope: ValType
    #y_intercept: ValType
    """def show_with_err(self, error: Self) -> None:
        func_name = "Linear"
        para_names = ["slope"]


        self._print_params_and_errs(func_name, para_names, error)
        return"""
    
    def para_names(self):
        return ["slope"]

    def params(self):
        return (self.slope)

    def calc(self, x: cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.slope.value * x
    
#--------------------------------exponentials--------------------------------

@dataclass(frozen = True)
class Exponential(FittingFunctionElement[ValType]):
    tau: ValType
    amplitude : ValType

    """def show_with_err(self, error: Self) -> None:
        func_name = "Exponential"
        para_names = ["tau", "amplitude"]


        self._print_params_and_errs(func_name, para_names, error)

        return"""
    def params(self):
        return (self.tau, self.amplitude)
    
    def calc(self, x: cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.amplitude.value * np.e ** ( x * -1 / self.tau.value)
    
    def para_names(self):
        return ["tau", "amplitude"]
    

#-----------------------------------sigmoids--------------------------------
@dataclass(frozen=True)
class Sigmoid(FittingFunctionElement[ValType]):
    center: ValType
    gain: ValType
    hieght: ValType

    
    def params(self):
        return (self.center, self.gain, self.hieght)
    
    def calc(self, x:cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.hieght.value * (np.tanh(self.gain.value * x /2) + 1)/2
    
    def para_names(self):
        return ["center", "gain", "hieght"]
     
#------------------------------ocilating functions--------------------------
@dataclass(frozen = True)
class OscilatedExponential(FittingFunctionElement[ValType]):
    
    
    _amplitude: ValType
    @property
    def amplitude(self):
        return self._amplitude
    
    _tau: ValType
    @property
    def tau(self):
        return self._tau

    _period: ValType
    @property
    def period(self):
        return self._period

    _initial_phase: ValType
    @property
    def initial_phase(self):
        return self._initial_phase
    
    
    def params(self):
        return (self.amplitude, self.tau, self.period, self.initial_phase)

    def calc(self, x: ndarray_like):
        return self.amplitude.value * np.sin(2*np.pi*x/self.period.value - self.initial_phase.value) * np.e ** ( x * -1 / self.tau.value)

    def para_names(self):
        return ["amplituide", "tau", "period", "initial_phase"]
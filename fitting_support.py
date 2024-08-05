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

from typing import Union,NewType, Optional
import abc

import BFC_libs.common as cmn

float_like = Union[int, float, np.float64, cmn.ValueObject]
ndarray_like = Union[list[float], np.ndarray, cmn.ValueObjectArray]
params = list[float]


class FittingFunctionElement(metaclass=abc.ABCMeta):
    
    def params(self)->tuple[float_like]:
        pass
    
    @property
    def para_len(self)->int:
        params = self.params()
        if type(params) == tuple:
            return len(params)
        elif isinstance(params, float_like):
            return 1
        else:
            raise TypeError("The type of parameter is invalid: {}".format(type(params)))

    
    def calc(self, x: ndarray_like)->ndarray_like:
        pass

    def __call__(self, x: ndarray_like)->ndarray_like:
        return self.calc(x)
    
    pass


def to_float_tuples(funcs: list[FittingFunctionElement])->list[tuple[float_like]]:
    params: list[tuple[float]] = []
    for func in funcs:
        if not hasattr(func.params(), "__iter__"):
            params.append(func.params())
        else:
            for param in func.params():
                params.append(param)
    return params


class TotalFunc:
    func_components: list[FittingFunctionElement]
    minimum_bounds: Optional[list[FittingFunctionElement]]
    maximum_bounds: Optional[list[FittingFunctionElement]]

    def __init__(
            self, 
            func_components: list[FittingFunctionElement],
            minimum_bounds: Optional[list[FittingFunctionElement]] = None,
            maximum_bounds: Optional[list[FittingFunctionElement]] = None
    ):
        self.func_components = func_components
        self.minimum_bounds = minimum_bounds
        self.maximum_bounds = maximum_bounds


    def calc(self, x: np.ndarray):
        sum = 0.0
        for func in self.func_components:
            sum += func(x)
        return sum
    
    def __call__(self, x: ndarray_like)->ndarray_like:
        return self.calc(x)
    
    
    
    def fit_func(self):

        def func(x, *params):
            
            sum :np.ndarray =0.0
            para_index = 0
            #params = to_float_list(self.fitted_funcs)

            for i, func in enumerate(self.func_components):
                sliced_params = []
                for j in range(0, func.para_len):
                    sliced_params.append(params[para_index])
                    para_index+=1
                #print(x)        
                sum += type(func)(*sliced_params)(np.array(x))
            return sum
    
        return func
   

FittingResult = NewType("FittingResult", TotalFunc)


def fitting(
        x, 
        y, 
        func_and_initial_params: TotalFunc
        )->tuple[FittingResult, list[FittingFunctionElement]]:
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
    
    params = to_float_tuples(func_and_initial_params.func_components)
    if func_and_initial_params.minimum_bounds is None:
        minimums = None
    else:
        minimums = to_float_tuples(func_and_initial_params.minimum_bounds)
    
    if func_and_initial_params.maximum_bounds is None:
        maximums = None
    else:
        maximums = to_float_tuples(func_and_initial_params.maximum_bounds)
    
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
    opt_para, covariance = opt.curve_fit(
        func_and_initial_params.fit_func(),
        x, 
        y, 
        bounds=(minimums, maximums),
        p0 = params, 
        maxfev = 10000
        )

    opt_func_list = []
    para_index = 0
    for i, func in enumerate(func_and_initial_params.func_components):
        func_type = type(func)
        opt_para_sliced = opt_para[para_index:para_index+func.para_len]
        #print(opt_para_sliced)
        opt_func_list.append(func_type(*opt_para_sliced))
        para_index += func.para_len
    
    fit_res = TotalFunc(opt_func_list, func_and_initial_params.minimum_bounds, func_and_initial_params.maximum_bounds)
    #print(fit_res)

    err = np.sqrt(np.diag(covariance))

    

    return (fit_res, err)

#--------------------------------peak functions--------------------------------

@dataclass(frozen=True)
class PeakFunction(FittingFunctionElement, metaclass=abc.ABCMeta):
    _center: float_like
    @property
    def center(self):
        return self._center
    
    _width: float_like
    @property
    def width(self):
        return self._width
    
    _ampletude: float_like
    @property
    def ampletude(self):
        return self._ampletude
    
    
    def params(self):
        return (self._center, self.width, self._ampletude)
    
    def calc(self, x: ndarray_like)->ndarray_like:
        pass
    
    pass

class LorentzPeak(PeakFunction):
    def calc(self, x: ndarray_like)->ndarray_like:
        return 1/np.pi * self._width/((x-self._center)**2 + self._width**2) * self._ampletude
    
class GaussianPeak(PeakFunction):
    def calc(self, x: ndarray_like)->ndarray_like:
        sigma = self._width /2  / np.sqrt(2*np.log(2))
        return np.exp(-(x - self._center)**2 / 2/sigma**2)/ np.sqrt(2*np.pi) / sigma * self._ampletude

#--------------------------------linear and poly--------------------------------

@dataclass(frozen=True)
class Constant(FittingFunctionElement):
    value: float_like


    def params(self):
        return (self.value)

    def calc(self, x: cmn.ValueObjectArray) -> cmn.ValueObjectArray:
        return self.value

@dataclass(frozen=True)
class Linear(FittingFunctionElement):
    slope: float_like
    #y_intercept: float_like


    def params(self):
        return (self. slope)

    def calc(self, x: cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.slope * x
    
#--------------------------------exponentials--------------------------------

@dataclass(frozen = True)
class Exponential(FittingFunctionElement):
    tau: float_like
    amplitude : float_like

    
    def params(self):
        return (self.tau, self.amplitude)
    
    def calc(self, x: cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.amplitude * np.e ** ( x * -1 / self.tau)
    

#-----------------------------------sigmoids--------------------------------
@dataclass(frozen=True)
class Sigmoid(FittingFunctionElement):
    center: float_like
    gain: float_like
    hieght: float_like

    
    def params(self):
        return (self.center, self.gain, self.hieght)
    
    def calc(self, x:cmn.ValueObjectArray)->cmn.ValueObjectArray:
        return self.hieght * (np.tanh(self.gain * x /2) + 1)/2
    

"""
### BFC_libs.common
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from copy import deepcopy as copy
from functools import wraps
import codecs


try:
    from .quantity import Quantity
    from .quantity_array import QArray
except ImportError:
    # リポジトリ直下から直接 import する場合(パッケージ外実行)
    from quantity import Quantity
    from quantity_array import QArray



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
# データ階層クラス(DataSeriese / DataArray / DataFile)は data_object.py に
# 分離した。従来 common に属していたが、汎用関数群と役割が異なるため。


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_tupple(self):
        return (self.x, self.y)


#------------------------------------------------------
#---------------common object values-------------------
#------------------------------------------------------	
        
class Time(Quantity):
    """時間 [s]。新 Quantity 系(実体 float)の物理量。"""
    pass

TimeArray = QArray[Time]

#------------------------------------------------------
#-------------------functions--------------------------
#------------------------------------------------------


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

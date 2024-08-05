import pandas as pd
import numpy as np
from os.path import dirname, basename
#from copy import deepcopy as copy

from dataclasses import dataclass
from typing import Optional, Self
from enum import Enum

from .. import common as cmn
from .. import XAS as xas

class FileContentError(BaseException):
    pass

@dataclass(frozen=True)
class AthenaProjectEXAFS(cmn.DataFile[xas.InversedEXAFS]):
    pass

def load_athena_EXAFS_file(chir_mag_file_path: str):
    """
    .chir_magファイルは読み込める前提。
    他のファイルは無くても機能する。
    """
    class ChiRVals(Enum):
        MAGNITUDE=0
        REAL = 1
        IMAGINARY = 2
        PHASE = 3

    dir = dirname(chir_mag_file_path)
    file_name_base = cmn.extract_filename(basename(chir_mag_file_path)).split('.')[-2]

    skip_row_num = cmn.find_line_with_key(chir_mag_file_path, "#------------------------")

    with open(chir_mag_file_path, 'r', encoding='UTF-8') as file:
        raw_txt_lines = file.readlines()
        columns = raw_txt_lines[skip_row_num].replace("  ", " ").split(" ")[1:] # "#"だけの列名を消す
        skipped_lines = raw_txt_lines[0:skip_row_num]
    
    #print(columns)
    dfs:dict[ChiRVals, Optional[pd.DataFrame]] = {}

    dfs[ChiRVals.MAGNITUDE] = pd.read_csv(
        chir_mag_file_path, 
        delim_whitespace=True, 
        skiprows=skip_row_num+1,
        names=columns)

    #同名拡張子違いのファイルがあると仮定して読み込みを試行
    #失敗すれば、対応するDataFrameをNoneに
    #ただし、現時点では読み込み中のエラーはキャッチせず、そのまま上げる
    try:
        dfs[ChiRVals.REAL] = \
            pd.read_csv(dir+"/"+file_name_base+ ".chir_re", delim_whitespace=True, skiprows=skip_row_num+1, names=columns)
    except(FileNotFoundError):
        dfs[ChiRVals.REAL] = None

    try:
        dfs[ChiRVals.IMAGINARY] = \
            pd.read_csv(dir+"/"+file_name_base+ ".chir_im", delim_whitespace=True, skiprows=skip_row_num+1, names=columns)
    except(FileNotFoundError):
        dfs[ChiRVals.IMAGINARY] = None

    try:
        dfs[ChiRVals.PHASE] = \
            pd.read_csv(dir+"/"+file_name_base+ ".chir_pha", delim_whitespace=True, skiprows=skip_row_num+1, names=columns)
    except(FileNotFoundError):
        dfs[ChiRVals.PHASE] = None

    # R行が一致しない場合に例外送出
    for chir_val in ChiRVals:
        # magnitude以外に対して、magnitudeのdfとR列が一致するか検査。
        if chir_val == ChiRVals.MAGNITUDE:
            continue

        
        if isinstance(dfs[chir_val], pd.DataFrame):
            #ここに入ったならNoneじゃない
            if (len(dfs[chir_val]["r"].values) != len(dfs[ChiRVals.MAGNITUDE]["r"].values)) or \
                any(x != y for x,y in zip(dfs[chir_val]["r"].values, dfs[ChiRVals.MAGNITUDE]["r"].values)):
                raise FileContentError(f"A ChiR file was found but r column is not the same. \
                                    Check the column content and length.\n\
                                    The error was raised while loading {chir_val}")
        
    r_array = dfs[ChiRVals.MAGNITUDE]["r"].values
    data_names = dfs[ChiRVals.MAGNITUDE].columns.drop("r")

    data_array:list[xas.InversedEXAFS] = []
    for data_name in data_names:
        data_array.append(
            xas.InversedEXAFS(
                _comment = [],
                _condition = [],
                _original_file_path = chir_mag_file_path,
                _data_name = data_name,
                _distance = xas.DistanceArray(r_array),
                _chir_magnitude = xas.ChiR_MagArray(dfs[ChiRVals.MAGNITUDE][data_name].values),
                _chir_real = (xas.ChiR_ReArray(dfs[ChiRVals.REAL][data_name].values) \
                            if dfs[ChiRVals.REAL]is not None else None),
                _chir_imaginary = (xas.ChiR_ImArray(dfs[ChiRVals.IMAGINARY][data_name].values) \
                            if dfs[ChiRVals.IMAGINARY]is not None else None),
                _chir_phase = (xas.ChiR_PhaseArray(dfs[ChiRVals.PHASE][data_name].values) \
                            if dfs[ChiRVals.PHASE]is not None else None),
                
            )
        )
    
    return AthenaProjectEXAFS(
        _data = cmn.DataArray(data_array),
        _comment = [],
        _condition = [],
        _file_path = chir_mag_file_path,
        _data_name = file_name_base
    )
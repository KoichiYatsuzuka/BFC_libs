import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy as copy

from typing import Optional, Self, NewType
from dataclasses import dataclass
from enum import Enum

from ... import common as cmn
from ... import photoelectron as pes

class VasaProbeData(cmn.DataFile[pes.PhotoelectronSpectrum]):

    
    
    @classmethod
    def load_file(cls, file_path)->Self:
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()

        class FileLoadStatus:
            SKIP = 0
            DATA = 1
        
        file_load_status: FileLoadStatus = FileLoadStatus.SKIP # mutable

        data:list[pes.PhotoelectronSpectrum] = []
        tmp_array:tuple[list[float]] = ([], []) #何度か初期化する。tmp_array[0]がx、tmp_array[1]がy　namedtupleがいいか？
        tmp_condition: list[str] = [] #何度か初期化する

        for line in lines:
            match(file_load_status):
                case FileLoadStatus.SKIP:
                    tmp_condition.append(line)
                    if line.find("file#") != -1:
                        file_load_status = FileLoadStatus.DATA
                    continue
                
                case FileLoadStatus.DATA:
                    
                    #空行の時はデータの境目なので、データを整理して抜ける
                    if line == "\n":
                        file_load_status = FileLoadStatus.SKIP
                        data.append(pes.PhotoelectronSpectrum(
                            _comment = ["photoelectron enrgy: in binding energy"],
                            _condition = copy(tmp_condition),
                            _original_file_path = file_path,
                            _data_name = tmp_condition[2].replace("\n", ""), # C1sとか書いてある行
                            _photoelectron_energy = pes.PhotoelectronEnergyArray(tmp_array[0]),
                            _photoelectron_intensity = pes.PhotoelectronIntensityArray(tmp_array[1])
                        ))
                        # 一時変数初期化
                        tmp_condition = []
                        tmp_array = ([], [])
                        continue

                    #以降、データ処理

                    #実はここ以降安全ではない
                    #本当はtry-catch節がいる    
                    values_str = line.split(',')
                
                    x = float(values_str[0]) #キャストできるかの保証とindex範囲内の保証がない
                    y = float(values_str[1])
                    if x != 0.0 or y != 0.0: # なぜか全部0のデータが入るのでそれをskip
                        tmp_array[0].append(x)
                        tmp_array[1].append(y)
                    else: #両方とも0.0の時
                        continue
        
        return VasaProbeData(
            data, 
            _comment = ["file: {}".format(file_path)], 
            _condition = tmp_condition, 
            _file_path = file_path, 
            _data_name = cmn.extract_filename(file_path)
            )

    pass
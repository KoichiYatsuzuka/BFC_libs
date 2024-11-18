import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Optional, Self, Final
from dataclasses import dataclass

from ... import common as cmn
from ... import photoelectron as pe

#めんどかったので積み
"""class JEOL_XPS_DataFile(pe.PhotoelectronSpectrum):

    @classmethod
    def load_file(cls, file_path: str):
        skip_line_num = 18 #ファイルに依らない？
    
        file = open(file_path, 'r', encoding='Shift-JIS')
        conditions = file.readlines()[0:skip_line_num]
        file.close()
        
        df = pd.read_csv(file_path, sep=",", skiprows=skip_line_num)

        column_names=df.keys()

        bindingenergy_column_names = column_names[::2]
        spectrum_column_names = column_names[1::2]

        spectra_tmp: list[pe.PhotoelectronSpectrum] = []
        for i in range(len(bindingenergy_column_names)-1):
            spectra_tmp.append(
                pe.PhotoelectronSpectrum(
                    _comment = [spectrum_column_names[i]],
                    _condition = conditions,
                    _original_file_path = file_path, 
                    _data_name = spectrum_column_names[i],
                    _photoelectron_energy = pe.PhotoelectronEnergyArray(df[[i]].values),
                    _photoelectron_intensity= pe.PhotoelectronIntensityArray(df[df[i]].values)
                )
            )"""
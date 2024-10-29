import numpy as np
import pandas as pd
from typing import overload, NewType
from dataclasses import dataclass
from copy import deepcopy as copy
import codecs

from .. import common as cmn
from .. import Raman as rmn
from .. Raman import RamanSpectrum


class Jasco1DRamanDataFile(cmn.DataFile[rmn.RamanSpectrum]):

    @classmethod
    def load_file(cls, file_path: str):
        skip_line_num = \
            cmn.find_line_with_key(file_path, "XYDATA")+1
        
        stop_line_num = \
            cmn.find_line_with_key(file_path, "##### Extended Information")-1

        file = codecs.open(file_path, 'r', encoding='UTF-8', errors='ignore')
        file_lines = file.readlines()
        conditions = file_lines[0:skip_line_num]
        file.close()
        
        df = pd.read_csv(file_path, sep=",", skiprows=skip_line_num, skipfooter=len(file_lines)-stop_line_num, encoding_errors='ignore', header=None, names=[rmn.COLUMN_NAME_WAVENUMBER, rmn.COLUMN_NAME_INTENSITY], engine='python')


        spectrum = RamanSpectrum(
            _comment = conditions[0][6:],
            _condition = conditions[1:],
            _original_file_path = file_path, 
            _data_name = conditions[0][6:],
            _wavenumber = rmn.WavenumberArray(df[rmn.COLUMN_NAME_WAVENUMBER].values),
            _intensity= rmn.RammanIntensityArray(df[rmn.COLUMN_NAME_INTENSITY].values)
        )

        return cls(
            _comment = ["Raw data file: "+cmn.extract_filename(file_path)],
            _condition = conditions[1:],
            _file_path = file_path,
            _data_name = cmn.extract_filename(file_path),
            _data = cmn.DataArray[RamanSpectrum]([spectrum], RamanSpectrum, file_path)
        )
    




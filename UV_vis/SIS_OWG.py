from .. import common as cmn
from .. import data_object as do
from dataclasses import dataclass
from typing import NewType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from copy import deepcopy as copy
from typing import Union, Optional, Final, Self
from .. import UV_vis

SKIPPING_ROWS_NUM: Final[int] = 11

#そのうち
@dataclass(frozen=True)
class SIS_OWG_Spectra(do.DataFile[UV_vis.UV_VisSpectrum]):

    
    @classmethod
    def load_data(
        cls, 
        file_path: str, 
        wavelength_range: tuple[float, float] = (350, 750)
        )->Self:
        
        file = open(file_path, 'r', encoding='Shift-JIS')
        header = file.readlines()[0:SKIPPING_ROWS_NUM]
        file.close()

        raw_df = pd.read_csv(
            file_path,
            
            skiprows= SKIPPING_ROWS_NUM,
            header=None,
            quoting=csv.QUOTE_NONE,
            sep="\t",
            na_values=["       NaN"],

            names=["wavelength", "absorption"],
            encoding='Shift-JIS'
        )

        nan_deleted_df = raw_df.fillna(0.0)


        data = UV_vis.UV_VisSpectrum(
            _comment=[],
            _condition = header,
            _original_file_path = file_path,
            _data_name = "",
            _wavelength=UV_vis.WavelengthArray(nan_deleted_df["wavelength"]),
            _absorption = UV_vis.AbsorptionArray(nan_deleted_df["absorption"])
        ).slice(wavelength_range[0], wavelength_range[1])



        tmp = SIS_OWG_Spectra(
            do.DataArray[UV_vis.UV_VisSpectrum](
                [data], UV_vis.UV_VisSpectrum
            ),
            _comment = [],
            _condition = [],
            _file_path = "",
            _data_name = ""
        )

        return tmp

    @classmethod
    def load_data_series(
        cls, 
        path_with_format: str, 
        file_number_start_end_step: tuple[int, int, int], 
        wavelength_range: tuple[float, float] = (350, 750)
        )->Self:
        
        l: list[UV_vis.UV_VisSpectrum] = []
        
        for i in range(file_number_start_end_step[0], file_number_start_end_step[1]+1, file_number_start_end_step[2]):
            l.append(cls.load_data(path_with_format.format(i), wavelength_range)[0])



        tmp = SIS_OWG_Spectra(
            do.DataArray[UV_vis.UV_VisSpectrum](l, UV_vis.UV_VisSpectrum),
            _comment = [],
            _condition = [],
            _file_path = path_with_format.format(0),
            _data_name = ""
        )

        return tmp

    def to_df(self):
        wavelength = np.array([self[0]._wavelength.float_array()])

        column_name_list:list[str] = ["Wavelength [nm]"]
        spectra_list:list[np.ndarray] = []
        
        for i, spc in enumerate(self.data):
            spectra_list.append(spc.absorption.float_array())
            column_name_list.append(spc.data_name)
        
        spectra_array = np.array(spectra_list)

        merged_array = np.append(wavelength, spectra_array, axis=0).T

        df = pd.DataFrame(
            merged_array,
            columns=column_name_list
        )

        return df

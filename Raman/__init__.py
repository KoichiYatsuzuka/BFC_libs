"""
# Library for files from Nanophoton Raman microscope
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Self
from dataclasses import dataclass

from .. import common as cmn
from .. common import ValueObject, ValueObjectArray

class Wavenumber(cmn.ValueObject):
    pass

class WavenumberArray(ValueObjectArray):
    def __new__(cls, obj, dtype=Wavenumber, meta: Optional[str] = None):
        
        return super().__new__(cls, obj, dtype, meta)
    pass

class RammanIntensity(ValueObject):
    pass

class RammanIntensityArray(ValueObjectArray):
    def __new__(cls, obj, dtype=RammanIntensity, meta: Optional[str] = None):
        
        return super().__new__(cls, obj, dtype, meta)
    pass


@dataclass(frozen=True, repr=False)
class RamanSpectrum(cmn.DataSeriese[Wavenumber, RammanIntensity]):
    _wavenumber: WavenumberArray
    _intensity: RammanIntensityArray
    
    @property
    def wavenumber(self):
        return self._wavenumber
    
    @property
    def intensity(self):
        return self._intensity
    
    def __hash__(self) -> int:
        first_part_of_intensity = self._intensity[:10]
        hash_str = ""
        for current in first_part_of_intensity:
            tmp = str(current)
            tmp=tmp.replace(".", "")
            print(tmp)
            hash_str += tmp

        return int(hash_str)
    
    @property
    def x(self):
        return self._wavenumber
    
    @property
    def y(self):
        return self._intensity
    
    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.stack([
                self._wavenumber.float_array(),
                self._intensity.float_array()
            ], 1),
            columns = [
                "wavenumber",
                "intensity"
            ]
        )
    
    @classmethod
    def from_data_frame(
        cls, 
        df: pd.DataFrame, 
        comment: list[str] = [],
        condition: list[str] = [],
        original_file_path: str = "",
        ) -> Self:
        return RamanSpectrum(
            _comment = comment,
            _data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
            _condition = condition,
            _original_file_path = original_file_path,
            _wavenumber = WavenumberArray(df["wavenumber"]),
            _intensity = RammanIntensityArray(df["intensity"])
        )

    
    def plot(self, fig: Optional[plt.Figure]=None, ax: Optional[plt.Axes]=None, **kargs)->tuple[plt.Figure, plt.Axes]:
        (_fig, _ax) = super().plot(fig, ax, **kargs)
        if (ax == None):
            _ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            _ax.set_ylabel("Intensity (a.u.)")

        return (_fig, _ax)


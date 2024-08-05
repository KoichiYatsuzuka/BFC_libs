import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Optional, Self, NewType, Final
from dataclasses import dataclass

from .. import common as cmn




class PhotoelectronEnergy(cmn.ValueObject):
    pass

class PhotoelectronEnergyArray(cmn.ValueObjectArray[PhotoelectronEnergy]):
    def __new__(cls, obj, dtype=PhotoelectronEnergy, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)

class PhotoelectronIntensity(cmn.ValueObject):
    pass
class PhotoelectronIntensityArray(cmn.ValueObjectArray[PhotoelectronEnergy]):
    def __new__(cls, obj, dtype=PhotoelectronIntensity, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)

@dataclass(frozen=True, repr=False)
class PhotoelectronSpectrum(cmn.DataSeriese[PhotoelectronEnergy, PhotoelectronIntensity]):
    _photoelectron_energy: PhotoelectronEnergyArray
    _photoelectron_intensity: PhotoelectronIntensityArray

    @property
    def x(self):
        return self._photoelectron_energy
    
    @property
    def y(self):
        return self._photoelectron_intensity
    
    def to_data_frame(self):
        return pd.DataFrame(
            np.stack([
                self._photoelectron_energy.float_array(),
                self._photoelectron_intensity.float_array()
            ], 1),
            columns = [
                "photoelectron energy",
                "photoelectron intensity"
            ]
        )
        
        pass

    @classmethod
    def from_data_frame(
        cls, 
        df: pd.DataFrame, 
        comment: list[str] = ..., 
        condition: list[str] = ..., 
        original_file_path: str = ""
        ) -> Self:


        return PhotoelectronSpectrum(
            _comment = [comment],
            _condition = [condition],
            _original_file_path = original_file_path,
            _data_name = cmn.extract_filename(original_file_path).split(".")[0],
            _photoelectron_energy = PhotoelectronEnergyArray(df["photoelectron energy"].values),
            _photoelectron_intensity = PhotoelectronIntensityArray(df["photoelectron intensity"].values)
        )
        
    def plot(
            self,
            fig: Optional[plt.Figure]=None, 
            ax: Optional[plt.Axes]=None, 
            **kargs
		):
        fig, ax = super().plot()
        ax.set_xlim(self.x.max().value, self.x.min().value)
        





XPSpectrum = NewType("XPSpectrum", PhotoelectronSpectrum)

UPSpectrum = NewType("UPSpectrum", PhotoelectronSpectrum)

HELIUM_UV_ENERGY: Final[PhotoelectronEnergy] = PhotoelectronEnergy(21.22)

import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from copy import deepcopy as copy
from typing import Union, Optional, Self

from .. import common as cmn 
from typing import NewType

#----------------------------------------
#-------------classes--------------------
#----------------------------------------

class Theta(cmn.ValueObject):
	pass

class DiffractionIntensity(cmn.ValueObject):
	pass

#Theta_Array = NewType("Theta_Array", cmn.ValueObjectArray[Theta])
#Diffracton_Intensity_Array = NewType("Diffracton_Intensity_Array", cmn.ValueObjectArray[DiffractionIntensity])

class ThetaArray(cmn.ValueObjectArray[Theta]):
	def __new__(cls, obj, dtype=Theta, meta: Optional[str] = None):		
		return super().__new__(cls, obj, dtype, meta)
	pass

class DiffractionIntensityArray(cmn.ValueObjectArray[DiffractionIntensity]):
	def __new__(cls, obj, dtype=DiffractionIntensity, meta: Optional[str] = None):		
		return super().__new__(cls, obj, dtype, meta)
	pass



@dataclass(frozen=True)
class XRDPattern(cmn.DataSeriese[Theta, DiffractionIntensity]):
	_two_theta: ThetaArray
	_intensity: DiffractionIntensityArray

	@property
	def two_theta(self):
		return self._two_theta
	
	@property
	def intensity(self):
		return self._intensity
	
	@property
	def x(self):
		return self._two_theta
	
	@property
	def y(self):
		return self._intensity

	def to_data_frame(self) -> DataFrame:
		return pd.DataFrame(
			np.stack([
				self._two_theta.float_array(),
				self._intensity.float_array()
			], 1),
			columns = [
				"two theta",
				"intensity"
			]
		)
	
	@classmethod
	def from_data_frame(
		cls, 
		df: DataFrame, 
		comment: list[str] = [], 
		condition: list[str] = [], 
		original_file_path: str = "") -> Self:
		return XRDPattern(
			_comment = comment,
			_condition = condition,
			_original_file_path = original_file_path,
			_data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
			_two_theta = ThetaArray(df["two theta"].values),
			_intensity = DiffractionIntensityArray(df["intensity"].values)
		)
	
	def plot(self,
			fig: Optional[plt.Figure] = None, 
			ax: Optional[plt.Axes] = None,
			**kargs
		)->tuple[plt.Figure, plt.Axes]:
		
		_fig, _ax = super().plot(fig, ax, **kargs)

		ax.set_xlabel(r"2$theta$")
		ax.set_xlabel("Diffraction intensity")

		return (_fig, _ax)
	



	pass



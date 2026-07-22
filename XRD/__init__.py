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

class Theta(cmn.Quantity):
	"""回折角 2theta [deg]。"""
	pass

class DiffractionIntensity(cmn.Quantity):
	"""回折強度 [a.u.]。"""
	pass

ThetaArray = cmn.QArray[Theta]
DiffractionIntensityArray = cmn.QArray[DiffractionIntensity]



@dataclass(frozen=True)
class XRDPattern(cmn.DataSeriese[ThetaArray, DiffractionIntensityArray]):
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



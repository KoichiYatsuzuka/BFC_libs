from pandas import DataFrame
import pandas as pd
import numpy as np
from copy import deepcopy as copy
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Self
from typing import Union, Optional

from .. import common as cmn

class Absorption(cmn.Quantity):
	"""吸光度 [a.u.]。"""
	pass

class Wavelength(cmn.Quantity):
	"""波長 [nm]。"""
	pass

AbsorptionArray = cmn.QArray[Absorption]

WavelengthArray = cmn.QArray[Wavelength]


@dataclass(frozen=True)
class UV_VisSpectrum(cmn.DataSeriese[WavelengthArray, AbsorptionArray]):
	
	_wavelength: WavelengthArray
	@property
	def wavelength(self):
		return self._wavelength
	
	_absorption: AbsorptionArray
	@property
	def absorption(self):
		return self._absorption
	
	@property
	def x(self):
		return self._wavelength
	
	@property
	def y(self):
		return self._absorption
	
	@classmethod
	def from_data_frame(
		cls, 
		df: DataFrame, 
		comment: list[str] = ..., 
		condition: list[str] = ..., 
		original_file_path: str = ""
		) -> Self:


		return cls.__init__(
			_comment = comment,
			_condition = condition,
			_original_file_path = original_file_path,
			_data_name = cmn.extract_filename(original_file_path).split(".")[0:-1],
			_wavelength = WavelengthArray(df["wavelength"].values),
			_absorption = AbsorptionArray(df["absorption"].values)
		)
	
	def to_data_frame(self) -> DataFrame:
		return pd.DataFrame(
			np.stack([
				self._wavelength.float_array(),
				self._absorption.float_array()
			], 1),
			columns = [
				"wavelength",
				"absorption"
			]
		)
	
	def plot(
			self, 
			fig: Optional[plt.Figure] = None, 
			ax: Optional[plt.Axes] = None,
			**kwargs)->tuple[plt.Figure, plt.Axes]:
		
		_fig, _ax = super().plot(fig, ax)

		_ax.plot(
			self._wavelength,
			self._absorption,
			**kwargs
		)

		return (_fig, _ax)
	

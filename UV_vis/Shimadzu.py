from .. import common as cmn
from dataclasses import dataclass
from typing import NewType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from copy import deepcopy as copy
from typing import Union, Optional
from .. import UV_vis

@dataclass(frozen=True)
class Kinetics(cmn.DataSeriese):
	
	_time: cmn.TimeArray
	@property
	def time(self):
		return self._time
	
	_absorption: UV_vis.AbsorptionArray
	@property
	def absorption(self):
		return self._absorption
	
	"""_wavelength : UV_vis.Wavelength
	@property
	def wavelength(self):
		return self._wavelength"""
	
	def plot(self, fig: plt.Figure, ax: plt.Axes)->tuple[plt.Figure, plt.Axes]:
		_fig, _ax = super().plot(fig, ax)

		_ax.plot(
			self._time,
			self._absorption
		)

		return (_fig, _ax)

#そのうち
@dataclass(frozen=True)
class ShimadzuSpectra(cmn.DataFile[UV_vis.UV_VisSpectrum]):
	"""_spectra: cmn.DataArray[UV_vis.UV_VisSpectrum]
	@property
	def spectra(self):
		return self._spectra"""
	pass

@dataclass(frozen=True)
class ShimadzuKinetics(cmn.DataFile):
	_kinetic_profile: Kinetics
	@property
	def kinetic_profile(self):
		return self._kinetic_profile


def load_kinetic_profile(file_path: str)->ShimadzuKinetics:

	file = open(file_path, 'r', encoding='Shift-JIS')
	comment = file.readlines()[0]
	file.close()

	raw_df = pd.read_csv(
		file_path,
		
		skiprows= 2,
		quoting=csv.QUOTE_NONE,
		names=["time", "absorption"],
		encoding='Shift-JIS'
		)
	
	kinetic_data = Kinetics(
		_comment = comment,
		_condition = "",
		_original_file_path = file_path,
		_data_name = cmn.extract_filename(file_path),
		_time = cmn.TimeArray(raw_df["time"].values),
		_absorption = UV_vis.AbsorptionArray(raw_df["absorption"]),
		
	)

	file_data = ShimadzuKinetics(
		_comment = comment,
		_condition = [""],
		_file_path = file_path,
		_data_name = cmn.extract_filename(file_path),
		_kinetic_profile = kinetic_data
	)

	return file_data

def load_spectrum_data(
		file_path: str, 
		encoding = "Shift-JIS",
		comment = "",
		conditions = ""
		):
	df = pd.read_csv(
		file_path,
		encoding=encoding
	)

	wavelength_column = df.iloc[:,0]
	spectra = df.iloc[:,1:]

	wavelength_array = UV_vis.WavelengthArray(wavelength_column.values)

	spectra_list = []
	for key in spectra.keys():
		spectra_list.append(
			UV_vis.UV_VisSpectrum(
				_comment = comment,
				_condition = conditions,
				_original_file_path = file_path,
				_data_name = cmn.extract_filename(file_path),
				_wavelength = wavelength_array,
				_absorption = UV_vis.AbsorptionArray(spectra[key].values)
			)
		)
	spectra_data = ShimadzuSpectra(
		_comment = comment,
		_condition = conditions,
		_file_path = file_path,
		_data_name = cmn.extract_filename(file_path),
		_data = cmn.DataArray[UV_vis.UV_VisSpectrum](spectra_list)
	)
	
	return spectra_data

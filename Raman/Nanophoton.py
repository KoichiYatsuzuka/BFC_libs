import numpy as np
import pandas as pd
from typing import overload, NewType
from dataclasses import dataclass
from copy import deepcopy as copy
import xarray as xr

from .. import common as cmn
from .. import Raman as rmn
from .. Raman import RamanSpectrum


# @dataclass(frozen=True)
# class NanophotonDataFile(cmn.DataFile):
# 	_raman_spectra: cmn.DataArray[rmn.RamanSpectrum]
# 	@property
# 	def raman_spectra(self):
# 		return self._raman_spectra
	
# 	@cmn.immutator
# 	def update(self, raman_spectra: list[RamanSpectrum], added_comment: list[str]):
# 		"""
#         ### This method does not modify the instance itself.
#         Use the returned instance.
#         """
# 		return NanophotonDataFile(
# 			_raman_spectra = raman_spectra,
# 			_file_path = self._file_path,
# 			_condition = self._condition, 
# 			_comment = self.comment.append(added_comment)
#         )

Nanophoton1DDataFile = cmn.DataFile[rmn.RamanSpectrum]
	
def read_peakfit_result(file_path: str):
	raw_table = pd.read_csv(file_path, sep='\t')
	new_tabe_list : list[pd.DataFrame] = []
	data_names = raw_table["Name"].unique()

	for data_name in data_names:
		peak_list = raw_table.query(f"Name == '{data_name}'").drop(columns="Name")
		new_tabe_list.append(copy(peak_list))

	new_index = np.linspace(0, len(new_tabe_list[0]["No."]), len(new_tabe_list[0]["No."]), dtype=int)
	
	peak_data = xr.DataArray(
		new_tabe_list, 
		coords=[data_names, new_index, raw_table.columns.drop("Name")], 
		dims=["spectrum", "row", "column"]
		)
	
	return peak_data


def read_1D_data(file_path: str, encoding = "Shift-JIS")->Nanophoton1DDataFile:


	"""extension = common.extract_extension(file_path)
	
	if extension != "txt":
		return None"""

	skip_line_num = \
		cmn.find_line_with_key(file_path, "wn", encoding=encoding)
	
	file = open(file_path, 'r', encoding=encoding)
	conditions = file.readlines()[0:skip_line_num]
	file.close()
	
	df = pd.read_csv(file_path, sep="\t", skiprows=skip_line_num)

	column_names=df.keys()

	wavenumber_column_names = column_names[::2]
	spectrum_column_names = column_names[1::2]

	spectra_tmp: list[RamanSpectrum] = []
	for i in range(len(wavenumber_column_names)-1):
		spectra_tmp.append(
			RamanSpectrum(
				_comment = [spectrum_column_names[i]],
				_condition = conditions,
				_original_file_path = file_path, 
				_data_name = spectrum_column_names[i],
				_wavenumber = rmn.WavenumberArray(df[wavenumber_column_names[i]].values),
				_intensity= rmn.RammanIntensityArray(df[spectrum_column_names[i]].values)
			)
		)
		

	return Nanophoton1DDataFile(
		_comment = ["Raw data file: "+cmn.extract_filename(file_path)],
		_condition = conditions,
		_file_path = file_path,
		_data_name = cmn.extract_filename(file_path),
		_data = cmn.DataArray[RamanSpectrum](spectra_tmp, RamanSpectrum, file_path)
	)


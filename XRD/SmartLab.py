from .. import XRD as xrd
from .. import common
from typing import Optional, Literal
from dataclasses import dataclass
from copy import deepcopy as copy
import pandas as pd
import csv

"""@dataclass(frozen=True)
class SmartLabData(common.DataFile[xrd.XRDPattern]):
	_xrd_data: xrd.XRDPattern
	@property
	def xrd_data(self):
		return self._xrd_data
	pass"""
SmartLabData = common.DataFile[xrd.XRDPattern]

def read_smart_lab_1D_data(file_path: str, sep: Literal[" ", ","]=",")->SmartLabData:
	skp_row_num = common.find_line_with_key(file_path,"#", encoding='Shift-JIS') + 1 # #Intensity_unit=cpsまで飛ばす
	
	if skp_row_num == None:
		raise common.FileContentError

	file = open(file_path, 'r', encoding='Shift-JIS')
	conditions = file.readlines()[0:skp_row_num]
	file.close()

	raw_df = pd.read_csv(
		file_path,
		sep=sep,
		skiprows= skp_row_num,
		quoting=csv.QUOTE_NONE,
		names=["two_theta", "intensity"],
		encoding='Shift-JIS'
		)
	
	
	two_theta_array = xrd.ThetaArray(raw_df["two_theta"].values)
	intensity_array = xrd.DiffractionIntensityArray(raw_df["intensity"].values)
	

	xrd_data = xrd.XRDPattern(
		_data_name = common.extract_filename(file_path),
		_comment = [common.extract_filename(file_path)],
		_original_file_path = file_path,
		_condition = conditions,
		_two_theta=two_theta_array,
		_intensity=intensity_array,
	)

	file_data = SmartLabData(
		_comment = [common.extract_filename(file_path)],
		_condition = copy(conditions),
		_file_path = file_path,
		_data_name = common.extract_filename(file_path),
		_data = [xrd_data]
	)


	return file_data
	
#%%
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import electrochemistry as ec

from copy import deepcopy as copy
from dataclasses import dataclass
from typing import Union, Optional, Any
from collections import namedtuple
from nptyping import NDArray


from .. import common as cmn
from ..common import immutator
from . import ReferenceElectrode, Registance, Voltammogram, Potential, Current, EIS
from .. import electrochemistry as ec

BiologicFileInfo = namedtuple("BiologicFileInfo", ['skip_line_num', 'conditions'])

def biologic_file_info(file_path)->BiologicFileInfo:
	with open(file_path, 'r', encoding='UTF-8') as file:
		raw_txt_lines = file.readlines()
		num_skip_lines = int(raw_txt_lines[1][18:])-1
	
	return BiologicFileInfo(num_skip_lines, num_skip_lines)

@dataclass(frozen=True)
class BioLogicVoltammogramData(cmn.DataFile):

	_data: cmn.DataArray[Voltammogram]

	@property
	def data(self):
		return self._data
	
	@cmn.immutator
	def update(self, data: list[Voltammogram], added_comment: list[str]):
		"""
		### This method does not modify the instance itself.
		Use the returned instance.
		"""
		return BioLogicVoltammogramData(
			_data= data,
			_file_path = self._file_path,
			_condition = self._condition, 
			_data_name = self._data_name,
			_comment = self.comment + [added_comment]
			)
	
	@cmn.immutator
	def iR_correction(self, registance: ec.Registance)->BioLogicVoltammogramData:
		tmp_voltammograms = []
		for voltammogram in self._data:
			tmp_voltammograms.append(voltammogram.IR_correction(registance))
		return self.update(
			cmn.DataArray[Voltammogram]((tmp_voltammograms)),
			f"iR_corrected {registance} Ohm"
		)

	pass

@dataclass(frozen=True)
class BiologicEISData(cmn.DataFile[ec.EIS]):
	""""""
	"""_eis: EIS

	@property
	def eis(self):
		return self._eis"""
	pass

@dataclass(frozen=True)
class BiologicCAData(cmn.DataFile[ec.ChronoAmperogram]):
	
	@classmethod
	def load_file(cls, file_path: str):
		skipped_line_num, condition = biologic_file_info(file_path)
		df = pd.read_csv(file_path, sep="\t", skiprows=skipped_line_num)

		# preparing to slice CVs
		#cycle_border_list = [0]
		"""for i in range(0, df["Ns"].size-2):
			if df["Ns"][i] != df["Ns"][i+1]:
				cycle_border_list.append(i+1)
		cycle_border_list.append(len(df["Ns"])-1)
		"""
		
		delta_Ns = df["Ns"].values[:-1] - df["Ns"].values[1:] #変わり目だけ
		#print(delta_Ns)
		ends_indexes = np.append(np.where(delta_Ns == -1), [len(df["Ns"].values)]) #-1が入っているところはそのcycleが終わるところ + データの一番最後

		starts_indexes = np.append(0, ends_indexes+1)

		CycleRange = namedtuple("Cyclerange", ["begin", "end"])
		cycle_range_list: list[CycleRange] = []
		
		for i in range(len(starts_indexes)-1):
			cycle_range_list.append(
				CycleRange(starts_indexes[i], ends_indexes[i])
				)
		
		ca_list:list[ec.ChronoAmperogram] = []
		for i, cycle_range in enumerate (cycle_range_list):
			#print(cycle_range)
			ca_list.append(
				ec.ChronoAmperogram(
					_comment = ["file: "+cmn.extract_filename(file_path)+"[{}]".format(i)],
					_condition =condition,
					_original_file_path = file_path,
					_data_name = cmn.extract_filename(file_path),
					_time = cmn.TimeArray(np.array(df["time/s"][cycle_range.begin:cycle_range.end])),
					_potential = ec.PotentialArray(np.array(df["Ewe/V"][cycle_range.begin:cycle_range.end])),
					_current = ec.CurrentArray(np.array(df["<I>/mA"][cycle_range.begin:cycle_range.end]))
				)
			)
		return BiologicCAData(
			
			cmn.DataArray[ec.ChronoAmperogram](ca_list),
			_comment = ["file: "+cmn.extract_filename(file_path)],
			_condition = condition,
			_file_path = file_path,
			_data_name = cmn.extract_filename(file_path)
			)
	
	pass

def load_biologic_CV(
		file_path: str, 
		reference_electrode: Optional[ec.ReferenceElectrode] = None,
		data_name: str = "")->BioLogicVoltammogramData:
	
	"""with open(file_path, 'r', encoding='UTF-8') as file:
		raw_txt_lines = file.readlines()
		num_skip_lines = int(raw_txt_lines[1][18:])-1"""
	
	if data_name == "":
		data_name = cmn.extract_filename(file_path)
	
	num_skip_lines, condition = biologic_file_info(file_path)
	
	df = pd.read_csv(file_path, sep="\t", skiprows=num_skip_lines)

	# preparing to slice CVs
	cycle_border_list = [0]
	for i in range(0, df["cycle number"].size-2):
		if df["cycle number"][i] != df["cycle number"][i+1]:
			cycle_border_list.append(i+1)
	cycle_border_list.append(len(df["cycle number"])-1)

	CycleRange = namedtuple("Cyclerange", ["begin", "end"])
	cycle_range_list: list[CycleRange] = []
	for i in range(len(cycle_border_list)-1):
		cycle_range_list.append(
			CycleRange(cycle_border_list[i], cycle_border_list[i+1]-1)
			)

	# slicing
	voltammograms_list = []
	for i, cycle_range in enumerate (cycle_range_list):
		tmp_voltammogram = Voltammogram(
			#_data_name = cmn.extract_filename(file_path) + " " + str(i+1) + "th cycle",
			_data_name = data_name,
			_potential = ec.PotentialArray(np.array(df["Ewe/V"][cycle_range.begin:cycle_range.end], dtype = Potential).astype(Potential)),
			_current = ec.CurrentArray(np.array(df["<I>/mA"][cycle_range.begin:cycle_range.end]/1000)),
			_RE = reference_electrode,
			#_conditions = raw_txt_lines[:num_skip_lines-1],
			_time = cmn.TimeArray(np.array(df["time/s"][cycle_range.begin:cycle_range.end])),
			_others_data = df.drop(columns=["Ewe/V", "<I>/mA", "time/s"])[cycle_range.begin:cycle_range.end].reset_index(),
			_original_file_path = file_path,
			_condition = condition,
			_comment = ["file: "+cmn.extract_filename(file_path)+"[{}]".format(i)],
		)
		voltammograms_list.append(copy(tmp_voltammogram))
	


	bl_voltammogram_data = BioLogicVoltammogramData(
		_data = cmn.DataArray[Voltammogram](voltammograms_list), 
		_data_name = "",
		_condition = condition,
		_comment = ["file: "+cmn.extract_filename(file_path)],
		_file_path = file_path
	)
	
	return bl_voltammogram_data

def load_biologic_EIS(file_path: str)->Optional[BiologicEISData]:

	with open(file_path, 'r', encoding='UTF-8') as file:
		raw_txt_lines = file.readlines()
		num_skip_lines = int(raw_txt_lines[1][18:])-1

	condition = raw_txt_lines[:num_skip_lines-1]
	df = pd.read_csv(file_path, sep="\t", skiprows=num_skip_lines)


	# preparing to slice CVs
	cycle_border_list = [0]
	for i in range(0, df["cycle number"].size-2):
		if df["cycle number"][i] != df["cycle number"][i+1]:
			cycle_border_list.append(i+1)
	cycle_border_list.append(len(df["cycle number"])-1)

	CycleRange = namedtuple("Cyclerange", ["begin", "end"])
	cycle_range_list: list[CycleRange] = []
	for i in range(len(cycle_border_list)-1):
		cycle_range_list.append(
			CycleRange(cycle_border_list[i], cycle_border_list[i+1]-1)
			)
		
	eis_list = []

	for i, cycle_range in enumerate (cycle_range_list):
		tmp_eis = EIS(
			_real_Z = ec.ImpedanceArray(np.array(df["Re(Z)/Ohm"][cycle_range.begin:cycle_range.end]), meta=file_path),
			_imaginary_Z = ec.ImpedanceArray(np.array(df["-Im(Z)/Ohm"][cycle_range.begin:cycle_range.end]), meta=file_path),
			_frequency = ec.FrequencyArray(np.array(df["freq/Hz"][cycle_range.begin:cycle_range.end]), meta=file_path),
			_data_name = "",
			_others_data = df.drop(columns=["Re(Z)/Ohm", "-Im(Z)/Ohm", "freq/Hz"])[cycle_range.begin:cycle_range.end].reset_index(),
			_comment = ["file: "+cmn.extract_filename(file_path)],
			_condition = condition,
			_original_file_path = file_path
		)
		eis_list.append(copy(tmp_eis))


	return BiologicEISData(
		_data = cmn.DataArray[EIS](eis_list),
		_data_name = cmn.extract_filename(file_path),
		_comment = ["file: "+cmn.extract_filename(file_path)],
		_condition = condition,
		_file_path = file_path
	)

	return BiologicEISData(
		_data = [EIS(
			_real_Z = ec.ImpedanceArray(np.array(df["Re(Z)/Ohm"]), meta=file_path),
			_imaginary_Z = ec.ImpedanceArray(np.array(df["-Im(Z)/Ohm"]), meta=file_path),
			_frequency = ec.FrequencyArray(np.array(df["freq/Hz"]), meta=file_path),
			_data_name = "",
			_others_data = df.drop(columns=["Re(Z)/Ohm", "-Im(Z)/Ohm", "freq/Hz"]),
			_comment = ["file: "+cmn.extract_filename(file_path)],
			_condition = condition,
			_original_file_path = file_path
		)],
		_data_name = cmn.extract_filename(file_path),
		_comment = ["file: "+cmn.extract_filename(file_path)],
		_condition = condition,
		_file_path = file_path
	)
	pass

"""def load_biologic_mpt(file_path: str)->Union[Voltammogram, BiologicEIS]:
	with open(file_path, 'r', encoding='UTF-8') as file:
		lines = file.readlines()
	
	determinator = lines[3]
	match determinator:
		case determinator if determinator == "Cyclic Voltammetry" or determinator == "Cyclic Voltammetry Advanced":
			load_biologic_CV(file_path)
		
		case _:
			print("Not made")


"""
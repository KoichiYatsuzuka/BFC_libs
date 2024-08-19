# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]

"""
# Common modules for electrochemical measurements.
There are the common things among Biologic, HokutoDenko, and so on.
For example, a class for voltammogram.

class list
	Potential [V]
	Current [A]
	Registance [Ohm]

"""
from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from collections import namedtuple
from dataclasses import dataclass
from typing import Final, NewType, Optional, Union, TypeAlias, Self
from enum import Enum

from .. import common as cmn
from ..common import ValueObject, ValueObjectArray

# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]

#---------------------------------------------------------------
#Value object classes
#---------------------------------------------------------------


class Current(ValueObject):
	"""
	note: iR correction can be performed via iR_correction()
	"""
	@cmn.immutator
	def log10(self)->LogCurrent:
		return LogCurrent(float(np.log10(np.abs(self.value))))
	
	@cmn.immutator
	def iR_correction(self, registance: Registance):
		return Potential(self.value*registance.value)
	pass

class CurrentArray(ValueObjectArray[Current]):

	def __new__(cls, obj, dtype=Current, meta: Optional[str] = None):
		
		return super().__new__(cls, obj, dtype, meta)
	pass

	@cmn.immutator
	def log10(self)->LogCurrentArray:
		"""list_tmp = []
		for val in self:
			list_tmp.append(val.log10())"""
		

		return LogCurrentArray(np.log10(np.abs(self.float_array())))
	
	@cmn.immutator
	def iR_correction(self, registance: Registance):
		list_tmp = []
		for val in self:
			list_tmp.append(val.iR_correction(registance))
		return PotentialArray(list_tmp)
	
	pass



class Potential(ValueObject):
	pass

#PotentialArray= ValueObjectArray[Potential]
class PotentialArray(ValueObjectArray):
	def __new__(cls, obj, dtype=Potential, meta: Optional[str] = None):
		
		return super().__new__(cls, obj, dtype, meta)
	pass


class LogCurrent(ValueObject):
	pass

LogCurrentArray = NewType("LogCurrentArray", ValueObjectArray[LogCurrent])

class Registance(ValueObject):
	pass

Impedance = Registance

class ImpedanceArray(ValueObjectArray):
	def __new__(cls, obj, dtype=Impedance, meta: Optional[str] = None):
		
		return super().__new__(cls, obj, dtype, meta)
	pass

class Frequency(ValueObject):
	pass

class FrequencyArray(ValueObjectArray):
	def __new__(cls, obj, dtype=Frequency, meta: Optional[str] = None):
		
		return super().__new__(cls, obj, dtype, meta)
	pass

#---------------------------------------------------------------
#Objects relating to potential
#---------------------------------------------------------------
ReferenceElectrode = Potential
"""
The type for conversion of potential.
Use ReferenceElectrodes enum or RHE() function.
"""

"""
*Only RHE is defined as a function.*
"""

SHE = ReferenceElectrode(0.0)
"""Standard Hydrogen Electrode"""
NHE = ReferenceElectrode(0.0)
"""Natural Hydrogen Electrode"""

SSCE = ReferenceElectrode(0.2)
"""
Ag/AgCl (Silver/Silver chloride electorde)
"""

def RHE(pH: float)->ReferenceElectrode:
	""" Retruens -0.059*pH
	"""
	return -0.059*pH

#---------------------------------------------------------------
#Value object classes
#---------------------------------------------------------------
@dataclass(frozen=True, repr=False)
class Voltammogram(cmn.DataSeriese[Potential, Current]):
	"""
	## The dataclass to treat voltammograms
	Can be used for LSV and CV.\n
	This class is frozen, so you cannot re-write class members after initialization.\n
	If you want to adjust some things (like current to current density, or Ag/AgCl to SHE), use class methods.
	### member values and properties
	potential: numpy.ndarray
	current: numpy.ndarray
	### member methods
	"""

	_potential: PotentialArray
	"""in [V]"""
	@property
	def potential(self):
		return self._potential

	_current: CurrentArray
	"""in [A]"""
	@property
	def current(self):
		return self._current
	
	_RE: Optional[ReferenceElectrode]
	@property
	def reference_electrode(self):
		return self._RE
	
	"""_conditions: list[str]
	@property
	def conditions(self):
		return copy(self._conditions)"""
	
	_time: cmn.TimeArray
	@property
	def time(self):
		return self._time
	
	_others_data: pd.DataFrame

	@property
	def x(self):
		return self._potential
	
	@property
	def y(self):
		return self._current
	
	def to_data_frame(self):
		return pd.concat([
			pd.DataFrame(
				np.stack([
					self._potential.float_array(),
					self._current.float_array(),
					self._time.float_array()
				], 1),
				columns = [
					"potential",
					"current",
					"time"
				]
			),
			self._others_data],
			axis=1
		)

	@property
	def Tafel_slope(self):
		"""
		現在の仕様: 電流値を100 plot、4次関数でsavgol filterかけてから、∂E/∂log|I|のseriese作って返す。電流記録頻度によってfilterのかかる電位幅が変わる
		"""
		#もし既に計算済みならそれを返す
		if "Tafel slope" in self._others_data.keys():
			return self._others_data["Tafel slope"]
		#計算したことないならそれを計算する
		len_array = len(self.current.log10())
		
		#電流値を平滑化
		current_tmp = savgol_filter(
			self.current.float_array(),
			100,
			4,
			deriv=0
			) 

		#log iの計算
		logI_for = np.delete(np.log10(np.abs(current_tmp)), [0, 1, 2, 3, 4, 5])
		logI_back = np.delete(np.log10(np.abs(current_tmp)), [len_array-6, len_array-5, len_array-4, len_array-3, len_array-2, len_array-1])
		delta_logI = logI_for - logI_back
		
		E_for = np.delete(self.potential, [0, 1, 2, 3, 4, 5])
		E_back = np.delete(self.potential, [len_array-6, len_array-5, len_array-4, len_array-3, len_array-2, len_array-1])
		delta_E: PotentialArray = E_for-E_back

		Tafel_slope_raw = (delta_E*1000).float_array() / delta_logI

		tafel_slope_nparray = np.append(np.append([0, 0, 0], Tafel_slope_raw), [0, 0, 0])

		return copy(tafel_slope_nparray)

	@property
	def others_data(self):
		return self._others_data

	def __hash__(self) -> int:
		first_part_of_current = self.current[:10]
		hash_str = ""
		for current in first_part_of_current:
			tmp = str(current)
			tmp=tmp.replace(".", "")
			print(tmp)
			hash_str += tmp

		return int(hash_str)
	
	"""def _data_finalize(original_obj, new_obj: Voltammogram)->Voltammogram:
		return new_obj"""
	
	"""@cmn.immutator
	def __make_modified_object(self, modified_parameter: dict):
		new_obj = copy(self)

		all_dict = self.__dict__

		for par_key in all_dict.keys():
			if par_key in modified_parameter.keys():
				new_obj.__setattr__(par_key, modified_parameter[par_key])
			
		return new_obj"""

	
	@cmn.immutator
	def modify_voltammogram(
		self,
		new_potential: Optional[Potential] = None,
		new_current: Optional[Current] = None,
		new_time: Optional[cmn.Time] = None,
		new_others_data: Optional[np.ndarray] = None,
		prefix_to_data_name: Optional[str] = None
	):
		tmp_prefix_candidate: str = "_"

		if new_potential == None:
			_new_potential = self._potential
		else:
			_new_potential = new_potential
			tmp_prefix_candidate = tmp_prefix_candidate + "potential_mod_"
			
		#match new_potential:
		#	case None:
		#		_new_potential = self._potential
		#	case _:
		#		_new_potential = new_potential
		#		tmp_prefix_candidate = tmp_prefix_candidate + "potential_mod_"
		
		if new_current == None:
			_new_current = self.current
		else:
			_new_current = new_current
			tmp_prefix_candidate = tmp_prefix_candidate + "current_mod_"
		
		#match new_current: 
		#	case None:
		#		_new_current = self.current
		#	case _:
		#		_new_current = new_current
		#		tmp_prefix_candidate = tmp_prefix_candidate + "current_mod_"
			
		if new_time == None:
			_new_time = self.time
		else:
			_new_time = new_time
		
		"""match new_time:
			case None:
				_new_time = self.time
			case _:
				_new_time = new_time"""
		if new_others_data == None:
			_new_other_data = self.others_data
		else:
			_new_other_data = new_others_data
			tmp_prefix_candidate = tmp_prefix_candidate + "others_mod_"

		# match new_others_data:
		# 	case None:
		# 		_new_other_data = self.others_data
		# 	case _:
		# 		_new_other_data = new_others_data
		# 		tmp_prefix_candidate = tmp_prefix_candidate + "others_mod_"

		if prefix_to_data_name == None:
			_prefix_to_data_name = tmp_prefix_candidate
		else:
			_prefix_to_data_name = prefix_to_data_name

		# match prefix_to_data_name:
		# 	case None:
		# 		_prefix_to_data_name = tmp_prefix_candidate
		# 	case _:
		# 		_prefix_to_data_name = prefix_to_data_name

		return Voltammogram(
			self._data_name,
			_new_potential,
			_new_current,
			self._RE,
			_new_time,
			#self._conditions,
			_new_other_data
		)

		
		"""mod_attr_dict = {
			"_data_name": self.data_name + _prefix_to_data_name,
			"_potential": _new_potential,
			"_current": _new_current,
			"_other_data": _new_other_data
		}

		return self.__make_modified_object(mod_attr_dict)"""

	@cmn.immutator
	def convert_potential_reference(
			self, 
			RE_before: ReferenceElectrode, 
			RE_after: ReferenceElectrode):
		"""
		"""
		RE_diff = RE_after-RE_before
		new_voltammogram = Voltammogram(
			_potential = self.potential-RE_diff,
			_current = self.current,
			_data_name = self.data_name+"_RE_corrected_"+str(RE_diff)+"V",
			_RE = RE_after,
			_others_data = self.others_data,
			_time = self.time,
			_condition = self.condition,
			_comment = self.comment+["RE corrected: {} -> {}".format(RE_before, RE_after)],
			_original_file_path = self._original_file_path
			#_conditions = self.conditions
		)
		return new_voltammogram
	
	@cmn.immutator
	def IR_correction(self, resistance: Registance):
		new_voltammogram = Voltammogram(
			_potential = self.potential - self.current.iR_correction(resistance),
			_current = self.current,
			_data_name = self.data_name + "_iR_corrected_"+str(resistance)+"Ohm",
			_RE = self._RE,
			_others_data = self.others_data,
			_time = self.time,
			_comment = self.comment,
			_condition = self.condition,
			_original_file_path= self._original_file_path
		)
		return new_voltammogram
	
	
	def plot(self, fig: Optional[plt.Figure]=None, ax: Optional[plt.Axes]=None, **args)->tuple[plt.Figure, plt.Axes]:
		(_fig, _ax) = super().plot(fig, ax)
		if (ax == None):
			_ax.set_xlabel("Potential")
			_ax.set_ylabel("Current")

		#_ax.plot(self.potential, self.current, **args)

		return (_fig, _ax)

	
@dataclass(frozen=True, repr=False)
class EIS(cmn.DataSeriese[Registance, Registance]):
	_real_Z: ImpedanceArray
	"""in Ohm"""
	@property
	def real_Z(self):
		return self._real_Z

	_imaginary_Z : ImpedanceArray
	"""in Ohm"""
	@property
	def imaginary_Z(self):
		return self._imaginary_Z
	
	_frequency: cmn.ValueObjectArray
	@property
	def frequency(self):
		return self._frequency
	
	"""_phase: cmn.ValueObjectArray
	@property
	def phase(self):
		return self._phase"""

	_others_data: pd.DataFrame

	@property
	def x(self):
		return self._real_Z
	
	@property
	def y(self):
		return self._imaginary_Z

	def to_data_frame(self) -> pd.DataFrame:
		return pd.DataFrame(
			np.stack([
				self._frequency.float_array(),
				self._real_Z.float_array(),
				self._imaginary_Z.float_array()
			], 1),
			columns = [
				"frequency",
				"real Z",
				"imaginary Z"
			]
		)
	
	@classmethod
	def from_data_frame(
		cls, 
		df: pd.DataFrame, 
		comment: list[str] = [], 
		condition: list[str] = [], 
		original_file_path: str = ""
		) -> Self:
		return EIS(
			_comment = comment,
			_condition = condition,
			_original_file_path = original_file_path,
			_data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
			_real_Z = ImpedanceArray(df["real Z"]),
			_imaginary_Z = ImpedanceArray(df["imaginary Z"]),
			_frequency = df["frequency"],
			_other_data = pd.DataFrame([]),
		)

	def plot(self, fig: Optional[plt.Figure]=None, ax: Optional[plt.Axes]=None, **args)->tuple[plt.Figure, plt.Axes]:
		"""
		Nyquest plot
		"""
		(_fig, _ax) = super().plot(fig, ax)
		if (ax == None):
			_ax.set_xlabel("Real Z [Ohm]")
			_ax.set_ylabel("-Imaginary Z [Ohm]")
			_ax.axvline(0, linewidth = 1, color = "#303030")
			_ax.axhline(0, linewidth = 1, color = "#303030")
		#_ax.plot(self._real_Z, self._imaginary_Z)

		if (self.get_resistance() != None):
			_ax.axvline(self.get_resistance().value, -0.2, 0.2, linestyle = "dotted")
		

		return (_fig, _ax)


	def get_resistance(self)->Optional[Registance]: 
		higher: ImpedanceArray = np.delete(self.imaginary_Z, 0, 0)
		lower: ImpedanceArray = np.delete(self.imaginary_Z, -1, 0)
		product : ImpedanceArray = higher * lower
		cross_indexes = np.where(product.float_array() < 0.0)[0]+1
		
		if len(cross_indexes>0):
			return self._real_Z[cross_indexes].min()
		else:
			return None


@dataclass(frozen=True, repr=False)
class ChronoAmperogram(cmn.DataSeriese[cmn.Time, Current]):
	
	_time : cmn.TimeArray
	@property
	def time(self):
		return self._time
	
	_potential: PotentialArray
	@property
	def potential(self):
		return self._potential
	
	_current: CurrentArray
	@property
	def current(self):
		return self._current
	
	@property
	def x(self):
		return self._time
	
	@property
	def y(self):
		return self._current
	
	def to_data_frame(self) -> pd.DataFrame:
		return pd.DataFrame(
            np.stack([
                self._time.float_array(),
                self._current.float_array(),
				self._potential.float_array()
            ], 1),
            columns = [
                "time",
                "current",
				"potential"
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
		
		return ChronoAmperogram(
			_comment = comment,
			_condition = condition,
			_original_file_path = original_file_path,
			_data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
			_time = df["time"],
			_potential = df["potential"],
			_current = df["current"]
		)
	
"""
To do
to_data_frame関連の引数を修正
"""
@dataclass(frozen=True, repr=False)
class ChronoPotentiogramn(cmn.DataSeriese[cmn.Time, Current]):
	
	_time : cmn.TimeArray
	@property
	def time(self):
		return self._time
	
	_potential: PotentialArray
	@property
	def potential(self):
		return self._potential
	
	_potential_CE: PotentialArray
	@property
	def potential_CE(self):
		return self._potential_CE
	
	
	_current: CurrentArray
	@property
	def current(self):
		return self._current
	
	@property
	def x(self):
		return self._time
	
	@property
	def y(self):
		return self._current
	
	def to_data_frame(self) -> pd.DataFrame:
		return pd.DataFrame(
            np.stack([
                self._time.float_array(),
                self._current.float_array(),
				self._potential.float_array()
            ], 1),
            columns = [
                "time",
                "current",
				"potential"
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
		
		return ChronoAmperogram(
			_comment = comment,
			_condition = condition,
			_original_file_path = original_file_path,
			_data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
			_time = df["time"],
			_potential = df["potential"],
			_current = df["current"]
		)

#---------------------------------------------------------------
#functions
#---------------------------------------------------------------

def convert_potential_reference(
		voltammogram: Voltammogram, 
		RE_before: ReferenceElectrode, 
		RE_after: ReferenceElectrode
		)->Voltammogram:
	
	potential_diff = RE_before - RE_after


	return Voltammogram(
		
		_comment = voltammogram.comment + ["RE converted: {} -> {}".format(RE_before, RE_after)],
		_condition = voltammogram.condition,
		_original_file_path = voltammogram.original_file_path,
		_data_name = voltammogram.data_name,
		_RE = RE_after,
		_others_data = voltammogram.others_data,
		_potential = voltammogram.potential + potential_diff,
		_current = voltammogram.current,
		_time = voltammogram.time
	)


def mupltiply_current(
		voltammogram: Voltammogram,
		multiplied_value: float
)->Voltammogram:
	pass
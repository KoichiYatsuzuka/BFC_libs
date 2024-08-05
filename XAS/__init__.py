import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, Self

from .. import common as cmn


"""
To do: ChiR空間の変数の演算（実部^2 + 虚部^2 = 絶対値^2など）
"""
class PhotonEnergy(cmn.ValueObject):
    pass

class PhotonEnergyArray(cmn.ValueObjectArray[PhotonEnergy]):
    def __new__(cls, obj, dtype=PhotonEnergy, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)
    
class ChiMu(cmn.ValueObject):
    pass

class ChiMuArray(cmn.ValueObjectArray[ChiMu]):
    def __new__(cls, obj, dtype=ChiMu, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)

class Frequency(cmn.ValueObject):
    """in angstrom-1"""
    pass

class FrequencyArray(cmn.ValueObjectArray[Frequency]):
    def __new__(cls, obj, dtype=Frequency, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)

class Distance(cmn.ValueObject):
    pass

class DistanceArray(cmn.ValueObjectArray[Distance]):
    def __new__(cls, obj, dtype=Distance, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)

class ChiR_Mag(cmn.ValueObject):
    pass

class ChiR_MagArray(cmn.ValueObjectArray[ChiR_Mag]):
    def __new__(cls, obj, dtype=ChiR_Mag, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)


class ChiR_Re(cmn.ValueObject):
    pass

class ChiR_ReArray(cmn.ValueObjectArray[ChiR_Re]):
    def __new__(cls, obj, dtype=ChiR_Re, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)


class ChiR_Im(cmn.ValueObject):
    pass

class ChiR_ImArray(cmn.ValueObjectArray[ChiR_Im]):
    def __new__(cls, obj, dtype=ChiR_Im, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)


class ChiR_Phase(cmn.ValueObject):
    pass

class ChiR_PhaseArray(cmn.ValueObjectArray[ChiR_Phase]):
    def __new__(cls, obj, dtype=ChiR_Phase, meta: Optional[str] = None):
        return super().__new__(cls, obj, dtype, meta)



@dataclass(frozen=True, repr=False)    
class XAS(cmn.DataSeriese[PhotonEnergy, ChiMu]):
    pass

@dataclass(frozen=True, repr=False)
class InversedEXAFS(cmn.DataSeriese[DistanceArray, ChiR_MagArray]):
    
    _distance: DistanceArray
    @property
    def distance(self):
        return self._distance
    
    _chir_magnitude: ChiR_MagArray
    @property
    def chir_magnitude(self):
        return self._chir_magnitude
    
    _chir_real: Optional[ChiR_ReArray]
    @property
    def chir_real(self):
        return self._chir_real
    
    _chir_imaginary: Optional[ChiR_ImArray]
    @property
    def chir_imaginary(self):
        return self._chir_imaginary
    
    _chir_phase: Optional[ChiR_PhaseArray]
    @property
    def chir_phase(self):
        return self._chir_phase

    @property
    def x(self)->FrequencyArray:
        return self._distance
    
    @property
    def y(self)->ChiR_MagArray:
        return self._chir_magnitude
    
    def to_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.stack([
                self._distance.float_array(),
                self._chir_magnitude.float_array(),
                self._chir_real.float_array() if self._chir_real is not None else [],
                self._chir_imaginary.float_array() if self._chir_imaginary is not None else [],
                self._chir_phase.float_array() if self._chir_phase is not None else []
            ], 1),
            columns = [
                "distance",
                "ChiR magnituite",
                "ChiR real",
                "ChiR imaginary",
                "ChiR phase",
            ]
        )
    
    @classmethod
    def from_data_frame(
        cls, 
        df: pd.DataFrame, 
        comment: list[str] = "", 
        condition: list[str] = "", 
        original_file_path: str = ""
        ) -> Self:
        
        return InversedEXAFS(
            _comment = comment,
            _condition =condition,
            _original_file_path = original_file_path,
            _data_name = f"Generated from dataframe({cmn.extract_filename(original_file_path)})",
            _distance = df["distance"],
            _chir_magnitude = df["ChiR magnituite"],
            _chir_real = df["ChiR real"],
            _chir_imaginary = df["ChiR imaginary"],
            _chir_phase = df["ChiR phase"]
        )

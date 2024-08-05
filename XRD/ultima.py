from .. import XRD as xrd
from typing import Optional, Literal
import pandas as pd
import csv

def read_XRD_data(file_path: str, sep: Literal[" ", ","]=" ")->Optional[xrd.XRDPattern]:
    try:
        raw_df = pd.read_csv(
            file_path,
            sep=sep,
            skiprows= 24,
            quoting=csv.QUOTE_NONE,
            names=["two_theta", "intensity"]
            )
    except FileNotFoundError:
        return None
    
    two_theta_array = raw_df["two_theta"].values
    intensity_array = raw_df["intensity"].values

    converted_columns = xrd.XRDPattern(
        two_theta_array,
        intensity_array
    )

    return converted_columns
    


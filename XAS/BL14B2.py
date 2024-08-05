#%%
import pandas as pd
import numpy as np
import csv
from .. import common
from copy import deepcopy as copy
from typing import Optional


def read_XAFS_data(file_path: str)->Optional[pd.DataFrame]:
	
	extension= common.extract_extension(file_path)
	if extension is None:
		raise(ValueError("Cannot idenify file extension.\n\
		   Check the path.\n\
		   The paht is \"{}\".".format(file_path)))
	
	# key to count non-data lines
	key_word:str
	if extension == "nor":
		key_word = "#------------------------"
	elif extension == "chir_mag":
		key_word = "#------------------------"
		
	else:
		print("Cannot read this file.")
		return None
		
	skip_line_num = common.find_line_with_key(file_path, key_word)
	
	if skip_line_num is None:
		raise(
			ValueError("File content is invalid.\n\
			The path is {}, \n\
			program tried to find \"{}\" from this file, but not found.".format(file_path, key_word)))
	
	
	# Start to read
	raw_df = pd.read_csv(
		file_path,
		delim_whitespace=True,
		skiprows= skip_line_num,
		quoting=csv.QUOTE_NONE
		)
	# But this DataFrame's strucutres is invalid, because of the first " #".
	# So some modificaion is required.

	columns_raw = raw_df.columns.values
	
	last_coulmn_name = columns_raw[len(columns_raw)-1]
	df_modified = raw_df.drop(last_coulmn_name, axis=1)
	new_columns = columns_raw[1:]
	df_modified.set_axis(new_columns, axis=1, inplace=True)
	
	return copy(df_modified)
	




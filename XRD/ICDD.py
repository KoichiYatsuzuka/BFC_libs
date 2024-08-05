#%%
from .. import common as cmn
from .. import XRD as xrd
from dataclasses import dataclass
from typing import NewType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from typing import Union, Optional


#PDFはファイルごとに体裁が異なるうえに、得られている表が崩れているときがあるので断念

# def load_ICDD_PDF(file_path: str):
#     doc = fitz.open(file_path)
#     pages = []
#     for i in range(10):
#         try:
#             pages.append(doc.load_page(i))
#         except ValueError:
#             break
    
#     df_list = []
#     for i,page in enumerate(pages):
#         table:fitz.table.Table = page.find_tables().tables[0]
#         df = table.to_pandas() #very mutable variable
        
#         #1ページ目時のみPDFのデータあるので例外処理
#         if i == 0:
#             # PDFカード番号:XX-XXX-XXX Quality:X
#             pdf_no = df[df.keys()[1]]
            
#             material_info = df[df.keys()[0]][1] 

#             #結晶の情報
#             crystal_info = df[df.keys()[0]][2]

#             #どのように得たデータかの情報
#             data_info = df[df.keys()[0]][3]

#             #コメント
#             comments = df[df.keys()[0]][4]

#             df = df.drop(index=[0,1,2,3,4,5,27,28])
#             df = df.drop(columns = df.keys()[0])
#             """dropped_column = [2,3,  4,  6,7, 9, 10, 12, 14, 17, 18, 20, 22, 23, 24, 26, 27, 29, 30, 34, 35]
#             for col_num in dropped_column:                
#                 df = df.drop(columns = [f"Col{col_num}"])"""
#             df = df.drop(index=6)
#             """df = df.rename(columns={
#                 "Col1": "No.",
#                 "Col5": "2theta",
#                 "Col8": "d value",
#                 "Col11": "relative intensity",
#                 "Col13": "h",
#                 "Col15": "k",
#                 "Col16": "l",
#                 "Col19": "No.",
#                 "Col21": "2theta",
#                 "Col25": "d value",
#                 "Col28": "relative intensity",
#                 "Col31": "h",
#                 "Col32": "k",
#                 "Col33": "l",
#             })"""

#         else:
#             df = df.drop(index=[0, 61, 62])
#             df = df.drop(columns = df.keys()[0])

#             """dropped_column = [2, 3, 5, 6,  8, 9, 11, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24] #このうちのどれかは右側のカラム
#             for col_num in dropped_column:                
#                 df = df.drop(columns = [f"Col{col_num}"])"""
#             """df = df.rename(columns={
#                 "Col1": "No.",
#                 "Col4": "2theta",
#                 "Col7": "d value",
#                 "Col10": "relative intensity",
#                 "Col13": "h",
#                 "Col15": "k",
#                 "Col17": "l",
                
#             })"""
#             """df = df.dropna(how="all").dropna(how="all")
#             df.index = range(len(df))
#             df_list.append(copy(df))"""

#         df = df.replace('', float('nan'))
#         df = df.replace(None, float('nan'))
#         df = df.dropna(how="all").dropna(how="all", axis=1)
#         """df = pd.concat(
#             [df.iloc[:, 0:7], df.iloc[:, 7:]]
#         )"""
        
#         #df.index = range(len(df))
#         df_list.append(copy(df))

#     total_df = pd.concat(df_list)

#     return total_df

@dataclass(frozen=True)
class MillerIndex:
    h: int
    k: int
    l: int

@dataclass(frozen=True)
class ICDD(xrd.XRDPattern):
    
    d_value: cmn.ValueObjectArray
    Miller_index: MillerIndex

"""
必須要件: Origin 2019以降がインストールされている（外部pythonからOriginのAPIが叩ける）こと
それ以前のOriginだとエラーはいて止まる。

originproパッケージが微妙に使いにくいので、改良したAPIっぽいものを提供する。
たとえば、シートを選択するときにはシートの有無を先に検査する必要があるが、シートが無かった場合には自動で作成する関数を提供したりする。

Originのインスタンスは一個しか管理できない。
これはoriginproライブラリの仕様に基づく。

外部PythonからOriginを起動したとき、PythonがOiriginをバインドしたままになるので、手動でウィンドウを閉じれなくなることに注意。

"""
#%%
#import originpro as op
import OriginExt as oext
import OriginExt._OriginExt as op_API

import sys
import os

from typing import Generator
from enum import Enum

# originproよりコピー
# originproは読み込むだけでoriginを起動する
class APP:
    'OriginExt.Application() wrapper'
    def __init__(self):
        self._app = None
        self._first = True
    def __getattr__(self, name):
        try:
            return getattr(oext, name)
        except AttributeError:
            pass
        if self._app is None:
            self._app = oext.Application()
            self._app.LT_execute('sec -poc') # wait until OC ready
        return getattr(self._app, name)
    def __bool__(self):
        return self._app is not None
    def Exit(self, releaseonly=False):
        'Exit if Application exists'
        if self._app is not None:
            self._app.Exit(releaseonly)
            self._app = None
    def Attach(self):
        'Attach to exising Origin instance'
        releaseonly = True
        if self._first:
            releaseonly = False
            self._first = False
        self.Exit(releaseonly)
        self._app = oext.ApplicationSI()
    def Detach(self):
        'Detach from Origin instance'
        self.Exit(True)

class OriginPath(Enum):
    USER_FILES_DIR = "u",
    ORIGIN_EXE_DIR = "e",
    PROJECT_DIR = "p",
    ATTACHED_FILE_DIR = "a",
    LEANING_CENTER = "l"


class OriginNotFoundError(BaseException):
    pass

class OriginInstanceGenerationError(BaseException):
    pass

class OriginTooManyInstancesError(BaseException):
    pass

# #便利っていうんならデフォルトでオンにしとけ
# def origin_shutdown_exception_hook(exctype, value, traceback):
#     '''Ensures Origin gets shut down if an uncaught exception'''
#     op.exit()
#     sys.__excepthook__(exctype, value, traceback)
# if op and op.oext:
#     sys.excepthook = origin_shutdown_exception_hook

# if op.oext:
#     op.set_show(True)

if "ORIGIN_INSTANCE_LIMIT" not in globals():
    ORIGIN_INSTANCE_LIMIT = 5



class OriginInstance:
    """
    ## Abstract
    Originのインスタンスを管理するクラス。
    ここからファイル構造を辿ったり、シートを取得したりする。

    ## Member variables
    __core (op.APP): originproのAPPクラスのインスタンス。

    path (str): このインスタンスが管理しているOriginプロジェクトファイルのパス。
    パスはグローバル変数で管理しており、同じパスのものは複数生成できない。


    ## note
    デストラクタが呼ばれた際、自動的に保存するので、保存したくないときには明示的にclose(False)を呼び出さなければならない。



    余談だが、originproはその設計上、インスタンスを一度だけ、一個だけ呼べるようになっており、そのインスタンスそのものを操作することがサポートされていない。
    オリジナルのoriginproではop.poの実態がモジュールではなくAPPクラスになっている。
    実質的にこれがOriginのインスタンスとなるのだが、このクラスのインスタンス生成がライブラリ呼び出し時の一回だけである。
    インテリジェンスの予測もop.poをモジュールとして認識してしまっているため、わかりにくくなっている。
    """
    __core: APP
    @property
    def get_API(self):
        return self.__core
    
    path: str
    @property
    def get_path(self):
        return self.path
    
    # 疑似static変数
    __instance_count: int = 0
    __instance_path_list: set[str] = set()

    def __init__(
            self,
            path_to_origin_file: str,
            create_new_if_not_exist: bool = True
            ):
        """
        Originのインスタンスを生成する。
        Args:
            path_to_origin_file (str): 開きたいOriginプロジェクトファイルのパス(**フルパス、絶対参照**)
            create_new_if_not_exist (bool, optional): ファイルが無かった場合に新規作成するかどうか. Defaults to True.

        Raises:
            FileNotFoundError: path_to_origin_fileで指定したパスにファイルが存在しない場合に発生
            OriginTooManyInstancesError: Originのインスタンス数が多すぎる場合に発生
        Returns:
            このクラスのインスタンス
        """

        # //があるとフォルダだと認識されないため、置換
        self.path = path_to_origin_file.replace("//", "\\")
        
        dir = os.path.dirname(self.path)
        if not os.path.exists(dir):
            raise OriginNotFoundError(
                "The directory was not found:\n\
                {}".format(dir)
            )
        
        # 同パスへのインスタンスが既にあればエラー
        if path_to_origin_file in OriginInstance.__instance_path_list:
            raise OriginInstanceGenerationError(
                f"An Origin instance with the path {path_to_origin_file} already active."
                )
        OriginInstance.__instance_path_list.add(path_to_origin_file)
        

        # インスタンス数が多すぎたらエラー
        if OriginInstance.__instance_count > ORIGIN_INSTANCE_LIMIT:
            raise OriginTooManyInstancesError(
                "Too many Origin instances are being created.\n"+ \
                "Run close() function to free instances\n"+ \
                "To increase the limit of the number of instances,"+ \
                "define ORIGIN_INSTANCE_LIMIT variable before importing this module."
                )
        
        # インスタンス生成開始
        print("Generating Origin instance...")
        self.__core = APP()
        
        # すでにあるファイルへのパスならばそれを読み込み
        if os.path.isfile(self.path):
            print("Opening the file...")
            r = self.__core.Load(self.path, False) 
            # 書き込み可で開く
            
            if r == False:
                self.close()
                raise OriginInstanceGenerationError(
                    "Failed to load.\n \
                    Please check the extension, the version of Origin, and so on.\n"+\
                    "The path: {}".format(os.path)
                    )

        # 新規作成がFalseでパスが見つからない場合
        elif not create_new_if_not_exist:
            raise OriginNotFoundError(
                "The file was not found\n \
                The path: {}".format(path_to_origin_file)
            )
        # 一旦新規作成して保存
        else:
            print("Generating a new file...")
            self.__core.Save(self.path)
            
        
        OriginInstance.__instance_count += 1

        self.__core.LT_set_var("@VIS", 100)

        print("Origin booted")

        # エラーが投げられたら即座に終了
        def origin_shutdown_exception_hook(exctype, value, traceback):
            '''Ensures Origin gets shut down if an uncaught exception'''
            self.close()
            sys.__excepthook__(exctype, value, traceback)
        if oext:
            sys.excepthook = origin_shutdown_exception_hook

    def __del__(self):
        self.close()
    
    def close(self, save_flag: bool = True):
        '''Originのインスタンスを終了する'''
        if save_flag:
            self.__core.Save(self.path)
        self.__core.Exit()

        OriginInstance.__instance_count = OriginInstance.__instance_count - 1
        
        OriginInstance.__instance_path_list.remove(self.path)
    
    def get_root_dir(self):
        '''Originのルートディレクトリを取得する'''
        return self.__core.GetRootFolder()
    
    def lt_exec_cmnd(self, command: str)->None:
        """
        ## 概要
        OriginにLab talkのコマンドを送って実行する。
        一部のコマンドには例外処理を挟み、バグを減らしているが、網羅しているわけではないので利用は計画的に。

        ## 例外処理
        exit: 代わりにclose()処理を実行する
        """

        match (command):
            case "exit":
                self.close()
            case _:
                self.__core.lt_exec(command)
    
    def is_valid(self):
        return bool(self.__core)


# def open_origin(
#     path_to_origin_file: str,
#     #readonly: bool = False,
#     create_new_if_not_exist: bool = True,
# )-> APP:
#     """
#     Originのプロジェクトファイルを開く。
#     path_to_origin_fileで指定したパスにファイルが存在しない場合、create_new_if_not_existがTrueなら新規作成する。
#     Falseなら例外を投げる。

#     Args:
#         path_to_origin_file (str): 開きたいOriginプロジェクトファイルのパス
#         create_new_if_not_exist (bool, optional): ファイルが無かった場合に新規作成するかどうか. Defaults to True.

#     Raises:
#         FileNotFoundError: path_to_origin_fileで指定したパスにファイルが存在しない場合に発生

#     Returns:
#         None
#     """
    
#     #なぜか//をファイル名の一部として認識するので変換
#     path = path_to_origin_file.replace("//", "\\")


#     if os.path.isfile(path):

#         return op.open(path)

#     else:
#         if create_new_if_not_exist:
#             op.new()
#             return op.save(path)
#         else:
#             raise FileNotFoundError(f"File not found: {path_to_origin_file}")
        

# def get_work_sheet(
#     sheet_name: str,
#     create_new_if_not_exist: bool = True,
# ) -> op.WSheet:
#     """
#     指定した名前のシートを取得する。
#     long nameも可だが、long nameは重複可能なことに注意。
#     一番最初に見つけた1枚を返す。
#     シートが存在しない場合、create_new_if_not_existがTrueなら新規作成する。
#     Falseなら例外を投げる。

#     Args:
#         sheet_name (str): 取得したいシートの名前
#         create_new_if_not_exist (bool, optional): シートが無かった場合に新規作成するかどうか. Defaults to True.

#     Raises:
#         ValueError: sheet_nameで指定した名前のシートが存在しない場合に発生

#     Returns:
#         op.WSheet: 指定した名前のシート
#     """

#     list_book:list[op.WBook] = []
#     book_generator: Generator[op.WBook, None, None] = op.pages("w") # type: ignore #これはoriginproの方の返り値が違う
#     for sheet in sheet_generator:
#         list_sheet.append(sheet)
    
#     #long nameを優先で検査
#     for sheet in list_sheet:
#         if sheet.lname == sheet_name:
#             return sheet.unembed_sheet
        
#     #short nameで検査
#     for sheet in list_sheet:
#         if sheet.name == sheet_name:
#             return sheet

#     #どれにも該当しなかった場合、ファイルを作るかError送出
#     if create_new_if_not_exist:
#         new_sheet: op.WSheet = op.new_sheet("w", sheet_name)
#         return new_sheet
#     else:
#         raise OriginNotFoundError("A sheet named as \"{}\" was not found.".format(sheet_name))


# #def get_work_book()
"""
# Libraries for Biofunctional Catalyst research team (Nakamura lab.)
\n
まだ日本語のdoc_stringしか書けてない。
英語はできたらやる。

### Python version: 3.12.2
match制御構文の使用 (>3.10)\n
typing.Self型の使用 (>3.11)\n


## データ階層
疑似的な抽象クラスを継承した、次のような階層になる
    DataFile
        :-DataSeriese(or list[DataSeriese])
            :-ValueObjectArray(numpy.array)
                :-[ValueObject, ValueObject, ValueObject, ..., ValueObject,]
            :-ValueObjectArray  
            :-ValueObjectArray  
            ...
## 基本理念
・ 直観でPythonによるデータ読み込み、処理を可能とすること(pandas.read_csv()及びカラム処理の自動化)\n
・ プリミティブ型を極力使わないこと（値は値オブジェクトを用いる）\n
・ 値の編集を許可しないこと（再代入の禁止は仕様上無理だが、主たるオブジェクトは基本メンバ書き換えを禁止する）\n
・ 型ヒントにAnyを表示させない\n
・ 上記を達成したうえで極力高速化（numpyの使用、for文を減らす、など）\n

## やりたいこと
・ CythonかRustへの一部移行（高速化のため）\n
"""
#%%
import struct
import csv

mdp_path = "20211214 FK002LSV-.MDP" #ここを入力ファイルに書き換え
out_csv = "MDP_extracted_columns.csv"

base = 22425   # 測定ブロックの先頭オフセット（今回のファイル）
stride = 69    # レコード長（バイト）

rows = []
with open(mdp_path, "rb") as f:
    data = f.read()
    n_records = (len(data) - base) // stride
    for i in range(n_records):
        off = base + i * stride
        c1 = struct.unpack_from("<d", data, off + 0)[0]
        c2 = struct.unpack_from("<d", data, off + 10)[0]
        c3 = struct.unpack_from("<d", data, off + 20)[0]
        c4 = struct.unpack_from("<d", data, off + 30)[0]
        rows.append((c1, c2, c3, c4))

with open(out_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["C1","C2","C3","C4"])
    writer.writerows(rows)

print("Saved", len(rows), "records to", out_csv)
import pandas as pd
import os, warnings
import multiprocessing
warnings.filterwarnings("ignore")

# 子进程需要共享的全局变量 - 文件列表
rtn_files = os.listdir('./cache/dist/info/')

def handle_cross_section(i):
    global rtn_files
    factor = pd.read_csv('./cache/dist/factor/' + rtn_files[i-1], low_memory=False)
    info = pd.read_csv('./cache/dist/info/' + rtn_files[i], low_memory=False)

    factor.drop_duplicates(subset=['permno', 'yyyymm'], inplace=True)

    info.drop_duplicates(subset=['PERMNO', 'DATE'], inplace=True)

    factor['permno'] = factor['permno'].astype('int')
    info['Permno'] = info['PERMNO'].astype('int')
    info.rename(columns={'PERMNO': 'permno'}, inplace=True)
    
    merged = info.dropna(subset=['RET'])[['permno','RET']].merge(factor, on='permno', how='left')

    merged.to_csv('./cache/dist/merge/' + rtn_files[i-1], index=False)

    print("Merged file:", rtn_files[i-1], "with", rtn_files[i], "and saved to cache/dist/merge/")


def get_file(file_path):

    return pd.read_csv(file_path, low_memory=False)

if __name__ == "__main__":

    n_subprocesses = 10; pool = multiprocessing.Pool(processes=n_subprocesses)

    print("Start merging cross-section data..., please wait...")

    print("Total files to process:", rtn_files.__len__() - 1)

    print("Using", n_subprocesses, "subprocesses...")

    pool.map(handle_cross_section, list(range(1, rtn_files.__len__())))

    pool.close(); pool.join()


    files_to_concat = os.listdir('./cache/dist/merge/')

    files_to_concat = ['./cache/dist/merge/' + file for file in files_to_concat]

    pool = multiprocessing.Pool(processes=8)

    print("Start concatenating files..., please wait...")

    print("Total files to concatenate:", len(files_to_concat))

    print("Using 8 subprocesses...")

    data = pool.map(get_file, files_to_concat)
    
    print("Concatenation completed, total rows:", data.shape[0], "and columns:", data.shape[1])
    
    # 删除 observation 太少的列 无法处理参与训练
    # 选择空值最少的前 30 列 保留这 30 列

    # 计算每列的空值数量并排序
    null_counts = data.isnull().sum().sort_values()

    # 选择空值最少的前30列（除了permno和RET列，它们必须保留）
    essential_cols = ['permno', 'RET', 'yyyymm']
    other_cols = [col for col in null_counts.index if col not in essential_cols]
    selected_cols = essential_cols + other_cols[:30]  # 28 + 2 = 30列

    # 保留选定的列
    data = data[selected_cols]

    print(f"Selected {len(selected_cols)} columns with least missing values")

    data.to_csv('./cache/FactorAndReturn.csv', index=False)


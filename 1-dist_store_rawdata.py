# 分布式存储

print('Raw data alreadly stored in "./cache/dist/raw/".' \
'\nPlease unlock this script mannually to re-run it.')

# with open('./data/signed_predictors_dl_wide.csv', 'r') as f:
#     columns = f.readline()
#     content = f.readline()
#     file_dict = {}
#     while content:
#         yyyymm = content.split(',')[1]
#         # 如果不存在这个csv就创建一个
#         if yyyymm not in file_dict.keys():
#             file = open(f'./cache/dist/factor/{yyyymm}.csv', 'w')
#             file.write(columns)
#             file_dict[yyyymm] = file
#         # 写入内容
#         file = file_dict[yyyymm]
#         file.write(content)
#         # 读取下一行
#         content = f.readline()
    
#     # 写完后 关闭全部文件
#     for file in file_dict.values():
#         file.close()

print('Firm-specific info alreadly stored in "./cache/dist/info/".' \
'\nPlease unlock this script mannually to re-run it.')

# with open('./data/raw_info_and_returns.csv', 'r') as f:
#     columns = f.readline()
#     content = f.readline()
#     file_dict = {}
#     while content:
#         yyyymm = content.split(',')[1]
#         # 修改 yymm 的格式
#         yyyymm = yyyymm[:yyyymm.rfind('-')].replace('-', '')
#         # 如果不存在这个csv就创建一个
#         if yyyymm not in file_dict.keys():
#             file = open(f'./cache/dist/info/{yyyymm}.csv', 'w')
#             file.write(columns)
#             file_dict[yyyymm] = file
#         # 写入内容
#         file = file_dict[yyyymm]
#         file.write(content)
#         # 读取下一行
#         content = f.readline()
    
#     # 写完后 关闭全部文件
#     for file in file_dict.values():
#         file.close()


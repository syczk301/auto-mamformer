from bsm2_python import BSM2OL
import pandas as pd

# 创建开环模型（默认就是官方609天动态进水）
bsm2 = BSM2OL()

# 运行一次（生成进水时间序列）
bsm2.simulate()

# 读取进水数据
influent = bsm2.y_in_all     # 进水矩阵
time = bsm2.simtime         # 时间（单位：天）

# 官方进水变量名（按BSM2标准顺序）
cols = [
    "SI","SS","XI","XS","XBH","XBA","XP","SO","SNO","SNH","SND","XND","SALK",
    "TSS","Q","TEMP","SD1","SD2","SD3","XD4","XD5"
]

df = pd.DataFrame(influent, columns=cols)
df.insert(0, "time_day", time)

# 保存为CSV
df.to_csv("bsm2_609day_influent.csv", index=False)

print("已生成官方609天动态进水文件: bsm2_609day_influent.csv")

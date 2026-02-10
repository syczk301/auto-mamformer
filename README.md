# Auto-Mamformer

基于 Auto-Mamformer 的废水处理预测项目，包含两套任务：

- `Water` 数据集：预测 `COD-S` / `BOD-S`
- `BSM2` 数据集：预测 `COD` / `BOD5`

同时提供了 Web 结果看板（Water/BSM2 两个可切换页面）。

## 1. 目录结构

```text
code/
  water/
    auto_mamformer_cod.py
    auto_mamformer_bod.py
  BSM2/
    auto_mamformer_bsm2.py
    auto_mamformer_bsm2_cod.py
    auto_mamformer_bsm2_bod.py
data/
  water/water-treatment_model_cleaned.csv
  BSM2/bsm2_full_data.csv
web/
  index.html
  bsm2.html
result/
model/
```

## 2. 环境准备

建议 Python 3.10+，安装依赖：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

如果需要自己生成 BSM2 故障数据，额外安装：

```bash
pip install bsm2-python
```

## 3. 如何启动训练

请在项目根目录执行命令（即本 README 所在目录）。

### 3.1 Water 数据

训练 COD：

```bash
python code/water/auto_mamformer_cod.py
```

训练 BOD：

```bash
python code/water/auto_mamformer_bod.py
```

说明：

- 脚本会自动定位 `data/water/water-treatment_model_cleaned.csv`
- 结果默认输出到 `result/`，模型默认保存到 `model/`

### 3.2 BSM2 数据

统一入口（可选 cod / bod / both）：

```bash
python code/BSM2/auto_mamformer_bsm2.py --target both
```

仅训练 COD：

```bash
python code/BSM2/auto_mamformer_bsm2.py --target cod
```

仅训练 BOD5：

```bash
python code/BSM2/auto_mamformer_bsm2.py --target bod
```

快捷入口脚本：

```bash
python code/BSM2/auto_mamformer_bsm2_cod.py
python code/BSM2/auto_mamformer_bsm2_bod.py
```

常用加速参数（BSM2）：

```bash
python code/BSM2/auto_mamformer_bsm2.py --target cod --batch-size 128 --val-every 10 --r2-every 10 --num-workers 0
```

## 4. 如何启动 Web 看板

先在项目根目录启动本地静态服务：

```bash
python -m http.server 8000
```

然后在浏览器打开：

- Water 页面：`http://localhost:8000/web/index.html`
- BSM2 页面：`http://localhost:8000/web/bsm2.html`

## 5. 结果文件说明

主要输出在 `result/`：

- Water：`auto_mamformer_cod_results.png`、`auto_mamformer_bod_results.png`
- BSM2：`auto_mamformer_bsm2_cod_results.png`、`auto_mamformer_bsm2_bod_results.png`
- BSM2 汇总：`auto_mamformer_bsm2_summary.json`
- Water 汇总：`auto_mamformer_water_summary.json`

## 6. 常见问题

### Q1: 报错找不到 `water-treatment_model_cleaned.csv`

请确认文件存在于：

```text
data/water/water-treatment_model_cleaned.csv
```

并从项目根目录启动脚本。

### Q2: BSM2 训练每轮耗时较长

建议先用单目标 + 大 batch + 降低验证频率：

```bash
python code/BSM2/auto_mamformer_bsm2.py --target cod --batch-size 128 --val-every 10 --r2-every 10
```

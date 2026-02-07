# Results Web Dashboard

## 说明
`web/` 目录提供一个静态结果看板，用于系统展示 `result/` 下的指标、图像和工件文件。

## 启动方式
在项目根目录运行：

```bash
python -m http.server 8000
```

然后在浏览器访问：

```text
http://localhost:8000/web/
```

## 页面入口
- Water 页面: `http://localhost:8000/web/index.html`
- BSM2 页面: `http://localhost:8000/web/bsm2.html`

两个页面顶部有切换导航，可互相跳转。

## 页面内容
- 核心指标卡
- 模型指标对比表
- 预测轨迹（Water 页面）
- 图像结果库（自动链接到 `../result/*.png`）
- 全部工件清单（image/json/npy）

## 数据来源
- `web/results-manifest-water.json`
- `web/results-manifest-bsm2.json`
- `result/compare_results.json`
- `result/auto_mamformer_bsm2_summary.json`

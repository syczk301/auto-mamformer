# Results Web Dashboard

## 说明
`web/` 目录提供一个静态结果看板，用于系统展示模型指标与结果图像。
图像资源使用网页静态目录 `web/assets/results/`，不再直接从 `../result` 加载。

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
- 图像结果库（链接到 `./assets/results/*.png`）

## 数据来源
- `web/results-manifest-water.json`
- `web/results-manifest-bsm2.json`
- `result/auto_mamformer_water_summary.json`
- `result/auto_mamformer_bsm2_summary.json`

## 更新网页图片
训练后如果有新图，请把 `result/` 下对应 PNG 同步到网页静态目录：

```powershell
Copy-Item result/*.png web/assets/results -Force
```

const PAGE = document.body.dataset.page || "water";

const CONFIG = {
  water: {
    manifest: "./results-manifest-water.json",
    summaryPath: "../result/auto_mamformer_water_summary.json",
    title: "Auto-Mamformer Water (BOD / COD)",
    showCompare: false,
    showTracks: true
  },
  bsm2: {
    manifest: "./results-manifest-bsm2.json",
    summaryPath: "../result/auto_mamformer_bsm2_summary.json",
    title: "Auto-Mamformer BSM2 (COD / BOD5)",
    showCompare: false,
    showTracks: false
  }
};

const CURRENT = CONFIG[PAGE] || CONFIG.water;

const safeNumber = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
};

async function fetchJson(path) {
  if (!path) return null;
  try {
    const resp = await fetch(path);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

function renderMeta(artifacts) {
  const now = new Date();
  const updated = document.getElementById("meta-updated");
  const count = document.getElementById("meta-count");
  updated.textContent = `页面刷新: ${now.toLocaleString()}`;
  count.textContent = `工件总数: ${artifacts.length}`;
}

function renderKpiCards(summary) {
  const kpiGrid = document.getElementById("kpi-grid");
  const tpl = document.getElementById("kpi-card-template");
  kpiGrid.innerHTML = "";

  const targets = [];
  if (summary) {
    if (summary.cod) {
      targets.push({
        name: PAGE === "water" ? "COD-S" : "COD",
        m: summary.cod
      });
    }
    if (summary.bod) {
      targets.push({
        name: PAGE === "water" ? "BOD-S" : "BOD5",
        m: summary.bod
      });
    }
  }

  if (!targets.length) {
    const node = tpl.content.cloneNode(true);
    node.querySelector(".kpi-label").textContent = "暂无数据";
    node.querySelector(".kpi-value").textContent = "-";
    node.querySelector(".kpi-sub").textContent = "请先运行训练代码生成结果";
    kpiGrid.appendChild(node);
    return;
  }

  targets.forEach((t) => {
    const items = [
      { label: `${t.name} R2`, value: safeNumber(t.m.r2), sub: "决定系数" },
      { label: `${t.name} MAPE`, value: `${safeNumber(t.m.mape, 2)}%`, sub: "平均百分比误差" },
      { label: `${t.name} MAE`, value: safeNumber(t.m.mae), sub: "ug/m3" },
      { label: `${t.name} RMSE`, value: safeNumber(t.m.rmse), sub: "ug/m3" }
    ];
    items.forEach((item) => {
      const node = tpl.content.cloneNode(true);
      node.querySelector(".kpi-label").textContent = item.label;
      node.querySelector(".kpi-value").textContent = item.value;
      node.querySelector(".kpi-sub").textContent = item.sub;
      kpiGrid.appendChild(node);
    });
  });
}

function getSummaryRows(summary) {
  if (!summary) return [];
  const rows = [];
  if (summary.cod) {
    rows.push({
      name: PAGE === "water" ? "COD-S" : "COD",
      r2: Number(summary.cod.r2),
      mape: Number(summary.cod.mape),
      mae: Number(summary.cod.mae),
      rmse: Number(summary.cod.rmse),
      preds: Array.isArray(summary.cod.preds) ? summary.cod.preds : [],
      trues: Array.isArray(summary.cod.trues) ? summary.cod.trues : []
    });
  }
  if (summary.bod) {
    rows.push({
      name: PAGE === "water" ? "BOD-S" : "BOD5",
      r2: Number(summary.bod.r2),
      mape: Number(summary.bod.mape),
      mae: Number(summary.bod.mae),
      rmse: Number(summary.bod.rmse),
      preds: Array.isArray(summary.bod.preds) ? summary.bod.preds : [],
      trues: Array.isArray(summary.bod.trues) ? summary.bod.trues : []
    });
  }
  return rows;
}

function renderCompareTable(rows) {
  const compareSection = document.getElementById("compare-section");
  if (!compareSection) return;
  if (!CURRENT.showCompare) {
    compareSection.style.display = "none";
    return;
  }

  compareSection.style.display = "";
  const tbody = document.querySelector("#compare-table tbody");
  const bars = document.getElementById("r2-bars");
  tbody.innerHTML = "";
  bars.innerHTML = "";

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">暂无可显示的指标数据</td>`;
    tbody.appendChild(tr);
    return;
  }

  rows.sort((a, b) => b.r2 - a.r2);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.name}</td>
      <td>${safeNumber(row.r2, 4)}</td>
      <td>${safeNumber(row.mape, 2)}</td>
      <td>${safeNumber(row.mae, 4)}</td>
      <td>${safeNumber(row.rmse, 4)}</td>
    `;
    tbody.appendChild(tr);
  });

  const maxR2 = Math.max(...rows.map((r) => (Number.isFinite(r.r2) ? r.r2 : 0)), 1);
  rows.forEach((row) => {
    const widthPct = Math.max(0, (row.r2 / maxR2) * 100);
    const wrap = document.createElement("div");
    wrap.className = "bar-row";
    wrap.innerHTML = `
      <span class="bar-label">${row.name}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${widthPct.toFixed(1)}%"></div></div>
      <span class="bar-value">${safeNumber(row.r2, 4)}</span>
    `;
    bars.appendChild(wrap);
  });
}

function drawTrack(canvas, trues, preds) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!trues.length || !preds.length) {
    ctx.fillStyle = "#6b7d78";
    ctx.font = "12px SimSun";
    ctx.fillText("No data", 12, 20);
    return;
  }

  const n = Math.min(trues.length, preds.length, 240);
  const t = trues.slice(0, n);
  const p = preds.slice(0, n);
  const minV = Math.min(...t, ...p);
  const maxV = Math.max(...t, ...p);
  const span = Math.max(maxV - minV, 1e-9);
  const padX = 12;
  const padY = 10;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(30,42,58,0.14)";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  const drawLine = (arr, color) => {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    arr.forEach((v, i) => {
      const x = padX + (i / (n - 1)) * plotW;
      const y = padY + (1 - (v - minV) / span) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  drawLine(t, "#245a9a");
  drawLine(p, "#2f7c8f");
}

function renderTrackCards(rows) {
  const section = document.getElementById("track-section");
  const wrap = document.getElementById("track-grid");
  wrap.innerHTML = "";

  if (!CURRENT.showTracks) {
    section.style.display = "none";
    return;
  }

  section.style.display = "";
  rows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "track-card";
    card.innerHTML = `
      <h3>${row.name}</h3>
      <canvas width="360" height="160"></canvas>
      <p class="track-hint">
        <span class="tag">蓝线: true</span>
        <span class="tag">青线: pred</span>
      </p>
    `;
    wrap.appendChild(card);
    drawTrack(card.querySelector("canvas"), row.trues, row.preds);
  });
}

function renderFigureGallery(artifacts) {
  const gallery = document.getElementById("figure-gallery");
  gallery.innerHTML = "";
  const images = artifacts.filter((a) => a.type === "image");
  images.forEach((img) => {
    const src = `../result/${img.name}`;
    const card = document.createElement("article");
    card.className = "figure-card";
    card.innerHTML = `
      <a href="${src}" target="_blank" rel="noopener noreferrer">
        <img src="${src}" alt="${img.name}" loading="lazy">
        <div class="figure-meta">
          <p class="figure-name">${img.name}</p>
          <p class="figure-group">${img.group} · ${img.description}</p>
        </div>
      </a>
    `;
    gallery.appendChild(card);
  });
}

function renderArtifactTable(artifacts) {
  const tbody = document.querySelector("#artifact-table tbody");
  tbody.innerHTML = "";
  artifacts.forEach((item) => {
    const href = `../result/${item.name}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.name}</td>
      <td>${item.type}</td>
      <td>${item.group}</td>
      <td>${item.description}</td>
      <td><a href="${href}" target="_blank" rel="noopener noreferrer">打开</a></td>
    `;
    tbody.appendChild(tr);
  });
}

function setTitle() {
  const title = document.getElementById("compare-title");
  if (title) title.textContent = CURRENT.title;
}

async function boot() {
  setTitle();
  const manifest = await fetchJson(CURRENT.manifest);
  const artifacts = manifest?.artifacts ?? [];
  const summary = await fetchJson(CURRENT.summaryPath);

  const rows = getSummaryRows(summary);

  renderMeta(artifacts);
  renderKpiCards(summary);
  renderCompareTable(rows);
  renderTrackCards(rows);
  renderFigureGallery(artifacts);
  renderArtifactTable(artifacts);
}

boot();

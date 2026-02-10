const PAGE = document.body.dataset.page || "water";

const CONFIG = {
  water: {
    manifest: "./results-manifest-water.json",
    summaryPath: [
      "./assets/results/auto_mamformer_water_summary.json",
      "../result/auto_mamformer_water_summary.json"
    ],
    title: "Auto-Mamformer Water (BOD / COD)",
    showCompare: false,
    showTracks: true,
    stackImagesVertically: false,
    fallbackImages: [
      {
        name: "auto_mamformer_bod_results.png",
        type: "image",
        group: "water",
        description: "Water BOD 预测结果图"
      },
      {
        name: "auto_mamformer_cod_results.png",
        type: "image",
        group: "water",
        description: "Water COD 预测结果图"
      }
    ]
  },
  bsm2: {
    manifest: "./results-manifest-bsm2.json",
    summaryPath: [
      "./assets/results/auto_mamformer_bsm2_summary.json",
      "../result/auto_mamformer_bsm2_summary.json"
    ],
    title: "Auto-Mamformer BSM2 (COD / BOD5)",
    showCompare: false,
    showTracks: false,
    stackImagesVertically: true,
    fallbackImages: [
      {
        name: "auto_mamformer_bsm2_cod_results.png",
        type: "image",
        group: "bsm2",
        description: "BSM2 COD 预测结果图"
      },
      {
        name: "auto_mamformer_bsm2_bod_results.png",
        type: "image",
        group: "bsm2",
        description: "BSM2 BOD5 预测结果图"
      }
    ]
  }
};

const CURRENT = CONFIG[PAGE] || CONFIG.water;

const safeNumber = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
};

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function metricKeyFromLabel(label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("r2")) return "r2";
  if (key.includes("mape")) return "mape";
  if (key.includes("mae")) return "mae";
  if (key.includes("rmse")) return "rmse";
  return "default";
}

function joinPath(root, name) {
  const cleanRoot = String(root || "").replace(/[\\]+/g, "/").replace(/\/+$/, "");
  const cleanName = String(name || "").replace(/^\/+/, "");
  if (!cleanRoot) return cleanName;
  if (!cleanName) return cleanRoot;
  return `${cleanRoot}/${cleanName}`;
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  values.forEach((v) => {
    const text = String(v || "").trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    out.push(text);
  });
  return out;
}

function buildImageCandidates(imageRoot, name) {
  return uniqueStrings([
    joinPath(imageRoot, name),
    joinPath("./assets/results", name),
    joinPath("../result", name),
    name
  ]);
}

function imageOrderKey(name) {
  const text = String(name || "").toLowerCase();
  if (text.includes("cod")) return 0;
  if (text.includes("bod")) return 1;
  return 2;
}

function assignImageCandidates(imgEl, candidates) {
  if (!imgEl) return;
  const list = uniqueStrings(candidates || []);
  if (!list.length) return;
  let idx = 0;
  imgEl.src = list[idx];
  imgEl.onerror = () => {
    idx += 1;
    if (idx < list.length) {
      imgEl.src = list[idx];
      return;
    }
    imgEl.onerror = null;
  };
}

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

async function fetchJsonFirst(paths) {
  const list = Array.isArray(paths) ? paths : [paths];
  for (const p of list) {
    const data = await fetchJson(p);
    if (data) return data;
  }
  return null;
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

function renderSnapshot(artifacts, rows) {
  const now = new Date();
  const imageCount =
    artifacts.filter((item) => item.type === "image").length ||
    (CURRENT.fallbackImages || []).length;
  const validRows = rows.filter((row) => Number.isFinite(row.r2));
  const bestRow = validRows.sort((a, b) => b.r2 - a.r2)[0];

  setText("meta-updated", now.toLocaleString());
  setText("meta-artifacts", String(artifacts.length));
  setText("meta-images", String(imageCount));
  setText("meta-models", String(rows.length));
  setText(
    "meta-best-r2",
    bestRow ? `${bestRow.name} ${safeNumber(bestRow.r2, 4)}` : "-"
  );
  setText("meta-page-tag", PAGE.toUpperCase());
}

function renderKpiCards(summary) {
  const kpiGrid = document.getElementById("kpi-grid");
  const tpl = document.getElementById("kpi-card-template");
  if (!kpiGrid || !tpl) return;
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
    const card = node.querySelector(".kpi-card");
    if (card) card.dataset.metric = "default";
    node.querySelector(".kpi-label").textContent = "暂无数据";
    node.querySelector(".kpi-value").textContent = "-";
    node.querySelector(".kpi-sub").textContent = "请先运行训练代码生成结果";
    kpiGrid.appendChild(node);
    return;
  }

  targets.forEach((target) => {
    const items = [
      { label: `${target.name} R2`, value: safeNumber(target.m.r2), sub: "决定系数" },
      {
        label: `${target.name} MAPE`,
        value: `${safeNumber(target.m.mape, 2)}%`,
        sub: "平均百分比误差"
      },
      { label: `${target.name} MAE`, value: safeNumber(target.m.mae), sub: "ug/m3" },
      { label: `${target.name} RMSE`, value: safeNumber(target.m.rmse), sub: "ug/m3" }
    ];

    items.forEach((item) => {
      const node = tpl.content.cloneNode(true);
      const card = node.querySelector(".kpi-card");
      if (card) card.dataset.metric = metricKeyFromLabel(item.label);
      node.querySelector(".kpi-label").textContent = item.label;
      node.querySelector(".kpi-value").textContent = item.value;
      node.querySelector(".kpi-sub").textContent = item.sub;
      kpiGrid.appendChild(node);
    });
  });
}

function renderCompareTable(rows) {
  const compareSection = document.getElementById("compare-section");
  if (!compareSection) return;
  if (!CURRENT.showCompare) {
    compareSection.style.display = "none";
    return;
  }

  const tbody = document.querySelector("#compare-table tbody");
  const bars = document.getElementById("r2-bars");
  if (!tbody || !bars) return;

  compareSection.style.display = "";
  tbody.innerHTML = "";
  bars.innerHTML = "";

  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">暂无可显示的指标数据</td>`;
    tbody.appendChild(tr);
    return;
  }

  const sortedRows = [...rows].sort((a, b) => b.r2 - a.r2);
  sortedRows.forEach((row, index) => {
    const tr = document.createElement("tr");
    if (index === 0) tr.classList.add("is-best");
    tr.innerHTML = `
      <td>${row.name}</td>
      <td>${safeNumber(row.r2, 4)}</td>
      <td>${safeNumber(row.mape, 2)}</td>
      <td>${safeNumber(row.mae, 4)}</td>
      <td>${safeNumber(row.rmse, 4)}</td>
    `;
    tbody.appendChild(tr);
  });

  const maxR2 = Math.max(...sortedRows.map((r) => (Number.isFinite(r.r2) ? r.r2 : 0)), 1);
  sortedRows.forEach((row) => {
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
    ctx.fillStyle = "#5a7398";
    ctx.font = "12px 'IBM Plex Mono'";
    ctx.fillText("No data", 12, 20);
    return;
  }

  const n = Math.min(trues.length, preds.length, 260);
  if (n < 2) {
    ctx.fillStyle = "#5a7398";
    ctx.font = "12px 'IBM Plex Mono'";
    ctx.fillText("No data", 12, 20);
    return;
  }

  const trueSeries = trues.slice(0, n);
  const predSeries = preds.slice(0, n);
  const minV = Math.min(...trueSeries, ...predSeries);
  const maxV = Math.max(...trueSeries, ...predSeries);
  const span = Math.max(maxV - minV, 1e-9);

  const padX = 12;
  const padY = 10;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  ctx.fillStyle = "#fafdff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(17, 68, 120, 0.16)";
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

  drawLine(trueSeries, "#0f4c81");
  drawLine(predSeries, "#cf7b11");
}

function renderTrackCards(rows) {
  const section = document.getElementById("track-panel");
  const wrap = document.getElementById("track-grid");
  if (!section || !wrap) return false;
  wrap.innerHTML = "";

  if (!CURRENT.showTracks) {
    section.style.display = "none";
    return false;
  }

  const validRows = rows.filter(
    (row) =>
      Array.isArray(row.trues) &&
      Array.isArray(row.preds) &&
      row.trues.length > 1 &&
      row.preds.length > 1
  );

  if (!validRows.length) {
    section.style.display = "none";
    return false;
  }

  section.style.display = "";
  validRows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "track-card";
    card.innerHTML = `
      <h3>${row.name}</h3>
      <canvas width="420" height="180"></canvas>
      <p class="track-hint">
        <span class="tag">蓝线: true</span>
        <span class="tag">橙线: pred</span>
      </p>
    `;
    wrap.appendChild(card);
    drawTrack(card.querySelector("canvas"), row.trues, row.preds);
  });
  return true;
}

function renderFigureGallery(artifacts, imageRoot = "./assets/results", fallbackImages = []) {
  const gallery = document.getElementById("figure-gallery");
  const preview = document.getElementById("figure-preview");
  const previewImg = document.getElementById("figure-preview-img");
  const previewTitle = document.getElementById("figure-preview-title");
  const previewDesc = document.getElementById("figure-preview-desc");
  if (!gallery) return false;

  gallery.innerHTML = "";
  const images = artifacts.filter((a) => a.type === "image");
  const sourceImages = images.length ? images : fallbackImages;

  if (!sourceImages.length) {
    if (preview) preview.hidden = true;
    gallery.innerHTML = `<article class="figure-empty">暂无图像结果，请先运行训练脚本生成 PNG 文件。</article>`;
    return false;
  }

  const sortedImages = [...sourceImages].sort((a, b) => {
    const diff = imageOrderKey(a?.name) - imageOrderKey(b?.name);
    if (diff !== 0) return diff;
    return String(a?.name || "").localeCompare(String(b?.name || ""));
  });

  if (CURRENT.stackImagesVertically) {
    gallery.classList.add("gallery-stacked");
    if (preview) preview.hidden = true;

    sortedImages.forEach((img) => {
      const candidates = buildImageCandidates(imageRoot, img.name);
      const card = document.createElement("article");
      card.className = "figure-card figure-card-static";
      card.innerHTML = `
        <img alt="${img.name}" loading="lazy">
        <div class="figure-meta">
          <p class="figure-name">${img.name}</p>
          <p class="figure-group">${img.group} · ${img.description}</p>
        </div>
      `;
      const imageEl = card.querySelector("img");
      assignImageCandidates(imageEl, candidates);
      gallery.appendChild(card);
    });
    return true;
  }

  gallery.classList.remove("gallery-stacked");

  const cards = [];
  const updatePreview = (item) => {
    if (!item || !preview || !previewImg || !previewTitle || !previewDesc) return;
    preview.hidden = false;
    previewTitle.textContent = item.name || "图像结果";
    previewDesc.textContent = `${item.group || "default"} · ${item.description || ""}`;
    assignImageCandidates(previewImg, item.candidates);
    cards.forEach((entry) => {
      entry.card.classList.toggle("active", entry.item.name === item.name);
    });
  };

  sortedImages.forEach((img, index) => {
    const candidates = buildImageCandidates(imageRoot, img.name);
    const item = { ...img, candidates };

    const card = document.createElement("article");
    card.className = "figure-card";
    card.innerHTML = `
      <button class="figure-card-btn" type="button" aria-label="查看 ${img.name}">
        <img alt="${img.name}" loading="lazy">
        <div class="figure-meta">
          <p class="figure-name">${img.name}</p>
          <p class="figure-group">${img.group} · ${img.description}</p>
        </div>
      </button>
    `;

    const button = card.querySelector(".figure-card-btn");
    const imageEl = card.querySelector("img");
    assignImageCandidates(imageEl, candidates);
    button.addEventListener("click", () => updatePreview(item));

    cards.push({ card, item });
    gallery.appendChild(card);

    if (index === 0) updatePreview(item);
  });

  return true;
}

function syncVizLayout(hasTracks) {
  const layout = document.getElementById("viz-layout");
  const figurePanel = document.getElementById("figure-panel");
  if (layout) layout.classList.toggle("track-hidden", !hasTracks);
  if (figurePanel) figurePanel.classList.toggle("full-width", !hasTracks);
}

function setTitle() {
  const title = document.getElementById("compare-title");
  if (title) title.textContent = CURRENT.title;
}

async function boot() {
  setTitle();
  const manifest = await fetchJsonFirst(CURRENT.manifest);
  const artifacts = manifest?.artifacts ?? [];
  const imageRoot = manifest?.image_root || "./assets/results";
  const summary = await fetchJsonFirst(CURRENT.summaryPath);
  const rows = getSummaryRows(summary);

  renderSnapshot(artifacts, rows);
  renderKpiCards(summary);
  renderCompareTable(rows);
  const hasTracks = renderTrackCards(rows);
  renderFigureGallery(artifacts, imageRoot, CURRENT.fallbackImages || []);
  syncVizLayout(hasTracks);
}

boot();

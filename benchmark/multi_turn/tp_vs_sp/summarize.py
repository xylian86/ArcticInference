#!/usr/bin/env python3
"""Parse benchmark log files and generate an interactive HTML dashboard."""

import os
import re
import json
import glob
from collections import defaultdict

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(LOG_DIR, "benchmark_dashboard.html")


def parse_filename(filename):
    """Extract input_len, output_len, num_requests, strategy, run from filename.

    Supports both original files (e.g. i128_o128_n32_SP.log) and per-run
    files (e.g. i128_o128_n32_sp_run2.log).  Returns ``run=0`` for the
    original files and ``run=N`` for per-run files.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Per-run file: i128_o128_n32_sp_run2
    m = re.match(r"i(\d+k?)_o(\d+k?)_n(\d+)_(\w+)_run(\d+)", basename)
    if m:
        return {
            "input_len": m.group(1),
            "output_len": m.group(2),
            "num_requests": int(m.group(3)),
            "strategy": m.group(4).upper(),
            "run": int(m.group(5)),
            "label": basename,
        }

    # Original file: i128_o128_n32_SP
    m = re.match(r"i(\d+k?)_o(\d+k?)_n(\d+)_(\w+)", basename)
    if not m:
        return None
    return {
        "input_len": m.group(1),
        "output_len": m.group(2),
        "num_requests": int(m.group(3)),
        "strategy": m.group(4),
        "run": 0,
        "label": basename,
    }


def parse_size(s):
    """Convert '8k' -> 8192, '512' -> 512."""
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def parse_log(filepath):
    """Extract benchmark metrics from the tail of a log file."""
    with open(filepath, "r") as f:
        content = f.read()

    result = {}

    patterns = {
        "benchmark_duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "successful_requests": r"Successful requests:\s+(\d+)",
        "failed_requests": r"Failed requests:\s+(\d+)",
        "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_tok_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            result[key] = float(m.group(1))

    if len(result) < len(patterns):
        return None
    return result


def main():
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    print(f"Found {len(log_files)} log files")

    # Parse all log files, grouping by (config_key, strategy, run)
    all_parsed = []
    parse_failures = 0
    for lf in sorted(log_files):
        meta = parse_filename(lf)
        if not meta:
            parse_failures += 1
            continue
        metrics = parse_log(lf)
        if not metrics:
            parse_failures += 1
            continue
        meta.update(metrics)
        meta["input_tokens"] = parse_size(meta["input_len"])
        meta["output_tokens"] = parse_size(meta["output_len"])
        all_parsed.append(meta)

    # Build averaged records: for strategies with run2+run3 data, use their
    # average; otherwise fall back to the original (run0) data.
    METRIC_KEYS = [
        "benchmark_duration_s", "successful_requests", "failed_requests",
        "request_throughput", "output_tok_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    ]

    grouped = defaultdict(list)
    for rec in all_parsed:
        key = (rec["input_len"], rec["output_len"], rec["num_requests"],
               rec["strategy"])
        grouped[key].append(rec)

    records = []
    for key, recs in grouped.items():
        input_len, output_len, num_requests, strategy = key
        by_run = {r["run"]: r for r in recs}

        # Prefer average of run2 & run3; fall back to run0
        avg_runs = [r for r in [2, 3] if r in by_run]
        if len(avg_runs) >= 2:
            avg = {}
            for mk in METRIC_KEYS:
                avg[mk] = sum(by_run[r][mk] for r in avg_runs) / len(avg_runs)
            base = by_run[avg_runs[0]]
            label = f"i{input_len}_o{output_len}_n{num_requests}_{strategy}"
            rec = {
                "input_len": input_len,
                "output_len": output_len,
                "num_requests": num_requests,
                "strategy": strategy,
                "label": label,
                "input_tokens": base["input_tokens"],
                "output_tokens": base["output_tokens"],
            }
            rec.update(avg)
            records.append(rec)
        elif 0 in by_run:
            rec = dict(by_run[0])
            rec.pop("run", None)
            records.append(rec)

    print(f"Parsed {len(all_parsed)} log entries, {parse_failures} failures")
    print(f"Produced {len(records)} dashboard records (avg of run2+run3 where available)")

    strategies = sorted(set(r["strategy"] for r in records))
    io_combos = sorted(
        set((r["input_len"], r["output_len"]) for r in records),
        key=lambda x: (parse_size(x[0]), parse_size(x[1])),
    )
    n_values = sorted(set(r["num_requests"] for r in records))

    records_json = json.dumps(records)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Results Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card-bg: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-muted: #8b8fa3;
    --accent-sp: #4fc3f7;
    --accent-tp: #ff7043;
    --accent-shift: #66bb6a;
    --accent-sp-bg: rgba(79,195,247,0.12);
    --accent-tp-bg: rgba(255,112,67,0.12);
    --accent-shift-bg: rgba(102,187,106,0.12);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .subtitle {{
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 24px;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 24px;
    align-items: flex-end;
  }}
  .control-group {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .control-group label {{
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  select, button {{
    background: var(--card-bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
  }}
  select:focus, button:focus {{ outline: 2px solid var(--accent-sp); }}
  button {{ transition: background 0.15s; }}
  button:hover {{ background: var(--border); }}
  button.active {{ background: var(--accent-sp); color: #000; border-color: var(--accent-sp); }}

  .tab-bar {{
    display: flex;
    gap: 2px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }}
  .tab-bar button {{
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 6px 6px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    background: transparent;
    color: var(--text-muted);
  }}
  .tab-bar button.active {{
    color: var(--accent-sp);
    border-bottom-color: var(--accent-sp);
    background: var(--accent-sp-bg);
  }}

  .view {{ display: none; }}
  .view.active {{ display: block; }}

  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
  }}
  .chart-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }}
  .chart-card h3 {{
    font-size: 0.95rem;
    margin-bottom: 12px;
    color: var(--text-muted);
  }}
  .chart-container {{
    position: relative;
    height: 320px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    background: var(--card-bg);
    border-radius: 10px;
    overflow: hidden;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th {{
    background: #14161e;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    position: sticky;
    top: 0;
    cursor: pointer;
  }}
  th:hover {{ color: var(--text); }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .strategy-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.75rem;
  }}
  .strategy-SP {{ background: var(--accent-sp-bg); color: var(--accent-sp); }}
  .strategy-TP {{ background: var(--accent-tp-bg); color: var(--accent-tp); }}
  .strategy-SHIFT {{ background: var(--accent-shift-bg); color: var(--accent-shift); }}

  .table-wrapper {{
    max-height: 75vh;
    overflow-y: auto;
    border-radius: 10px;
    border: 1px solid var(--border);
  }}

  .summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .summary-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }}
  .summary-card .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-sp);
  }}
  .summary-card .label {{
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
  }}
  .legend-inline {{
    display: flex; gap: 18px; margin-bottom: 12px;
    font-size: 0.8rem; color: var(--text-muted);
  }}
  .legend-dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
  }}
</style>
</head>
<body>

<h1>Benchmark Results Dashboard</h1>
<p class="subtitle">Parsed from {len(records)} log files &middot; Strategies: {', '.join(strategies)} &middot; I/O combos: {len(io_combos)}</p>

<div class="summary-cards" id="summaryCards"></div>

<div class="tab-bar" id="tabBar">
  <button class="active" data-tab="charts">Charts</button>
  <button data-tab="table">Full Table</button>
  <button data-tab="comparison">SP vs TP vs SHIFT</button>
</div>

<div id="viewCharts" class="view active">
  <div class="controls">
    <div class="control-group">
      <label>I/O Configuration</label>
      <select id="ioSelect"></select>
    </div>
    <div class="control-group">
      <label>Metric</label>
      <select id="metricSelect">
        <option value="mean_ttft_ms">Mean TTFT (ms)</option>
        <option value="median_ttft_ms">Median TTFT (ms)</option>
        <option value="p99_ttft_ms">P99 TTFT (ms)</option>
        <option value="mean_tpot_ms">Mean TPOT (ms)</option>
        <option value="median_tpot_ms">Median TPOT (ms)</option>
        <option value="p99_tpot_ms">P99 TPOT (ms)</option>
        <option value="benchmark_duration_s">Duration (s)</option>
        <option value="output_tok_throughput">Output Throughput (tok/s)</option>
      </select>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card">
      <h3 id="chartTitle1">Metric vs Num Requests</h3>
      <div class="chart-container"><canvas id="chart1"></canvas></div>
    </div>
    <div class="chart-card">
      <h3 id="chartTitle2">TTFT Breakdown</h3>
      <div class="chart-container"><canvas id="chart2"></canvas></div>
    </div>
    <div class="chart-card">
      <h3 id="chartTitle3">TPOT Breakdown</h3>
      <div class="chart-container"><canvas id="chart3"></canvas></div>
    </div>
    <div class="chart-card">
      <h3 id="chartTitle4">Duration vs Num Requests</h3>
      <div class="chart-container"><canvas id="chart4"></canvas></div>
    </div>
  </div>
</div>

<div id="viewTable" class="view">
  <div class="controls">
    <div class="control-group">
      <label>Filter Strategy</label>
      <select id="tableStrategyFilter">
        <option value="all">All</option>
      </select>
    </div>
    <div class="control-group">
      <label>Filter I/O</label>
      <select id="tableIOFilter">
        <option value="all">All</option>
      </select>
    </div>
  </div>
  <div class="table-wrapper">
    <table id="dataTable">
      <thead><tr>
        <th data-col="label">Config</th>
        <th data-col="strategy">Strategy</th>
        <th data-col="input_tokens">Input</th>
        <th data-col="output_tokens">Output</th>
        <th data-col="num_requests">N</th>
        <th data-col="benchmark_duration_s">Duration(s)</th>
        <th data-col="mean_ttft_ms">TTFT Mean</th>
        <th data-col="median_ttft_ms">TTFT Med</th>
        <th data-col="p99_ttft_ms">TTFT P99</th>
        <th data-col="mean_tpot_ms">TPOT Mean</th>
        <th data-col="median_tpot_ms">TPOT Med</th>
        <th data-col="p99_tpot_ms">TPOT P99</th>
        <th data-col="output_tok_throughput">Out Thr</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div id="viewComparison" class="view">
  <div class="controls">
    <div class="control-group">
      <label>I/O Configuration</label>
      <select id="cmpIOSelect"></select>
    </div>
  </div>
  <div class="legend-inline">
    <span><span class="legend-dot" style="background:var(--accent-sp)"></span>SP</span>
    <span><span class="legend-dot" style="background:var(--accent-tp)"></span>TP</span>
    <span><span class="legend-dot" style="background:var(--accent-shift)"></span>SHIFT</span>
  </div>
  <div class="charts-grid">
    <div class="chart-card">
      <h3>Mean TTFT (ms) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart1"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>P99 TTFT (ms) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart2"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Mean TPOT (ms) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart3"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>P99 TPOT (ms) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart4"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Duration (s) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart5"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Output Throughput (tok/s) vs N</h3>
      <div class="chart-container"><canvas id="cmpChart6"></canvas></div>
    </div>
  </div>
</div>

<script>
const DATA = {records_json};

const COLORS = {{
  SP: '#4fc3f7',
  TP: '#ff7043',
  SHIFT: '#66bb6a',
}};
const COLORS_BG = {{
  SP: 'rgba(79,195,247,0.15)',
  TP: 'rgba(255,112,67,0.15)',
  SHIFT: 'rgba(102,187,106,0.15)',
}};

const strategies = [...new Set(DATA.map(d => d.strategy))].sort();
const ioOptions = [...new Set(DATA.map(d => d.input_len + '_' + d.output_len))];

function parseSize(s) {{
  return s.endsWith('k') ? parseInt(s) * 1024 : parseInt(s);
}}
ioOptions.sort((a, b) => {{
  const [ai, ao] = a.split('_').map(parseSize);
  const [bi, bo] = b.split('_').map(parseSize);
  return ai - bi || ao - bo;
}});

// Populate selects
function populateSelect(sel, opts, formatFn) {{
  opts.forEach(o => {{
    const opt = document.createElement('option');
    opt.value = o;
    opt.textContent = formatFn ? formatFn(o) : o;
    sel.appendChild(opt);
  }});
}}

const ioFormat = v => 'i=' + v.split('_')[0] + ' o=' + v.split('_')[1];
populateSelect(document.getElementById('ioSelect'), ioOptions, ioFormat);
populateSelect(document.getElementById('cmpIOSelect'), ioOptions, ioFormat);
populateSelect(document.getElementById('tableIOFilter'), ioOptions, ioFormat);
populateSelect(document.getElementById('tableStrategyFilter'), strategies);

// Tabs
document.querySelectorAll('#tabBar button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#tabBar button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('view' + btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)).classList.add('active');
  }});
}});

// Summary cards
const totalRecs = DATA.length;
const avgTTFT = (DATA.reduce((s, d) => s + d.mean_ttft_ms, 0) / totalRecs).toFixed(1);
const avgTPOT = (DATA.reduce((s, d) => s + d.mean_tpot_ms, 0) / totalRecs).toFixed(2);
const avgDur = (DATA.reduce((s, d) => s + d.benchmark_duration_s, 0) / totalRecs).toFixed(1);
document.getElementById('summaryCards').innerHTML = `
  <div class="summary-card"><div class="value">${{totalRecs}}</div><div class="label">Total Benchmarks</div></div>
  <div class="summary-card"><div class="value">${{strategies.length}}</div><div class="label">Strategies</div></div>
  <div class="summary-card"><div class="value">${{ioOptions.length}}</div><div class="label">I/O Configurations</div></div>
  <div class="summary-card"><div class="value">${{avgTTFT}} ms</div><div class="label">Avg Mean TTFT</div></div>
  <div class="summary-card"><div class="value">${{avgTPOT}} ms</div><div class="label">Avg Mean TPOT</div></div>
  <div class="summary-card"><div class="value">${{avgDur}} s</div><div class="label">Avg Duration</div></div>
`;

// Chart helpers
const chartInstances = {{}};
function makeChart(canvasId, cfg) {{
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  cfg.options = cfg.options || {{}};
  cfg.options.responsive = true;
  cfg.options.maintainAspectRatio = false;
  cfg.options.plugins = cfg.options.plugins || {{}};
  cfg.options.plugins.legend = cfg.options.plugins.legend || {{ labels: {{ color: '#8b8fa3' }} }};
  cfg.options.scales = cfg.options.scales || {{}};
  for (const axis of ['x', 'y']) {{
    cfg.options.scales[axis] = cfg.options.scales[axis] || {{}};
    cfg.options.scales[axis].ticks = cfg.options.scales[axis].ticks || {{}};
    cfg.options.scales[axis].ticks.color = '#8b8fa3';
    cfg.options.scales[axis].grid = {{ color: 'rgba(255,255,255,0.05)' }};
  }}
  chartInstances[canvasId] = new Chart(ctx, cfg);
}}

function getFiltered(io) {{
  const [il, ol] = io.split('_');
  return DATA.filter(d => d.input_len === il && d.output_len === ol);
}}

function updateCharts() {{
  const io = document.getElementById('ioSelect').value;
  const metric = document.getElementById('metricSelect').value;
  const metricLabel = document.getElementById('metricSelect').selectedOptions[0].textContent;
  const filtered = getFiltered(io);

  document.getElementById('chartTitle1').textContent = metricLabel + ' vs Num Requests (' + ioFormat(io) + ')';

  const datasets = strategies.map(s => {{
    const pts = filtered.filter(d => d.strategy === s).sort((a, b) => a.num_requests - b.num_requests);
    return {{
      label: s,
      data: pts.map(p => ({{ x: p.num_requests, y: p[metric] }})),
      borderColor: COLORS[s],
      backgroundColor: COLORS_BG[s],
      tension: 0.3,
      pointRadius: 4,
    }};
  }});

  makeChart('chart1', {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      scales: {{
        x: {{ type: 'logarithmic', title: {{ display: true, text: 'Num Requests (log)', color: '#8b8fa3' }} }},
        y: {{ title: {{ display: true, text: metricLabel, color: '#8b8fa3' }} }},
      }},
    }},
  }});

  // TTFT breakdown (mean, median, p99) for each strategy
  const ttftDatasets = [];
  strategies.forEach(s => {{
    const pts = filtered.filter(d => d.strategy === s).sort((a, b) => a.num_requests - b.num_requests);
    ['mean_ttft_ms', 'p99_ttft_ms'].forEach((m, i) => {{
      ttftDatasets.push({{
        label: s + ' ' + (i === 0 ? 'Mean' : 'P99'),
        data: pts.map(p => ({{ x: p.num_requests, y: p[m] }})),
        borderColor: COLORS[s],
        backgroundColor: COLORS_BG[s],
        borderDash: i === 1 ? [6, 3] : [],
        tension: 0.3,
        pointRadius: 3,
      }});
    }});
  }});
  document.getElementById('chartTitle2').textContent = 'TTFT Mean & P99 (' + ioFormat(io) + ')';
  makeChart('chart2', {{
    type: 'line',
    data: {{ datasets: ttftDatasets }},
    options: {{
      scales: {{
        x: {{ type: 'logarithmic', title: {{ display: true, text: 'Num Requests (log)', color: '#8b8fa3' }} }},
        y: {{ title: {{ display: true, text: 'TTFT (ms)', color: '#8b8fa3' }} }},
      }},
    }},
  }});

  // TPOT breakdown
  const tpotDatasets = [];
  strategies.forEach(s => {{
    const pts = filtered.filter(d => d.strategy === s).sort((a, b) => a.num_requests - b.num_requests);
    ['mean_tpot_ms', 'p99_tpot_ms'].forEach((m, i) => {{
      tpotDatasets.push({{
        label: s + ' ' + (i === 0 ? 'Mean' : 'P99'),
        data: pts.map(p => ({{ x: p.num_requests, y: p[m] }})),
        borderColor: COLORS[s],
        backgroundColor: COLORS_BG[s],
        borderDash: i === 1 ? [6, 3] : [],
        tension: 0.3,
        pointRadius: 3,
      }});
    }});
  }});
  document.getElementById('chartTitle3').textContent = 'TPOT Mean & P99 (' + ioFormat(io) + ')';
  makeChart('chart3', {{
    type: 'line',
    data: {{ datasets: tpotDatasets }},
    options: {{
      scales: {{
        x: {{ type: 'logarithmic', title: {{ display: true, text: 'Num Requests (log)', color: '#8b8fa3' }} }},
        y: {{ title: {{ display: true, text: 'TPOT (ms)', color: '#8b8fa3' }} }},
      }},
    }},
  }});

  // Duration
  const durDatasets = strategies.map(s => {{
    const pts = filtered.filter(d => d.strategy === s).sort((a, b) => a.num_requests - b.num_requests);
    return {{
      label: s,
      data: pts.map(p => ({{ x: p.num_requests, y: p.benchmark_duration_s }})),
      borderColor: COLORS[s],
      backgroundColor: COLORS_BG[s],
      tension: 0.3,
      pointRadius: 4,
    }};
  }});
  document.getElementById('chartTitle4').textContent = 'Duration (s) vs Num Requests (' + ioFormat(io) + ')';
  makeChart('chart4', {{
    type: 'line',
    data: {{ datasets: durDatasets }},
    options: {{
      scales: {{
        x: {{ type: 'logarithmic', title: {{ display: true, text: 'Num Requests (log)', color: '#8b8fa3' }} }},
        y: {{ title: {{ display: true, text: 'Duration (s)', color: '#8b8fa3' }} }},
      }},
    }},
  }});
}}

document.getElementById('ioSelect').addEventListener('change', updateCharts);
document.getElementById('metricSelect').addEventListener('change', updateCharts);
updateCharts();

// Table
let sortCol = 'num_requests';
let sortAsc = true;

function renderTable() {{
  const stratFilter = document.getElementById('tableStrategyFilter').value;
  const ioFilter = document.getElementById('tableIOFilter').value;
  let rows = DATA;
  if (stratFilter !== 'all') rows = rows.filter(d => d.strategy === stratFilter);
  if (ioFilter !== 'all') {{
    const [il, ol] = ioFilter.split('_');
    rows = rows.filter(d => d.input_len === il && d.output_len === ol);
  }}
  rows = [...rows].sort((a, b) => {{
    const va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'number') return sortAsc ? va - vb : vb - va;
    return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  }});

  const tbody = document.querySelector('#dataTable tbody');
  tbody.innerHTML = rows.map(r => `<tr>
    <td style="text-align:left">${{r.label}}</td>
    <td><span class="strategy-badge strategy-${{r.strategy}}">${{r.strategy}}</span></td>
    <td>${{r.input_tokens}}</td>
    <td>${{r.output_tokens}}</td>
    <td>${{r.num_requests}}</td>
    <td>${{r.benchmark_duration_s.toFixed(2)}}</td>
    <td>${{r.mean_ttft_ms.toFixed(2)}}</td>
    <td>${{r.median_ttft_ms.toFixed(2)}}</td>
    <td>${{r.p99_ttft_ms.toFixed(2)}}</td>
    <td>${{r.mean_tpot_ms.toFixed(2)}}</td>
    <td>${{r.median_tpot_ms.toFixed(2)}}</td>
    <td>${{r.p99_tpot_ms.toFixed(2)}}</td>
    <td>${{r.output_tok_throughput.toFixed(1)}}</td>
  </tr>`).join('');
}}

document.querySelectorAll('#dataTable th').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else {{ sortCol = col; sortAsc = true; }}
    renderTable();
  }});
}});
document.getElementById('tableStrategyFilter').addEventListener('change', renderTable);
document.getElementById('tableIOFilter').addEventListener('change', renderTable);
renderTable();

// Comparison view
function updateComparison() {{
  const io = document.getElementById('cmpIOSelect').value;
  const filtered = getFiltered(io);
  const metrics = [
    ['cmpChart1', 'mean_ttft_ms', 'Mean TTFT (ms)'],
    ['cmpChart2', 'p99_ttft_ms', 'P99 TTFT (ms)'],
    ['cmpChart3', 'mean_tpot_ms', 'Mean TPOT (ms)'],
    ['cmpChart4', 'p99_tpot_ms', 'P99 TPOT (ms)'],
    ['cmpChart5', 'benchmark_duration_s', 'Duration (s)'],
    ['cmpChart6', 'output_tok_throughput', 'Output Throughput (tok/s)'],
  ];
  metrics.forEach(([canvasId, metric, label]) => {{
    const datasets = strategies.map(s => {{
      const pts = filtered.filter(d => d.strategy === s).sort((a, b) => a.num_requests - b.num_requests);
      return {{
        label: s,
        data: pts.map(p => ({{ x: p.num_requests, y: p[metric] }})),
        borderColor: COLORS[s],
        backgroundColor: COLORS_BG[s],
        tension: 0.3,
        pointRadius: 4,
        fill: false,
      }};
    }});
    makeChart(canvasId, {{
      type: 'line',
      data: {{ datasets }},
      options: {{
        scales: {{
          x: {{ type: 'logarithmic', title: {{ display: true, text: 'Num Requests (log)', color: '#8b8fa3' }} }},
          y: {{ title: {{ display: true, text: label, color: '#8b8fa3' }} }},
        }},
      }},
    }});
  }});
}}
document.getElementById('cmpIOSelect').addEventListener('change', updateComparison);
updateComparison();
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()

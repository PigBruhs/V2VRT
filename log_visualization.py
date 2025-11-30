"""log_visualization.py

Reads logs/pipeline.log and logs/pipeline_app.log and produces visualizations comparing
cloud (pipeline.log) and mobile/local (pipeline_app.log) module timings.

Usage:
    python log_visualization.py

Outputs PNG files into ./outputs/plots/.
"""
import re
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception:
    print('matplotlib is required. Please install matplotlib.')
    raise

# Optional: pandas makes aggregation easier but we keep dependency minimal.

LOG_PIPELINE = os.path.join('logs', 'pipeline.log')
LOG_APP = os.path.join('logs', 'pipeline_app.log')
OUT_DIR = os.path.join('outputs', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# Regex patterns
RE_LINE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| (?P<level>[^|]+) \| (?P<name>[^|]+) \| (?P<msg>.*)$")
RE_DURATION = re.compile(r"duration_ms=([0-9]+\.?[0-9]*)")
RE_ASR = re.compile(r"ASR\.transcribe success")
RE_TR = re.compile(r"Translator\.translate success")
RE_TTS = re.compile(r"TTS\.synthesize success")
RE_LOCAL_PIPELINE = re.compile(r"Local\.pipeline .*total_ms=(?P<total>[0-9]+) \|? ?asr_ms=(?P<asr>[0-9]+) \|? ?tr_ms=(?P<tr>[0-9]+) \|? ?tts_ms=(?P<tts>[0-9]+)")
RE_CLOUD_PIPELINE = re.compile(r"Cloud\.pipeline .*total_ms=(?P<total>[0-9]+)")
RE_CLOUD_REQUEST = re.compile(r"Cloud\.request .*duration_ms=(?P<dur>[0-9]+)")

# Helper to parse timestamp
def parse_ts(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        return None

# Parse generic "module success" lines
def parse_log_file(path, source_label):
    """Return list of entries: dict with keys: ts,module,duration_ms,chars,lang,src"""
    entries = []
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return entries

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            m = RE_LINE.match(line)
            if not m:
                continue
            ts = parse_ts(m.group('ts'))
            # 如果是云端日志，按要求将含日期的时间加 8 小时（时区偏移），以便于可视化对齐
            if ts is not None and source_label == 'cloud':
                ts = ts + timedelta(hours=8)
            msg = m.group('msg')

            # Check specific module lines
            if RE_ASR.search(msg):
                dur_m = RE_DURATION.search(msg)
                duration = float(dur_m.group(1)) if dur_m else None
                lang_m = re.search(r"lang=([^ |]+)", msg)
                chars_m = re.search(r"chars=([0-9]+)", msg)
                entries.append({'ts': ts, 'module': 'asr', 'duration_ms': duration, 'lang': lang_m.group(1) if lang_m else None, 'chars': int(chars_m.group(1)) if chars_m else None, 'source': source_label})
            elif RE_TR.search(msg):
                dur_m = RE_DURATION.search(msg)
                duration = float(dur_m.group(1)) if dur_m else None
                src_m = re.search(r"src=([^ |]+)", msg)
                tgt_m = re.search(r"tgt=([^ |]+)", msg)
                chars_m = re.search(r"chars=([0-9]+)", msg)
                entries.append({'ts': ts, 'module': 'translator', 'duration_ms': duration, 'src': src_m.group(1) if src_m else None, 'tgt': tgt_m.group(1) if tgt_m else None, 'chars': int(chars_m.group(1)) if chars_m else None, 'source': source_label})
            elif RE_TTS.search(msg):
                dur_m = RE_DURATION.search(msg)
                duration = float(dur_m.group(1)) if dur_m else None
                lang_m = re.search(r"lang=([^ |]+)", msg)
                file_m = re.search(r"file=([^ |]+)", msg)
                chars_m = re.search(r"chars=([0-9]+)", msg)
                entries.append({'ts': ts, 'module': 'tts', 'duration_ms': duration, 'lang': lang_m.group(1) if lang_m else None, 'file': file_m.group(1) if file_m else None, 'chars': int(chars_m.group(1)) if chars_m else None, 'source': source_label})
            else:
                # Check Local.pipeline entries (contains per-module ms)
                mloc = RE_LOCAL_PIPELINE.search(msg)
                if mloc:
                    ts_val = ts
                    entries.append({'ts': ts_val, 'module': 'local_pipeline_total', 'duration_ms': float(mloc.group('total')), 'asr_ms': float(mloc.group('asr')), 'tr_ms': float(mloc.group('tr')), 'tts_ms': float(mloc.group('tts')), 'source': source_label})
                else:
                    mcloud = RE_CLOUD_PIPELINE.search(msg)
                    if mcloud:
                        entries.append({'ts': ts, 'module': 'cloud_pipeline_total', 'duration_ms': float(mcloud.group('total')), 'source': source_label})
                    else:
                        mcr = RE_CLOUD_REQUEST.search(msg)
                        if mcr:
                            entries.append({'ts': ts, 'module': 'cloud_request', 'duration_ms': float(mcr.group('dur')), 'source': source_label})

    return entries


def aggregate_by_module(entries):
    d = defaultdict(list)
    for e in entries:
        if 'duration_ms' in e and e['duration_ms'] is not None:
            d[(e['module'], e['source'])].append(e)
    return d


def make_boxplot(module, cloud_vals, local_vals, out_path):
    plt.figure(figsize=(6,4))
    data = []
    labels = []
    if cloud_vals:
        data.append(cloud_vals)
        labels.append('cloud')
    if local_vals:
        data.append(local_vals)
        labels.append('mobile')
    if not data:
        print(f'no data for {module}')
        return
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(f'{module.upper()} duration_ms (cloud vs mobile)')
    plt.ylabel('ms')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def timeseries_plot(module_entries_cloud, module_entries_local, out_path):
    plt.figure(figsize=(10,4))
    if module_entries_cloud:
        xs = [e['ts'] for e in module_entries_cloud if e['ts']]
        ys = [e['duration_ms'] for e in module_entries_cloud]
        plt.plot(xs, ys, label='cloud', marker='.', alpha=0.6)
    if module_entries_local:
        xs2 = [e['ts'] for e in module_entries_local if e['ts']]
        ys2 = [e['duration_ms'] for e in module_entries_local]
        plt.plot(xs2, ys2, label='mobile', marker='.', alpha=0.6)
    # determine module name safely
    module_name = None
    if module_entries_cloud and len(module_entries_cloud) > 0:
        module_name = module_entries_cloud[0].get('module')
    elif module_entries_local and len(module_entries_local) > 0:
        module_name = module_entries_local[0].get('module')
    module_name = module_name or ''
    plt.title(f'Timeseries: {module_name} durations')
    plt.ylabel('ms')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def stacked_local_bars(local_pipeline_entries, out_path):
    # show distribution of asr/tr/tts for local pipeline entries (per entry stacked bar)
    if not local_pipeline_entries:
        print('no local pipeline data')
        return
    local_pipeline_entries = sorted(local_pipeline_entries, key=lambda x: x['ts'] or datetime.min)
    labels = [e['ts'].strftime('%H:%M:%S') if e['ts'] else str(i) for i,e in enumerate(local_pipeline_entries)]
    asr = [e.get('asr_ms',0) for e in local_pipeline_entries]
    tr = [e.get('tr_ms',0) for e in local_pipeline_entries]
    tts = [e.get('tts_ms',0) for e in local_pipeline_entries]
    x = range(len(labels))
    plt.figure(figsize=(max(8,len(labels)*0.4),4))
    plt.bar(x, asr, label='asr')
    plt.bar(x, tr, bottom=asr, label='translator')
    bottom2 = [a+b for a,b in zip(asr,tr)]
    plt.bar(x, tts, bottom=bottom2, label='tts')
    plt.xticks(x, labels, rotation=90)
    plt.title('Local pipeline per-request component breakdown')
    plt.ylabel('ms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def scatter_cloud_vs_local(local_totals, cloud_totals, out_path):
    # local_totals and cloud_totals are lists of dicts with ts,duration_ms
    # We'll match by nearest time: for each local record find closest cloud record within 60s
    cloud_ts = [c['ts'] for c in cloud_totals]
    cloud_vals = [c['duration_ms'] for c in cloud_totals]
    pairs = []
    for l in local_totals:
        lt = l['ts']
        if lt is None:
            continue
        # find nearest cloud
        best = None
        best_dt = None
        for c in cloud_totals:
            if c['ts'] is None:
                continue
            dt = abs((c['ts'] - lt).total_seconds())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = c
        if best and best_dt <= 60:
            pairs.append((l['duration_ms'], best['duration_ms']))
    if not pairs:
        print('no matched cloud/local totals for scatter')
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel('local total_ms')
    plt.ylabel('cloud total_ms')
    plt.title('Local vs Cloud pipeline total_ms (matched by time)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def main():
    pipeline_entries = parse_log_file(LOG_PIPELINE, 'cloud')
    app_entries = parse_log_file(LOG_APP, 'mobile')

    entries = pipeline_entries + app_entries
    agg = aggregate_by_module(entries)

    # Prepare per-module lists
    cloud_asr = [e for e in agg.get(('asr','cloud'), [])]
    mobile_asr = [e for e in agg.get(('asr','mobile'), [])]
    cloud_tr = [e for e in agg.get(('translator','cloud'), [])]
    mobile_tr = [e for e in agg.get(('translator','mobile'), [])]
    cloud_tts = [e for e in agg.get(('tts','cloud'), [])]
    mobile_tts = [e for e in agg.get(('tts','mobile'), [])]
    local_pipeline = [e for e in entries if e.get('module') == 'local_pipeline_total']
    cloud_pipeline = [e for e in entries if e.get('module') == 'cloud_pipeline_total']

    # Boxplots
    make_boxplot('ASR', [e['duration_ms'] for e in cloud_asr if e['duration_ms'] is not None], [e['duration_ms'] for e in mobile_asr if e['duration_ms'] is not None], os.path.join(OUT_DIR, 'asr_box.png'))
    make_boxplot('Translator', [e['duration_ms'] for e in cloud_tr if e['duration_ms'] is not None], [e['duration_ms'] for e in mobile_tr if e['duration_ms'] is not None], os.path.join(OUT_DIR, 'translator_box.png'))
    make_boxplot('TTS', [e['duration_ms'] for e in cloud_tts if e['duration_ms'] is not None], [e['duration_ms'] for e in mobile_tts if e['duration_ms'] is not None], os.path.join(OUT_DIR, 'tts_box.png'))

    # Timeseries for each module
    timeseries_plot(cloud_asr, mobile_asr, os.path.join(OUT_DIR, 'asr_timeseries.png'))
    timeseries_plot(cloud_tr, mobile_tr, os.path.join(OUT_DIR, 'translator_timeseries.png'))
    timeseries_plot(cloud_tts, mobile_tts, os.path.join(OUT_DIR, 'tts_timeseries.png'))

    # Stacked bars for local pipeline
    stacked_local_bars(local_pipeline, os.path.join(OUT_DIR, 'local_pipeline_stacked.png'))

    # Scatter cloud vs local totals
    local_totals = [e for e in entries if e.get('module') == 'local_pipeline_total']
    cloud_totals = [e for e in entries if e.get('module') == 'cloud_pipeline_total']
    scatter_cloud_vs_local(local_totals, cloud_totals, os.path.join(OUT_DIR, 'local_vs_cloud_scatter.png'))

    print('Done. Plots written to', OUT_DIR)

if __name__ == '__main__':
    main()

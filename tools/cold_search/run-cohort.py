import argparse
import json
from pathlib import Path
import subprocess

import sys
from paths import workspace as w
scripts = Path(__file__).resolve().parent
python = sys.executable
p = argparse.ArgumentParser()
p.add_argument('prefix')
p.add_argument('--pairs', type=int, default=1)
p.add_argument('--name', default='cold_search_qv_768')
p.add_argument('--release-wait', type=float, default=62)
a = p.parse_args()
labels = []
for i in range(a.pairs):
    for scenario in ('id-only', 'with-vector'):
        label = f'{a.prefix}-{scenario}-{i}'
        subprocess.run([python, str(scripts / 'bench.py'), '--scenario', scenario,
                        '--label', label, '--release-wait', str(a.release_wait), '--name', a.name,
                        '--rounds', '1'], cwd=w, check=True)
        with (w / 'results' / f'{label}-0-stages.txt').open('w') as out:
            subprocess.run([python, str(scripts / 'metric_delta.py'), label, '0'], cwd=w, stdout=out, check=True)
        subprocess.run([python, str(scripts / 'extract_all.py'), label], cwd=w, check=True)
        labels.append(label)
        subprocess.run([python, str(scripts / 'summarize_scenarios.py'), '--output', a.prefix + '-summary',
                        *labels], cwd=w, check=True, stdout=subprocess.DEVNULL)
        row = json.loads((w / 'results' / f'{a.prefix}-summary.json').read_text())[-1]
        keys = ['label', 'latency_ms', 'readiness_ms', 'execution_ms', 'wal_impl_ms',
                'qn_load_ms', 'minio_requests', 'minio_wave_request_counts', 'go_load_count', 'native_load_count']
        print(json.dumps({k: row[k] for k in keys}), flush=True)

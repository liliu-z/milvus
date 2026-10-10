"""Summarize the two requested output scenarios without mixing scalar trials."""
import csv
import argparse
import json
from pathlib import Path

from paths import results as root
parser = argparse.ArgumentParser()
parser.add_argument('labels', nargs='*', default=['scenario-id-only', 'scenario-with-vector'])
parser.add_argument('--output', default='requested-scenarios-summary')
parser.add_argument('--round', type=int, default=0)
args = parser.parse_args()
rows = []
for label in args.labels:
    raw_path = root / f'{label}.jsonl'
    if not raw_path.exists():
        continue
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    run = raw[args.round]
    assert run['ok'] and run['before'] == 'NotLoad' and run['after'] == 'Loaded'
    assert not run['manual_load']
    assert run.get('id_only_verified') or run.get('vectors_verified')
    stages = json.loads((root / f'{label}-{args.round}-stages.json').read_text())

    def select(metric, **labels):
        return [s for s in stages if s['metric'] == metric
                and all(s['labels'].get(k) == v for k, v in labels.items())]

    def value(metric, **labels):
        return sum(s['sum_ms'] for s in select(metric, **labels))

    row = {k: run[k] for k in ['label', 'scenario', 'latency_ms', 'output_fields']}
    row['round'] = args.round
    for stage in ['readiness', 'execution', 'retry_wait', 'total']:
        row[stage + '_ms'] = value('milvus_qv_request_stage_duration_seconds', stage=stage)
    assert abs(row['total_ms'] - sum(row[s + '_ms'] for s in ['readiness', 'execution', 'retry_wait'])) < 0.001
    row['sdk_grpc_residual_ms'] = row['latency_ms'] - row['total_ms']
    row['wal_impl_ms'] = value('milvus_wal_append_message_stage_duration_seconds', stage='wal_impl', message_type='AlterLoadConfig')
    row['qn_load_ms'] = value('milvus_qv_stage_duration_seconds', operation='segment_load_attempt', stage='total')
    row['native_load_ms'] = value('internal_core_segment_load_duration_seconds', stage='total')
    row['vector_prefetch_ms'] = value('internal_core_query_stage_duration_seconds', stage='vector_prefetch_run')
    for name, samples in [
        ('go_load_count', select('milvus_qv_stage_duration_seconds', operation='segment_load_attempt', stage='total')),
        ('native_load_count', select('internal_core_segment_load_duration_seconds', stage='total')),
    ]:
        row[name] = sum(s['count'] for s in samples)
        assert row[name] == 1
        assert all(s['labels']['result'] == 'success' for s in samples)
    phases = select('internal_core_segment_load_duration_seconds')
    assert len(phases) == 15 and all(s['count'] == 1 and s['labels']['result'] == 'success' for s in phases)
    assert abs(row['native_load_ms'] - sum(s['sum_ms'] for s in phases if s['labels']['stage'] != 'total')) < 0.0001
    spans = json.loads((root / f'{label}-{args.round}-spans.json').read_text())
    search = next(s for s in spans if s['name'] == 'milvus.proto.milvus.MilvusService/Search')
    target = [s for s in spans if s['traceId'] == search['traceId']]
    row['bf_compute_ms'] = sum(s['ms'] for s in target if s['name'] == 'knowhere bf search with buf')
    row['post_execute_ms'] = sum(s['ms'] for s in target if s['name'] == 'Proxy-Search-PostExecute')
    for method in ['GetQueryPlan', 'SearchOnView', 'QueryOnView']:
        calls = [s for s in target if s['kind'] == 'SPAN_KIND_CLIENT' and s['name'].endswith('/' + method)]
        row[method + '_rpc_count'] = len(calls)
        row[method + '_rpc_ms'] = sum(s['ms'] for s in calls)
    row['wal_messages'] = sorted(set(
        a['value']['stringValue'] for s in target if s['name'] == 'wal.appendimpl'
        for a in s.get('attributes', []) if a['key'] == 'message.type'))
    assert row['wal_messages'] == ['AlterLoadConfig'], row['wal_messages']
    all_s3 = json.loads((root / f'{label}-{args.round}-minio.json').read_text())
    collection_id = str(run.get('collection_id', 469658685339075637))
    segment_ids = set()
    for e in all_s3:
        parts = e['path'].split('/')
        if 'insert_log' in parts:
            i = parts.index('insert_log')
            if parts[i+1] == collection_id:
                segment_ids.add(parts[i+3])
    if collection_id == '469658685339075637':
        segment_ids.add('469658685339276649')
    assert len(segment_ids) == 1, segment_ids
    s3 = [e for e in all_s3 if any('/'+sid+'/' in e['path'] for sid in segment_ids)]
    row['segment_ids'] = sorted(segment_ids)
    row['minio_requests'] = len(s3)
    row['minio_window_requests'] = len(all_s3)
    row['minio_other_window_requests'] = [
        {'api': e['api'], 'path': e['path'], 'offset_ms': e['offset_ms']}
        for e in all_s3 if e not in s3
    ]
    assert len(s3) >= 2, 'A physically cold vector search must read fresh data objects'
    if len(s3) == 2:
        row['minio_common_overlap_ms'] = max(0, min(e['offset_ms'] + e['duration_ms'] for e in s3) - max(e['offset_ms'] for e in s3))
    else:
        row['minio_common_overlap_ms'] = None
    row['minio_returned_bytes'] = sum(e['callStats']['tx'] for e in s3)
    row['minio_server_sum_ms'] = sum(e['duration_ms'] for e in s3)
    waves = []
    for e in sorted(s3, key=lambda e: e['offset_ms']):
        start, end = e['offset_ms'], e['offset_ms'] + e['duration_ms']
        if waves and start < waves[-1]['end_ms']:
            waves[-1]['end_ms'] = max(waves[-1]['end_ms'], end)
            waves[-1]['requests'] += 1
        else:
            waves.append({'start_ms': start, 'end_ms': end, 'requests': 1})
    # Interval-connected components are not causal dispatch stages when a
    # slow speculative GET overlaps a later foreground fallback (v81).
    row['minio_wave_definition'] = 'interval-connected; inspect fallback dispatch separately'
    row['minio_waves'] = len(waves)
    row['minio_wave_request_counts'] = [w['requests'] for w in waves]
    row['returned_vector_count'] = run.get('returned_vector_count', 0)
    rows.append(row)

(root / (args.output + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
if rows:
    with (root / (args.output + '.csv')).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
for row in rows:
    print(json.dumps(row, ensure_ascii=False))

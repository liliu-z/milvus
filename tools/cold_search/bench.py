#!/usr/bin/env python3
"""One Release-cold Search per round; validate data outside the SDK timer."""
import argparse
import gzip
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from paths import results, workspace


def faults():
    marker = workspace / 'milvus-active.pid'
    if not marker.exists():
        return None
    stat = Path('/proc', marker.read_text().strip(), 'stat').read_text().rsplit(') ', 1)[1].split()
    return {'minor': int(stat[7]), 'major': int(stat[9])}


def metrics(label, round_id, when):
    data = urllib.request.urlopen('http://127.0.0.1:9091/metrics', timeout=10).read()
    with gzip.open(results / f'{label}-{round_id}-{when}.prom.gz', 'wb') as stream:
        stream.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup', action='store_true', help='create a new collection; refuses to overwrite')
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--label', required=True)
    p.add_argument('--release-wait', type=float, default=62)
    p.add_argument('--name', default='cold_search_qv_768')
    p.add_argument('--scenario', choices=['id-only', 'with-vector'], default='id-only')
    a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', a.label) or a.label in ('.', '..'):
        p.error('label must be a simple unique filename component')
    if a.rounds < 0 or a.release_wait < 0:
        p.error('rounds and release-wait must be nonnegative')
    target = results / f'{a.label}.jsonl'
    if target.exists():
        raise RuntimeError('Use a new label; refusing to mix runs')
    connections.connect(host='127.0.0.1', port=19530)
    rng = np.random.default_rng(20261009)
    vectors = rng.normal(size=(1000, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    query = vectors[123:124]
    if a.setup:
        if utility.has_collection(a.name):
            raise RuntimeError('Refusing to overwrite an existing collection')
        fields = [FieldSchema('id', DataType.INT64, is_primary=True),
                  FieldSchema('vector', DataType.FLOAT_VECTOR, dim=768),
                  FieldSchema('age', DataType.INT64), FieldSchema('score', DataType.DOUBLE),
                  FieldSchema('active', DataType.BOOL), FieldSchema('tag', DataType.VARCHAR, max_length=64),
                  FieldSchema('payload', DataType.JSON)]
        c = Collection(a.name, CollectionSchema(fields, enable_dynamic_field=False),
                       shards_num=1, consistency_level='Strong')
        c.insert([list(range(1000)), vectors.tolist(), [i % 100 for i in range(1000)],
                  [i / 1000 for i in range(1000)], [i % 2 == 0 for i in range(1000)],
                  [f'tag_{i % 10}' for i in range(1000)],
                  [{'i': i, 'group': i % 10} for i in range(1000)]])
        # Preparation only. No flush or explicit load occurs in a timed search.
        c.flush(timeout=120)
        c.create_index('vector', {'index_type': 'FLAT', 'metric_type': 'COSINE', 'params': {}}, timeout=120)
        assert c.num_entities == 1000 and str(utility.load_state(a.name)) == 'NotLoad'
        np.savez_compressed(results / f'{a.name}-dataset.npz', vectors=vectors, query=query)
    else:
        c = Collection(a.name)
    collection_id = c.describe()['collection_id']
    output_fields = [] if a.scenario == 'id-only' else ['vector']
    for round_id in range(a.rounds):
        release_ms = None
        if str(utility.load_state(a.name)) != 'NotLoad':
            started = time.perf_counter_ns()
            c.release(timeout=120)
            release_ms = (time.perf_counter_ns() - started) / 1e6
        state = str(utility.load_state(a.name))
        assert state == 'NotLoad', state
        time.sleep(a.release_wait + float(rng.uniform(0, .3)))
        (workspace / 'active-trace-label').write_text(a.label)
        metrics(a.label, round_id, 'before')
        out = {'label': a.label, 'round': round_id, 'start_ns': time.time_ns(), 'before': state,
               'manual_load': False, 'release_wait_s': a.release_wait, 'release_rpc_ms': release_ms,
               'scenario': a.scenario, 'output_fields': output_fields, 'collection_id': collection_id,
               'index_type': 'FLAT'}
        before_faults = faults()
        started = time.perf_counter_ns()
        try:
            response = c.search(query.tolist(), 'vector', {'metric_type': 'COSINE', 'params': {}},
                                limit=10, output_fields=output_fields, timeout=120, consistency_level='Strong')
            out['latency_ms'] = (time.perf_counter_ns() - started) / 1e6
            out['return_ns'] = time.time_ns()
            after_faults = faults()
            if before_faults is not None and after_faults is not None:
                out['process_fault_delta'] = {k: after_faults[k] - before_faults[k] for k in before_faults}
            out['ids'] = [hit.id for hit in response[0]]
            out['distances'] = [hit.distance for hit in response[0]]
            scores = vectors @ query[0]
            expected = np.argsort(-scores)[:10].tolist()
            assert out['ids'] == expected, (out['ids'], expected)
            assert np.allclose(out['distances'], scores[expected], atol=1e-5)
            if a.scenario == 'with-vector':
                for hit in response[0]:
                    value = np.asarray(hit.entity.get('vector'), dtype=np.float32)
                    assert value.shape == (768,) and np.array_equal(value, vectors[hit.id]), hit.id
                out.update(vectors_verified=True, returned_vector_count=10, returned_vector_values=7680)
            else:
                for hit in response[0]:
                    for field in ('vector', 'age', 'score', 'active', 'tag', 'payload'):
                        assert hit.entity.get(field) is None, (hit.id, field)
                out['id_only_verified'] = True
            out['ok'] = True
        except Exception as error:
            # Preserve the search duration if validation, rather than Search, failed.
            out.setdefault('latency_ms', (time.perf_counter_ns() - started) / 1e6)
            out.update(ok=False, error=str(error))
        out['end_ns'] = time.time_ns()
        metrics(a.label, round_id, 'after')
        out['after'] = str(utility.load_state(a.name))
        print(json.dumps(out), flush=True)
        with target.open('a') as stream:
            stream.write(json.dumps(out) + '\n')
        if not out['ok']:
            raise RuntimeError(out['error'])
        assert out['after'] == 'Loaded'
    # Exporter completion is outside the SDK timer and before switching labels.
    time.sleep(6)


if __name__ == '__main__':
    main()

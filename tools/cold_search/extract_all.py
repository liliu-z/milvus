"""One streaming pass over compressed OTLP capture for all completed rounds."""
import gzip,orjson,json,sys,datetime
from pathlib import Path
from paths import results as w;label=sys.argv[1]
runs=list(map(json.loads,(w/f'{label}.jsonl').read_text().splitlines()));buckets=[[] for _ in runs]
trace_path=w/f'{label}-traces.jsonl.gz'
for line in (gzip.open(trace_path,'rb') if trace_path.exists() else []):
 d=orjson.loads(line)
 for rs in d.get('resourceSpans',[]):
  for ss in rs.get('scopeSpans',[]):
   for s in ss.get('spans',[]):
    t=int(s['startTimeUnixNano'])
    for r,out in zip(runs,buckets):
     if r['start_ns']-10**6<=t<=r.get('return_ns',r['end_ns'])+10**6:
      s['offset_ms']=(t-r['start_ns'])/1e6;s['ms']=(int(s['endTimeUnixNano'])-t)/1e6;out.append(s)
def iso_ns(x):
 head,frac=x.rstrip('Z').split('.') if '.' in x else (x.rstrip('Z'),'0')
 return int(datetime.datetime.fromisoformat(head).replace(tzinfo=datetime.timezone.utc).timestamp())*10**9+int(frac.ljust(9,'0')[:9])
requests=[[] for _ in runs]
for l in (w/'minio-trace.jsonl').open():
 try:d=json.loads(l);t=iso_ns(d['time'])
 except (ValueError,KeyError):continue
 if '/cold-search-qv/' not in d.get('path',''):continue
 for r,out in zip(runs,requests):
  if r['start_ns']-10**6<=t<=r.get('return_ns',r['end_ns'])+10**6:
   d['offset_ms']=(t-r['start_ns'])/1e6;d['duration_ms']=d.get('duration',0)/1e6;out.append(d)
for i,(spans,m) in enumerate(zip(buckets,requests)):
 spans.sort(key=lambda s:int(s['startTimeUnixNano']))
 (w/f'{label}-{i}-spans.json').write_text(json.dumps(spans,indent=2));(w/f'{label}-{i}-minio.json').write_text(json.dumps(m,indent=2))
 root=next((s for s in spans if s['name']=='milvus.proto.milvus.MilvusService/Search'),None);target=[s for s in spans if root and s['traceId']==root['traceId']]
 lines=[f"{s['offset_ms']:10.3f} {s['ms']:10.3f} {s['name']}" for s in target]
 (w/f'{label}-{i}-target-timeline.txt').write_text('\n'.join(lines))
 print(label,i,'spans',len(spans),'target spans',len(target),'S3 requests',len(m))

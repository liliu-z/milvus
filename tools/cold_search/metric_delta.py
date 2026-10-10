"""Exact counter deltas around one request; no rate(), extrapolation or quantiles."""
import gzip,json,sys
from pathlib import Path
from prometheus_client.parser import text_string_to_metric_families
from paths import workspace as w
def read(path):
    out={}
    with gzip.open(path,'rt') as f:
        for family in text_string_to_metric_families(f.read()):
            for sample in family.samples:
                if sample.name.endswith(('_sum','_count','_total')):
                    out[(sample.name,tuple(sorted(sample.labels.items())))]=sample.value
    return out

def compare(label,round):
    before=read(w/'results'/f'{label}-{round}-before.prom.gz')
    after=read(w/'results'/f'{label}-{round}-after.prom.gz')
    counters=[{'metric':name,'labels':dict(labels),'delta':value-before.get((name,labels),0)} for (name,labels),value in after.items() if name.endswith('_total') and value-before.get((name,labels),0)>0]
    (w/'results'/f'{label}-{round}-counters.json').write_text(json.dumps(counters,indent=2))
    out=[]
    for (name,labels),value in after.items():
        if not name.endswith('_sum'):continue
        base=name[:-4]
        count_key=(base+'_count',labels)
        count=after.get(count_key,0)-before.get(count_key,0)
        if count<=0:continue
        val=value-before.get((name,labels),0)
        scale=1000 if '_seconds' in base else .001 if '_microseconds' in base else 1
        out.append({'metric':base,'labels':dict(labels),'count':count,'sum_ms':val*scale,'mean_ms':val*scale/count})
    out.sort(key=lambda x:(x['metric'],str(x['labels'])))
    (w/'results'/f'{label}-{round}-stages.json').write_text(json.dumps(out,indent=2))
    for x in out:
        if 'stage' in x['metric'] or 'segment_load_duration' in x['metric']:
            print(f"{x['sum_ms']:10.3f} ms  n={x['count']:3g}  {x['metric']} {x['labels']}")
    go=[x for x in out if x['labels'].get('operation')=='segment_load_attempt' and x['labels'].get('stage')=='total']
    native=[x for x in out if x['metric']=='internal_core_segment_load_duration_seconds' and x['labels'].get('stage')=='total']
    print('PHYSICAL_LOAD_COUNTS',json.dumps({'go_attempts':sum(x['count'] for x in go),'native_loads':sum(x['count'] for x in native)}))
    return out
if __name__=='__main__':compare(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 0)

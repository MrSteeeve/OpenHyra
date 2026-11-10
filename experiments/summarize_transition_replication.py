#!/usr/bin/env python3
"""Read-only progress and cost summary; mechanism claims remain unaudited."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def summary(out):
    protocol=json.loads((out/'protocol.json').read_text())
    cells={(m,a):dict(started=0,completed=0,evaluated=0,failed=0,planned=0,agent_calls=0,
                      call_seconds=0.,reported_tokens=0,outputs_without_token_count=0)
           for m in protocol['models'] for a in protocol['arms']}
    failures=Counter()
    for path in (out/'runs').glob('*/*/*/r*'):
        model,seed,arm,round_name=path.relative_to(out/'runs').parts
        c=cells[model,arm]
        c['started']+=int((path/'started.json').exists())
        c['planned']+=int((path/'plan.json').exists())
        if (path/'record.json').exists():
            r=json.loads((path/'record.json').read_text())
            c['completed']+=1
            c['evaluated']+=int(r['status']=='ok')
            c['failed']+=int(r['status']!='ok')
            if r['status']!='ok':failures[(r.get('failure_stage'),r.get('failure'))]+=1
        for file in (path/'calls').glob('*-output.json'):
            r=json.loads(file.read_text())
            c['agent_calls']+=1
            c['call_seconds']+=r['wall_seconds']
            match=re.search(r'tokens used\s*\n([\d,]+)',r.get('stderr',''))
            if match:c['reported_tokens']+=int(match[1].replace(',',''))
            else:c['outputs_without_token_count']+=1
    return dict(scheduled_attempts=protocol['scheduled_attempts'],
                full_rounds_completed=len(list(out.glob('round-*-complete.json'))),
                public_complete=(out/'public_complete.json').exists(),
                complete=(out/'complete.json').exists(),
                pauses=[str(p.name) for p in out.glob('pause-*.json')],
                cells=[dict(model=m,arm=a,**v) for (m,a),v in cells.items()],
                failures=[dict(stage=k[0],reason=k[1],count=v) for k,v in failures.items()],
                limitation='Interim operational counts; successful fit/predict does not establish method transition. Token reports are CLI-reported and may omit failed calls.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--pid',type=int)
    args=p.parse_args()
    if not args.watch:
        print(json.dumps(summary(args.output),ensure_ascii=False,indent=2))
    else:
        import os,time,datetime
        pid=args.pid or json.loads((args.output/'launch.json').read_text())['pid']
        while True:
            value=summary(args.output)
            value['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:alive=False
            value['supervisor_alive']=alive
            text=json.dumps(value,ensure_ascii=False,indent=2)
            tmp=args.output/'LIVE_STATUS.json.tmp'
            tmp.write_text(text+'\n')
            os.replace(tmp,args.output/'LIVE_STATUS.json')
            rows=['# Experiment progress (derived; interim)', '', 'Updated UTC: '+value['updated_utc'],
                  '', 'Supervisor running: '+str(alive)+'. Complete paired rounds: '+str(value['full_rounds_completed'])+'/20.',
                  '', '| Model | Condition | Started | Finished | Fit/predict OK | Failed |',
                  '|---|---|---:|---:|---:|---:|']
            for c in value['cells']:
                rows.append('| '+c['model']+' | '+c['arm']+' | '+str(c['started'])+' | '+str(c['completed'])+' | '+str(c['evaluated'])+' | '+str(c['failed'])+' |')
            rows += ['', 'Scheduled: 3,200 attempts. Every cell schedules 200 attempts across ten seeds.',
                     '', 'Operational success is not verified method transition. Mechanism audit and held-out results are pending.',
                     '', 'Resource/worker pauses: '+str(value['pauses']), '', 'Protocol: [PROTOCOL.md](PROTOCOL.md)']
            tmp=args.output/'LIVE_STATUS.md.tmp'
            tmp.write_text('\n'.join(rows)+'\n')
            os.replace(tmp,args.output/'LIVE_STATUS.md')
            if not alive or value['complete']:break
            time.sleep(30)

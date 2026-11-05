"""Native V2 greedy lifetime checks."""
import concurrent.futures
import time
from urllib.error import HTTPError

from verify_server import http


def verify(url, prompts, algorithm):
    records = []
    def generate(index, limit, *, rid, long=False, sampling=None, **kwargs):
        payload = {'rid':rid,'input_ids':prompts[index%len(prompts)]*(40 if long else 1),
                   'sampling_params':{'temperature':0,'max_new_tokens':limit,**(sampling or {})}, **kwargs}
        return http(url,'/generate',payload)
    def flush(): return http(url,'/flush_cache?timeout=15',{})

    for repeat in range(2):
        limits = [0,1,2,3,4,5,9,17]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i, limit in enumerate(limits):
                futures.append(pool.submit(generate,i,limit,rid=f'life-{repeat}-{i}'))
                time.sleep(.013)
            outputs = [f.result() for f in futures]
        for output, limit in zip(outputs,limits,strict=True):
            assert output['meta_info']['completion_tokens']==limit, output
            assert len(output['output_ids'])==limit, output
        records.append({'uneven':repeat,'outputs':outputs});flush()
    baseline = generate(0,32,rid='stop-control')
    stop = baseline['output_ids'][5]
    stopped = generate(0,32,rid='stop-token',sampling={'stop_token_ids':[stop]})
    count = baseline['output_ids'].index(stop)+1
    assert stopped['output_ids']==baseline['output_ids'][:count], stopped
    assert stopped['meta_info']['finish_reason']['type']=='stop', stopped
    records.append({'token_stop':stopped,'control':baseline});flush()

    for delay in (.03,.4):
        rid = f'abort-{delay}'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            doomed = pool.submit(generate,0,512,rid=rid,long=True)
            time.sleep(delay)
            http(url,'/abort_request',{'rid':rid})
            survivor = pool.submit(generate,1,12,rid=f'survivor-{delay}')
            try:
                aborted = doomed.result(timeout=120)
                assert aborted['meta_info']['finish_reason']['type']=='abort', aborted
            except HTTPError as e:
                body = e.read().decode();assert e.code in (400,500) and 'abort' in body.lower(),body
                aborted = {'status':e.code,'body':body}
            live = survivor.result(timeout=120)
            assert live['meta_info']['completion_tokens']==12, live
        records.append({'delay':delay,'aborted':aborted,'survivor':live});flush()
    flush()
    return {'passed':True,'cases':records}

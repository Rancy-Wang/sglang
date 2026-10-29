"""Finalize complete tables only after all 20 independent captures are analyzed."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1])
subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),str(root)],check=True)
performance=json.loads((root/'performance.json').read_text())
assert len(performance)==20
profiles=[]
canonical=json.loads((root/'native-reference-r1/prompts.json').read_text())['prompts']
for row in performance:
    name=f"{row['algorithm']}-b{row['bs']}-{row['pattern']}"
    for phase in ['performance','profile']:
        p=root/phase/name
        m=json.loads((p/'manifest.json').read_text())
        prompts=json.loads((p/'prompts.json').read_text())['prompts']
        assert [x['input_ids'] for x in prompts]==[x['input_ids'] for x in canonical[:row['bs']]]
        assert m['args']['max_new_tokens']==512 and m['args']['thinking']
        assert m['sha']=='da56f8103931e5b3a2451219b4e0c4b3dcb16286'
    summary=json.loads((root/'profile'/name/'summary.json').read_text())
    extra=json.loads((root/'profile'/name/'token-intervals.json').read_text())
    assert not extra['failures'] and extra['decode_outputs']==extra['expected_decode_outputs']
    profiles.append({'name':name,**{k:row[k] for k in ['algorithm','bs','pattern']},
        'profile_e2e_ms':summary['makespan_ms']['median'],
        'gpu_kernel_busy_union_ms':summary['gpu_kernel_busy_union_ms'],
        'depths':extra['depths'],'prefill':extra['prefill'],'coalesced_tokens':extra['coalesced_tokens']})
(root/'profile-summary.json').write_text(json.dumps(profiles,indent=2)+'\n')
base=(root/'report.md').read_text()
base=base.replace('E2E 为客户端发起到最后 token；', '模板渲染与tokenization在计时前由客户端完成，标准/generate请求传入input_ids；E2E为客户端发起HTTP请求到最后token，不含上述预处理；')
base=base.replace('Profile 单独运行，不能代替下表延迟。', '精细Profile每格单独采集1次完整请求组，另预热2次；阶段/token路径分位数是在该次采集的token之间统计，不是5次运行间统计。Profile不能代替下表延迟，调度和生成路径也可能与无profiler运行不同。')
base=base.replace('同时发送延迟全 0；', '同时发送为 bs 个独立 HTTP 请求共用计划发送时刻，实际发送存在毫秒级偏差，非一个 HTTP 批量数组；')
base=base.replace('AR decode 均执行完整 4 loop；', 'LoopSpec 的 depth3 路径前三个 loop 已在之前的流水线步骤提前计算，target 仍完成全部4个loop；AR decode 均执行完整 4 loop；')
base+='''
## TTFT、decode 和吞吐

下表来自无 profiler 的5次测量。TTFT/decode为全部请求记录的中位数；decode=最后token到达时间减首token到达时间。每请求详细值见 requests.csv。所有620个测量请求都达到512-token上限，因此属于固定输出预算测试，未测完整AIME答案或准确率。

|发送|bs|算法|TTFT ms|decode s|整批吞吐 token/s|最大计划发送偏差 ms|
|---|---:|---|---:|---:|---:|---:|
'''
for r in performance:
    base+=f"|{r['pattern']}|{r['bs']}|{r['algorithm']}|{r['ttft_median_ms']:.2f}|{r['decode_median_ms']/1000:.3f}|{r['throughput_tokens_s']:.2f}|{r['max_send_lateness_ms']:.2f}|\n"
base+='''
各阶段含义：prepare_rows包括分支KV分配、prefix映射、必要的新token embedding与状态打包；execute_core是一次48层共享core循环；finish为endpoint归一化和LM head读出；decision为草稿选择/验证决策；commit把完整target KV所有权转交原生请求；retain清理拒绝分支和不再引用的slot。AR的start/prepare负责常规forward状态与输入准备，随后完整执行4次core。准备、core、readout等嵌套区间不能累加为互斥耗时。

## 按 depth 对应的 token 到达间隔与其余 CPU 阶段

这些数值来自独立的 profile 请求。客户端 ITL 是相邻输出token的到达间隔，含网络/IPC/缓冲；不等于纯GPU计算时间。CPU时间为墙钟区间，含等待GPU，不能直接解读为CPU忙碌时间。CPU其余时间是父区间减去直接子区间，分组包含多个未单独打点的函数，不能视为某一个函数的独占耗时。共享调用同样不可跨请求累加。

AR 的 sample/publish/copy 分组包含采样、FutureMap发布与结果拷贝提交等主机操作；LoopSpec 的 plan/collect 分组包含分支规划、流水线收集与结果装配，scatter/readback 分组包含hidden/logit散布、元数据更新和批量决策D2H读取。结果处理是递归调用中的实际 native result handler，排除只有部分loop完成的外层空结果。

|发送|bs|算法|入口depth|decode输出数|新执行loop数中位|客户端ITL中位/p95 ms|结果处理CPU ms|sample/publish/copy CPU ms|plan/collect CPU ms|scatter/readback CPU ms|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
for r in profiles:
    for depth,group in r['depths'].items():
        def val(k):return f"{group[k]['median']:.3f}" if k in group else '—'
        itl=group['client_itl_ms']
        base+=f"|{r['pattern']}|{r['bs']}|{r['algorithm']}|{depth}|{group['outputs']}|{group['new_cycles']['median']}|{itl['median']:.3f}/{itl['p95']:.3f}|{val('native.result_handler_cpu_ms')}|{val('native.sample_publish_copy_host_ms')}|{val('loopspec.plan_and_collect_host_ms')}|{val('loopspec.scatter_and_decision_readback_host_ms')}|\n"
base += "\n## 首token / Prefill阶段\n\n每请求所经历的共享prefill阶段中位数，单位ms；各请求首个token均经过4个core loop。CPU/GPU区间为嵌套/共享测量，不能相加为TTFT。\n\n|发送|bs|算法|prepare CPU/GPU|4个core CPU/GPU|readout CPU/GPU|LoopSpec prefill接管 CPU/GPU|\n|---|---:|---|---:|---:|---:|---:|\n"
import statistics
for r in profiles:
    def pre_value(stage):
        groups=[x['stages'][stage] for x in r['prefill'] if stage in x['stages']]
        if not groups:return '—'
        return f"{statistics.median(x['cpu_ms'] for x in groups):.3f}/{statistics.median(x['gpu_stream_ms'] for x in groups):.3f}"
    base+=f"|{r['pattern']}|{r['bs']}|{r['algorithm']}|"+'|'.join(pre_value(s) for s in ['recurrent.prepare','recurrent.execute_core','recurrent.finish','loopspec.prefill_complete'])+'|\n'
base += "\n## Profile开销与kernel忙碌区间\n\n本表显式展示profiler开销。CPU墙钟/流区间在打点环境下测得，不能直接视为无profiler的阶段占比。kernel busy union仅是trace中kernel事件的去重时间，不包含所有memcpy或全部GPU活动，也不能将剩余墙钟时间全部称为GPU bubble。\n\n|发送|bs|算法|无profiler整批 s|profile整批 s|profile/无profiler|kernel busy union s|\n|---|---:|---|---:|---:|---:|---:|\n"
for r in profiles:
    baseline=next(x for x in performance if (x['algorithm'],x['bs'],x['pattern'])==(r['algorithm'],r['bs'],r['pattern']))['makespan_ms']
    base+=f"|{r['pattern']}|{r['bs']}|{r['algorithm']}|{baseline/1000:.3f}|{r['profile_e2e_ms']/1000:.3f}|{r['profile_e2e_ms']/baseline:.3f}|{r['gpu_kernel_busy_union_ms']/1000:.3f}|\n"
small=json.loads((root/'profile/loopspec-b1-simultaneous/summary.json').read_text())['stage_time_per_output']['3']
large=json.loads((root/'profile/loopspec-b16-simultaneous/summary.json').read_text())['stage_time_per_output']['3']
def gpu_pair(stage):return '/'.join(f"{x[stage]['gpu_stream_ms']['median']:.3f}" for x in [small,large])
base+=f"\n## 观察与后续定位方向\n\nbs1到bs16的depth3共享阶段GPU流区间中位数（ms）为：准备 {gpu_pair('loopspec.prepare_rows')}，core {gpu_pair('recurrent.execute_core')}，决策 {gpu_pair('loopspec.decision')}。大batch下决策区间已接近core本身，准备成本也明显增加。可优先对LoopSpecWorker._prepare_rows和逐endpoint的RequestContext.queue_decision做低开销测量，并评估批量决策/映射准备。上述是trace定位证据，包含随分支数增长的profiler开销，不能直接预测某项优化的无profiler收益。\n\nbs16整批E2E/吞吐基本持平，但单请求E2E中位数下降约7.5%（同时）和8.9%（交错）；TTFT则更高。不能只用单一吞吐或单一平均延迟概括所有收益。\n"
base+='''
## 完整性、正确性与边界

- 测量版本：da56f8103931e5b3a2451219b4e0c4b3dcb16286，SGLang v0.5.22 fork，AR和LoopSpec均使用普通CDB执行基础；本任务只修改实验/分析工具，serving实现未改动。单卡RTX5090，BF16、Triton、R=4、CUDA graph、overlap、KV容量12000、最大请求数16，LoopSpec draft depth=1。
- 模型：ByteDance/Ouro-2.6B-Thinking，revision f1edd81e7ac41355db670500ceaf204e0f73af68。checkpoint chat template，enable_thinking=True，add_generation_prompt=True；所有40组performance/profile的输入ID已与独立参考核对。模板、完整prompt IDs和停止原因均已保存。
- 独立HF4.54.1 eager检查q1/q16的prefix0/15/63，6个位置均完整执行R4，最终argmax全部与原生AR一致；logit最大绝对误差0.33594、最大RMSE0.07427。这是抽查，不是全512-token HF验收。
- 独立普通KV/完整R4原生控制禁用CDB，覆盖各bs、branch-width4行形状与graph等价prefill补齐，11264个同前缀位置，48个argmax变化。数值范围由这些控制的实测最大误差加25%余量决定，在LoopSpec评估前固定；详细范围保留在native-reference-r4/numerical_envelope.json。
- 实际LoopSpec graph分支计算覆盖16请求×64位置，1024位置通过上述范围；5个argmax变化，最大参考margin0.125。引用前缀物化、所有权、释放断言通过；bs1/16、两种发送方式的HTTP严格内存检查通过。不是所有512-token自由生成前缀的数值证明。
- 完整512-token序列仅52/310对AR/LoopSpec完全一致。bs1第3个token为1303/588，native参考margin0.125；独立原生AR以一个真实行加三个dummy行计算也选择588，说明至少这个早期分歧可由批形状舍入复现。其余首差异逐项保存在token-comparison.csv，未逐个做HF裁决。不能将本次速度称为严格逐token无损加速。
- 两种算法均有相同输出长度，但生成内容/接受率可能不同。AR整块先测、随后LoopSpec整块测，非ABBA交替设计；硬件telemetry和各次波动保留，bs16微小差异不作稳定加速结论。
- 直接阶段GPU值是CUDA event流区间，可能包含流等待；Chrome trace的kernel busy union才是去重kernel忙碌区间。两者不能混用或与CPU时间相加。CPU调度、HTTP/IPC等未完整单独插桩的时间保留为分组/未归因部分，不强行凑成E2E。
- 所有原始结果：/home/ruoxi/repo/test/results/loopspec/2026-10-29-thinking-batch-profile/。每格trace/包含CPU/GPU Chrome trace及recurrent JSONL；output-paths.json记录共享step IDs，token-intervals.json记录每个decode输出的对应关系。错误尝试与日志保留。正式loopspec分支未晋升。
'''
(root/'report.md').write_text(base)
(root/'final-status.json').write_text(json.dumps({'performance_cells':20,'profile_cells':20,
    'profile_decode_output_coverage':'all','profile_decode_tokens':sum(g['outputs'] for r in profiles for g in r['depths'].values()),'measured_requests':sum(r['requests_measured'] for r in performance),
    'all_inputs_match':True,'full_sequence_exact_pairs':52,'full_sequence_pairs':310,
    'full_sequence_exact_equivalence':False},indent=2)+'\n')
print(json.dumps({'complete_cells':20,'profile_cells':20,'inputs_verified':True}),flush=True)

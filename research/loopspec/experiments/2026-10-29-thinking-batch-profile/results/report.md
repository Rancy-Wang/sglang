# Ouro-2.6B-Thinking：AR / LoopSpec AIME profile

BF16，完整 R=4，LoopSpec draft depth=1。每题最多 512 新 token，保留自然 EOS；达到 length 上限为截断，不能解释成完整解题耗时。

同时发送为 bs 个独立 HTTP 请求共用计划发送时刻，实际发送存在毫秒级偏差，非一个 HTTP 批量数组；交错发送每题间隔 100ms。每格 2 次预热、5 次测量。模板渲染与tokenization在计时前由客户端完成，标准/generate请求传入input_ids；E2E为客户端发起HTTP请求到最后token，不含上述预处理；整批耗时从最早发送到最后 token。精细Profile每格单独采集1次完整请求组，另预热2次；阶段/token路径分位数是在该次采集的token之间统计，不是5次运行间统计。Profile不能代替下表延迟，调度和生成路径也可能与无profiler运行不同。

|发送|bs|AR 整批 s|LoopSpec 整批 s|AR/LS|AR 每请求 E2E 中位 s|LS 每请求 E2E 中位 s|AR/LS 输出 token|
|---|---:|---:|---:|---:|---:|---:|---:|
|simultaneous|1|14.032|6.252|2.244×|14.032|6.252|512/512|
|simultaneous|2|17.600|8.055|2.185×|17.594|7.655|1024/1024|
|simultaneous|4|17.892|9.906|1.806×|17.889|9.478|2048/2048|
|simultaneous|8|18.669|13.651|1.368×|18.662|12.626|4096/4096|
|simultaneous|16|24.826|24.820|1.000×|24.754|22.905|8192/8192|
|staggered|1|14.058|6.253|2.248×|14.058|6.253|512/512|
|staggered|2|17.522|8.150|2.150×|17.405|7.735|1024/1024|
|staggered|4|18.042|10.007|1.803×|17.769|9.377|2048/2048|
|staggered|8|19.143|13.801|1.387×|18.542|12.604|4096/4096|
|staggered|16|25.378|25.174|1.008×|24.242|22.075|8192/8192|

逐请求、逐重复 token 完全一致：52/310。首个差异保留于 token-comparison.csv；不能仅凭此速度表宣称逐 token 无损。

## 精细 Profile（每输出 token 路径所经历的共享阶段）

depth 从 0 计数：0 表示尚需 4 个 core loop，3 表示已经提前算完 3 个 loop，仍需 1 个。LoopSpec 的 depth3 路径前三个 loop 已在之前的流水线步骤提前计算，target 仍完成全部4个loop；AR decode 均执行完整 4 loop；prefill 首 token 独立记录。
CPU 为调用墙钟区间；GPU 为当前计算流上的 CUDA event 区间，含流等待。嵌套阶段不可相加，多请求共同经历的阶段不是每请求独占成本，亦不可跨请求累加。原始共享 step IDs 见每格 output-paths.json；全局不重复统计见 summary.json。

|发送|bs|算法|入口 depth|输出数|准备 CPU/GPU ms|core CPU/GPU ms|readout CPU/GPU ms|决策 CPU/GPU ms|提交 CPU/GPU ms|回收 CPU/GPU ms|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|simultaneous|1|ar|0|511|0.354/0.016|5.484/28.616|0.187/0.152|—|—|—|
|simultaneous|1|loopspec|0|42|2.387/2.352|6.495/37.574|0.776/1.066|2.511/2.205|0.062/0.049|0.050/0.040|
|simultaneous|1|loopspec|3|469|0.735/0.724|1.029/9.624|0.184/0.259|1.272/1.184|0.058/0.047|0.048/0.039|
|simultaneous|2|ar|0|1022|0.642/0.033|4.949/34.463|0.374/0.305|—|—|—|
|simultaneous|2|loopspec|0|93|3.310/3.274|5.441/40.823|0.752/1.053|7.206/6.792|0.288/0.234|0.206/0.170|
|simultaneous|2|loopspec|3|929|0.881/0.873|1.076/10.076|0.185/0.261|2.424/2.310|0.115/0.094|0.082/0.066|
|simultaneous|4|ar|0|2044|0.522/0.033|4.188/34.927|0.336/0.286|—|—|—|
|simultaneous|4|loopspec|0|185|4.836/4.801|5.469/42.788|0.762/1.061|14.365/13.792|0.590/0.479|0.486/0.402|
|simultaneous|4|loopspec|3|1859|1.298/1.289|1.239/10.850|0.187/0.263|4.432/4.284|0.193/0.158|0.156/0.127|
|simultaneous|8|ar|0|4088|0.590/0.034|4.535/36.168|0.382/0.287|—|—|—|
|simultaneous|8|loopspec|0|404|8.162/8.121|6.353/41.605|0.756/1.034|30.818/29.952|1.323/1.065|1.118/0.940|
|simultaneous|8|loopspec|3|3684|2.120/2.111|1.480/10.746|0.190/0.265|8.496/8.265|0.382/0.298|0.312/0.259|
|simultaneous|16|ar|0|8176|0.534/0.039|4.192/46.408|0.282/0.268|—|—|—|
|simultaneous|16|loopspec|0|917|15.437/15.400|6.100/67.005|0.792/1.060|62.192/60.681|2.962/2.249|2.465/2.031|
|simultaneous|16|loopspec|3|7259|4.034/4.024|1.513/17.128|0.195/0.262|16.184/15.795|0.770/0.592|0.650/0.540|
|staggered|1|ar|0|511|0.253/0.016|3.902/28.410|0.170/0.152|—|—|—|
|staggered|1|loopspec|0|42|2.295/2.261|5.390/37.004|0.747/1.047|2.473/2.170|0.058/0.047|0.041/0.033|
|staggered|1|loopspec|3|469|0.680/0.671|0.976/9.333|0.184/0.260|1.248/1.160|0.058/0.047|0.041/0.033|
|staggered|2|ar|0|1022|0.305/0.016|4.250/34.441|0.180/0.156|—|—|—|
|staggered|2|loopspec|0|103|3.306/3.272|5.586/40.622|0.763/1.062|7.237/6.828|0.290/0.235|0.215/0.174|
|staggered|2|loopspec|3|919|0.882/0.872|1.169/10.112|0.187/0.263|2.428/2.315|0.115/0.094|0.082/0.065|
|staggered|4|ar|0|2044|0.277/0.016|4.120/34.956|0.174/0.155|—|—|—|
|staggered|4|loopspec|0|184|4.902/4.856|5.479/43.211|0.768/1.063|14.854/14.262|0.616/0.496|0.510/0.420|
|staggered|4|loopspec|3|1860|1.297/1.284|1.260/10.813|0.188/0.264|4.071/3.922|0.183/0.149|0.158/0.130|
|staggered|8|ar|0|4088|0.521/0.035|4.110/36.215|0.327/0.286|—|—|—|
|staggered|8|loopspec|0|417|8.305/8.235|6.515/42.421|0.784/1.054|30.908/30.049|1.316/1.053|1.133/0.948|
|staggered|8|loopspec|3|3671|2.176/2.155|1.607/10.906|0.197/0.270|8.580/8.340|0.373/0.301|0.317/0.261|
|staggered|16|ar|0|8176|0.774/0.055|4.142/46.723|0.457/0.420|—|—|—|
|staggered|16|loopspec|0|912|16.463/16.091|6.315/67.840|0.813/1.078|62.030/60.537|2.935/2.232|2.511/2.081|
|staggered|16|loopspec|3|7264|4.222/4.155|1.576/17.010|0.203/0.271|16.222/15.856|0.764/0.597|0.652/0.546|

## TTFT、decode 和吞吐

下表来自无 profiler 的5次测量。TTFT/decode为全部请求记录的中位数；decode=最后token到达时间减首token到达时间。每请求详细值见 requests.csv。所有620个测量请求都达到512-token上限，因此属于固定输出预算测试，未测完整AIME答案或准确率。

|发送|bs|算法|TTFT ms|decode s|整批吞吐 token/s|最大计划发送偏差 ms|
|---|---:|---|---:|---:|---:|---:|
|simultaneous|1|ar|33.84|13.997|36.49|0.09|
|simultaneous|1|loopspec|42.26|6.210|81.89|0.09|
|staggered|1|ar|34.22|14.024|36.42|0.09|
|staggered|1|loopspec|42.07|6.211|81.87|0.09|
|simultaneous|2|ar|76.28|17.519|58.18|0.40|
|simultaneous|2|loopspec|87.32|7.567|127.13|0.40|
|staggered|2|ar|52.66|17.356|58.44|0.09|
|staggered|2|loopspec|65.11|7.670|125.64|0.09|
|simultaneous|4|ar|121.40|17.770|114.46|1.25|
|simultaneous|4|loopspec|138.10|9.343|206.74|1.43|
|staggered|4|ar|72.51|17.714|113.51|0.16|
|staggered|4|loopspec|89.53|9.309|204.65|0.10|
|simultaneous|8|ar|228.88|18.426|219.40|7.56|
|simultaneous|8|loopspec|237.12|12.391|300.06|2.96|
|staggered|8|ar|75.80|18.470|213.97|0.10|
|staggered|8|loopspec|95.24|12.553|296.79|0.10|
|simultaneous|16|ar|710.84|24.090|329.97|8.84|
|simultaneous|16|loopspec|815.55|22.110|330.06|8.14|
|staggered|16|ar|95.23|24.144|322.80|0.10|
|staggered|16|loopspec|125.98|21.913|325.41|0.09|

各阶段含义：prepare_rows包括分支KV分配、prefix映射、必要的新token embedding与状态打包；execute_core是一次48层共享core循环；finish为endpoint归一化和LM head读出；decision为草稿选择/验证决策；commit把完整target KV所有权转交原生请求；retain清理拒绝分支和不再引用的slot。AR的start/prepare负责常规forward状态与输入准备，随后完整执行4次core。准备、core、readout等嵌套区间不能累加为互斥耗时。

## 按 depth 对应的 token 到达间隔与其余 CPU 阶段

这些数值来自独立的 profile 请求。客户端 ITL 是相邻输出token的到达间隔，含网络/IPC/缓冲；不等于纯GPU计算时间。CPU时间为墙钟区间，含等待GPU，不能直接解读为CPU忙碌时间。CPU其余时间是父区间减去直接子区间，分组包含多个未单独打点的函数，不能视为某一个函数的独占耗时。共享调用同样不可跨请求累加。

AR 的 sample/publish/copy 分组包含采样、FutureMap发布与结果拷贝提交等主机操作；LoopSpec 的 plan/collect 分组包含分支规划、流水线收集与结果装配，scatter/readback 分组包含hidden/logit散布、元数据更新和批量决策D2H读取。结果处理是递归调用中的实际 native result handler，排除只有部分loop完成的外层空结果。

|发送|bs|算法|入口depth|decode输出数|新执行loop数中位|客户端ITL中位/p95 ms|结果处理CPU ms|sample/publish/copy CPU ms|plan/collect CPU ms|scatter/readback CPU ms|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|simultaneous|1|ar|0|511|4|28.901/31.579|19.993|0.529|—|—|
|simultaneous|1|loopspec|0|42|4.0|50.327/54.663|0.138|—|0.309|33.034|
|simultaneous|1|loopspec|3|469|1|12.967/15.405|0.123|—|0.165|8.654|
|staggered|1|ar|0|511|4|28.713/31.524|22.749|0.436|—|—|
|staggered|1|loopspec|0|42|4.0|47.573/50.606|0.133|—|0.301|32.565|
|staggered|1|loopspec|3|469|1|12.544/13.491|0.119|—|0.160|8.596|
|simultaneous|2|ar|0|1022|4.0|35.012/37.594|12.938|0.815|—|—|
|simultaneous|2|loopspec|0|93|4|57.985/62.548|0.121|—|0.748|36.586|
|simultaneous|2|loopspec|3|929|1|15.436/16.915|0.115|—|0.266|9.274|
|staggered|2|ar|0|1022|4.0|34.771/37.143|28.271|0.482|—|—|
|staggered|2|loopspec|0|103|4|57.847/63.237|0.125|—|0.768|36.390|
|staggered|2|loopspec|3|919|1|15.554/17.172|0.116|—|0.268|9.279|
|simultaneous|4|ar|0|2044|4.0|35.517/37.990|14.108|0.703|—|—|
|simultaneous|4|loopspec|0|185|4|75.249/84.777|0.119|—|1.378|39.172|
|simultaneous|4|loopspec|3|1859|1|20.191/22.380|0.096|—|0.447|9.980|
|staggered|4|ar|0|2044|4.0|35.303/37.783|29.184|0.465|—|—|
|staggered|4|loopspec|0|184|4.0|77.148/84.555|0.120|—|1.445|39.438|
|staggered|4|loopspec|3|1860|1.0|20.100/22.550|0.096|—|0.421|9.977|
|simultaneous|8|ar|0|4088|4.0|36.782/39.374|14.130|0.737|—|—|
|simultaneous|8|loopspec|0|404|4.0|104.971/122.902|0.125|—|2.864|37.551|
|simultaneous|8|loopspec|3|3684|1.0|27.373/32.894|0.093|—|0.777|9.889|
|staggered|8|ar|0|4088|4.0|36.944/39.506|7.511|0.683|—|—|
|staggered|8|loopspec|0|417|4|107.062/123.709|0.126|—|2.854|38.336|
|staggered|8|loopspec|3|3671|1|28.203/33.070|0.088|—|0.803|9.957|
|simultaneous|16|ar|0|8176|4.0|47.089/51.075|10.898|0.683|—|—|
|simultaneous|16|loopspec|0|917|4|187.738/223.524|0.127|—|5.708|64.968|
|simultaneous|16|loopspec|3|7259|1|48.452/57.729|0.077|—|1.509|16.580|
|staggered|16|ar|0|8176|4.0|47.694/51.506|10.191|0.858|—|—|
|staggered|16|loopspec|0|912|4.0|191.927/227.465|0.130|—|5.867|65.856|
|staggered|16|loopspec|3|7264|1.0|49.394/59.759|0.085|—|1.529|16.453|

## 首token / Prefill阶段

每请求所经历的共享prefill阶段中位数，单位ms；各请求首个token均经过4个core loop。CPU/GPU区间为嵌套/共享测量，不能相加为TTFT。

|发送|bs|算法|prepare CPU/GPU|4个core CPU/GPU|readout CPU/GPU|LoopSpec prefill接管 CPU/GPU|
|---|---:|---|---:|---:|---:|---:|
|simultaneous|1|ar|0.158/0.137|6.135/30.795|0.312/0.157|—|
|simultaneous|1|loopspec|0.175/0.157|5.137/31.656|0.328/0.157|25.736/0.371|
|staggered|1|ar|0.151/0.129|6.107/31.569|0.320/0.158|—|
|staggered|1|loopspec|0.150/0.131|4.923/31.608|0.304/0.157|25.798/0.158|
|simultaneous|2|ar|0.164/0.082|6.871/37.696|0.352/0.159|—|
|simultaneous|2|loopspec|0.126/0.070|4.961/38.414|0.261/0.156|28.833/0.145|
|staggered|2|ar|0.179/0.080|6.913/37.730|0.355/0.158|—|
|staggered|2|loopspec|0.152/0.134|6.216/40.774|0.337/0.156|17.851/0.245|
|simultaneous|4|ar|0.143/0.009|11.402/63.028|0.334/0.156|—|
|simultaneous|4|loopspec|0.109/0.009|11.546/77.789|0.311/0.153|17.747/0.215|
|staggered|4|ar|0.183/0.008|6.981/38.398|0.342/0.158|—|
|staggered|4|loopspec|0.108/0.085|5.417/41.322|0.269/0.158|10.508/0.132|
|simultaneous|8|ar|0.132/0.009|27.293/130.811|0.420/0.156|—|
|simultaneous|8|loopspec|0.114/0.008|15.379/171.920|0.242/0.153|41.971/0.298|
|staggered|8|ar|0.168/0.008|7.134/40.031|0.330/0.158|—|
|staggered|8|loopspec|0.124/0.103|7.138/43.695|0.292/0.158|10.242/0.146|
|simultaneous|16|ar|0.144/0.011|18.114/353.219|0.393/0.152|—|
|simultaneous|16|loopspec|0.115/0.011|25.041/224.854|0.284/0.155|54.901/0.386|
|staggered|16|ar|0.111/0.009|6.169/40.750|0.234/0.158|—|
|staggered|16|loopspec|0.118/0.100|7.339/47.076|0.264/0.158|10.107/0.148|

## Profile开销与kernel忙碌区间

本表显式展示profiler开销。CPU墙钟/流区间在打点环境下测得，不能直接视为无profiler的阶段占比。kernel busy union仅是trace中kernel事件的去重时间，不包含所有memcpy或全部GPU活动，也不能将剩余墙钟时间全部称为GPU bubble。

|发送|bs|算法|无profiler整批 s|profile整批 s|profile/无profiler|kernel busy union s|
|---|---:|---|---:|---:|---:|---:|
|simultaneous|1|ar|14.032|14.510|1.034|12.898|
|simultaneous|1|loopspec|6.252|8.366|1.338|4.944|
|staggered|1|ar|14.058|14.449|1.028|12.883|
|staggered|1|loopspec|6.253|7.873|1.259|4.937|
|simultaneous|2|ar|17.600|18.013|1.023|16.030|
|simultaneous|2|loopspec|8.055|9.977|1.239|5.637|
|staggered|2|ar|17.522|17.922|1.023|15.980|
|staggered|2|loopspec|8.150|10.643|1.306|6.017|
|simultaneous|4|ar|17.892|18.274|1.021|16.302|
|simultaneous|4|loopspec|9.906|13.504|1.363|6.602|
|staggered|4|ar|18.042|18.418|1.021|16.459|
|staggered|4|loopspec|10.007|13.655|1.365|6.696|
|simultaneous|8|ar|18.669|18.992|1.017|17.017|
|simultaneous|8|loopspec|13.651|19.483|1.427|7.922|
|staggered|8|ar|19.143|19.619|1.025|17.576|
|staggered|8|loopspec|13.801|20.139|1.459|8.203|
|simultaneous|16|ar|24.826|24.809|0.999|22.837|
|simultaneous|16|loopspec|24.820|36.239|1.460|14.264|
|staggered|16|ar|25.378|25.987|1.024|23.838|
|staggered|16|loopspec|25.174|37.543|1.491|14.609|

## 观察与后续定位方向

bs1到bs16的depth3共享阶段GPU流区间中位数（ms）为：准备 0.724/4.024，core 9.624/17.128，决策 1.184/15.795。大batch下决策区间已接近core本身，准备成本也明显增加。可优先对LoopSpecWorker._prepare_rows和逐endpoint的RequestContext.queue_decision做低开销测量，并评估批量决策/映射准备。上述是trace定位证据，包含随分支数增长的profiler开销，不能直接预测某项优化的无profiler收益。

bs16整批E2E/吞吐基本持平，但单请求E2E中位数下降约7.5%（同时）和8.9%（交错）；TTFT则更高。不能只用单一吞吐或单一平均延迟概括所有收益。

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

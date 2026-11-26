# Current task context

Active goal: compare ResidualQuant with VQ+ at matching steps and use up to20 newly authorized measured optimization rounds. Current11/20; prior two10-round budgets remain immutable. Chinese conversation; source authored locally, commit/push origin loopspec-test and remote Gitpull authorized. Never touch formal loopspec branch, useredits, otherjobs, existingSSHmasters. No subagents authorized. Six-part plan before every candidate edit/config, record.md is authoritative ledger.

## Source and contract

Local /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang, loopspec-test HEAD56fbb58de; remote BUS2 sameHEAD. Baseline7833b357a8d4554f90cd4eba309b64282c0b83ce. Productionsource stillunchanged. Officialreference readonly ../ResidualQuant@f4c3cccb76cb6df37eb6233fa92c7aaa054d6695. Task research/loopspec/experiments/2026-11-27-residualquant-profile. Previous harnesses ../2026-11-26-vqplus-residual.

Ouro2.6BThinking,48layers×4loops,TP2localH8,D128,preRoPEK,BF16centers/sharedjoint8bitindices,perloopINT4[-7,7]/FP16group32scales,page8/HPprefix8recent64,FP32composition+RoPE thenBF16round/FP32QKsoftmaxPV. .002/.01output,2e-5LSE,bitwiseencoder,servinggreedyIDs/lifecycle. No newcodec/retraining/qualitybenchmark. Gates >=10%B16N2k kernelgain,no>5%smallbatchregressionwithoutguard,then>=5%samehostfullservingthroughput.

## Runtime and sessions

BUS2 exec36805,ControlMaster/tmp/vqrq-20261127-bus2.sock,hostnameis-dbhptku7mdxq7ayk-devmachine-0,userwangruoxi. Repo/share/wangruoxi/repo/sglang-vqplus-kernel; RUN=/dev/shm/wangruoxi-vqrq-20261127 temporarytmpfs,archiveevidence locally. GPUsA80080GBSM80/108SM,driver535.161.08. GPU0 UUID79630815-938c-ed94-d02d-77a40b9834a0,GPU1ce3c4e31-7c9b-e23b-0c73-bbe43ffc2bc1,GPU2 2c74d358-1c1a-d483-1a48-1662ab847796,GPU3 d43d1df1-5064-25e9-446a-36f788d752d0. EightGPUsidlebeforeR11;R11uses0/1,2/3reservedfullserving.

BUS1exec29655/socket/tmp/vqrq-20261127-bus1.sock;Tengexec61593/socket/tmp/vqrq-20261127-teng.sock;X5exec90355/socket/tmp/vqpage-20261126-x5.sock;A12exec24610/socket/tmp/vqpage-20261126-a12.sock. Keepallmasters. A12read-onlycommandauto-rejectedtwicebecauseapprovalrequired/policyNever; do notbypassexecutionviaotherchannel. Userinformed;continueauthorizedBUS2.

NCUrootprobesall3BUShostsexit1ERR_NVGPUCTRPERM:containerCapEff0xa80c65fb lacksSYS_ADMIN; noNCUreports/metrics. Driver535needs hostenablement/containercapability, no systemchanges authorized. CUDAprobespass. Evidenceprofile/20261127-{bus1,bus2,teng}-counter-probe. CUDAevents/clock64areavailable,notNCUstall/trafficdata.

Microenvironment RUN/env/bin/pythonTorch2.5.1cu121,Triton3.4manualpairing(customkernelsverified,notfullservice/torchcompile),CUDA12.2/GCC9.4. ShellCUDA_HOME=/usr/local/cuda-12.2,TORCH_EXTENSIONS_DIR=RUN/extensions,TRITON_CACHE_DIR=RUN/triton34-cache,SGLANG_CACHE_DIR=RUN/sglang-cache,MAX_JOBS2,OMP_NUM_THREADS8,PYTHONPATH=repo/python,TORCH_CUDA_ARCH_LIST8.0. DO NOTchange whenpreparingserving;useprocesssubshells.

GitHubremoteHTTPpullstalled;localforkpushworks. Fallback: strictlocalgitdaemon127.0.0.1:39437 --base-path=<localrepo> --strict-paths <localrepo>/.git; verifySSHmasterthen -O forward -R127.0.0.1:39437:127.0.0.1:39437. Remote git pull --ff-only git://127.0.0.1:39437/.git loopspec-test. Afterpullcancelonlytemporaryforwardandverifieddaemon,nevermaster. R11temporarydaemon63198/forwardclosed. No sourceSCP orremoteedits.

## Evidence and current findings

Realinputs allhashchecked RUN/{2k,1024,4096,8192},16requests/layer23/all4loops only; manifestsalsolist0/47notcopied. RUN/books Ksha bb20d06b1c187bfaf7f03456ce2536fb505429fe365a4a2ee7a1cedf7e97a2a7,V c5becdb3b882ee8584cb8cf163ca96a928fb838c459ba417270a6ccb295b06db. Local evidence/inputs same. Model RUN/model fixedHFf1edd81e7ac41355db670500ceaf204e0f73af68,all8filesmatchhistoricalprovenance,weightsha c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080. Provenance RUN/compat/training-provenance.json,model-hashes.json. OfficialHFnetworkunreachable;hf-mirrordownloadhashverified.

R1 baselineB16N2kpage222.09us/native97.78,100boundary/LSE/dynamicgraph/paddingpass. R2matchedfull222/dense101/scalar133.6;meanCTAcyclesQK~63%,PV~36%,softmax~1%,clockoverhead<1%;notadditivewalltime/NCUstalls. Full63regs,dense/scalar40.
R3split-onlytarget5=178.37us;R4smaller128/256CTA~190/187rejected. R5extra8Ktarget10oldreader617.4vs627.6immaterial. StoredRUN/r3/b16-n8192-l0,needlocalcopy.
R6combinedpayloadpair8+pv16 helpers:PVonly40regs144us,combined53regs154us. R7__launch_bounds__(512,3)combined40regs124.12but8BstackactualSTL/LDL,diagnosticnotpromoted. R8explicitFMAreconstruction mathematicallyexactfiniteFP16scale×INT4product,131858432bitwisepropertycasespass,40regs0stack/local;combinedtarget5=123.82vs222.09us,44.25%lower. All4loops~123.6–123.8. N1kbesttarget4=71.8vs114.55,N4ktarget5=219.22vs342.91;B8target5=61.27vs111.96.
R9moretargets: R8N8ktarget10=422.65vs627.69/native388.22;B1target24=16.23vs18.46/native16.98,B4target10=32.37vs52.13/native33.90. B2R1target12=21.77vs33.41butactualproductionB2R2notcompared. Page_auto inpayload_probe isR1,NOTactualResidualAttentionB2dispatch.
R10QKdot2FMAchain7d72a4a2a:output/LSEpass40regs0stack;2k123.25vsR8same-run123.72,8k420.76vs422.60 (~0.4%). RejectpromotionasimmaterialbenefitforchangedFP32association,retainR8originaldot.
R11shared_source.py stagescode-majorK+PVpermutedV64KiBtablewithcp.async8B;R8payload/mathunchanged. Expected2CTAresidencyvs3global;actualperformancepending. Source56fbb58de. Jobs2629057GPU0B16N2k,2629059GPU1N8k;--fma --min-blocks3 --shared-table --modes both_r8 both --targets2 3 4 5 8 10. RUN/r11/*.{json,log,exit}. Checkfailurebeforeanynextcandidate.

Localr1/r2/r3early/r4/r6/r7/r8copied;R9SCPexec91974andR10exec5817pending/check. R8SASSnotyetarchived,resourcesfilesvqrq_payload_r6_both_t512_r7_b3_r8_fma-resources.txt. Preservelargeevidenceuntracked. No productionchanges yet;boundary/memcheck/fullserviceacceptanceoutstanding.

## Full serving environment (not successful yet)

Readonlyoldakv/share/public/wangruoxi/local/akv/bin/pythonTorch2.13cu130,sglang-kernel0.4.7. Task-localofficialNVIDIAcuda-compat13-0_580.196.11extracted(noinstall)atRUN/compat/root/usr/local/cuda-13.0/compat allowsCUDAcomputeandserverimportswithdriver535. PackageSHAa5be2f03bd67550f7264e6b44863c5252d3e9be82bbfefe504a73ee389990aa2. SourceNVIDIAforwardcompatmatrix;cuMemMap13featuresmayrequirebase580,checkactualerrors.

Baselineattempt1RUN/serving-baseline/server.logexit2 obsolete --cuda-graph-bs ambiguous;attempt2server-v2.logexit1requireskernel>=0.4.9;neverbypasscheck. NoGPUserveractive. FreshRUN/serve-envPython3.12.15,uv0.13.10installingmatchedtorch2.14.1/sglang-kernel0.4.9/transformers5.19.0;wrapper2627895uv2627961,logcompat/install-core.log/exit. UV_CACHE_DIR=RUN/uv-cache,publicPyPI;largepackagesdownloadingsteadily~2GBunpackedat20min. Do notduplicatereinstallormodifyakv. Aftercompletionaddoldreadonlysite-packagesvia.pthforotherdeps(corefreshenvpriority);testABI/importsCUDA,handleactualfailures. Oldsitepath/share/public/wangruoxi/local/akv/lib/python3.12/site-packages.

ServingCUDA_HOME=/share/public/wangruoxi/local/context-r2-20260929/cuda-toolkit-full/nvidia/cu13;CC/CXX=/share/wangruoxi/runtime/pd-drop-c8-20260919/compiler/bin/x86_64-conda-linux-gnu-{gcc,g++}; LD_LIBRARY_PATH=RUN/compat/root/usr/local/cuda-13.0/compat:compiler/lib:CUDA_HOME/lib;LIBRARY_PATH=/share/public/wangruoxi/local/akv/cuda-link:CUDA_HOME/lib. SeparateHF_HOME=RUN/hf-cache,XDG_CACHE_HOME=RUN/serve-cache,SGLANG_CACHE_DIR=RUN/serve-cache/sglang,TORCH_EXTENSIONS_DIR=RUN/serve-extensions,TRITON_CACHE_DIR=RUN/serve-triton,PYTHONPATHrepo/python. GPU2,3onlyper-processsubshell.

Nextbaselinev3launch: modelRUN/model,trustremote,host127.0.0.1port30664,TP2,BF16,Tritonattention,pytorchsampling,page8,context32768,totalKV65536,maxrunning16,disable-radix-cache,chunked-prefill-size-1,max-prefill-tokens8192,memfraction.90,decodegraphfull --cuda-graph-bs-decode1 2 4 8 16,prefillgraphdisabled,metrics,decode-log-interval1,incrementalstreaming,ouro-decode-profileoptimized,triton-attention-num-kv-splits48,booksRUN/books,prefix8recent64,residualbits4,codebookglobal,layoutpage-head,pagekernelcuda. ExistingOLD/check_serving.py lifecycle/timing/profileunchanged. Needbaselineactualproductionunchangedthenbestcandidate,samehost/env/GPU,matchedgreedyIDs. SeparateBF16serverlaternativeKVsplits8. Do notsubstituteoldA12performance.

## Next action

InspectR11results/resources;checkinstallprogress. ThenplanproductionintegrationofthebestmeasuredcandidatewithGPU-length-awarepartitionpolicyandlowbatchguards;validate100boundaries/LSE/dynamicgraph/memcheck,allloops/contexts/actualdispatch. Collectfullservingbaselineandcandidatepluscross-layerprofiling. Updateplans/resultswithhonestunknowns;cannotclaimglobaloptimumorfullservicegainfrommicrobenchmarks.

# Additional optimization budget

User: “我再批准20轮优化”. Baseline7833b357a8d4554f90cd4eba309b64282c0b83ce, branchloopspec-test. Earlier10/10 batches retained. Current1/20 started; no production kernel changes in this round.

## Round1/20 — isolated measurement entry

1. Purpose: replay existing production CUDA/Triton functions on BUS2 without importing unrelated model/server dependencies; establish counter access and real-input baseline.
2. Steps: add a thin task-local bootstrap for existing diagnose_paged/check_paged/check_residual scripts; expose real package paths; provide only hardware-query utilities needed by decode_attention; run unchanged harness and unchanged kernels. Add reproducible ncu_report metric/source extraction, not implementation modifications.
3. Files: task-local run_isolated.py and analyze_ncu.py. No production math/dispatch edits.
4. Success: real snapshot/book hashes validated by existing harness, oracle pass at unchanged .002/.01, NCU report collected and parsed. No performance improvement claimed for measurement infrastructure.
5. Risks: bypassing package initialization could alter environment; record all supplied utilities, use real environ module, assert SM80/CUDA, and compare source kernel functions unchanged. CUDA12.1Torch/Triton3.1 may expose source compatibility gaps; fix in a separately counted round if required. NCUroot may still lack container capabilities.
6. Validation: Python compile, source review, Git push/pull, isolated real B16/N2048 baseline and boundary tests; sameGPU matched native/page profiles; full/source sections supported by NCU2023, no fabricated PM sampling. Store unique profile/<run> dirs under task workspace with harness/reports/analysis. Include versions and GPU UUID in logs.

Environment observation: existing akv environment uses Torch2.13cu130 and cannot initialize on driver535. New isolated Torch2.5.1cu121 environment is being prepared without modifying existing environments. BUS2 /share/wangruoxi currently resolves to overlay with only4.5GiB free; install paused before completion to choose mounted storage.

Before first GPU validation, extend this same measurement-infrastructure candidate with counter_probe.cu: a checked, tiny CUDA launch to test NCU counter permission independently of the ongoing Python download. It is not a performance benchmark. Compile locally authored source remotely through Git using nvcc sm80; compare non-root/root NCU basic report status. Storage selected: task-owned /dev/shm/wangruoxi-vqrq-20261127 (executable tmpfs,381GiB free, host816GiB available), since persistent shared mounts are read-only; retain results locally before teardown. No system changes.

Round1 counter result: a95c8b1014e39a1ddfaf4a5192798ac6aaa7cb9f; nvcc probe succeeds, root NCU exits1 ERR_NVGPUCTRPERM on BUS1 GPU0, BUS2 GPU0 and Teng GPU4. No report/metric was collected. All root CapEff0xa80c65fb omit CAP_SYS_ADMIN. Official NVIDIA documentation requires host enablement or container SYS_ADMIN for driver535. Counter failure is infrastructure, not kernel failure. Local Python syntax passed; full real replay pending dependency/data setup. Measurement bootstrap retained.

## Round2/20 — stage cycles and matched arithmetic control (planned before edits)

Environment/baseline checkpoint for rounds1/2: remote01228fe427d3e11e82b171147b758b47384e4cd5, Torch2.5.1cu121/Triton3.1/CUDA12.2/GCC9.4. Round1 boundary runner2625215 exited0: all100 shape cases plus LSE/dynamic graph/padding passed. Real runner2625214 exited1 while compiling unchanged native BF16 decode: Triton3.1 lacks tl.extra.cuda.gdc_wait, even though the GDC path is disabled on SM80. VQ+ output checks occurred before this native failure, but no timing accepted. Upstream Triton3.4 exports this symbol; upgrading only the isolated environment's standalone Triton to3.4 (no source edits, same candidates). Torch2.5's package metadata pins3.1, so this is an explicitly tested manual pairing for eager tensors/custom Triton kernels, not a supported torch.compile/full-serving environment. Rerun both boundary and baseline after upgrade. Preserve original failure logs; no counter evidence or speedup claimed.

Round1 final baseline (same code, Triton3.4): baseline-t34.exit0 and boundary-t34.exit0. GPU0 A800 UUID79630815-938c-ed94-d02d-77a40b9834a0, B16/N2048/layer23/loop0,30 paired graph samples,32inner calls. Median current227.435us, page_head_auto222.088us, native_serving97.780us (actual5splits/max8). Page range222.001–222.243us, native97.601–97.872us; current first sample246.478us then~227.4us (preserved warmup transient). All real output checks passed .002/.01; page max_abs9.54e-7, native max_abs.00390625 but within relative tolerance. This numerical difference must be retained when interpreting native-vs-VQ costs; same-mapping FP32 dense control is essential. All100 boundaries/LSE/dynamicgraph/padding checks passed again. Page compiled63registers/thread,0stack/local memory. No production kernel improvement in round1; round2 stage measurement currently compiling.

1. Purpose: obtain stage-level evidence despite unavailable hardware counters. Distinguish compressed reconstruction/RoPE cost from the existing thread mapping/softmax/reduction cost.
2. Steps: create task-only full/scalar-diagnostic/dense-matched CUDA readers using the unchanged production source; dense control reads materialized post-RoPE BF16 and removes RoPE, preserving QK/PV ownership/reductions. Add an optional separate instrumented compilation writing clock64 timestamps at existing QK/softmax/PV synchronization boundaries. Compare instrumented vs uninstrumented graph timing; report overhead, not NCU stalls or traffic.
3. Scope: task-local stage_profile.py only. No production dispatch/codec changes. Real trained-codebook input and existing oracle/timing helpers reused. Preserve the original native Triton reader as an additional whole-attention control.
4. Success: full and dense controls satisfy unchanged output tolerance, scalar against its own independent reconstruction oracle; stage cycles from active CTAs only. Timing speedups from uninstrumented readers only. If instrumentation changes latency materially (>10%), treat cycle fractions as perturbative evidence and require independent ablations before optimization conclusions.
5. Risks: clock/global-store overhead, register allocation differences, cycle counts are CTA residence time not additive global latency; scalar differs in numerical values and is not a deployment candidate. Same-source dense control is not the production BF16 baseline. Setup/materialization excluded and explicitly labeled.
6. Validation: local syntax/generated-source checks, Git synchronization, real B16/N2048 layer23loop0 output checks, paired warmed graphs and repeated CTA-clock samples, compiled resource extraction. Extend layers/loops after a useful hypothesis emerges. Primary performance comparison remains actual native serving kernel and final full serving; no speed claim from instrumentation alone.


Round2 result: source01228fe42, BUS2GPU0, exit0. Thirty paired warmed graph samples (32 inner calls), B16/N2048/layer23/loop0: production222.070us, full222.084us, full_clock222.235us, dense101.328us, dense_clock102.138us, scalar133.566us, scalar_clock134.058us. All unchanged output checks passed. Median mean-CTA cycles QK/softmax/PV: full103312.54/1864.68/58174.35; dense67073.21/2022.58/52665.14; scalar98478.05/2390.27/42704.25. Instrumentation latency overhead below1%. Full63registers/thread vs dense/scalar40, all0stack/local. Clock samples use individual launches rather than graph hot loops, so cross-variant cycle differences cannot directly decompose graph latency. Evidence evidence/r2/{stages.json,stages.log,stages.exit,resources.txt}. No NCU metrics, no production change.

## Round3/20 — existing-kernel split scheduling sweep (before edits)

1. Purpose: test whether the full reader's higher register use and underfilled CTA waves contribute to B16 latency; distinguish this from reconstruction instruction/traffic costs.
2. Steps: reuse unchanged production page-head reader, override only its existing active_target launch argument; preplanned targets1,2,3,4,5,6,8,12,16,24,32. Compare page auto and native. Test B16/N2048 first, then B1/2/4/8/16 and N1024/4096/8192, loops0..3 for promising settings with same unchanged harness. Preserve every point.
3. Scope: task-local schedule_probe.py; no production source or cache-format edits. Real input/codebook hashes and materialization/oracle reused from stage_profile.py.
4. Success: output .002/.01 and logsumexp2e-5/2e-5 for every tested point; >=10% lower B16/N2048 median before considering production policy. Lower batch regressions must be handled by explicit dispatch, not omitted.
5. Risks: changing split changes reduction order; excess partitions incur launch/reduction overhead. Resource-derived occupancy is a hypothesis, not a measured hardware stall diagnosis. Short sequences clamp target, document effective splits.
6. Validation: local syntax/review, Git push/pull, GPU/process/memory preflight, paired CUDA graphs30samples/32inner, sameGPU baseline, retained samples and sourceSHA. LSE checked against independent dense FP32 scores. No full-serving claim from microbenchmarks.


Round3 first result:42d3effc8, BUS2GPU0, B16N2048layer23loop0 exit0. All output/LSE checks passed. Mediansus: auto222.103,native97.853; targets1/2/3/4/5/6/8/12/16/24/32 =222.041/222.104/179.324/185.046/178.368/190.630/184.294/206.078/203.133/238.675/248.252. Target5 lowers latency19.69%, target3~19.26%. This establishes a scheduling effect, not its exact stall mechanism. Source unchanged, broader loop/shape sweeps running wrappers2626278GPU0 and2626280GPU1; actual effective splits retained. Baseline CPU/GPU environment unchanged, allGPU idle before launches.

## Round4/20 — CTA size and split co-design (before edits)

1. Purpose: test whether reducing the existing512-thread CTA improves work distribution for the63-register full reader, after R3 demonstrated a19.7% scheduling gain.
2. Steps: expose existing prepare(threads=128/256/512) in the task sweep; fixed preplanned128 and256thread candidates across targets2,3,4,5,6,8,12,16,24,32; keep512threadauto/native controls. Same noncompact page-head source and arithmetic, unlike prior rejected compact/layout1 experiment.
3. Scope: schedule_probe.py argument and explicit candidate thread selection; existing CUDA compile specialization only, no production dispatch change. Do not change R3 baseline or old evidence.
4. Success: output .002/.01 and LSE2e-5; beat both current512auto and best R3 schedule atB16N2048, then useful across1K–8K/loops/batches. Record compiled registers/spills per specialization.
5. Risks: fewer warps perCTA can reduce latency hiding, increase per-thread serial work, and alter FP32 reduction order. Split sweep must be considered jointly; parameter search is bounded and does not establish a global optimum.
6. Validation: local syntax/diff, Git sync only after existingGPU jobs finish, idleGPU preflight, same30sample/32inner paired graphs and independent oracle per point; resource extraction. Subsequent production policy requires separately counted implementation and boundary/serving validation.


Round3 extension all10cases exit0, source42d3effc8; GPU0 B16 loops0..3,N1024/2048/4096 andGPU1 B1/2/4/8,N2048. B16N2k all4loops target5~178us vsauto~222us. N1kauto114.60/target3 98.79/target5 107.53us;N4kauto344.12/target5 315.63us. B8N2kauto112.00/target3 89.94us;B4auto52.08/target5 47.29/target12 46.78us;B1auto18.46/target24 18.55us;B2R1auto33.43/target12 25.82us (productionB2 usesR2, so R1auto is not its serving baseline). All points retained; no universal fixed-target rule accepted. Native timing at4k198.62us differs from earlierhost/toolchain and is reported as measured, not substituted. N8k pending data upload.


Environment-only full-serving investigation: A12 original session read-only status was automatically rejected (approval required, policyNever). No alternative execution channel used to bypass this. Continue on authorized BUS2. NVIDIA official forward-compatibility matrix marks cuda-compat-13-0 compatible with535/datacenter GPUs. Download official580.196.11-1ubuntu1 package, SHA256a5be2f03bd67550f7264e6b44863c5252d3e9be82bbfefe504a73ee389990aa2,65405870bytes, extract under runtime/compat/root with dpkg-deb -x only. No system installation/driver modification; test existing read-only akvcu130Python using per-process LD_LIBRARY_PATH. Keep original microbenchmark environment/library path unchanged. Source https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html and NVIDIAubuntu2204Packages.gz. Compatibility remains unverified until a CUDA launch succeeds. R4Gitpull currently waiting on network; no newGPU run started yet.


Git transport recovery: forkpush ofa2549f84f9783e8137b729ce497c051f0c21ac4e succeeded; BUS2publicGitHubfetch twice stalled (bounded retry returns low-speed error). Terminated only our verified stuckgit-remote-http2626948. Started task-local Gitdaemon exec28796 on127.0.0.1:39437 with strict-paths restricted to localsglang, forwarded same loopbackport through verified BUS2ControlMaster; remote git pull --ff-only git://127.0.0.1:39437/sglang loopspec-test preserves the exact pushed commit and Git object provenance. No source SCP/remote edits. Cancel temporaryforward/daemon after sync, retainSSHmaster. Compatibility probe succeeded: existingread-onlyakv Torch2.13.0cu130 seesA800 andCUDAarange.sum=120 usingtask-localcompatlib; fullserving dependencies still unverified.


Round4 result:a2549f84f, bothB16N2klayer23loop0 runs exit0, all output/LSE checks pass. T256GPU0 best187.112us(target3),T128GPU1 best189.902us(target32), paired512auto222.584/222.074us andnative97.861/97.862us. Both64regs/thread,0spills. Neither improves onR3T512target5 178.368us: reject smallerCTA candidate, retainallpoints. TemporaryGitforward anddaemon closed after successfulpull, SSHmaster retained. Existing akvfullserver import succeeds fromtargetrepo withtaskcompat; CUDA13toolkit andGCCcompiler already exist viaactivate-bus2.sh, no environment mutation needed. FrozenHFmodel revisionf1edd81e7ac41355db670500ceaf204e0f73af68 download started, hash will match historical provenance before use.

## Round5/20 — additional 8K schedule points

Accounting correction before payload edits: the just-launched8K sweep on unchangeda2549f84f includes targets9/10 beyondR3's enumerated targetlist, alongside8/12/16/24/32. Charge this added behavioral sweep asR5 instead of hiding it inR3. Its purpose is long-context wave-fill sensitivity, scope runtime/schedule_probe.py unchanged, success output/LSEpass andlower same-run latency, risk extra partitions/rounding, validation same30samples/32inner native+auto paired controls. No source change. The9/10 additions were launched before this written addendum; retain this procedural deviation explicitly. Output remains runtime/r3/b16-n8192-l0.* for immutable provenance, logicallyR5. Payload code has not yet been edited and is nowR6.

## Round6/20 — combine token payload reads (before edits)

1. Purpose: reduce actual repeated scalar loads/address calculations and expose independent centroid gathers; stage evidence places63%CTAcycles inQK and36%PV, andcuobjdump confirms repeatedU8index/U16residual/U16scale loads inPV.
2. Steps: add paired-QK row8 helper sharing page/HP branch and row address for low/highRoPE groups; load twoindices/residuals/scales andcenters before reconstruction. Add PV16 helper loading fourindices asuint32,16residual nibbles asuint2 and one group32scale, then fourcentroid gathers and unchangedFP32accumulation. Preplanned ablationsQK-only/PV-only/combined, same512threads and targets3/5, currentauto andoriginaltarget3/5 controls. No mathematical regrouping.
3. Scope: task-only payload_source.py andpayload_probe.py. Productionreader untouched until candidate passes. Source transformation asserts exact anchors; compilecache uniqueR6names.
4. Success: unchanged output .002/.01 andLSE2e-5/2e-5; lower latency thanbestR3sameGPUpairedcontrol, then multi-context/loop/batch validation. No improvement claim from static instruction counts alone.
5. Risks: preloading increasesregisterliveness/occupancycost; vector alignment and64bitaddressing; HPbranchmust preserveBF16 boundaries; potentialcompileralias/inlining effects. Reject any numericalfailure orperformance regression.
6. Validation: generated-source/syntaxreview, Gitversion sync, GPUpreflight, realhashedB16N2k input,30pairedgraphsamples/32inner, scalarcompiledresources andSASS. Usefulcandidate subsequently goes through100boundary/LSE/dynamicgraph/memcheck andfullserving/lifecycle in a separately counted integration round.


Round5 result:a2549f84f,BUS2GPU0,B16N8192layer23loop0 exit0,alloutput/LSEpass. Auto627.562us,native388.321us,targets8/9/10/12/16/24/32=627.541/635.864/617.417/647.888/628.686/666.546/671.515us. Target10 gain1.6% fails10%optimizationgate; do not justify an8Kspecialcase from this tiny single-cellgain. Preserveevidence under r3/b16-n8192-l0.* (see accountingnote). Round6 sourcegeneration anchors/braces andPythoncompile passed forqk/pv/both beforeGPUtest. Helpers use read-only loads, so changes include cache/load scheduling as well as vector packing; no isolatedvector-only attribution claimed.


Round6 result:0d76bfbfd,BUS2GPU0,B16N2048layer23loop0 exit0, allunchangedoutput/LSEchecks pass. Mediansus:auto222.095,native97.971;originaltarget3/5=179.259/178.163;qktarget3/5=163.856/164.789;pvtarget3/5=190.538/143.999;bothtarget3/5=152.673/154.337. PVtarget5 reducesbaseline35.16% andbestprevious19.18%. Resourcesqk63,pv40,both53regs/thread,all0spills. These arepairedmicrobenchmarks, notfullserving. KeepPVcandidate;combinedmorearithmetic-localitybenefit doesnotoutweighitsresource/schedulecost atthese settings.

## Round7/20 — constrain combined registers and retune new payload scheduling (before edits)

1. Purpose: test whether combinedQK/PV canretainitsinstruction/branchbenefit whilematchingPV-only's40register allocation andthreeCTAresourcecapacity.
2. Steps: add optional __launch_bounds__(512,3) forpayloadcompilation; no arithmetic change. Preplanned unboundedPV/both/original andboundedboth at targets2,3,4,5,6,8,10,12,16 onB16N2048. This is a separatelycountednewcompiler/configcandidate. Thenunchangedwinning settings across1K/4K/8K,loops0..3,B1/2/4/8/16.
3. Scope: payload_source.py optionalmin_blocks transform/cachekey, payload_probe.py flag/metadata. Productionstillunchanged. Default0 usesexistingR6source/cacheidentity.
4. Success: output .002/.01,LSE2e-5 andlowerpairedlatency thanPV-onlybest144us; zero spills preferred, rejectspillingoranyperformance regression. Verify actualcompiledresources ratherthanassumingcompilerhonorsdesiredresidency.
5. Risks: compiler spills/rematerialization, reducedILP, scheduling overhead, cross-GPUclockdifferences; eachcandidate hasownsameGPUbaseline andrawsamples.
6. Validation: syntax/generated-sourceanchors, Gitprovenance, idleGPUpreflight, B16N2kpaired30samples/32inner,resourceusage/SASS. Preserveallvariantsandfailedruns; broadenonlypromisingcandidate, thenindependentboundary/memcheck/fullservingintegration.


Full-serving baseline preparation (no productioncandidate/sourceedit): existingakvTorch2.13cu130/Triton3.7.1/Transformers5.12.1/FlashInfer0.6.18 importshttp_server andmodel_runner successfully. FixedHFrevision downloaded viahf-mirror withimplicittokensdisabled becauseofficialHFnetworkunreachable; beforeuseverifyall8historicalprovenancehashes (including5.336GBmodel). UseGPU2/3samepair forbaseline/candidate,localport30664,TP2R4 BF16model, Tritonprefill/decode,page8,context32768,totalKV65536,maxbatch16,fullgraphs1/2/4/8/16,prefillgraphdisabled,radixdisabled,chunkedprefill-1,optimizedOuroprofile,VQ+INT4globalpage-head,prefix8/recent64,48nativeVQsplitsflag. Baseline productionreader unchanged atR7commit; separate servingextension/cache dirs andper-processcompat/CUDA13compiler env. Existingcheck_serving.py lifecycle/timing suites unchanged. Model/server outputchecks mustpassbefore performanceclaim. Runtimeerrors retained; do notsilently patchmodel orweakenchecks.


Round7 result:e40e0cb0a,bothruns exit0,alloutput/LSEpass. BoundedbothT512minblocks3 reaches124.12us(target5),sameGPUauto222.10/native97.97/PVbase144.01us;unboundedbothbest152.74us(target3),PVbest144.11us(target5). Bound3resources40regs,8Bstack,LOCAL0; SASS provesoneSTL beforePV andoneLDL insideloop per kernel specialisation. This isrealinvariantspill, notzero-spill. Retain fastestdiagnostic butdo notpromoteunderR7'sstatedspillcriterion. Fullmatrixrawr7JSON retained. Fullserverinitiallaunch exits2 beforeGPUwork: obsolete --cuda-graph-bs ambiguousbetweenphase-specificoptions. CorrectCLIto --cuda-graph-bs-decode,explicitdecodefull/prefilldisabled; this restoresalreadyplannedconfiguration, notanoptimizationcandidateornewnumerics.

## Round8/20 — exact fused reconstruction (before edits)

1. Purpose: reduce decodearithmetic andregisterpressure withoutchangingreconstruction rounding. ForanyfiniteFP16scale andsignedINT4code, theexactproduct needs<=15significandbits andhasFP32representableexponent; thereforeFP32productisexact. FP32add(center,product) andfmaf(scale,code,center) roundthesameexactsum forfiniteBF16centers. KeepseparateRoPEproducts/adds andallBF16boundaries.
2. Steps: optionalexplicitfmafonlyforcenter+scale*code inrow4,pair8,pv16. Retain--fmad=false globally. AddindependentCUDApropertytest usingexplicit__fmul_rn/__fadd_rn versus__fmaf_rn: allfinitehalfbitpatterns×16signedcodes×64edgeBF16centers, andallfiniteBF16patterns×16codes×64edgehalfscales. Thenpairedunfused/fusedPV andbound3combined atpreplannedtargets2,4,5,8; fulloutput/LSEoracle. Unfusedreferences staysameprocess/control.
3. Scope: taskpayload_source/probe plusfma_check.py. No productionedit,quantizationortrainingchange. Defaultflags retainR6/R7cacheidentity; fusedcacheuniquer8suffix.
4. Success: zero bitwisepropertymismatches, unchangedattentiontolerances, fewerinstructions/noPVspill andlowerlatencythanR7fastdiagnostic124.12us orR6spill-free144us. Ifspillsremain,retainasdiagnosticonlyunderexistinggate; do notrelaxnumericorservicechecks.
5. Risks: nonfiniteinputsoutsidefinitecodecproof, signedzero/subnormalcompilerbehavior, accidentalRoPEfusion, compilerallocationchanges. Testcodeincludesnegative8althoughencoderuses-7..7; finitehalfscalesofbothsigns tested. No--use_fast_math.
6. Validation: Python/generated-sourceanchors, CUDApropertytestfirst, realtrainedB16N2klayer23loop0 output/LSE,30pairedgraphsamples/32inner,resources+SASS; winningcandidateextendedtoall4loops/lengths/batches beforeproductionboundary/memcheck/fullserving. Fullserverbaselinebeingpreparedfromunchangedproductionreaderwithverifiedmodelhashes.


Round8 propertytest63c8fe7c2 onBUS2GPU0 exit0,131858432finitecases,zero bitwiseFP32mismatches. Cases covereveryfinitehalfscale×16codes×64BF16edgecenters andeveryfiniteBF16center×16codes×64halfedges,includingbothsigns/zeros/subnormals. Attentiontimings nowrunningGPU0boundedcombined andGPU1unboundedPV withsame-processunfusedcontrols; proofdoesnotreplaceattention/servingchecks. Fullserverv2failedrequiredsglang-kernel0.4.9versioncheck;freshserve-envinstallingmatchedTorch2.14.1/sglang-kernel0.4.9/Transformers5.19.0,oldakvuntouched. No version-checkbypass.


Round8 attentionresults:63c8fe7c2,bothruns exit0. GPU0fusedcombinedbound3target5=123.821us vsunfusedbound3 124.093us,PVbase144.114us,production222.094us,native97.978us. GPU1fusedPVtarget5=142.188us vsPVbase144.001us. Bothfusedvariants40regs,0stack/LOCAL; theFMAchange removesR7spillbutitsadditional~0.2%combinedtimingdifference isnotclaimedmaterial. Bestspill-freecandidate44.25%lowerlatencythanproduction. Alloutput/LSEpass. Extendedunchangedcandidate startedGPU0loops1/2/3,target5;GPU1N1024/4096/8192,targets2/4/5/8;GPU4B1/2/4/8N2k,targets2/4/5/8. Lowbatchrequiresmoreparallelismoroldpathguardiftheseconfigurationsregress. No productionpromotionyet.


Round8 extensions all10cases exit0: loops1/2/3 target5=123.80/123.69/123.58us; B16 N1k besttarget4=71.80 vs114.55us; N4k target5=219.22 vs342.91; N8k effective8=449.87 vs627.13. B8 target5=61.27 vs111.96; B4 target8=35.95 vs52.02. B1 target8=26.45 vs18.44 is a regression; B2 R1 target8=27.06 vs33.34 does not establish improvement over actual productionR2. Fixedtarget5 rejected as universal policy. Correctness allpass, scheduling pending.

## Round9/20 — complete resource-aware schedule measurements (before changes)

1. Purpose: choose dispatch from measured length/batch behavior for the40-register, zero-spill R8 reader; avoid smallbatch underparallelization and test8K wavefill.
2. Steps: unchanged R8 fusedcombinedminblocks3; additional targets3,10,12,16,20,24,32 across B16N1024/2048/4096/8192 and B1/2/4/8N2048. Preserveauto/native controls and all samples. No mathematical/source changes.
3. Scope: runtime configuration of existing payload_probe.py only; recordthis as a separate optimizationround. ActualproductionB2R2 willbe compared before integration; page_auto hereis R1 diagnostic.
4. Success: unchanged .002/.01 output and2e-5LSE checks; improvepairedlatency and deriveboundedpolicy, with no unsupported >5% smallbatch regression. No globaloptimum claim.
5. Risks: extra splits/merge overhead andchangedreductionorder, data/clock variation; use sameGPUpairedbaselines and retainedrawsamples.
6. Validation: existinghashed real snapshots/books,30samples/32innerCUDAgraphs,idleGPUpreflight,exitstatus andJSON. Candidate policy is a later countedimplementationround requiringboundary/dynamicgraphs/memcheck/fullserving.


## Round10/20 — FP32 QK dot instruction chain (before edits)

1. Purpose: test whether removing one arithmetic instruction per low/high pair reduces QK cost after payload reconstruction and register allocation improvements.
2. Steps: optional replacement dot += fmaf(kl,qlo,kh*qhi) with dot=fmaf(kl,qlo,dot); dot=fmaf(kh,qhi,dot). BF16 Q and K products are individually exact in FP32; accumulation association changes and is not claimed bitwise equivalent. Keep reconstruction, RoPE arithmetic and BF16 conversions unchanged. Compare R8 combined bounded fused control with new chain at targets3/5/10, B16N2k and8k, then useful shape/loop extensions.
3. Scope: task-local payload_source.py optional source transform/cache suffix; payload_probe.py flag and explicit both_r8 control. No production edit.
4. Success: existing .002/.01 output and2e-5LSE tolerance; lower paired latency and zero spills. Reject failed numerics or immaterial improvement before promotion. No precision reduction or tensor-core approximation.
5. Risks: FP32 reduction association affects softmax and final greedy ties; fullserving greedy comparison remains required. Compiler may increase live values despite fewer instructions.
6. Validation: Python source/anchor checks, exact Git synchronization after R9 jobsfinish; idle GPUs;30pairedgraphsamples/32inner, compiled resources, all4loops/multiplecontexts/boundary tests ifpromising. R8 reference remains unchanged andcache-distinct.

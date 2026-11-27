# Active continuation checkpoint (R18 launch pending)

Latest user requires generalization acrossGPU,bs,maxtokens. No final B16/108SM-only policy. Fixedtargets are diagnostic; finalselection mustuseactualdevicecapacity andshape,withfallback/explicitunverifiedstatusforotherGPUs. ExpandfullservingB32/N1K,B64/N512withinmatchedKV/graphlimits. Preservetrained/numericalcontract. Finishremainingrounds18–20;14–17done,noNCUwait.

R17 b0afc5752 allcorrectnesspassedbut2Kqshared135.765,pv8qshared142.510,b4130.0usvsR13128.082; reject. b432regs0stackbothwidths,othermodes40/38regs0stack. R18source3ee08a534 pushed; remoteff-onlypullinprogress. PlanrecordedbeforeGPU. Fourgeometries×targets3/5/10/20; probeexplicitrepeatedrequestprovenanceallowsB32/64andproperactualgroupedcontrolbuffers. No productionchange yet.

BUS2 exec20202/socket/tmp/vqrq-r14-bus2.sock/master82096. Main /share/wangruoxi/repo/sglang-vqplus-kernel; RUN /dev/shm/wangruoxi-vqrq-20261127. CUDA13/int64 probes serve-env/serve-extensions. TemporaryGitdaemon82709/39437 +reverseforwardOPEN;closeonlyafterfinalsync. Sourceonlylocal+Git,noagents.

R13refreshserver2638845 stopped; lifecycleandtimingpassed, B16=371.713token/s. FrozenR13 worktree/share/wangruoxi/repo/sglang-vqplus-r13@4743e9430. BF16refreshserverwrapper2640803 currentlyidleGPU2/3 port30664; lifecycle/timingexit0, B16median419.653647. Client2641396done. AllR17microjobsdone. R14/R15/R16 archivedlocally;R17andBF16refreshneedarchive. NextlaunchR18,adaptR19integrationtowell-supportedgeneralpolicy,completeR20andfullserving. Historicalcontextbelowissuperseded.

# Final task context

Completed13 of maximum20 newly authorized rounds; earlier budgets immutable. Retained runtime source4743e9430 on local/fork loopspec-test. Analysis/results commit777c5bb1a is pushed and synchronized to remote main; production code is unchanged from4743e9430. No formal loopspec promotion. No subagents. All candidate plans/results in ../record.md; detailed conclusions in design-review.md; reproducible acceptance in plan.md.

Local target /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang. Task research/loopspec/experiments/2026-11-27-residualquant-profile. ResidualQuant read-only sibling atf4c3cccb76cb6df37eb6233fa92c7aaa054d6695. Original production control56fbb58de, candidate4743e9430. Remote main/share/wangruoxi/repo/sglang-vqplus-kernel is synchronized through777c5bb1a; baseline and candidate dedicated remote worktrees remain at their measured revisions.

Contract: Ouro2.6BThinking48layers×4loops,TP2,H8/D128,SM80/A800108SM,page8,HP8/64,context≤32768; trainedpre-RoPEK/sharedUINT8index/BF16center/INT4[-7,7]/FP16group32scales,FP32composition/RoPE thenBF16round,FP32QK/PV. Output .002/.01,LSE2e-5,encoderunchanged,servinggreedyIDs/lifecycle. No retraining/newcodec/broadqualityeval.

R13 pair8/pv16+exactFMA,40registers/0stack/local onCUDA12 and13,actualcapacity324,429B GPU-length schedule table. Automaticdispatchonlylayout2/defaultgroup/dynamic/H8/108SM/capacity324/B4..16; otherpaths retained includingB2R2. All8batches×4contexts pass,all4loopsat2K;100boundary/LSE/Graph cases andmemcheck0errors. B16N2K125.121us vsold221.404/native97.862; B1/2/3within1%. Sharedcodebookstaginganddotreassociationrejected.

Fullserving sameGPU2/3/env/model/books/seed605366612,TP2/R4/graphs. Baseline56fbb58de andcandidate4743e9430lifecyclegreedyIDsidentical includingpad3/5/15/16 andabort/reuse. Allthree methods lifecycle/timing/profileexit0, identicalinputfixture. MediansB1/2/4/8/16: old49.055/85.577/129.056/189.413/293.573; new49.118/85.558/168.965/264.672/371.748; BF1647.368/90.435/170.092/262.932/419.645 token/s. B16+26.63%vsold,still11.41%belowBF16. Original/retained/BF16historicalreadertraceTP0 means184.195/127.557/103.910us; traces notcounters/walltime. FulltraceN>2048 oldsplitcount≥3 unlikefixed2Kmicrotarget2; no cross-environment absolute comparison.

BUS2persistent exec36805,socket/tmp/vqrq-20261127-bus2.sock,userwangruoxi,hostis-dbhptku7mdxq7ayk-devmachine-0. RUN/dev/shm/wangruoxi-vqrq-20261127 temporarytmpfs. EightA80080GBdriver535.161.08. GPU2UUID2c74d358-1c1a-d483-1a48-1662ab847796,GPU3d43d1df1-5064-25e9-446a-36f788d752d0. Alltaskservers andmicrojobs stopped,port30664free,GPUcomputeprocesslistempty. PreserveSSHmasters:BUS1exec29655,Teng61593,X590355,A1224610. A12read-onlyexecutionauto-rejectedtwice;do notbypassanotherchannel.

NCUrootprobesall3BUS hostsERR_NVGPUCTRPERM,lackCAP_SYS_ADMIN. No NCUreports/traffic/stalls. CUDAevents/clock64/CUPTIavailable. Counterprobe reports in profile/. Next follow-uprequirescounter-capablehost; plan.md givesconcretecapturestep. No systemdriverchanges.

MicroenvRUN/env Torch2.5.1cu121/Triton3.4,CUDA12.2/GCC9.4. ServeenvRUN/serve-env Python3.12.15,Torch2.14.1cu130/Triton3.8,sgkernel0.4.9,Transformers5.19,Torchvision0.29.1,cutlass-dsl[cu13]4.8.0. Ownreadonly-akv.pthappendsolddeps;oldenvuntouched. Task-localcuda-compat13_580.196.11 andcuobjdump13.0.85 extractedwithoutinstall. Allfailedsetupattemptspreservedin evidence/compat. Newsourceauthoredlocally,Git-onlysync; evidence/dataSCPviaexistingmasterallowed.

Allrealinputs/booksverified; local evidence/inputs contains16requests/layer23/all4loops at1K/2K/4K/8K. Model revisionf1edd81e7ac41355db670500ceaf204e0f73af68,weighthashc506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080; fullprovenancearchived. Evidence/r1..r13 andserving-baseline/candidate/bf16,compat/compilerbuildflags/rawtraces savedlocally. analyze_serving.py reproduces summary; acceptance-manifest.json hasheskeyartifacts. Largeevidenceuntrackedbydesign;preserveotheruseruntrackedfiles.

TemporaryGitforward/daemoncurrentlyclosed. If finalGitpull needsfallback: strictlocalgitdaemon127.0.0.1:39437 servingrepo/.git,verifiedmasterreverseforward,remoteff-onlypull git://127.0.0.1:39437/.git loopspec-test,thencloseonlytempforward/daemon. Neverclosemasters/reset/force/clean. Allrequiredimplementation,acceptanceandanalysisworkiscomplete. Results777c5bb1aarecommitted/pushedandremoteff-onlysyncverified. Thisfinalcontextreceiptaddsnoimplementationchange. Anynewoptimizationmuststartanewrecordedroundandretaincurrentacceptedbaseline.

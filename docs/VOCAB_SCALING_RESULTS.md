# Feasibility evidence, 2026-11-04

# FP32 IsoFLOPs feasibility

Observed screening results; confirmation uses separate documents. No quality conclusion from feasibility.

## baseline8k

Final screening BPB: 2.710160; best screening: 2.710160; confirmation final: 2.5816273722930796
Consumed FLOPs: 50892756872976; unused: 0; raw bytes: 2006956
Vocabulary archive bytes: 8389916; median update seconds: 0.14609270001528785
Peak memory: {'cuda_allocated': 128442368, 'cuda_reserved': 293601280, 'peak_allocated': 252152832, 'peak_reserved': 293601280}. Complete storage/training-memory accounting: run/result.json; sparse spectra and factor diagnostics: diagnostics.jsonl.

## vocab16k

Final screening BPB: 2.559683; best screening: 2.559683; confirmation final: 2.380973348342717
Consumed FLOPs: 50741547070464; unused: 151209802512; raw bytes: 2188511
Vocabulary archive bytes: 8388120; median update seconds: 0.16184350004186854
Peak memory: {'cuda_allocated': 128434176, 'cuda_reserved': 427819008, 'peak_allocated': 319638528, 'peak_reserved': 427819008}. Complete storage/training-memory accounting: run/result.json; sparse spectra and factor diagnostics: diagnostics.jsonl.

## Complete-update profiles and audit

102 regression/correctness tests passed. Counter and resume-log audits passed for both arms. Profiler and timed sessions are separate.

| Metric | 8K Dense | 16K Low-rank r126 |
|---|---:|---:|
| Vocabulary archive bytes | 8389916 | 8388120 |
| Total inference archive bytes | 27875460 | 27873793 |
| Trainable parameters | 6962688 | 6962176 |
| Parameter bytes | 27850752 | 27848704 |
| Gradient bytes | 27850752 | 27848704 |
| AdamW state bytes | 55701784 | 55697692 |
| Warm profile median update seconds | 0.14434860000619665 | 0.13987660000566393 |
| Profile peak allocated CUDA bytes | 251628544 | 320031744 |
| Profile peak reserved CUDA bytes | 287309824 | 459276288 |

Low-rank / Dense warm profile ratios: update time 0.969x, peak allocated 1.272x. Memory slightly exceeds the 1.25x screening target; recipe unchanged.

Full feasibility active timings partly overlapped regression execution. Use the isolated warmed profiles for the runtime screen; feasibility timing logs remain actual observations, not clean performance benchmarks. Host peak RAM and isolated temporary allocation are unmeasured. No extended toy or large training occurred.

Active byte throughput: Dense 46436.4, Low-rank 48258.3 bytes/s. Token throughput: 12290.5, 11626.4 targets/s.

Confirmation evaluates the final checkpoint; best screening is a separate saved artifact. No interpolation is used. See verification.json for complete raw counters and memory payloads.


This checked-in record summarizes ignored artifacts; it is not a claim about converged quality.

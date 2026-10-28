> Standalone edition: the dataset protocol and historical numerical results below
> are retained for provenance. Historical `src/`, archive, migration gates and
> commands describe the producing repository; they are not runtime dependencies.
> Use `../README.md` for current commands and `PROVENANCE.md` for read-only replay.
> The standalone training/checkpoint path now explicitly supports this FineWeb
> validation protocol only. Historical WikiText references below record provenance.

# FineWeb v1 active data protocol

Gate 4B location note: the unchanged shared BPE operations now live in
`src/vocab_lm/tokenizer.py`; their original source is preserved at
`archive/legacy/src/tokenizers.py`. The Gate 4A commands below record the
pre-cleanup layout (`migration-pre-cleanup-gate4a`). Current read-only regression
and non-writing preservation commands are in the root README. Existing artifact
seals and their original implementation hashes remain unchanged; they describe
the source executed at Gate 4A, not the relocated working tree. Do not republish
that sealed generation merely to update source-path receipts.

Gate 4A prepares data only. No language model training, compression fitting,
recovery or test evaluation is authorized. Settings are in
[`migration_gate4a.yaml`](../configs/migration_gate4a.yaml). Historical WikiText
protocols and artifacts remain unchanged.

## Source and selection

Use standard [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/9bb295ddab0e05d785b879661af7260fed5140fc),
configuration `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`.
The pinned dataset card lists default, 10BT/100BT/350BT samples and individual
Common Crawl configurations; the complete list is retained in the source manifest.
The 10BT sample is an appropriate bounded general FineWeb source, not FineWeb-Edu.
Of its 15 Parquets, the smallest is `sample/10BT/014_00000.parquet`:

- Download: 574,962,194 bytes, estimated before transfer; HTTP HEAD returned 200
  after redirects, with matching repository revision and published LFS hash.
- SHA-256: `7168f92a680b3e28155a6b654ebf235b19bf4d2ab35f91dbab3fb47cbd629a8e`.
- Local preparation estimate: 2 GB, with roughly 273 GB free before preparation.
- 281,664 source rows. The entire bounded file is inspected locally.

This is hash sampling within one deliberately bounded source file. It is **not a
global random sample of FineWeb**. Sample-tail composition, web-domain mix and
near duplicates can affect representativeness. Only exact document duplicates
are removed; no near-duplicate or substring contamination claim is made.

Stable IDs are `dataset@revision/path#row=N`; FineWeb native `id` is also recorded.
For each pinned file, sort by SHA-256 of `42:order:source_id`, breaking ties with
the ID. Deduplicate exact UTF-8 hashes, retaining the first in that order.
Reserve validation when SHA-256 of `42:split:source_id`, interpreted as an integer,
is zero modulo 16. All other documents are train candidates. Domain separation
keeps split membership independent of the order hash. Empty documents follow
the same ordering, deduplication and split rules, and are counted explicitly.

Take complete-document prefixes until training reaches 200,000,000 bytes and
validation reaches 10,000,000 bytes. Overshoot is retained. Smoke is the shortest
training prefix reaching 2,000,000 bytes. No document is truncated.

| Tier/split | Target raw bytes | Actual raw bytes | Documents | Status |
| --- | ---: | ---: | ---: | --- |
| Smoke train | 2,000,000 | 2,000,916 | 574 | Prepared; pilot prefix view |
| Pilot train | 200,000,000 | 200,000,346 | 65,182 | Prepared |
| Fixed validation | 10,000,000 | 10,001,247 | 3,219 | Prepared; shared across tokenizers |
| Medium train | 1,000,000,000 | — | — | Future; not authorized |
| Large train | 5,000,000,000 | — | — | Future; not authorized |

Expansion must retain this pinned source-file batch as the first batch. Hash
order remaining candidates inside this batch first; append newly pinned files
after it in an explicitly frozen file-list order. Never globally sort an enlarged
candidate pool or insert files ahead of an established batch. This preserves the
selected training prefix. Freeze existing validation IDs/hashes; do not add later
validation-bucket candidates. Continue deduplication against previous candidates
and validation hashes. The synthetic extension test covers an appended source
file and unchanged train/validation prefixes. Medium/large materialization and
incremental append publication are future work, not executed by this gate.

An expanded artifact generation gets a new source/selection identity. Preserve
this generation; an eventual incremental writer should append new raw bytes and
shards while retaining old references, or publish a separate generation. The
current CLI refuses to overwrite prepared raw text, tokenizers or shards.

## Raw bytes and tokenizer family

`raw.bin` contains each selected `text.encode('utf-8')` exactly once, physically in
source-row order. `selection.json` stores byte offsets separately from the logical
hash order, raw lengths, raw hashes, native/source IDs, split and selection order.
No normalization, whitespace modification, inserted separator or reconstruction
is applied. Source Parquet is retained for provenance/replay. Smoke duplicates
neither raw text nor token arrays.

The tokenizer-training sample is **exactly all ordered pilot train documents**:
65,182 documents / 200,000,346 bytes. `training_sample.json` records all IDs,
lengths, hashes and order. Its SHA-256 is
`82f941e75dba2498e5ac54a42ef1e1532c5ccb12106ff1a8df3705ab75585fd1`.
Validation is excluded by both IDs and exact document hashes.

Reuse the unchanged `src/tokenizers.py` BPE implementation: ByteLevel alphabet,
regex pretokenization, no prefix space/normalizer, minimum frequency 2, EOS
`<|endoftext|>` at ID 0. Train 16,384 once; retain its first 8,192 IDs and first
7,935 merges for 8K. All special definitions and shared IDs remain identical.
Literal EOS spelling stays ordinary text; EOS is inserted explicitly. Two
independent full-sample fits reproduced identical parsed artifacts with tokenizer
parallelism disabled. Versions: Python 3.10.0, tokenizers 0.22.1, NumPy 2.2.6,
PyArrow 21.0.0. Configuration, versions, execution base commit and artifact
hashes are recorded; final verification also hashes the executed source files.

| Artifact | SHA-256 |
| --- | --- |
| 8K `base.json` | `8eb2caf9d3ca4d1390c60c12d11633b086682939a59110bf83e66adc39008649` |
| 16K `expanded.json` | `cf148983e4f5838d445c0eb0ccc6793997d03da9509910298b9094b29b1eb38c` |

Future 32K/64K families require an explicitly recorded tokenizer-training sample
protocol, even if the sample remains the same. A larger or differently sampled
training corpus may be necessary. Train that family's maximum size and derive
smaller merge prefixes anew; do not assume fitted 16K can be extended. Validate
IDs and replay within the new family; do not assume identity across families.

## Token shards and loader

The binary-shard concept follows the simple mmap approach reviewed for
SuperTinyLM at Gate 0. No external code or artifact is copied. New format
`vocab_lm_token_shards`, version 1:

- 64-byte header: `<8sIIQ` magic `VLMTSHRD`, version, token width and count;
  remaining header bytes are zero. Explicit little-endian uint16 IDs follow.
- Choose uint16 only for vocabulary sizes at most 65,536, including specials;
  uint32 otherwise. Synthetic uint32 serialization/mmap is tested end to end.
- `index.json` binds tokenizer/source/selection hashes, token width, ordered
  shard hashes, and each complete document's offset, token length, source ID,
  selection order, raw length/hash and tokenizer hash.
- Token byte lengths use a tokenizer-bound lookup table, avoiding a repeated
  length integer for every corpus token. EOS maps to zero bytes. Final verification
  compares this table with the tokenizer and reconstructs every document.
- Rotate near 2,000,000 stored IDs, only between complete documents. A single
  longer document may exceed this threshold. Each stored document is
  `[EOS] + text_tokens + [EOS]`; prefix EOS is context, closing EOS is a target.

`ShardedDocuments` maps token arrays read-only on CPU and supplies the existing
`TrainingData` Sequence interface. It verifies hashes, headers, ID ranges,
contiguous boundaries, document order and byte totals. One complete document is
converted to tuples and cached for sequential windows. Metadata is resident and
scales with document count; startup hashing, full identity construction and long
document conversion remain CPU costs. No full corpus is placed in GPU memory.

`load_training_data(root, config, vocab_size=8192|16384, tier='smoke'|'pilot')`
constructs only the loader. It never constructs a model or optimizer. Smoke
has its own prefix identity and references pilot shards. Training window policy
is unchanged: `document_eos_disjoint_targets_shuffled_v1`, configurable context
256 / stride 128, deterministic `Random(seed + epoch)` document permutations,
sequential disjoint targets, per-row document isolation and ignored padding.
The cursor retains epoch/permutation position/first target; its identity binds
tokenized content, tokenizer, selected corpus and settings. Corpus expansion
changes identity and cannot silently resume an old epoch permutation.

## Validation and statistics

FineWeb protocol `fineweb_document_eos_c256_s128_v1` uses context 256 / stride
128. The shared evaluator retains causal document-local scoring, prefix-only
EOS, scored zero-byte closing EOS, FP32 forward/CE, FP64 loss reduction and
exact UTF-8 denominators. No padding targets contribute loss. All targets,
including the first text token, are scored exactly once. Exhaustive prepared-data
verification checks coverage without a model; a synthetic uniform-logit test
checks analytical BPB. No FineWeb language-model BPB was computed.

WikiText remains context 128 / stride 64. Its BPB and FineWeb BPB describe
different distributions and must not be directly compared. Token context also
covers different raw byte lengths across vocabularies. At the original Gate 4A, the training controller used only WikiText validation.
The separately authorized standalone repair now dispatches by the explicit
checkpoint validation protocol without changing either scoring policy. FineWeb validation
can use `evaluate(..., FineWebEvaluationConfig(), expected_bytes=10001247)`.

Corpus token counts below include one closing EOS per document, excluding the
prefix-only EOS. Binary shards additionally store the prefix EOS.

| Split | 8K tokens | 16K tokens | Token reduction | 8K bytes/token | 16K bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke | 529,614 | 482,150 | 8.96% | 3.78 | 4.15 |
| Pilot train | 53,041,227 | 48,361,071 | 8.82% | 3.77 | 4.14 |
| Validation | 2,617,427 | 2,388,605 | 8.74% | 3.82 | 4.19 |

Raw document byte p50/p95/p99/max are 1,827 / 8,731 / 20,815.9 / 572,473
for train and 1,802 / 8,387.5 / 20,307.74 / 267,516 for validation. Selected
splits contain no empty documents; synthetic empty-document cases are tested.

| Vocabulary / split | Tokens/byte | Used vocabulary | Types with frequency 1 / at most 10 | Documents >4,096 text tokens | Windows/epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K train | 0.265206 | 8,136 (99.32%) | 0 / 10 | 1,120 | 446,618 |
| 16K train | 0.241805 | 16,323 (99.63%) | 7 / 43 | 952 | 409,783 |
| 8K validation | 0.261710 | 8,100 (98.88%) | 16 / 139 | 55 | 22,058 |
| 16K validation | 0.238831 | 16,177 (98.74%) | 93 / 1,247 | 46 | 20,218 |

Rare-type counts exclude unused types. One closing EOS per document is counted
in frequencies and windows. Training contains 59 documents above 65,536 raw
bytes; validation has six. Maximum train text-token lengths are 138,789 / 126,317
at 8K / 16K, respectively. Text-token p50/p95/p99 are 485 / 2,292.95 / 5,592.52
(8K train), 441 / 2,091 / 5,100.76 (16K train), 482 / 2,200.5 / 5,484.02
(8K validation), and 440 / 2,011.5 / 5,019.34 (16K validation).

`statistics.json` contains document byte/token percentiles, exact token frequency
arrays, utilization, unused/rare type counts, EOS/empty counts, long-document
counts/maxima, tokens/byte, bytes/token and estimated window counts for both
tokenizers and all prepared views. Tokenization savings imply no model-quality,
matched-budget or embedding-compression result.

## Reproduction

Use the existing project `.venv`; no packages need installation. From repository
root, for a fresh, approved destination with no prepared artifacts:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage source
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage download
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage raw
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage tokenizers
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage shards
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage verify
```

The first two stages access the network. Inspect the printed source transfer/disk
estimate before invoking download; downloads above 2 GB are refused. Existing
source downloads are checked before reuse. The executed source inspection used
the equivalent HF API/config-file listing and `curl.exe -I -L` accessibility check,
followed by a bounded curl download and published SHA-256 verification. Those
original metadata snapshots and inspection record are retained, so provenance
JSON hashes may differ from a fresh API receipt; selected IDs, texts, tokenizers
and token payloads follow the same pinned protocol.

For existing artifacts, offline verification is read-only. It checks the artifact
seal before replaying source hashes, selection, all tokenizations, validation
coverage and exact pilot loader continuation. Only `--publish` writes statistics,
a verification receipt and an artifact seal, and it refuses an existing seal.

The standalone implementation has been reorganized: source acquisition and raw
document selection now reside in `fineweb.py`, shard storage in `data.py`,
vocabulary representations in `model.py`, and budget policies in `train.py`.
This changes producing-code fingerprints without changing the dataset methodology.
Shared hash/JSON helpers reside in `runtime.py` to avoid circular imports.
Current fingerprints cover FineWeb, tokenizers, data, evaluation, shared runtime,
the preparation CLI and its pinned configuration. Historical manifests, seals and
recorded code hashes must remain unchanged.

Review the recorded/current provenance difference, then explicitly request a
compatibility replay. This option permits producing-code differences only;
artifact-integrity checks remain strict. Preserve console output outside the
artifact directory as a replay receipt:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_fineweb --stage verify --allow-code-provenance-mismatch *> .tmp/fineweb_compatibility_receipt.txt
```

The configured `data/fineweb_v1` must already exist; `--root PATH` can select an
existing generation. A missing generation cannot be verified offline. The
compatibility receipt is evidence of replay with current code, not a replacement
for historical producing-code provenance.

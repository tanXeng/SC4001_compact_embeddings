"""Dense training, exact resume and validation of prepared FineWeb checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from vocab_lm.accounting import storage_report
from vocab_lm.checkpoint import load_training_checkpoint
from vocab_lm.config import load_config
from vocab_lm.fineweb import read_json, sha256
from vocab_lm.data import TrainingData
from vocab_lm.evaluate import evaluate
from vocab_lm.fineweb import dataset_identity
from vocab_lm.model import CausalTransformer
from vocab_lm.runtime import seed_all
from vocab_lm.data import DocumentPrefix, ShardedDocuments
from vocab_lm.tokenizer import NestedBPE, tokenizer_identity
from vocab_lm.train import train


def load_data(config):
    """Bind prepared shards to the actual tokenizer, selection and corpus split."""
    root, directory = Path(config.data.root), Path(config.data.tokenizer)
    pair = NestedBPE.load(directory)
    expanded = config.vocab_size == pair.expanded.get_vocab_size()
    tokenizer = tokenizer_identity(directory, expanded)
    selected = pair.expanded if expanded else pair.base
    if selected.get_vocab_size() != config.vocab_size:
        raise ValueError("Configured vocabulary differs from tokenizer")
    byte_lengths = [len(pair.token_bytes(i, expanded=expanded)) for i in range(config.vocab_size)]
    identity = dataset_identity(root, "train")
    validation = dataset_identity(root, "validation")
    train_docs = ShardedDocuments(root / "shards" / str(config.vocab_size) / "train",
                                 tokenizer_sha256=tokenizer["sha256"], dataset_sha256=identity["sha256"])
    valid_docs = ShardedDocuments(root / "shards" / str(config.vocab_size) / "validation",
                                 tokenizer_sha256=tokenizer["sha256"], dataset_sha256=validation["sha256"])
    for documents, split in ((train_docs, "train"), (valid_docs, "validation")):
        if documents.metadata["dataset"].get("split") != split or documents.metadata["byte_lengths"] != byte_lengths:
            raise ValueError("Shard split or tokenizer byte table mismatch")
        if documents.metadata["vocab_size"] != config.vocab_size:
            raise ValueError("Shard vocabulary mismatch")
    selection = read_json(root / "selection.json")
    if config.data.tier == "smoke":
        count = selection["smoke_documents"]
        train_docs = DocumentPrefix(train_docs, count)
        identity = dict(identity, sha256=sha256((identity["sha256"] + f":prefix:{count}").encode()),
                        tier="smoke", documents=count)
    validation.update(protocol=asdict(config.evaluation), bytes=sum(d["raw_bytes"] for d in selection["validation"]))
    data = TrainingData(train_docs, context=config.training.context, stride=config.training.stride,
                        seed=config.training.seed, tokenizer=tokenizer, dataset=identity)
    return data, valid_docs, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cfg.yaml")
    parser.add_argument("--validate-only", action="store_true", help="Validate schema/device; no data or optimizer access")
    parser.add_argument("--resume", type=Path, help="Override checkpoint.resume")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a complete training checkpoint without updates")
    args = parser.parse_args(argv)
    if args.resume and args.evaluate:
        parser.error("Choose --resume or --evaluate")
    config = load_config(args.config)
    if args.validate_only:
        print("Configuration valid")
        return
    output = Path(config.checkpoint.output)
    resume_path = args.resume or config.checkpoint.resume
    if not args.evaluate and not resume_path and output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh training requires an empty output directory")
    data, documents, validation = load_data(config)
    payload = None
    if args.evaluate or resume_path:
        model, payload = load_training_checkpoint(args.evaluate or resume_path, config.device)
        if (model.config != config.model or model.vocabulary.specification() !=
                {"kind": "dense", "vocab_size": config.vocab_size, "hidden": config.model.hidden}):
            raise ValueError("Checkpoint model differs from configured dense model")
        if payload["dataset"] != validation or payload["tokenizer"] != data.identity["tokenizer"]:
            raise ValueError("Checkpoint dataset/tokenizer differs from configured data")
    else:
        seed_all(config.training.seed)
        model = CausalTransformer(config.model, config.vocab_size).to(config.device)
    if args.evaluate:
        print(json.dumps(evaluate(model, documents, config.evaluation, expected_bytes=validation["bytes"]), indent=2))
        return
    state = train(model, data, config.training, validation_documents=documents,
                  validation_identity=validation, output=output,
                  resume=payload["training_state"] if payload else None)
    result = evaluate(model, documents, config.evaluation, expected_bytes=validation["bytes"])
    print(json.dumps({"validation": result, "counters": state["counters"],
                     "storage": storage_report(model, tokenizer_paths=[Path(config.data.tokenizer) / name
                         for name in ("base.json", "expanded.json", "nested.json")],
                         optimizer_state=state["optimizer"], tokenizer=data.identity["tokenizer"], dataset=validation)}, indent=2))


if __name__ == "__main__":
    main()

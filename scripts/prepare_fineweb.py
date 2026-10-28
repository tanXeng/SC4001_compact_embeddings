"""FineWeb preparation stages; verification is read-only unless --publish."""
import argparse
from pathlib import Path
import json

import yaml

from vocab_lm.fineweb import inspect_source, download_source
from vocab_lm.fineweb import prepare_raw, prepare_tokenizers, prepare_shards, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fineweb.yaml")
    parser.add_argument("--stage", required=True, choices=("source", "download", "raw", "tokenizers", "shards", "verify"))
    parser.add_argument("--root", type=Path, help="Override artifact directory")
    parser.add_argument("--publish", action="store_true", help="Seal a new unsealed generation")
    parser.add_argument("--allow-code-provenance-mismatch", action="store_true",
                        help="Explicit read-only compatibility replay; artifact checks remain strict")
    args = parser.parse_args()
    if (args.publish or args.allow_code_provenance_mismatch) and args.stage != "verify":
        parser.error("Verification options require --stage verify")
    if args.publish and args.allow_code_provenance_mismatch:
        parser.error("Compatibility replay cannot publish")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = args.root or Path(config["root"])
    if (config["source"]["dataset"] != "HuggingFaceFW/fineweb"
            or not 0 < config["selection"]["pilot_bytes"] <= 200_000_000
            or not 1_000_000 <= config["selection"]["smoke_bytes"] <= 5_000_000
            or config["selection"]["validation_bytes"] != 10_000_000
            or (config["tokenizer"]["base_vocab_size"], config["tokenizer"]["expanded_vocab_size"]) != (8192, 16384)):
        raise PermissionError("Configuration exceeds Gate 4A scope")
    stages = {"source": inspect_source, "download": download_source, "raw": prepare_raw,
              "tokenizers": prepare_tokenizers, "shards": prepare_shards, "verify": verify}
    print(f"FineWeb {args.stage}", flush=True)
    if args.stage == "verify":
        result = verify(root, config, publish=args.publish,
                        allow_code_provenance_mismatch=args.allow_code_provenance_mismatch)
        print(json.dumps({k: result[k] for k in ("coverage", "evaluation", "code_provenance")}, indent=2))
    else:
        result = stages[args.stage](root, config)
    if args.stage == "source":
        print(f"Network estimate: {result['network_estimate_bytes']} bytes; disk estimate: {result['disk_estimate_bytes']} bytes")
    print(f"Completed {args.stage}", flush=True)


if __name__ == "__main__":
    main()

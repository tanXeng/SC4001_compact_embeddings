from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from vocab_lm.tokenizer import train_nested


@pytest.fixture(autouse=True)
def seeds():
    config = yaml.safe_load(Path("configs/testing.yaml").read_text(encoding="utf-8"))["tokenizer_tests"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # Tiny mathematical operations should not start dozens of CPU workers.
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def corpus():
    text = (
        " = Article one = \nCafé naïve 東京 🦊: banana banana abracadabra!\n\n"
        " = = Subsection = = \nNumbers 0123456789; English vocabulary and repeated characters aaaaaaaa.\n"
        " = Article two = \nWhitespace\t  preserved.\r\nEnd of document.\n"
    )
    return [text.encode("utf-8"), (text[::-1] + "\n").encode("utf-8")]


@pytest.fixture(scope="session")
def pair(corpus):
    config = yaml.safe_load(Path("configs/testing.yaml").read_text(encoding="utf-8"))["tokenizer_tests"]
    return train_nested(corpus, **config["tokenizer"])

import re
from decimal import Decimal
from typing import List, Tuple

from datasets import load_dataset

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def normalize_number_str(s: str) -> str:
    s = s.replace(",", "").strip()
    try:
        d = Decimal(s)
        q = d.normalize()
        s2 = format(q, "f")
        if "." in s2:
            s2 = s2.rstrip("0").rstrip(".")
        return s2
    except Exception:
        return s


def preprocess_answer(ans: str, normalize_numbers: bool, extract_first_number: bool) -> str:
    out = ans
    if extract_first_number:
        nums = extract_numbers(out)
        if nums:
            out = nums[0]
    if normalize_numbers:
        out = normalize_number_str(out)
    return out


def load_dataset_split(dataset_cfg, split: str, limit: int = None) -> List[Tuple[str, str]]:
    ds = load_dataset(
        dataset_cfg.name,
        dataset_cfg.subset,
        split=split,
        cache_dir=".cache/",
    )
    data: List[Tuple[str, str]] = []
    for ex in ds:
        if "Question" in ex:
            x = ex["Question"]
        elif "question" in ex:
            x = ex["question"]
        elif "Body" in ex:
            x = ex["Body"]
        elif "body" in ex:
            x = ex["body"]
        else:
            x = ex.get("input", ex.get("prompt", ""))
        if "Answer" in ex:
            g = str(ex["Answer"])
        elif "answer" in ex:
            g = str(ex["answer"])
        elif "label" in ex:
            g = str(ex["label"])
        elif "output" in ex:
            g = str(ex["output"])
        else:
            g = str(ex.get("target", ""))
        g = preprocess_answer(g, dataset_cfg.preprocessing.normalize_numbers, dataset_cfg.preprocessing.extract_first_number)
        data.append((x, g))
        if limit is not None and len(data) >= limit:
            break
    return data

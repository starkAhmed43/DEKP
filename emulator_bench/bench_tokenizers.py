import json
import regex as re
from pathlib import Path
from typing import Dict, Iterable, List


SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class ProteinSequenceTokenizer:
    def __init__(self) -> None:
        tokens = ["<pad>", "<unk>", "<bos>", "<eos>"] + list("ACDEFGHIKLMNPQRSTVWYX")
        self.stoi: Dict[str, int] = {token: idx for idx, token in enumerate(tokens)}
        self.itos: List[str] = list(tokens)
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, sequence: str, max_length: int | None = None, add_special_tokens: bool = True) -> List[int]:
        tokens = [self.stoi.get(char, self.unk_id) for char in str(sequence)]
        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"itos": self.itos}, handle, indent=2)


class RegexSmilesTokenizer:
    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.regex_tokenizer = re.compile(SMILES_PATTERN)
        self.stoi = stoi
        self.itos = itos
        self.pad_id = stoi["<pad>"]
        self.unk_id = stoi["<unk>"]
        self.bos_id = stoi["<bos>"]
        self.eos_id = stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, smiles: str) -> List[str]:
        return self.regex_tokenizer.findall(str(smiles).strip())

    def encode(self, smiles: str, max_length: int | None = None, add_special_tokens: bool = True) -> List[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in self.tokenize(smiles)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"itos": self.itos}, handle, indent=2)

    @classmethod
    def from_smiles(cls, smiles_values: Iterable[str], min_freq: int = 1) -> "RegexSmilesTokenizer":
        regex_tokenizer = re.compile(SMILES_PATTERN)
        counts: Dict[str, int] = {}
        for smiles in smiles_values:
            for token in regex_tokenizer.findall(str(smiles).strip()):
                counts[token] = counts.get(token, 0) + 1
        itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(
            [token for token, count in counts.items() if count >= min_freq]
        )
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def load(cls, path: Path) -> "RegexSmilesTokenizer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        itos = list(payload["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


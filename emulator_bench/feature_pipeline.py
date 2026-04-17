import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from torch import nn
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from transformers import T5EncoderModel, T5Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import normalize_sequence


RDLogger.DisableLog("rdApp.warning")

DEFAULT_TRFM_WEIGHTS = (REPO_ROOT.parent / "KcatNet" / "utils" / "trfm_12_23000.pkl")
DEFAULT_TRFM_VOCAB = REPO_ROOT / "DEKP" / "vocab.pkl"


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in {"TorchVocab", "Vocab", "WordVocab"}:
            return globals()[name]
        return super().find_class(module, name)


class TorchVocab:
    def __init__(self, counter, max_size=None, min_freq=1, specials=None):
        specials = specials or ["<pad>", "<oov>"]
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list(specials)
        for tok in specials:
            counter.pop(tok, None)

        max_size = None if max_size is None else max_size + len(self.itos)
        words_and_frequencies = sorted(counter.items(), key=lambda item: item[0])
        words_and_frequencies.sort(key=lambda item: item[1], reverse=True)
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)


class WordVocab(Vocab):
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as handle:
            return _CompatUnpickler(handle).load()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_size,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def _encode(self, src):
        embedded = self.embed(src)
        embedded = self.pe(embedded)
        output = embedded
        for idx in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[idx](output, None)
        penultimate = output.detach().cpu().numpy()
        output = self.trfm.encoder.layers[-1](output, None)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)
        output = output.detach().cpu().numpy()
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penultimate[0, :, :]])

    def encode(self, src):
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        outputs = [self._encode(src[:, :100])]
        for start in range(100, batch_size, 100):
            outputs.append(self._encode(src[:, start : start + 100]))
        return np.concatenate(outputs, axis=0)


def split_smiles_for_trfm(smiles: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|Si|Se|Na|Li|Mg|Ca|Zn|Fe|Cu|Mn|Al|As|Ag|Au|Ni|Rb|Ra|Xe|Sr|Ba|Bi|Be|Te|He|\%\d{2}|.)"
    tokens = [token for token in re_findall(pattern, str(smiles).strip()) if token]
    return " ".join(tokens)


def re_findall(pattern: str, text: str) -> List[str]:
    import regex as re

    return re.findall(pattern, text)


def get_trfm_inputs(smiles_values: Sequence[str], vocab: WordVocab, seq_len: int = 220) -> Tuple[torch.Tensor, torch.Tensor]:
    x_id, x_seg = [], []
    for smiles in smiles_values:
        tokens = split_smiles_for_trfm(smiles).split()
        if len(tokens) > seq_len - 2:
            half = (seq_len - 2) // 2
            tokens = tokens[:half] + tokens[-(seq_len - 2 - half) :]
        ids = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
        ids = [vocab.sos_index] + ids + [vocab.eos_index]
        seg = [1] * len(ids)
        padding = [vocab.pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        x_id.append(ids)
        x_seg.append(seg)
    return torch.tensor(x_id, dtype=torch.long), torch.tensor(x_seg, dtype=torch.long)


def load_smiles_transformer(weights_path: str | Path | None = None, vocab_path: str | Path | None = None, device: torch.device | None = None):
    weights_path = Path(weights_path or DEFAULT_TRFM_WEIGHTS)
    vocab_path = Path(vocab_path or DEFAULT_TRFM_VOCAB)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing SMILES Transformer weights: {weights_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing SMILES Transformer vocab: {vocab_path}")
    device = device or torch.device("cpu")
    vocab = WordVocab.load_vocab(str(vocab_path))
    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, vocab


def embed_smiles_trfm_batch(smiles_values: Sequence[str], model: TrfmSeq2seq, vocab: WordVocab, device: torch.device) -> Dict[str, np.ndarray]:
    xid, _ = get_trfm_inputs(smiles_values, vocab=vocab)
    xid = torch.t(xid).to(device)
    with torch.no_grad():
        embeddings = model.encode(xid)
    return {
        smiles: np.asarray(embeddings[idx], dtype=np.float32)
        for idx, smiles in enumerate(smiles_values)
    }


def load_prot_t5(model_name_or_path: str, device: torch.device):
    model = T5EncoderModel.from_pretrained(model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def build_prot_t5_batches(sequences: Sequence[str], max_residues: int = 4000, max_seq_len: int = 2500, max_batch: int = 16) -> List[List[str]]:
    ordered = sorted(sequences, key=len, reverse=True)
    batches: List[List[str]] = []
    batch: List[str] = []
    batch_residues = 0
    for sequence in ordered:
        sequence = normalize_sequence(sequence, max_len=max_seq_len)
        seq_len = len(sequence)
        if batch and (len(batch) >= max_batch or batch_residues + seq_len > max_residues):
            batches.append(batch)
            batch = []
            batch_residues = 0
        batch.append(sequence)
        batch_residues += seq_len
    if batch:
        batches.append(batch)
    return batches


def embed_prot_t5_batch(model, tokenizer, sequences: Sequence[str], device: torch.device) -> Dict[str, np.ndarray]:
    tokenized = [" ".join(list(normalize_sequence(seq))) for seq in sequences]
    encoding = tokenizer.batch_encode_plus(tokenized, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(encoding["input_ids"], device=device)
    attention_mask = torch.tensor(encoding["attention_mask"], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    embedded = {}
    for idx, sequence in enumerate(sequences):
        seq_len = len(normalize_sequence(sequence))
        array = outputs[idx, :seq_len].detach().cpu().numpy()
        embedded[sequence] = np.asarray(array.mean(axis=0), dtype=np.float32)
    return embedded


def load_pickle(path: Path) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_legacy_feature_pickles(feature_dir: Path, feature_names: Iterable[str]) -> Dict[str, Dict]:
    feature_dir = Path(feature_dir)
    payload = {}
    for name in feature_names:
        path = feature_dir / f"{name}.pkl"
        if path.exists():
            payload[name] = load_pickle(path)
    graph_path = feature_dir / "pyg_graph.pkl"
    if graph_path.exists():
        payload["graph"] = load_pickle(graph_path)
    return payload


def resolve_legacy_feature(mapping: Dict, key, fallback_key=None):
    if mapping is None:
        return None
    if key in mapping:
        return mapping[key]
    if fallback_key is not None and fallback_key in mapping:
        return mapping[fallback_key]
    return None


def _vectorize_atom(atom):
    atom_type = "CNOS$"
    aa_type = "ACDEFGHIKLMNPQRSTVWY$"
    atom_feature = np.zeros(len(atom_type) + len(aa_type), dtype=np.float32)
    symbol = str(atom.GetSymbol())
    symbol_idx = atom_type.find(symbol if symbol in atom_type else "$")
    atom_feature[max(0, symbol_idx)] = 1.0
    atom_feature[len(atom_type) + len(aa_type) - 1] = 1.0
    return atom_feature


def smiles_to_atom_features(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    atom_indices = []
    atom_features = []
    for atom in molecule.GetAtoms():
        atom_indices.append(atom.GetAtomicNum())
        atom_features.append(_vectorize_atom(atom))
    if not atom_indices:
        atom_indices = [0]
        atom_features = [np.zeros(27, dtype=np.float32)]
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    atom_features = np.asarray(atom_features, dtype=np.float32)
    return atom_indices, atom_features


def load_structure_graph_builder():
    from DEKP.Encode.extract_pdb_feature import get_geo_feat, parse_pdb

    return parse_pdb, get_geo_feat


def parse_pdb_to_array(pdb_path: Path, nneighbor: int = 32, atom_type: str = "CA") -> np.ndarray:
    parse_pdb, get_geo_feat = load_structure_graph_builder()
    try:
        pdb_array = parse_pdb(str(pdb_path), atom_type=atom_type, nneighbor=nneighbor, cal_cb=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to build graph from PDB `{pdb_path}`") from exc
    return pdb_array


def build_graph_from_array(pdb_array: np.ndarray, nneighbor: int = 32, device: str | torch.device = "cpu") -> Data:
    _, get_geo_feat = load_structure_graph_builder()
    graph_device = torch.device(device)
    x = torch.tensor(pdb_array, dtype=torch.float32, device=graph_device)
    ca_pos = x[:, 1]
    edge_index = radius_graph(ca_pos, r=15.0, loop=False, max_num_neighbors=nneighbor)
    node_feature, edge_feature = get_geo_feat(x, edge_index)
    return Data(
        x=node_feature.float().cpu(),
        edge_index=edge_index.long().cpu(),
        edge_attr=edge_feature.float().cpu(),
    )


def build_graph_from_pdb(pdb_path: Path, nneighbor: int = 32, atom_type: str = "CA", device: str | torch.device = "cpu") -> Data:
    pdb_array = parse_pdb_to_array(pdb_path, nneighbor=nneighbor, atom_type=atom_type)
    return build_graph_from_array(pdb_array, nneighbor=nneighbor, device=device)


def compute_dssp_feature(pdb_path: Path, dssp_executable: str | None = None) -> np.ndarray:
    from Bio.PDB import DSSP, PDBParser

    max_asa = {
        "G": 188,
        "A": 198,
        "V": 220,
        "I": 233,
        "L": 304,
        "F": 272,
        "P": 203,
        "M": 262,
        "W": 317,
        "C": 201,
        "S": 234,
        "T": 215,
        "N": 254,
        "Q": 259,
        "Y": 304,
        "H": 258,
        "D": 236,
        "E": 262,
        "K": 317,
        "R": 319,
    }
    ss_type = "HBEGITS-"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))
    model = structure[0]
    kwargs = {}
    if dssp_executable:
        kwargs["dssp"] = dssp_executable
    dssp = DSSP(model, str(pdb_path), **kwargs)
    features = []
    for record in dssp.property_list:
        ss_vec = np.zeros(8, dtype=np.float32)
        ss = record[2]
        ss_idx = ss_type.find(ss)
        ss_vec[ss_idx if ss_idx >= 0 else -1] = 1.0
        phi = float(record[4])
        psi = float(record[5])
        asa = float(record[3])
        aa_name = record[1]
        radian = np.array([phi, psi], dtype=np.float32) * (np.pi / 180.0)
        asa = min(float(asa) / max_asa.get(aa_name, 1.0), 1.0)
        feature = np.concatenate([np.sin(radian), np.cos(radian), np.asarray([asa], dtype=np.float32), ss_vec])
        features.append(feature)
    if not features:
        raise ValueError(f"DSSP returned no residues for {pdb_path}")
    return np.asarray(features, dtype=np.float32).mean(axis=0)

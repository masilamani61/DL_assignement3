from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


@dataclass
class Vocab:
    stoi: dict
    itos: list

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]


def _load_spacy_tokenizer(model_name: str, lang_code: str) -> Callable[[str], List[str]]:
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank(lang_code)
    return lambda text: [token.text.lower() for token in nlp(text.strip()) if token.text.strip()]


class TranslationDataset(Dataset):
    def __init__(self, src_sequences: Sequence[Sequence[int]], tgt_sequences: Sequence[Sequence[int]]) -> None:
        self.src_sequences = list(src_sequences)
        self.tgt_sequences = list(tgt_sequences)

    def __len__(self) -> int:
        return len(self.src_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_sequences[idx], dtype=torch.long),
            torch.tensor(self.tgt_sequences[idx], dtype=torch.long),
        )


class Multi30kDataset:
    def __init__(self, split: str = "train"):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        self.train_dataset = load_dataset("bentrevett/multi30k", split="train")
        self.src_tokenizer = _load_spacy_tokenizer("de_core_news_sm", "de")
        self.tgt_tokenizer = _load_spacy_tokenizer("en_core_web_sm", "en")
        self.src_vocab = None
        self.tgt_vocab = None

    def _extract_pair(self, sample) -> Tuple[str, str]:
        if "translation" in sample:
            pair = sample["translation"]
            return pair["de"], pair["en"]
        return sample["de"], sample["en"]

    def build_vocab(self, min_freq: int = 3):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for sample in self.train_dataset:
            src_text, tgt_text = self._extract_pair(sample)
            src_counter.update(self.src_tokenizer(src_text))
            tgt_counter.update(self.tgt_tokenizer(tgt_text))

        def make_vocab(counter: Counter) -> Vocab:
            tokens = SPECIAL_TOKENS.copy()
            for token, freq in counter.items():
                if freq >= min_freq and token not in tokens:
                    tokens.append(token)
            stoi = {token: idx for idx, token in enumerate(tokens)}
            return Vocab(stoi=stoi, itos=tokens)

        self.src_vocab = make_vocab(src_counter)
        self.tgt_vocab = make_vocab(tgt_counter)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab()

        src_data = []
        tgt_data = []
        for sample in self.dataset:
            src_text, tgt_text = self._extract_pair(sample)
            src_tokens = ["<sos>"] + self.src_tokenizer(src_text) + ["<eos>"]
            tgt_tokens = ["<sos>"] + self.tgt_tokenizer(tgt_text) + ["<eos>"]
            src_data.append([self.src_vocab[token] for token in src_tokens])
            tgt_data.append([self.tgt_vocab[token] for token in tgt_tokens])
        return TranslationDataset(src_data, tgt_data)


def build_collate_fn(pad_idx: int):
    def collate_fn(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded

    return collate_fn


def get_dataloaders(batch_size: int = 32, num_workers: int = 0):
    train_builder = Multi30kDataset(split="train")
    src_vocab, tgt_vocab = train_builder.build_vocab()

    val_builder = Multi30kDataset(split="validation")
    test_builder = Multi30kDataset(split="test")
    val_builder.src_vocab = test_builder.src_vocab = src_vocab
    val_builder.tgt_vocab = test_builder.tgt_vocab = tgt_vocab

    train_dataset = train_builder.process_data()
    val_dataset = val_builder.process_data()
    test_dataset = test_builder.process_data()

    collate_fn = build_collate_fn(pad_idx=src_vocab["<pad>"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    assets = {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_tokenizer": train_builder.src_tokenizer,
        "tgt_tokenizer": train_builder.tgt_tokenizer,
    }
    return train_loader, val_loader, test_loader, assets

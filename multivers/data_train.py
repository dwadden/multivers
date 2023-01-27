"""
Module to handle training data.

If you're just doing inference, look at `data.py` instead of this file.
"""

import os
import pathlib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
import random

from pytorch_lightning import LightningDataModule

from data_verisci import GoldDataset
from transformers import AutoTokenizer, BatchEncoding

import util


def get_tokenizer(hparams):
    "If using Arman's scilongformer checkpoint, need to add some tokens."
    # If we're not using the science model, just make the normal tokenizer.
    if hparams.encoder_name != "longformer-large-science":
        tokenizer = AutoTokenizer.from_pretrained(hparams.encoder_name)
        return tokenizer

    # Otherwise, add some extra tokens.
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")
    ADDITIONAL_TOKENS = {
        "section_start": "<|sec|>",
        "section_end": "</|sec|>",
        "section_title_start": "<|sec-title|>",
        "section_title_end": "</|sec-title|>",
        "abstract_start": "<|abs|>",
        "abstract_end": "</|abs|>",
        "title_start": "<|title|>",
        "title_end": "</|title|>",
        "sentence_sep": "<|sent|>",
        "paragraph_sep": "<|par|>",
    }
    tokenizer.add_tokens(list(ADDITIONAL_TOKENS.values()))
    return tokenizer


####################


class SciFactCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        "Collate all the data together into padded batch tensors."
        # NOTE(dwadden) Set missing values to 0 for `abstract_sent_idx` instead
        # of -1 because it's going to be used as an input to
        # `batched_index_select` later on in the modeling code.
        res = {
            "claim_id": self._collate_scalar(batch, "claim_id"),
            "abstract_id": self._collate_scalar(batch, "abstract_id"),
            "negative_sample_id": self._collate_scalar(batch, "negative_sample_id"),
            "dataset": [x["dataset"] for x in batch],
            "tokenized": self._pad_tokenized([x["tokenized"] for x in batch]),
            "abstract_sent_idx": self._pad_field(batch, "abstract_sent_idx", 0),
            "label": self._collate_scalar(batch, "label"),
            "rationale": self._pad_field(batch, "rationale", -1),
            "rationale_sets": self._pad_field(batch, "rationale_sets", -1),
            "weight": self._collate_scalar(batch, "weight"),
            "rationale_mask": self._collate_scalar(batch, "rationale_mask"),
        }
        # Make sure the keys match.
        assert res.keys() == batch[0].keys()
        return res

    @staticmethod
    def _collate_scalar(batch, field):
        "Collate scalars by concatting."
        return torch.tensor([x[field] for x in batch])

    def _pad_tokenized(self, tokenized):
        """
        Pad the tokenizer outputs. Need to do this manually because the
        tokenizer's default padder doesn't expect `global_attention_mask` as an
        input.
        """
        fields = ["input_ids", "attention_mask", "global_attention_mask"]
        pad_values = [self.tokenizer.pad_token_id, 0, 0]
        tokenized_padded = {}
        for field, pad_value in zip(fields, pad_values):
            tokenized_padded[field] = self._pad_field(tokenized, field, pad_value)

        return tokenized_padded

    def _pad_field(self, entries, field_name, pad_value):
        xxs = [entry[field_name] for entry in entries]
        return self._pad(xxs, pad_value)

    @staticmethod
    def _pad(xxs, pad_value):
        """
        Pad a list of lists to the length of the longest entry, using the given
        `pad_value`.
        """
        res = []
        max_length = max(map(len, xxs))
        for entry in xxs:
            to_append = [pad_value] * (max_length - len(entry))
            padded = entry + to_append
            res.append(padded)

        return torch.tensor(res)


####################


class SciFactDataset(Dataset):
    "Stores and tensorizes a list of claim / document entries."

    def __init__(self, entries, tokenizer, dataset_name, rationale_mask):
        self.entries = entries
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.rationale_mask = rationale_mask
        self.label_lookup = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        "Tensorize a single claim / abstract pair."
        entry = self.entries[idx]
        res = {
            "claim_id": entry["claim_id"],
            "abstract_id": entry["abstract_id"],
            "negative_sample_id": entry["negative_sample_id"],
            "dataset": self.dataset_name,
            "weight": entry["weight"],
            "rationale_mask": self.rationale_mask,
        }
        tensorized = self._tensorize(**entry["to_tensorize"])
        res.update(tensorized)
        return res

    def _tensorize(self, claim, sentences, label, rationales, title=None):
        """
        This function does the meat of the preprocessing work. The arguments
        should be self-explanatory, except `title`. We have abstract titles for
        SciFact, but not FEVER.
        """
        if "roberta" in self.tokenizer.name_or_path:
            tokenized, abstract_sent_idx = self._tokenize_roberta(
                claim, sentences, title
            )
        else:
            tokenized, abstract_sent_idx = self._tokenize_longformer(
                claim, sentences, title
            )

        # Get the label and the rationales.
        label_code = self.label_lookup[label]
        if label_code != self.label_lookup["NOT ENOUGH INFO"]:
            # If it's an evidence document, get the label and create an
            # evidence vector for the sentences. Each evidence set gets
            # its own digit, starting from 1.
            rationale_sets = torch.zeros(len(sentences), dtype=torch.int64)
            rationale_id = 1
            for this_rationale in rationales:
                rationale_sets[torch.tensor(this_rationale)] = rationale_id
                rationale_id += 1
        else:
            # If it's not an evidence document, the label is `NEI` and
            # none of the sentences are evidence.
            rationale_sets = torch.zeros(len(sentences), dtype=torch.int64)

        rationale = (rationale_sets > 0).int()
        # `rationale_sets` has a separate int ID for each rationale.
        rationale_sets = rationale_sets.tolist()
        # `rationale` is a 0 / 1 indicator for whether it's a rationale.
        rationale = rationale.tolist()

        return {
            "tokenized": tokenized,
            "abstract_sent_idx": abstract_sent_idx,
            "label": label_code,
            "rationale": rationale,
            "rationale_sets": rationale_sets,
        }

    def _tokenize_longformer(self, claim, sentences, title):
        cited_text = self.tokenizer.eos_token.join(sentences)
        if title is not None:
            cited_text = title + self.tokenizer.eos_token + cited_text
        tokenized = self.tokenizer(claim + self.tokenizer.eos_token + cited_text)
        tokenized["global_attention_mask"] = self._get_global_attention_mask(tokenized)
        abstract_sent_idx = self._get_abstract_sent_tokens(tokenized, title)

        # Make sure we've got the right number of abstract sentence tokens.
        assert len(abstract_sent_idx) == len(sentences)

        return tokenized, abstract_sent_idx

    def _tokenize_roberta(self, claim, sentences, title):
        "If we're using RoBERTa, we need to truncate the sentences to fit in window."

        def replace_bos_with_eos(sent):
            res = [
                word
                if word != self.tokenizer.bos_token_id
                else self.tokenizer.eos_token_id
                for word in sent
            ]
            return res

        # Claim and title aren't truncated.
        if title is not None:
            claim_and_title = claim + self.tokenizer.eos_token + title
        else:
            claim_and_title = claim

        # Strip off the trailing eos; will get it back when we concatenate the abstract.
        claim_and_title_tok = self.tokenizer(claim_and_title)["input_ids"][:-1]
        claim_and_title_len = len(claim_and_title_tok)

        # Need to subtract 1 for the final trailing EOS token.
        abstract_len = self.tokenizer.model_max_length - claim_and_title_len - 1

        sents = [self.tokenizer(sent)["input_ids"] for sent in sentences]
        sents = [replace_bos_with_eos(sent) for sent in sents]
        # Strip off the final eos so we don't duplicate.
        sents = [sent[:-1] for sent in sents]

        sent_lens = [len(sent) for sent in sents]
        length_so_far = sum(sent_lens)
        while length_so_far > abstract_len:
            longest = np.argmax(sent_lens)
            sents[longest] = sents[longest][:-1]
            sent_lens[longest] -= 1
            length_so_far = sum(sent_lens)

        sents_flat = util.flatten(sents)

        input_ids = claim_and_title_tok + sents_flat + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.tokenizer.model_max_length:
            raise Exception("Length is wrong.")

        tokenized = BatchEncoding(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        )
        # We don't use this, but setting it keeps the input consistent.
        tokenized["global_attention_mask"] = self._get_global_attention_mask(tokenized)
        abstract_sent_idx = self._get_abstract_sent_tokens(tokenized, title)

        # Make sure we've got the right number of abstract sentence tokens.
        assert len(abstract_sent_idx) == len(sentences)

        return tokenized, abstract_sent_idx

    def _get_global_attention_mask(self, tokenized):
        "Assign global attention to all special tokens and to the claim."
        input_ids = torch.tensor(tokenized.input_ids)
        # Get all the special tokens.
        is_special = (input_ids == self.tokenizer.bos_token_id) | (
            input_ids == self.tokenizer.eos_token_id
        )
        # Get all the claim tokens (everything before the first </s>).
        first_eos = torch.where(input_ids == self.tokenizer.eos_token_id)[0][0]
        is_claim = torch.arange(len(input_ids)) < first_eos
        # Use global attention if special token, or part of claim.
        global_attention_mask = is_special | is_claim
        # Unsqueeze to put in batch form, and cast like the tokenizer attention mask.
        global_attention_mask = global_attention_mask.to(torch.int64)
        return global_attention_mask.tolist()

    def _get_abstract_sent_tokens(self, tokenized, title):
        "Get the indices of the </s> tokens representing each abstract sentence."
        is_eos = torch.tensor(tokenized["input_ids"]) == self.tokenizer.eos_token_id
        eos_idx = torch.where(is_eos)[0]
        # If there's a title, the first two </s> tokens are for the claim /
        # abstract separator and the title. Keep the rest.
        # If no title, keep all but the first.
        start_ix = 1 if title is None else 2
        return eos_idx[start_ix:].tolist()


####################


class FactCheckingReader:
    def __init__(self, debug=False):
        self.debug = debug
        self.data_root = (
            pathlib.Path(os.path.realpath(__file__)).parent.parent / "data_train"
        )


class SciFactReader(FactCheckingReader):
    """
    Class to handle SciFact data. Not used directly; its subclasses handle cases with
    different numbers of negative samples.
    """

    def __init__(self, fewshot, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SciFact"
        self.rationale_mask = 1.0
        self.fewshot = fewshot

    def get_fold(self, fold, tokenizer):
        """
        Load in the dataset as a list of entries. Each entry is a single claim /
        cited document pair. Some cited documents have no evidence.
        """
        train_lookup = "train" if not self.fewshot else "fewshot"
        lookup = {"train": train_lookup, "val": "dev", "test": "test"}
        fold_name = lookup[fold]

        res = []
        # Get the data from the shuffled directory for training.
        claims_dir = self.data_dir
        corpus_file = self.data_dir / "corpus.jsonl"
        data_file = claims_dir / f"claims_{fold_name}.jsonl"
        ds = GoldDataset(corpus_file, data_file)

        for i, claim in enumerate(ds.claims):
            # Only read 10 if we're doing a fast dev run.
            if self.debug and i == 10:
                break
            # NOTE(dwadden) This is a hack because claim 1245 in the dev set
            # lists document 7662395 twice. Need to fix the dataset. For now,
            # I'll just do this check.
            seen = set()
            for cited_doc in claim.cited_docs:
                if cited_doc.id in seen:
                    # If we've seen it already, skip.
                    continue
                else:
                    seen.add(cited_doc.id)
                # Convert claim and evidence into form for function input.
                if cited_doc.id in claim.evidence:
                    ev = claim.evidence[cited_doc.id]
                    label = ev.label.name
                    rationales = ev.rationales
                else:
                    label = "NOT ENOUGH INFO"
                    rationales = []

                # Append entry.
                to_tensorize = {
                    "claim": claim.claim,
                    "sentences": cited_doc.sentences,
                    "label": label,
                    "rationales": rationales,
                    "title": cited_doc.title,
                }
                entry = {
                    "claim_id": claim.id,
                    "abstract_id": cited_doc.id,
                    "negative_sample_id": 0,  # No negative sampling for SciFact yet.
                    "to_tensorize": to_tensorize,
                }
                res.append(entry)

        return SciFactDataset(res, tokenizer, self.name, self.rationale_mask)


class SciFact10Reader(SciFactReader):
    "SciFact train data with 10 negative samples per positive."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "target/scifact_10"


class SciFact20Reader(SciFactReader):
    "SciFact train data with 20 negative samples per positive."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "target/scifact_20"


class HealthVerReader(SciFactReader):
    """
    HealthVer is formatted the same as SciFact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "target/healthver"
        self.name = "HealthVer"


class CovidFactReader(SciFactReader):
    """
    CovidFact is formatted the same as SciFact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "target/covidfact"
        self.name = "CovidFact"


class ExternalReader(FactCheckingReader):
    def get_fold(self, fold, tokenizer):
        """
        Load in the dataset as a list of entries. Each entry is a single claim /
        cited document pair. Some cited documents have no evidence.
        """
        lookup = {"train": "train", "val": "dev"}
        fold_name = lookup[fold]

        res = []
        if isinstance(self, PubMedQAReader):
            data_file = f"{self.data_dir}/{fold_name}.jsonl"
        else:
            data_file = f"{self.data_dir}/{fold_name}.jsonl"
        data = util.load_jsonl(data_file)

        for i, instance in enumerate(data):
            # Only read 10 if we're doing a fast dev run.
            if self.debug and i == 10:
                break

            # Assemble the data.
            to_tensorize = {
                "claim": instance["claim"],
                "sentences": instance["sentences"],
                "label": instance["label"],
                "rationales": instance["evidence_sets"],
                "title": None,
            }

            # Don't bother with FEVER abstract ID's; they're strings.
            if isinstance(self, FEVERReader):
                abstract_id = -1
            else:
                abstract_id = instance["abstract_id"]

            entry = {
                "claim_id": instance["id"],
                "abstract_id": abstract_id,
                "negative_sample_id": instance["negative_sample_id"],
                "to_tensorize": to_tensorize,
            }

            res.append(entry)

        return SciFactDataset(res, tokenizer, self.name, self.rationale_mask)


class FEVERReader(ExternalReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "pretrain/fever"
        self.name = "FEVER"
        self.rationale_mask = 1.0


class PubMedQAReader(ExternalReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "pretrain/pubmedqa"
        self.name = "PubMedQA"
        self.rationale_mask = 0.0


class EvidenceInferenceReader(ExternalReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_root / "pretrain/evidence_inference"
        self.name = "EvidenceInference"
        self.rationale_mask = 1.0


####################


class ConcatDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.tokenizer = get_tokenizer(hparams)
        self.num_workers = hparams.num_workers
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.collator = SciFactCollator(self.tokenizer)
        self.shuffle = not hparams.no_shuffle
        self.reweight_labels = not hparams.no_reweight_labels
        self.reweight_datasets = not hparams.no_reweight_datasets
        self.max_label_weight = hparams.max_label_weight
        self.max_dataset_weight = hparams.max_dataset_weight
        self.cap_fever_nsamples = hparams.cap_fever_nsamples
        self.debug = hparams.debug or hparams.fast_dev_run
        # And also for the `fewshot` flag.
        self.fewshot = getattr(hparams, "fewshot", False)

        # Get the readers.
        self.reader_lookup = {
            "scifact_20": SciFact20Reader,
            "scifact_10": SciFact10Reader,
            "healthver": HealthVerReader,
            "covidfact": CovidFactReader,
            "fever": FEVERReader,
            "pubmedqa": PubMedQAReader,
            "evidence_inference": EvidenceInferenceReader,
        }

        # Dataset re-weighting based on confidence.
        self.dataset_weights = {
            "SciFact": hparams.scifact_weight,
            "HealthVer": hparams.healthver_weight,
            "CovidFact": hparams.covidfact_weight,
            "FEVER": hparams.fever_weight,
            "PubMedQA": hparams.pubmedqa_weight,
            "EvidenceInference": hparams.evidence_inference_weight,
        }

        # Datasets with a test set
        self.datasets_with_test = ["scifact", "healthver", "covidfact"]

        self.dataset_names = hparams.datasets.split(",")
        for name in self.dataset_names:
            assert name in self.reader_lookup
        self.readers = []
        for name in self.dataset_names:
            # For SciFact, keep track of which subset to read.
            this_reader = self.reader_lookup[name]
            reader_args = dict(debug=self.debug)
            # If it's a SciFactReader or a subclass, need to pass in whether we're doing
            # fewshot.
            if issubclass(this_reader, SciFactReader):
                reader_args["fewshot"] = self.fewshot

            to_append = this_reader(**reader_args)

            self.readers.append(to_append)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        tokenizer: A Transformers tokenizer.
        num_workers: Number of workers for the dataloader.
        batch_size: The iterator batch size.
        data_dir: The data root directory. Defaults to env variable if not found.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=2)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--no_shuffle", action="store_true")
        parser.add_argument("--no_reweight_labels", action="store_true")
        parser.add_argument("--no_reweight_datasets", action="store_true")
        parser.add_argument("--max_label_weight", type=float, default=3.0)
        parser.add_argument("--max_dataset_weight", type=float, default=10.0)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--scifact_weight", type=float, default=1.0)
        parser.add_argument("--healthver_weight", type=float, default=1.0)
        parser.add_argument("--covidfact_weight", type=float, default=1.0)
        parser.add_argument("--fever_weight", type=float, default=1.0)
        parser.add_argument("--pubmedqa_weight", type=float, default=1.0)
        parser.add_argument("--evidence_inference_weight", type=float, default=1.0)
        parser.add_argument(
            "--cap_fever_nsamples",
            action="store_true",
            help="If given, make total # samples the same as the size of FEVER.",
        )
        parser.add_argument(
            "--fewshot",
            action="store_true",
            help=(
                "If given, use `claims_fewshot` as train file. "
                "Only applies to SciFactReader and subclasses."
            ),
        )

        return parser

    def setup(self, stage=None):
        # Not all datasets have test sets.
        if set(self.dataset_names) & set(self.datasets_with_test):
            test_fold = self._process_fold("test")
        else:
            test_fold = None

        self.folds = {
            "train": self._process_fold("train"),
            "val": self._process_fold("val"),
            "test": test_fold,
        }

    def _process_fold(self, fold):
        "Get the data from all the data readers."
        datasets = []
        for reader in self.readers:
            # Only subclasses of SciFactReader have a test set.
            if fold == "test" and not isinstance(reader, SciFactReader):
                continue
            datasets.append(reader.get_fold(fold, self.tokenizer))

        # Add instance weights.
        datasets = self._add_instance_weights(datasets)
        datasets = self._sample_instances(datasets, fold)

        return ConcatDataset(datasets)

    def train_dataloader(self):
        return self.get_dataloader("train", self.train_batch_size)

    def val_dataloader(self):
        return self.get_dataloader("val", self.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size)

    def get_dataloader(self, fold, batch_size):
        # Shuffle the train set if requested, but not dev and test.
        shuffle = self.shuffle if fold == "train" else False
        return DataLoader(
            self.folds[fold],
            num_workers=self.num_workers,
            batch_size=batch_size,
            collate_fn=self.collator,
            shuffle=shuffle,
            pin_memory=True,
        )

    def _add_instance_weights(self, datasets):
        "Add instance weights for label classes and datasets."
        if not self.reweight_datasets:
            dataset_weights = [1.0] * len(datasets)
        else:
            dataset_lengths = [len(x) for x in datasets]
            max_len = max(dataset_lengths)
            dataset_weights = [max_len / x for x in dataset_lengths]
            dataset_weights = [min(x, self.max_dataset_weight) for x in dataset_weights]

        for ds_weight_prelim, dataset in zip(dataset_weights, datasets):
            # Re-weight by our "prior" on dataset quality.
            ds_weight = ds_weight_prelim * self.dataset_weights[dataset.dataset_name]
            entries = dataset.entries
            # If we're not reweighting by label category, just give it the
            # weight of the dataset.
            if not self.reweight_labels:
                for entry in entries:
                    entry["weight"] = ds_weight
            # Otherwise, reweight by label frequency.
            else:
                # Re-weight so that supports and refutes are even, but don't
                # downweight the NEI's,
                labels = [x["to_tensorize"]["label"] for x in entries]
                label_counts = (
                    pd.Series(labels).value_counts().loc[["SUPPORTS", "REFUTES"]]
                )
                max_label_count = label_counts.max()
                label_weights = max_label_count / label_counts
                label_weights = np.minimum(label_weights, self.max_label_weight)
                label_weights["NOT ENOUGH INFO"] = 1.0
                # Loop over the entries and assign a weight.
                for entry in entries:
                    this_label = entry["to_tensorize"]["label"]
                    this_label_weight = label_weights[this_label]
                    entry["weight"] = ds_weight * this_label_weight

        return datasets

    def _sample_instances(self, datasets, fold):
        """
        If `cap_fever_nsamples` is True, cap PQA at 50K and have FEVER make up
        the rest.
        """
        # If not capping, just return. Also, don't cap for evaluation data.
        if not self.cap_fever_nsamples or (fold != "train"):
            return datasets

        # Otherwise, do some subsampling.
        original_lengths = {dataset.dataset_name: len(dataset) for dataset in datasets}
        if "FEVER" not in original_lengths:
            return datasets

        total_length = original_lengths["FEVER"]

        new_lengths = {k: v for k, v in original_lengths.items()}

        total_non_fever = sum([v for k, v in new_lengths.items() if k != "FEVER"])
        new_fever = total_length - total_non_fever
        new_lengths["FEVER"] = new_fever

        assert sum(new_lengths.values()) == total_length

        # Get new datasets by sampling the originals.
        for name, n_samples in new_lengths.items():
            this_dataset = [entry for entry in datasets if entry.dataset_name == name]
            assert len(this_dataset) == 1
            this_dataset = this_dataset[0]
            new_entries = random.sample(this_dataset.entries, n_samples)
            # Re-set the entries for the dataset
            this_dataset.entries = new_entries

        # Do a final length check.
        final_length = sum([len(x) for x in datasets])
        assert final_length == total_length

        return datasets

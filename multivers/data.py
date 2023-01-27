"""
Module to handle data for inference.
"""

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding
import torch
import numpy as np

import util


def get_tokenizer():
    "Need to add a few special tokens to the default longformer checkpoint."
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


class MultiVerSDataset(Dataset):
    "Stores and tensorizes a list of claim / document entries."

    def __init__(self, entries, tokenizer):
        self.entries = entries
        self.tokenizer = tokenizer
        self.label_lookup = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        "Tensorize a single claim / abstract pair."
        entry = self.entries[idx]
        res = {
            "claim_id": entry["claim_id"],
            "abstract_id": entry["abstract_id"],
        }
        tensorized = self._tensorize(**entry["to_tensorize"])
        res.update(tensorized)
        return res

    def _tensorize(self, claim, sentences, title=None):
        """
        This function does the meat of the preprocessing work. The arguments
        should be self-explanatory, except `title`. We have abstract titles for
        SciFact, but not FEVER.
        """
        if "roberta" in self.tokenizer.name_or_path:
            tokenized, abstract_sent_idx = self._tokenize_truncated(
                claim, sentences, title
            )
        else:
            tokenized, abstract_sent_idx = self._tokenize(
                claim, sentences, title
            )

        # Get the label and the rationales.
        return {
            "tokenized": tokenized,
            "abstract_sent_idx": abstract_sent_idx,
        }

    def _tokenize(self, claim, sentences, title):
        cited_text = self.tokenizer.eos_token.join(sentences)
        if title is not None:
            cited_text = title + self.tokenizer.eos_token + cited_text
        tokenized = self.tokenizer(claim + self.tokenizer.eos_token + cited_text)
        tokenized["global_attention_mask"] = self._get_global_attention_mask(tokenized)
        abstract_sent_idx = self._get_abstract_sent_tokens(tokenized, title)

        # Make sure we've got the right number of abstract sentence tokens.
        assert len(abstract_sent_idx) == len(sentences)

        return tokenized, abstract_sent_idx

    def _tokenize_truncated(self, claim, sentences, title):
        "If we're using RoBERTa, we need to truncate the sentences to fit in window."
        # Not used in final models, but may be helpful for people trying to train with
        # different encoders.

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


class MultiVerSReader:
    """
    Class to handle SciFact with retrieved documents.
    """
    def __init__(self, predict_args):
        self.data_file = predict_args.input_file
        self.corpus_file = predict_args.corpus_file
        # Basically, I used two different sets of labels. This was dumb, but
        # doing this mapping fixes it.
        self.label_map = {"SUPPORT": "SUPPORTS",
                          "CONTRADICT": "REFUTES"}

    def get_data(self, tokenizer):
        """
        Get the data for the relevant fold.
        """
        res = []

        corpus = util.load_jsonl(self.corpus_file)
        corpus_dict = {x["doc_id"]: x for x in corpus}
        claims = util.load_jsonl(self.data_file)

        for claim in claims:
            for doc_id in claim["doc_ids"]:
                candidate_doc = corpus_dict[doc_id]
                to_tensorize = {"claim": claim["claim"],
                                "sentences": candidate_doc["abstract"],
                                "title": candidate_doc["title"]}
                entry = {"claim_id": claim["id"],
                         "abstract_id": candidate_doc["doc_id"],
                         "to_tensorize": to_tensorize}
                res.append(entry)

        return MultiVerSDataset(res, tokenizer)


class Collator:
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
            "tokenized": self._pad_tokenized([x["tokenized"] for x in batch]),
            "abstract_sent_idx": self._pad_field(batch, "abstract_sent_idx", 0),
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


def get_dataloader(predict_args):
    "Main entry point to get the data loader. This can only be used at test time."
    reader = MultiVerSReader(predict_args)
    tokenizer = get_tokenizer()
    ds = reader.get_data(tokenizer)
    collator = Collator(tokenizer)
    return DataLoader(ds,
                      num_workers=predict_args.num_workers,
                      batch_size=predict_args.batch_size,
                      collate_fn=collator,
                      shuffle=False,
                      pin_memory=True)

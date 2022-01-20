from torch.utils.data import DataLoader

import data
from lib import util


# NOTE(DW): this only works for oracle retrievals. The reason is that it doesn't
# incorporate the gold data that the model failed to retrieve. As a result, the
# `metrics` won't properly evaluate when there's retrieval involved.


class SciFactRetrievedReader:
    """
    Class to handle SciFact with retrieved documents.
    """
    def __init__(self, predict_args):
        self.data_file = predict_args.input_file
        self.retrieval_file = predict_args.retrieval_file
        self.corpus_file = predict_args.scifact_corpus_file
        self.name = "SciFact-Retrieved"
        self.debug = predict_args.debug
        self.rationale_mask = 1.0
        # Basically, I used two different sets of labels. This was dumb, but
        # doing this mapping fixes it.
        self.label_map = {"SUPPORT": "SUPPORTS",
                          "CONTRADICT": "REFUTES"}

    def get_data(self, tokenizer):
        """
        Get the data for the relevant fold.
        """
        res = []
        # Get the data from the shuffled directory for training.
        data_file = self.data_file
        corpus_file = self.corpus_file

        corpus = util.load_jsonl(corpus_file)
        corpus_dict = {x["doc_id"]: x for x in corpus}
        claims = util.load_jsonl(data_file)
        retrievals = util.load_jsonl(self.retrieval_file)

        # Check that the claims and retrievals match up.
        claim_ids = [claim["id"] for claim in claims]
        retrieval_ids = [retrieval["claim_id"] for retrieval in retrievals]
        assert claim_ids == retrieval_ids

        for i, (claim, retrieved) in enumerate(zip(claims, retrievals)):
            # Only read 10 if we're doing a fast dev run.
            if self.debug and i == 10:
                break
            for doc_id in retrieved["doc_ids"]:
                retrieved_doc = corpus_dict[doc_id]
                if str(doc_id) in claim["evidence"]:
                    ev = claim["evidence"][str(doc_id)]
                    label = self.label_map[ev[0]["label"]]  # Need to re-map labels.
                    rationales = [entry["sentences"] for entry in ev]
                else:
                    label = "NOT ENOUGH INFO"
                    rationales = []

                to_tensorize = {"claim": claim["claim"],
                                "sentences": retrieved_doc["abstract"],
                                "label": label,
                                "rationales": rationales,
                                "title": retrieved_doc["title"]}
                entry = {"claim_id": claim["id"],
                         "abstract_id": retrieved_doc["doc_id"],
                         "negative_sample_id": 0,
                         "to_tensorize": to_tensorize,
                         "weight": 1.0}    # Istance weight is always 0.
                res.append(entry)

        return data.SciFactDataset(res, tokenizer, self.name, self.rationale_mask)


####################


def get_retrieved_dataloader(predict_args, model_hparams):
    "Main entry point to get the data loader. This can only be used at test time."
    reader = SciFactRetrievedReader(predict_args)
    tokenizer = data.get_tokenizer(model_hparams)
    ds = reader.get_data(tokenizer)
    collator = data.SciFactCollator(tokenizer)
    return DataLoader(ds,
                      num_workers=model_hparams.num_workers,
                      batch_size=predict_args.batch_size,
                      collate_fn=collator,
                      shuffle=False,
                      pin_memory=True)

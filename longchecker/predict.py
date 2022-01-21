from tqdm import tqdm
import argparse
from pathlib import Path

from model import SciFactModel
from data import ConcatDataModule
from retrieved_data import get_retrieved_dataloader
from lib import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--retrieval_file", type=str)
    parser.add_argument("--scifact_corpus_file", type=str,
                        default=util.scifact_data_dir / "corpus.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--use_last_checkpoint", action="store_true",
                        help="If given, use last instead of best.")
    parser.add_argument("--no_nei", action="store_true",
                        help="If given, never predict NEI.")
    parser.add_argument("--force_rationale", action="store_true",
                        help="If given, always predict a rationale for non-NEI.")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def get_predictions(args):
    args = get_args()
    model_dir = Path(args.model_dir)
    hparams_file = str(model_dir / "hparams.yaml")
    checkpoint_paths = [entry for entry in (model_dir / "checkpoint").iterdir()]
    if args.use_last_checkpoint:
        checkpoint_paths = [x for x in checkpoint_paths if "last.ckpt" in x.name]
    else:
        checkpoint_paths = [x for x in checkpoint_paths if "last.ckpt" not in x.name]
    assert len(checkpoint_paths) == 1
    checkpoint_path = str(checkpoint_paths[0])

    # Set up model and data.
    model = SciFactModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=hparams_file)
    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0

    # Since we're not running the training loop, gotta put model on GPU.
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    # Grab model hparams and override using new args, when relevant.
    hparams = model.hparams["hparams"]
    del hparams.precision   # Don' use 16-bit precision during evaluation.
    for k, v in vars(args).items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

    # Get the dataloader. If a retrieval file is specified, use the retrieval
    # dataloader. Otherwise use the normal one.
    dataloader = get_retrieved_dataloader(args, hparams)

    # Make predictions.
    predictions_all = []

    for batch in tqdm(dataloader):
        preds_batch = model.predict(batch, args.force_rationale)
        predictions_all.extend(preds_batch)

    return predictions_all


def format_predictions(args, predictions_all):
    # Need to get the claim ID's from the original file, since the data loader
    # won't have a record of claims for which no documents were retireved.
    claims = util.load_jsonl(args.input_file)
    claim_ids = [x["id"] for x in claims]
    assert len(claim_ids) == len(set(claim_ids))

    formatted = {claim: {} for claim in claim_ids}

    # Dict keyed by claim.
    for prediction in predictions_all:
        # If it's NEI, skip it.
        if prediction["predicted_label"] == "NEI":
            continue

        # Add prediction.
        formatted_entry = {
            prediction["abstract_id"]:
            {"label": prediction["predicted_label"],
             "sentences": prediction["predicted_rationale"]}}
        formatted[prediction["claim_id"]].update(formatted_entry)

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k,
                     "evidence": v}
        res.append(to_append)

    return res


def main():
    args = get_args()
    outname = Path(args.output_file)
    predictions = get_predictions(args)

    # Save the scores at pickle binary if requested.
    score_output = outname.parent / f"{outname.stem}.pkl"
    util.write_pickle(predictions, score_output)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)


if __name__ == "__main__":
    main()

from tqdm import tqdm
import argparse
from pathlib import Path

from model import MultiVerSModel
from data import get_dataloader
import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--no_nei", action="store_true", help="If given, never predict NEI."
    )
    parser.add_argument(
        "--force_rationale",
        action="store_true",
        help="If given, always predict a rationale for non-NEI.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def get_predictions(args):
    # Set up model and data.
    model = MultiVerSModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0

    # Since we're not running the training loop, gotta put model on GPU.
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    # Grab model hparams and override using new args, when relevant.
    hparams = model.hparams["hparams"]
    del hparams.precision  # Don' use 16-bit precision during evaluation.
    for k, v in vars(args).items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

    dataloader = get_dataloader(args)

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
            prediction["abstract_id"]: {
                "label": prediction["predicted_label"],
                "sentences": prediction["predicted_rationale"],
            }
        }
        formatted[prediction["claim_id"]].update(formatted_entry)

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k, "evidence": v}
        res.append(to_append)

    return res


def main():
    args = get_args()
    outname = Path(args.output_file)
    predictions = get_predictions(args)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)


if __name__ == "__main__":
    main()

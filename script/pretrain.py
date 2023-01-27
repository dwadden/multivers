"""
Kickoff pretraining on FEVER and weakly-labeled in-domain data.

NOTE: Training right now doesn't work with multiple GPU's and DDP. This is known issue;
see for instance https://lightning.ai/forums/t/gradient-checkpointing-ddp-nan/398/7.
"""


import argparse
import subprocess


def get_args():
    help_gpus = """GPU's used for training.
    If a single int, specifies the number of GPU's.
    If a comma-separated list, specifies the specific device ID's.
    For a single specific device, write it as `[device-num],`
    """
    help_datasets = """The datasets to train on, separated by a comma.
    To train on FEVER only, do `fever`. To do FEVER together with weakly-labeled
    in-domain data, do `fever,pubmedqa,evidence_inference`
    """

    parser = argparse.ArgumentParser("Kick off model training.")
    parser.add_argument("--datasets", type=str, help=help_datasets)
    parser.add_argument("--gpus", type=str, help=help_gpus)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Turning this on decreases memory usage at the cost of slower training",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    gpus = args.gpus

    # Deal with case of specific devides
    if "," in gpus:
        n_gpus = len([x for x in gpus.split(",") if x])
    else:
        n_gpus = int(gpus)

    if n_gpus not in [1, 2, 4, 8]:
        raise ValueError("The number of GPU's must be a power of 2.")

    epochs = 3
    workers_per_gpu = 4  # Number of CPU's per gpu.
    effective_batch_size = 8  # Desired effective batch size.
    accumulate_grad_batches = effective_batch_size // n_gpus
    num_workers = workers_per_gpu * n_gpus

    experiment_name = "-".join(args.datasets.split(","))

    cmd = [
        "python",
        "multivers/train.py",
        "--result_dir",
        "checkpoints_user",
        "--datasets",
        args.datasets,
        "--experiment_name",
        experiment_name,
        "--num_workers",
        num_workers,
        "--gpus",
        gpus,
        "--accumulate_grad_batches",
        accumulate_grad_batches,
        "--lr",
        "1e-5",
        "--precision",
        16,
        "--max_epochs",
        epochs,
        "--scheduler_total_epochs",
        epochs,
        "--train_batch_size",
        1,
        "--eval_batch_size",
        2,
        "--val_check_interval",
        "0.1",
        "--encoder_name",
        "longformer-large-science",
        "--monitor",
        "valid_abstract_rationalized_f1"
    ]

    # If training on more than 1 gpu, use DDP accelerator.
    if n_gpus > 1:
        cmd.extend(["--accelerator", "ddp"])

    # Turn on gradient checkpointing if requested.
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    subprocess.call(map(str, cmd))


if __name__ == "__main__":
    main()

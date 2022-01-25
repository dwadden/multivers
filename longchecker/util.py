"""
Shared utility functions.
"""

import json
import numpy as np
import pathlib
import os


def load_jsonl(fname, max_lines=None):
    res = []
    for i, line in enumerate(open(fname)):
        if max_lines is not None and i == max_lines:
            return res
        else:
            res.append(json.loads(line))

    return res


class NPEncoder(json.JSONEncoder):
    "Handles json encoding of Numpy objects."
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)


def write_jsonl(data, fname):
    with open(fname, "w") as f:
        for line in data:
            print(json.dumps(line, cls=NPEncoder), file=f)


def get_longformer_science_checkpoint():
    current_dir = pathlib.Path(os.path.realpath(__file__)).parent
    fname = current_dir.parent / "checkpoints/longformer_large_science.ckpt"

    return str(fname)

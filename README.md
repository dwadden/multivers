# The Longchecker model
Code and model checkpoints for the LongChecker model for scientific claim verification, described in the arXiv preprint [LongChecker: Improving scientific claim verification by modeling full-abstract context](https://arxiv.org/abs/2112.01640).

I'll have this repo populated by the end of December, including downloadable model checkpoints. If you need it sooner, email me and I'll help you get started: `dwadden@cs.washington.edu`.


## Setup

We recommend setting up a Conda environment:
```bash
conda create --name longchecker python=3.8 conda-build
```

Then, install required packages:
```bash
pip install -r requirements.txt
```

Next, clone this repo.

Then, download the Longformer checkpoint from which all the fact-checking models are finetuned by doing
```bash
python script/get_checkpoint.py longformer_large_science
```

Finally, this repo relies on the [scifact-evaluator](https://github.com/allenai/scifact-evaluator) for evaluation. To get things set up:
- Clone the `scifact-evaluator` repo.
- Set the environment variable `SCIFACT_EVALUATOR` to point at the [evaluation script](https://github.com/allenai/scifact-evaluator/blob/master/evaluator/eval.py) located at `evaluator/eval.py`. I accomplished this with:
  ```bash
  export SCIFACT_EVALUATOR=[path-to-scifact-evaluator-code]/evaluator/eval.py
  ```
  If you don't want to add an extra variable to your default shell environment, see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables) for instructions on how to set environment variables only within a specific Conda environment.


## Running inference with pre-trained models

- First, download the processed versions of the data by running `bash script/get_data.sh`. This will put the data in `data`.
- Then, download the model checkpoint you'd like to make predictions with using
  ```bash
  python script/get_checkpoint.py [checkpoint_name]
  ```
  Available models are listed in [model checkpoints](#model-checkpoints) section.


## Model checkpoints

The following model checkpoints are available. You can download them using the `script/get_checkpoint.sh`.

- `fever`: LongChecker trained on [FEVER](https://fever.ai/).
- `fever_sci`: LongChecker trained on FEVER, plus two weakly-supervised scientific datasets: [PubMedQA](https://pubmedqa.github.io/) and [Evidence Inference](https://evidence-inference.ebm-nlp.com/).
- `covidfact`: Finetuned on [CovidFact](https://github.com/asaakyan/covidfact), starting from the `fever_sci` checkpoint.
- `healthver`: Finetuned on [HealthVer](https://github.com/sarrouti/HealthVer).
- `scifact`: Finetuned on [SciFact](https://github.com/allenai/scifact).
- `longformer_large_science`: [Longformer](https://github.com/allenai/longformer) pre-trained on a corpus of scientific documents. This model has not been trained on any fact-checking data; it's the starting point for all other models.

You can also download all models by passing `all` to `get_checkpoint.sh`.

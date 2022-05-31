# The MultiVerS model

This is the repository for the MultiVerS model for scientific claim verification, described in the NAACL Findings 2022 paper [MultiVerS: Improving scientific claim verification with weak supervision and full-document context](https://arxiv.org/abs/2112.01640).

MultiVers was formerly known as LongChecker. It's the exact same model; we just changed the name to emphasize different aspects of the modeling approach. I'm still in the process of changing the filenames within this repo.

**Repository status**: We provide data, model checkpoints, and inference code for models trained on three scientific claim verification datasets: [SciFact](https://github.com/allenai/scifact), [CovidFact](https://github.com/asaakyan/covidfact), and [HealthVer](https://github.com/sarrouti/HealthVer) (see below for details).

While the SciFact test set is not public, predictions made using the SciFact checkpoint will reproduce the results in the preprint and on the [SciFact leaderboard](https://leaderboard.allenai.org/scifact/submissions/public).

**Update (May 2022)**: Apologies for the delay in getting the training code up. I will make sure that it is available by the time the work is presented at NAACL 2022, if not sooner.

**Disclaimer**: This software is intended to be used as a research protype, and its outputs shouldn't be used to inform any medical decisions.

## Setup

We recommend setting up a Conda environment:

```bash
conda create --name multivers python=3.8 conda-build
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

## Running inference with model checkpoints

- First, download the processed versions of the data by running `bash script/get_data.sh`. This will download the CovidFact, HealthVer, and SciFact datasets into the `data` directory.
- Then, download the model checkpoint you'd like to make predictions with using

  ```bash
  python script/get_checkpoint.py [checkpoint_name]
  ```

  Available models are listed in [model checkpoints](#model-checkpoints) section.
- Make predictions using the convenience wrapper script [script/predict.sh](script/predict.sh). This script accepts a dataset name as an argument, and makes predictions using the correct inputs files and model checkpoints for that dataset. For instance, to make predictions on the SciFact test set using the version of MultiVerS trained on Scifact, do:

  ```bash
  bash script/predict.sh scifact
  ```

- For more control over the models and datasets used for prediction, you can use [longchecker/predict.py](longchecker/predict.py).

## Model checkpoints

The following model checkpoints are available. You can download them using `script/get_checkpoint.sh`.

- `fever`: MultiVerS trained on [FEVER](https://fever.ai/).
- `fever_sci`: MultiVerS trained on FEVER, plus two weakly-supervised scientific datasets: [PubMedQA](https://pubmedqa.github.io/) and [Evidence Inference](https://evidence-inference.ebm-nlp.com/).
- `covidfact`: Finetuned on [CovidFact](https://github.com/asaakyan/covidfact), starting from the `fever_sci` checkpoint.
- `healthver`: Finetuned on [HealthVer](https://github.com/sarrouti/HealthVer).
- `scifact`: Finetuned on [SciFact](https://github.com/allenai/scifact).
- `longformer_large_science`: [Longformer](https://github.com/allenai/longformer) pre-trained on a corpus of scientific documents. This model has not been trained on any fact-checking data; it's the starting point for all other models.

You can also download all models by passing `all` to `get_checkpoint.sh`.

## Evaluating model predictions

The SciFact test set is private, but the test sets for HealthVer and CovidFact are included in the data download. To evaluate model predictions, use the [scifact-evaluator](https://github.com/allenai/scifact-evaluator) code. Clone the repo, then use the [evaluation script](https://github.com/allenai/scifact-evaluator/blob/master/evaluator/eval.py) located at `evaluator/eval.py`. This script accepts two files:

1. Predictions, as output by `longchecker/predict.py`
2. Gold labels, which are included in the data download.

It will evaluate the predictions with respect to gold and save metrics to a file. See the evaluation script for more details.

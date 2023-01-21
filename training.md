# MultiVerS model training

Code and data are now available to train MultiVerS.

TODO clean this up.


## Data

The training data are in `data_train`. There are subfolders for the pretraining datasets and target dataset.s

### Pretraining

These are in `pretrain`. For each of these, we use 4 negative examples per positive.

### Target

These are in `target`. For `CovidFact` and `HealthVer`, we don't do anything fancy with negative sampling. For SciFact, there are two options: 10 negative samples, or 20.

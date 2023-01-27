# Make model predictions.

function predict_scifact() {
    python multivers/predict.py \
        --checkpoint_path=checkpoints/scifact.ckpt \
        --input_file=data/scifact/claims_test_retrieved.jsonl \
        --corpus_file=data/scifact/corpus.jsonl \
        --output_file=prediction/scifact.jsonl
}

function predict_healthver() {
    python multivers/predict.py \
        --checkpoint_path=checkpoints/healthver.ckpt \
        --input_file=data/healthver/claims_test.jsonl \
        --corpus_file=data/healthver/corpus.jsonl \
        --output_file=prediction/healthver.jsonl
}

function predict_covidfact() {
    # NOTE: For covidfact, many of the claims are paper titles (or closely
    # related). To avoid data leakage for this dataset, we evaluate using a
    # version of the corpus with titles removed.
    python multivers/predict.py \
        --checkpoint_path=checkpoints/covidfact.ckpt \
        --input_file=data/covidfact/claims_test.jsonl \
        --corpus_file=data/covidfact/corpus_without_titles.jsonl \
        --output_file=prediction/covidfact.jsonl
}

########################################

model=$1

mkdir -p prediction

if [[ $model == "scifact" ]]
then
    predict_scifact
elif [[ $model == "covidfact" ]]
then
    predict_covidfact
elif [[ $model == "healthver" ]]
then
    predict_healthver
else
    echo "Allowed options are: {scifact,covidfact,healthver}."
fi

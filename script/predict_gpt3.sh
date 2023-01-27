# Make predictions on SciFact using GPT-3

python multivers/predict_gpt3.py \
    --train_file=data/scifact/claims_train_cited.jsonl \
    --test_file=data/scifact/claims_test_retrieved.jsonl \
    --corpus_file=data/scifact/corpus.jsonl \
    --output_file=prediction/scifact_gpt3.jsonl

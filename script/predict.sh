python longchecker/predict.py \
    --checkpoint_path=checkpoints/covidfact.ckpt \
    --input_file=data/covidfact/claims_test.jsonl \
    --corpus_file=data/covidfact/corpus.jsonl \
    --output_file=scratch/preds.jsonl \
    --batch_size=1 \

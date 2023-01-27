"""
Simple approach to make predictions using GPT-3.
"""


from tqdm import tqdm
import argparse
import os
import openai
import ast
from collections import Counter

import util

openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3Predictor:
    "Make preditions by calling the gpt-3 API."

    def __init__(self, prompt, model="text-davinci-003"):
        self.prompt = prompt
        self.model = model

    def get_response(self, full_prompt):
        response = openai.Completion.create(
            model=self.model, prompt=full_prompt, max_tokens=10, temperature=0
        )
        pred = response["choices"][0]["text"]
        return pred

    def predict_one(self, full_prompt):
        "Make predictions for a single claim / abstract pair."
        unparsed = self.get_response(full_prompt)
        splt = [x for x in unparsed.split("\n")]
        label_str = splt[0].strip().lower()
        lookup = {"true": "SUPPORT", "false": "CONTRADICT", "neither": "NEI"}
        label = lookup[label_str]
        rationale_str = splt[2].split()[-1]
        rationales = ast.literal_eval(rationale_str)

        return label, rationales

    def predict(self, test_data):
        "Make predictions for entire dataset."
        counts = Counter()
        preds = []
        for claim_id in tqdm(test_data.claims):
            claim = test_data.claims[claim_id]
            evidence = {}
            for doc_id in claim["doc_ids"]:
                test_input = test_data.format_instance(claim_id, doc_id)
                full_prompt = self.prompt + "\n\n" + "-" * 40 + "\n\n" + test_input
                # Get prediction.
                try:
                    # Make call to OpenAI and format the output.
                    label, rationales = self.predict_one(full_prompt)
                    counts["success"] += 1
                except Exception:
                    # If the call fails (probably due to badly-formatted output), just
                    # predict NEI. This doesn't happen that often.
                    counts["failure"] += 1
                    label = "NEI"
                    rationales = []
                # If not NEI, add to list.
                if label != "NEI":
                    evidence[str(doc_id)] = {"label": label, "sentences": rationales}

            # Append to predictinos.
            to_append = {"id": claim_id, "evidence": evidence}
            preds.append(to_append)

        print(counts)
        return preds


class Data:
    def __init__(self, input_file, corpus_file):
        self.claims = util.list_to_dict(util.load_jsonl(input_file), "id")
        self.corpus = util.list_to_dict(util.load_jsonl(corpus_file), "doc_id")


class TestData(Data):
    "Make prompts for test data."
    prompt_template = """Title: {title}

Abstract: {abstract}

Question: {claim} True, False, or Neither?

Answer:"""

    def make_single_prompt(self, claim, doc):
        "Format a single prompt instance."
        abstract_counter = [x + 1 for x in range(len(doc["abstract"]))]
        abstract_sents = [
            f"[{counter}] {sentence}"
            for counter, sentence in zip(abstract_counter, doc["abstract"])
        ]
        abstract_sents = " ".join(abstract_sents)

        the_prompt = self.prompt_template.format(
            title=doc["title"], abstract=abstract_sents, claim=claim
        )
        return the_prompt

    def format_instance(self, claim_id, doc_id):
        "Format test instance."
        claim = self.claims[claim_id]
        if doc_id not in claim["doc_ids"]:
            raise Exception("Doc ID not found.")

        doc = self.corpus[doc_id]
        single_prompt = self.make_single_prompt(claim["claim"], doc)

        return single_prompt


class TrainData(Data):
    "Create few-shot prompt to use as in-context example."
    prompt_template = """Title: {title}

Abstract: {abstract}

Question: {claim} True, False, or Neither?

Answer: {label}

Evidence: {highlights}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = self.make_prompt()

    def make_single_prompt(self, claim, doc, label, highlights):
        "Format a single prompt instance."
        abstract_counter = [x + 1 for x in range(len(doc["abstract"]))]
        abstract_sents = [
            f"[{counter}] {sentence}"
            for counter, sentence in zip(abstract_counter, doc["abstract"])
        ]
        abstract_sents = " ".join(abstract_sents)

        the_prompt = self.prompt_template.format(
            title=doc["title"],
            abstract=abstract_sents,
            claim=claim,
            label=label,
            highlights=highlights,
        )
        return the_prompt

    def format_instance(self, claim_id, doc_id, expected_label):
        "Format a training instance to use as part of a prompt."
        claim = self.claims[claim_id]
        if doc_id not in claim["doc_ids"]:
            raise Exception("Doc ID not found.")
        if expected_label == "NEI":
            if str(doc_id) in claim["evidence"]:
                raise Exception("NEI shouldn't be evidence.")
            highlights = []
            label = "Neither"
        else:
            lookup = {"SUPPORT": "True", "CONTRADICT": "False"}
            ev = claim["evidence"][str(doc_id)]
            highlights = util.flatten([x["sentences"] for x in ev])
            label = lookup[ev[0]["label"]]

        doc = self.corpus[doc_id]
        single_prompt = self.make_single_prompt(claim["claim"], doc, label, highlights)

        return single_prompt

    def make_prompt(self):
        "Make few-shot prompt from 3 randomly-chosen examples."
        # Three randomly-chosen examples; one per label.
        support = {"claim_id": 1023, "doc_id": 16927286, "expected_label": "SUPPORT"}
        refute = {"claim_id": 149, "doc_id": 6227220, "expected_label": "CONTRADICT"}
        nei = {"claim_id": 1400, "doc_id": 14706752, "expected_label": "NEI"}

        prompts_list = [
            self.format_instance(**instance) for instance in [support, refute, nei]
        ]
        sep = "\n\n" + "-" * 40 + "\n\n"
        prompt = sep.join(prompts_list)

        return prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--output_file", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    train_data = TrainData(args.train_file, args.corpus_file)
    test_data = TestData(args.test_file, args.corpus_file)
    predictor = GPT3Predictor(train_data.prompt)

    predictions = predictor.predict(test_data)
    util.write_jsonl(predictions, args.output_file)


if __name__ == "__main__":
    main()

# Data

## Data format

The data format for MultiVerS is similar to the format used in the [SciFact](https://github.com/allenai/scifact/blob/master/doc/data.md) repo, with a few small changes. Verifying a collection of claims requires two files:

- A `claims` file, containing the claims to be verified.
- A `corpus` file, containing the documents against which to verify them.

### Claims

The schema for the claim data is as follows:

```python
{
    "id": number,                   # An integer claim ID.
    "claim": string,                # The text of the claim.
    "doc_ids": number[]             # Documents from corpus on which to run MultiVerS.
    "evidence": {                   # OPTIONAL: The evidence for the claim.
        [doc_id]: [                 # The rationales for a single document, keyed by S2ORC ID.
            {
                "label": enum("SUPPORT" | "CONTRADICT"),
                "sentences": number[]
            }
        ]
    },
}
```

When making predictions (as opposed to training a model), the `evidence` field can be omitted or set to a placeholder value like `None`; this field will be ignored.

The `doc_ids` field is a list of document ID's, indicating which documents in the corpus should be run against each claim. Below is an example from the HealthVer datset.

```json
  "id": 1625,
  "claim": "A few cases of dogs tested weak positives of Coronavirus and there is evidence they can transmit to humans directly.",
  "doc_ids": [
    236
  ],
  "evidence": {
    "236": [
      {
        "sentences": [
          1
        ],
        "label": "CONTRADICT"
      }
    ]
  }
```

### Corpus

The schema is as follows:

```python
{
    "doc_id": number,               # A unique ID for the abstract.
    "title": string,                # The title.
    "abstract": string[],           # The abstract, written as a list of sentences.
}
```

Note that the `abstract` field could in principle be any text separated into sentences; it's called `abstract` because that's what we used in this work. Example from HealthVer below.

```json
{
  "doc_id": 236,
  "title": "Rising evidence of COVID-19 transmission potential to and between animals: do we need to be concerned?",
  "abstract": [
    "Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)--the virus that causes coronavirus disease (COVID-19)--has been detected in domestic dogs and cats, raising concerns of transmission from, to, or between these animals.",
    "There is currently no indication that feline- or canine-to-human transmission can occur, though there is rising evidence of the reverse.",
    ...
  ]
}
```

## Formatting a new dataset

To format a new dataset, you will need a claims file and a corpus file in the format described above.

### Choosing the `doc_ids` to associate with each claim

As described in the [claims](#claims) section, each claim must be associated with a list of documents from the corpus. MultiVers will only run on the documents in the provided list. If you already know which doucments you want to verify each claim against, you can just set the `doc_ids` field for each claim appropriately.

If you have a large corpus that you'd like to verify claims against, but don't know which documents are likely to contain evidence, you'll need to run an information retrieval step to select a list of candidate documents for each claim. A simple approach for this would be to use Skearln's implementation of [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). For more advanced retrieval techniques, [pyserini](https://github.com/castorini/pyserini) is a great place to start.

### Splitting documents into sentences

You'll need to format your documents as described in the [corpus](#corpus) section. This requires splitting your body text into sentences. If you're working with scientific text, [scispacy](https://allenai.github.io/scispacy/) provides a good option for this; see example below. For general text (Wikipedia, newswire, etc), you can use [spacy](https://spacy.io/).

```python
import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")
doc = nlp(doc_to_process)
sents = [sent.text for sent in doc.sents]
```

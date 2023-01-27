# GPT-3 baseline

For the GPT-3 baseline, I do few-shot learning with three demonstrations (one `SUPPORTS`, one `REFUTES`, one `NEI`). I chose the three examples randomly and did not them. I did not get good performance, and would welcome a PR exploring this further!

## Prompt template

```text
Title: {Paper title}

Abstract: [1] {Sentence 1} [2] {Sentence 2} ... [N] {Sentence N}

Question: {Claim}. True, False, or Neither?

Answer: {Label}

Evidence: {Rationales}
```

For prediction, the model is prompted with:

```text
Title: {Paper title}

Abstract: [1] {Sentence 1} [2] {Sentence 2} ... [N] {Sentence N}

Question: {Claim}. True, False, or Neither?

Answer:
```

## Performance

I got pretty poor performance from GPT-3. Metrics are below.

|          | Precision | Recall |   F1 |
|----------|-----------|--------|------|
| Abstract |      11.2 |   60.0 | 18.9 |
| Sentence |       4.3 |    7.6 |  5.5 |

## Full prompt

An example of a full prompt is shown below. It shows three demonstrations, followed by the instance to be predicted.

```text
Title: Large-Scale Movements of IF3 and tRNA during Bacterial Translation Initiation

Abstract: [1] In bacterial translational initiation, three initiation factors (IFs 1-3) enable the selection of initiator tRNA and the start codon in the P site of the 30S ribosomal subunit. [2] Here, we report 11 single-particle cryo-electron microscopy (cryoEM) reconstructions of the complex of bacterial 30S subunit with initiator tRNA, mRNA, and IFs 1-3, representing different steps along the initiation pathway. [3] IF1 provides key anchoring points for IF2 and IF3, thereby enhancing their activities. [4] IF2 positions a domain in an extended conformation appropriate for capturing the formylmethionyl moiety charged on tRNA. [5] IF3 and tRNA undergo large conformational changes to facilitate the accommodation of the formylmethionyl-tRNA (fMet-tRNA(fMet)) into the P site for start codon recognition.

Question: Recognition of start codons depends on the translation initiation factor IF3. True, False, or Neither?

Answer: True

Evidence: [4]

----------------------------------------

Title: Autophagy deficiency leads to protection from obesity and insulin resistance by inducing Fgf21 as a mitokine

Abstract: [1] Despite growing interest and a recent surge in papers, the role of autophagy in glucose and lipid metabolism is unclear. [2] We produced mice with skeletal muscle–specific deletion of Atg7 (encoding autophagy-related 7). [3] Unexpectedly, these mice showed decreased fat mass and were protected from diet-induced obesity and insulin resistance; this phenotype was accompanied by increased fatty acid oxidation and browning of white adipose tissue (WAT) owing to induction of fibroblast growth factor 21 (Fgf21). [4] Mitochondrial dysfunction induced by autophagy deficiency increased Fgf21 expression through induction of Atf4, a master regulator of the integrated stress response. [5] Mitochondrial respiratory chain inhibitors also induced Fgf21 in an Atf4-dependent manner. [6] We also observed induction of Fgf21, resistance to diet-induced obesity and amelioration of insulin resistance in mice with autophagy deficiency in the liver, another insulin target tissue. [7] These findings suggest that autophagy deficiency and subsequent mitochondrial dysfunction promote Fgf21 expression, a hormone we consequently term a 'mitokine', and together these processes promote protection from diet-induced obesity and insulin resistance.

Question: Autophagy deficiency in the liver increases vulnerability to insulin resistance. True, False, or Neither?

Answer: False

Evidence: [2, 5, 6]

----------------------------------------

Title: Gamma-Secretase Represents a Therapeutic Target for the Treatment of Invasive Glioma Mediated by the p75 Neurotrophin Receptor

Abstract: [1] The multifunctional signaling protein p75 neurotrophin receptor (p75(NTR)) is a central regulator and major contributor to the highly invasive nature of malignant gliomas. [2] Here, we show that neurotrophin-dependent regulated intramembrane proteolysis (RIP) of p75(NTR) is required for p75(NTR)-mediated glioma invasion, and identify a previously unnamed process for targeted glioma therapy. [3] Expression of cleavage-resistant chimeras of p75(NTR) or treatment of animals bearing p75(NTR)-positive intracranial tumors with clinically applicable gamma-secretase inhibitors resulted in dramatically decreased glioma invasion and prolonged survival. [4] Importantly, proteolytic processing of p75(NTR) was observed in p75(NTR)-positive patient tumor specimens and brain tumor initiating cells. [5] This work highlights the importance of p75(NTR) as a therapeutic target, suggesting that gamma-secretase inhibitors may have direct clinical application for the treatment of malignant glioma.

Question: p75 NTR - associated cell death executor (NADE) interacts with the p75 NTR death domain True, False, or Neither?

Answer: Neither

Evidence: []

----------------------------------------

Title: Prevalence, severity, and unmet need for treatment of mental disorders in the World Health Organization World Mental Health Surveys.

Abstract: [1] CONTEXT Little is known about the extent or severity of untreated mental disorders, especially in less-developed countries.
 [2] OBJECTIVE To estimate prevalence, severity, and treatment of Diagnostic and Statistical Manual of Mental Disorders, Fourth Edition (DSM-IV) mental disorders in 14 countries (6 less developed, 8 developed) in the World Health Organization (WHO) World Mental Health (WMH) Survey Initiative.
 [3] DESIGN, SETTING, AND PARTICIPANTS Face-to-face household surveys of 60 463 community adults conducted from 2001-2003 in 14 countries in the Americas, Europe, the Middle East, Africa, and Asia.
 [4] MAIN OUTCOME MEASURES The DSM-IV disorders, severity, and treatment were assessed with the WMH version of the WHO Composite International Diagnostic Interview (WMH-CIDI), a fully structured, lay-administered psychiatric diagnostic interview.
 [5] RESULTS The prevalence of having any WMH-CIDI/DSM-IV disorder in the prior year varied widely, from 4.3% in Shanghai to 26.4% in the United States, with an interquartile range (IQR) of 9.1%-16.9%. [6] Between 33.1% (Colombia) and 80.9% (Nigeria) of 12-month cases were mild (IQR, 40.2%-53.3%). [7] Serious disorders were associated with substantial role disability. [8] Although disorder severity was correlated with probability of treatment in almost all countries, 35.5% to 50.3% of serious cases in developed countries and 76.3% to 85.4% in less-developed countries received no treatment in the 12 months before the interview. [9] Due to the high prevalence of mild and subthreshold cases, the number of those who received treatment far exceeds the number of untreated serious cases in every country.
 [10] CONCLUSIONS Reallocation of treatment resources could substantially decrease the problem of unmet need for treatment of mental disorders among serious cases. [11] Structural barriers exist to this reallocation. [12] Careful consideration needs to be given to the value of treating some mild cases, especially those at risk for progressing to more serious disorders.

Question: 10-20% of people with severe mental disorder receive no treatment in low and middle income countries. True, False, or Neither?

Answer:
```

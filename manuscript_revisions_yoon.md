# Manuscript Revision Notes for Yoonsung-Marked Reviewer Responses

These notes list the manuscript text changes that should accompany the revised
responses in `ResponsesToReviewersComments.docx`. The submitted manuscript is
currently available as `SVS-S-26-00512.pdf`, so these are replacement-text
instructions rather than direct edits to the manuscript source.

## Outcome Variable

Replace the sentence that says sigmoid focal loss and threshold adjustment were
selectively applied.

Suggested replacement:

> The distribution of the binary outcome was mildly imbalanced, with 55.6% coded
> as non-completion and 44.4% coded as treatment completion. Final reported GNN
> models used cross-entropy loss, and class predictions were obtained using the
> standard argmax rule applied to model logits. Sigmoid focal loss and post-hoc
> probability-threshold adjustment were not used in the final reported analyses.

## Data Preprocessing / Graph Representation

Clarify that the seven-dimensional encoder applies to the original GAT baseline
and reviewer-requested graph-structure reruns, not every model pipeline.

Suggested replacement:

> In the original GAT baseline and in the reviewer-requested graph-structure
> reruns, categorical variables were encoded using variable-specific entity
> embeddings. The initial embedding width followed the square-root heuristic,
> ceil(sqrt(number of categories)), and each embedding was projected into a
> shared seven-dimensional space. A binary missingness indicator was appended to
> each node representation. This projection provided comparable node features for
> heterogeneous categorical variables while avoiding high-dimensional one-hot
> encodings.

## Training And Evaluation

Add a formal recall definition after precision or before F1-score.

Suggested insertion:

> Recall. Recall, also referred to as sensitivity, measures the proportion of
> actual positive cases correctly identified by the model. In this study, the
> positive class was treatment completion (REASON = 1). Recall is defined as TP /
> (TP + FN), where TP is the number of actual completions correctly predicted as
> completions and FN is the number of actual completions incorrectly predicted as
> non-completions.

## Results: T-GCN

Replace the temporal-model interpretation with a precision-recall trade-off
description.

Suggested replacement:

> For the temporal extension, we evaluated T-GCN against the GCN baseline. Table
> 2 shows that T-GCN increased precision but had lower accuracy, recall, and
> F1-score than GCN. We therefore interpret the T-GCN result as a
> precision-recall trade-off rather than an overall improvement. The temporal
> model reduced false positives but missed more true treatment completions. This
> limitation is likely related to the structure of TEDS-D after conversion to a
> temporal format, because each episode provided only two time points, admission
> and discharge, which was not sufficient for the recurrent component to learn
> rich longitudinal dynamics.

Also change the Table 2 model label from `TGCN` to `T-GCN` if the manuscript
source allows table edits.

## Results: Partial-Reporting Scenarios

Replace the current Scenario 01/02 explanation with a clearer definition of
missingness and thresholds.

Suggested replacement:

> To assess cross-state generalization, we trained models on
> comprehensive-reporting states and evaluated them on held-out states with
> higher missingness. For each state, average missingness was computed as the
> mean proportion of not-reported ("-9") values across the 20 predictor
> variables. The across-state mean missingness was 0.080 (SD = 0.083). Scenario
> 01 classified states as partial-reporting when average missingness was at or
> above mean + 1 SD (0.163), which selected Arizona, Wisconsin, Washington,
> Maryland, Louisiana, and California. Scenario 02 used the stricter mean + 2 SD
> threshold (0.245), which selected Arizona, Wisconsin, and Washington.

## Discussion

Replace over-positive language about T-GCN and GAT robustness.

Suggested replacement for T-GCN discussion:

> The addition of temporal modeling through T-GCN did not yield an overall
> performance improvement. T-GCN increased precision but reduced recall and
> F1-score, suggesting that it filtered out false positives at the cost of
> missing true treatment completions.

Suggested replacement for partial-reporting discussion:

> Generalization performance declined when models trained on
> comprehensive-reporting states were applied to partial-reporting states. GAT
> degraded less severely than GCN and GIN in Scenario 01, but its accuracy still
> remained low. These results indicate that incomplete and uneven reporting
> across states limits cross-state generalizability and should be addressed
> before using these models for accountability decisions.

## Abstract

If the abstract is revised, soften the robustness claim.

Suggested replacement:

> Performance declined in states with partial data reporting, indicating that
> uneven reporting quality remains a major limitation for cross-state
> generalization.

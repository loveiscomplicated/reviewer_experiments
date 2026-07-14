#!/usr/bin/env python3
"""Neutralize reviewer responses and append manuscript revision notes to DOCX."""

from __future__ import annotations

import copy
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


DOCX_PATH = Path("ResponsesToReviewersComments.docx")
BACKUP_PATH = Path("ResponsesToReviewersComments.before_neutral_notes.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}

NOTES_TITLE = "Manuscript Revision Notes for [윤성] Responses"


RESPONSES = {
    "T-GCN": (
        "[윤성] The reviewer is correct. Table 2 indicates that T-GCN increased "
        "precision relative to the GCN baseline (0.865 vs. 0.833), but accuracy "
        "(0.792 vs. 0.835), recall (0.629 vs. 0.785), and F1-score (0.729 vs. "
        "0.808) were lower. The result is therefore better interpreted as a "
        "precision-recall trade-off rather than an overall improvement. The "
        "temporal model reduced false positives but missed more true treatment "
        "completions. A likely reason is that, after TEDS-D was converted to a "
        "temporal format, each episode provided only two time points, admission "
        "and discharge, which limited the recurrent component's ability to learn "
        "longitudinal dynamics."
    ),
    "State-level missingness": (
        "[윤성] State-level missingness was computed as the average proportion of "
        'not-reported ("-9") values across the 20 predictor variables within each '
        "state (STFIPS). Across states, the mean missingness rate was 0.080 "
        "(SD = 0.083). Scenario 01 classified states at or above mean + 1 SD "
        "(0.163) as partial-reporting states: Arizona (0.381), Wisconsin (0.285), "
        "Washington (0.268), Maryland (0.244), Louisiana (0.205), and California "
        "(0.164), comprising 308,822 episodes (22.2% of the sample). Scenario 02 "
        "used the stricter mean + 2 SD threshold (0.245), which selected Arizona, "
        "Wisconsin, and Washington, comprising 182,341 episodes (13.1%). The "
        "variables with the highest overall missingness were LIVARAG_D (24.6%), "
        "MARSTAT (24.3%), DSMCRIT (20.2%), ARRESTS_D (19.3%), EMPLOY_D (17.9%), "
        "ARRESTS (17.3%), EDUC (15.7%), and LIVARAG (15.6%). Consistent with the "
        "reviewer's concern, performance in partial-reporting states was often "
        "near random guessing and, in several cases, below the majority-class "
        "baseline of 0.556. For example, under Scenario 01, GIN achieved accuracy "
        "0.313, recall 0.040, and F1-score 0.073; GCN achieved accuracy 0.447 and "
        "F1-score 0.413. GAT degraded less severely in Scenario 01 (accuracy "
        "0.536, F1-score 0.611), but its performance still remained limited. "
        "These results indicate that incomplete and uneven reporting across "
        "states limits cross-state generalizability and should be considered "
        "before using these models for accountability decisions."
    ),
    "Hyperparameters": (
        "[윤성] Hyperparameters were pipeline-specific. The original GAT baseline "
        "and the reviewer-requested graph-structure reruns used variable-specific "
        "entity embeddings projected to a shared seven-dimensional space, with a "
        "binary missingness indicator appended to each node representation. In the "
        "reviewer-requested reruns, GCN, GIN, GAT, and T-GCN used Adam "
        "(learning rate = 0.001), batch size 32, up to 100 epochs, cross-entropy "
        "loss, and a ReduceLROnPlateau scheduler on validation loss (factor = 0.5, "
        "patience = 7). Early stopping used validation loss with patience 20 for "
        "GCN, GIN, and GAT and patience 10 for T-GCN. The GAT architecture used "
        "hidden dimension 64, dropout 0.5, four attention heads in the first GAT "
        "layer, and one head in the final GAT layer. The Table 3 "
        "state-generalization GAT used a separate pipeline with a 64-dimensional "
        "ID embedding, hidden dimension 64, eight first-layer attention heads, "
        "one final-layer head, and dropout 0.1."
    ),
    "focal loss": (
        "[윤성] The final reported GNN models used cross-entropy loss, and "
        "predicted classes were obtained with the standard argmax rule applied to "
        "the model logits. Sigmoid focal loss and post-hoc probability-threshold "
        "adjustment were not used in the final reported results. The outcome "
        "distribution was only mildly imbalanced in the analytic sample (55.6% "
        "non-completion vs. 44.4% completion), so no additional "
        "imbalance-handling procedure was applied in the final reported models."
    ),
    "seven-dimensional": (
        "[윤성] The seven-dimensional projection was used in the original GAT "
        "baseline and in the reviewer-requested graph-structure reruns; it was not "
        "used in the Table 3 state-generalization experiment, which used a "
        "separate 64-dimensional ID-embedding pipeline. In the seven-dimensional "
        "encoder, each categorical variable first received a variable-specific "
        "entity embedding. The initial embedding width followed the square-root "
        "heuristic, ceil(sqrt(number of categories)), and each embedding was then "
        "projected into a shared seven-dimensional space. A binary missingness "
        "indicator was appended to each node representation. The purpose of this "
        "projection was to place heterogeneous categorical variables into a "
        "comparable node-feature space while keeping the input much smaller than a "
        "one-hot representation."
    ),
    "Recall": (
        "[윤성] Recall (sensitivity) is defined as TP / (TP + FN), where TP is the "
        "number of actual treatment completions (REASON = 1) correctly predicted "
        "as completions and FN is the number of actual treatment completions "
        "incorrectly predicted as non-completions. It measures the proportion of "
        "true treatment completions that the model correctly identifies."
    ),
    "Scenarios": (
        "[윤성] Scenarios 01 and 02 differ only in how strictly they define a "
        'partial-reporting state. For each state, average missingness was computed '
        'as the mean proportion of not-reported ("-9") values across the 20 '
        "predictor variables. The across-state distribution had a mean of 0.080 "
        "and an SD of 0.083. Scenario 01 classified states as partial-reporting "
        "when their average missingness was at or above mean + 1 SD (0.163), "
        "resulting in six states: Arizona, Wisconsin, Washington, Maryland, "
        "Louisiana, and California. Scenario 02 used the stricter mean + 2 SD "
        "threshold (0.245), resulting in three states: Arizona, Wisconsin, and "
        "Washington. In both scenarios, models were trained on "
        "comprehensive-reporting states and evaluated on held-out "
        "partial-reporting states to assess cross-state generalization. Here, "
        "standard deviation refers to the SD of the across-state "
        "average-missingness distribution, not to variation within a single "
        "variable."
    ),
}


MATCHERS = [
    ("T-GCN", ["T-GCN increased precision", "T-GCN improved precision", "interpretation of the T-GCN"]),
    ("State-level missingness", ["State-level missingness", "partial reporting materially"]),
    ("Hyperparameters", ["Hyperparameters were pipeline-specific", "Hyperparameters differed", "hyperparameter description"]),
    ("focal loss", ["final reported GNN models used cross-entropy", "final reported models were trained", "focal loss and threshold"]),
    ("seven-dimensional", ["seven-dimensional projection", "scope and rationale for the seven-dimensional"]),
    ("Recall", ["Recall (sensitivity) is defined"]),
    ("Scenarios", ["Scenarios 01 and 02 differ"]),
]


NOTE_PARAGRAPHS = [
    (NOTES_TITLE, True),
    ("Purpose", True),
    (
        "The reviewer-response paragraphs above answer the reviewer comments directly. "
        "The items below are the corresponding manuscript edits to apply in the manuscript source.",
        False,
    ),
    ("Outcome Variable", True),
    ("Replace the class-imbalance sentence in the Outcome Variable section.", False),
    (
        "Replacement text: The distribution of the binary outcome was mildly imbalanced, "
        "with 55.6% coded as non-completion and 44.4% coded as treatment completion. "
        "Final reported GNN models used cross-entropy loss, and class predictions were "
        "obtained using the standard argmax rule applied to model logits. Sigmoid focal "
        "loss and post-hoc probability-threshold adjustment were not used in the final "
        "reported analyses.",
        False,
    ),
    ("Data Preprocessing / Graph Representation", True),
    (
        "Clarify that the seven-dimensional encoder applies to the original GAT baseline "
        "and reviewer-requested graph-structure reruns, not every model pipeline.",
        False,
    ),
    (
        "Replacement text: In the original GAT baseline and in the reviewer-requested "
        "graph-structure reruns, categorical variables were encoded using "
        "variable-specific entity embeddings. The initial embedding width followed the "
        "square-root heuristic, ceil(sqrt(number of categories)), and each embedding was "
        "projected into a shared seven-dimensional space. A binary missingness indicator "
        "was appended to each node representation. This projection provided comparable "
        "node features for heterogeneous categorical variables while avoiding "
        "high-dimensional one-hot encodings.",
        False,
    ),
    ("Training And Evaluation", True),
    ("Add a formal recall definition after precision or before F1-score.", False),
    (
        "Insertion text: Recall, also referred to as sensitivity, measures the proportion "
        "of actual positive cases correctly identified by the model. In this study, the "
        "positive class was treatment completion (REASON = 1). Recall is defined as TP / "
        "(TP + FN), where TP is the number of actual completions correctly predicted as "
        "completions and FN is the number of actual completions incorrectly predicted as "
        "non-completions.",
        False,
    ),
    ("Results: T-GCN", True),
    (
        "Replacement text: For the temporal extension, T-GCN was evaluated against the GCN "
        "baseline. Table 2 shows that T-GCN increased precision but had lower accuracy, "
        "recall, and F1-score than GCN. The T-GCN result is therefore a precision-recall "
        "trade-off rather than an overall improvement. The temporal model reduced false "
        "positives but missed more true treatment completions. This limitation is likely "
        "related to the structure of TEDS-D after conversion to a temporal format, because "
        "each episode provided only two time points, admission and discharge, which was "
        "not sufficient for the recurrent component to learn rich longitudinal dynamics.",
        False,
    ),
    ("Results: Partial-Reporting Scenarios", True),
    (
        "Replacement text: To assess cross-state generalization, models were trained on "
        "comprehensive-reporting states and evaluated on held-out states with higher "
        'missingness. For each state, average missingness was computed as the mean '
        'proportion of not-reported ("-9") values across the 20 predictor variables. '
        "The across-state mean missingness was 0.080 (SD = 0.083). Scenario 01 "
        "classified states as partial-reporting when average missingness was at or above "
        "mean + 1 SD (0.163), which selected Arizona, Wisconsin, Washington, Maryland, "
        "Louisiana, and California. Scenario 02 used the stricter mean + 2 SD threshold "
        "(0.245), which selected Arizona, Wisconsin, and Washington.",
        False,
    ),
    ("Discussion", True),
    (
        "Replacement text for T-GCN discussion: The addition of temporal modeling through "
        "T-GCN did not yield an overall performance improvement. T-GCN increased "
        "precision but reduced recall and F1-score, suggesting that it filtered out false "
        "positives at the cost of missing true treatment completions.",
        False,
    ),
    (
        "Replacement text for partial-reporting discussion: Generalization performance "
        "declined when models trained on comprehensive-reporting states were applied to "
        "partial-reporting states. GAT degraded less severely than GCN and GIN in "
        "Scenario 01, but its accuracy still remained low. These results indicate that "
        "incomplete and uneven reporting across states limits cross-state generalizability "
        "and should be considered before using these models for accountability decisions.",
        False,
    ),
    ("Abstract", True),
    (
        "Optional replacement text: Performance declined in states with partial data "
        "reporting, indicating that uneven reporting quality remains a major limitation "
        "for cross-state generalization.",
        False,
    ),
]


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(t.text or "" for t in paragraph.findall(".//w:t", NS)).strip()


def register_existing_namespaces(xml_bytes: bytes) -> None:
    text = xml_bytes.decode("utf-8", errors="ignore")
    for prefix, uri in re.findall(r'xmlns(?::([^=]+))?="([^"]+)"', text):
        ET.register_namespace(prefix, uri)


def body_rpr(root: ET.Element) -> ET.Element:
    for paragraph in root.findall(".//w:p", NS):
        p_text = paragraph_text(paragraph)
        if "[윤성]" not in p_text or p_text == "[윤성]":
            continue
        for run in paragraph.findall("w:r", NS):
            run_text = "".join(t.text or "" for t in run.findall(".//w:t", NS)).strip()
            rpr = run.find("w:rPr", NS)
            if run_text and rpr is not None:
                return copy.deepcopy(rpr)

    rpr = ET.Element(f"{{{W_NS}}}rPr")
    rfonts = ET.SubElement(rpr, f"{{{W_NS}}}rFonts")
    for key in ("ascii", "eastAsia", "hAnsi", "cs"):
        rfonts.set(f"{{{W_NS}}}{key}", "Aptos")
    color = ET.SubElement(rpr, f"{{{W_NS}}}color")
    color.set(f"{{{W_NS}}}val", "000000")
    sz = ET.SubElement(rpr, f"{{{W_NS}}}sz")
    sz.set(f"{{{W_NS}}}val", "22")
    szcs = ET.SubElement(rpr, f"{{{W_NS}}}szCs")
    szcs.set(f"{{{W_NS}}}val", "22")
    return rpr


def make_paragraph(text: str, rpr: ET.Element, bold: bool = False) -> ET.Element:
    paragraph = ET.Element(f"{{{W_NS}}}p")
    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    run_rpr = copy.deepcopy(rpr)
    if bold and run_rpr.find("w:b", NS) is None:
        ET.SubElement(run_rpr, f"{{{W_NS}}}b")
    run.append(run_rpr)
    t = ET.SubElement(run, f"{{{W_NS}}}t")
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return paragraph


def make_page_break() -> ET.Element:
    paragraph = ET.Element(f"{{{W_NS}}}p")
    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    br = ET.SubElement(run, f"{{{W_NS}}}br")
    br.set(f"{{{W_NS}}}type", "page")
    return paragraph


def replace_paragraph(paragraph: ET.Element, text: str, rpr: ET.Element) -> None:
    ppr = paragraph.find("w:pPr", NS)
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    run.append(copy.deepcopy(rpr))
    t = ET.SubElement(run, f"{{{W_NS}}}t")
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text


def classify_response(text: str) -> str | None:
    if "[윤성]" not in text or text == "[윤성]":
        return None
    for key, snippets in MATCHERS:
        if any(snippet in text for snippet in snippets):
            return key
    return None


def remove_existing_notes(body: ET.Element) -> None:
    children = list(body)
    title_index = None
    for idx, child in enumerate(children):
        if child.tag == f"{{{W_NS}}}p" and paragraph_text(child) == NOTES_TITLE:
            title_index = idx
            break
    if title_index is None:
        return
    start = title_index
    if title_index > 0:
        prev = children[title_index - 1]
        if prev.find(".//w:br", NS) is not None:
            start = title_index - 1
    for child in children[start:]:
        if child.tag == f"{{{W_NS}}}sectPr":
            continue
        body.remove(child)


def append_notes(root: ET.Element, rpr: ET.Element) -> None:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("DOCX body not found")
    remove_existing_notes(body)
    sect_pr = body.find("w:sectPr", NS)
    insert_at = list(body).index(sect_pr) if sect_pr is not None else len(list(body))

    new_children = [make_page_break()]
    new_children.extend(make_paragraph(text, rpr, bold=bold) for text, bold in NOTE_PARAGRAPHS)
    for offset, child in enumerate(new_children):
        body.insert(insert_at + offset, child)


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)
    if not BACKUP_PATH.exists():
        shutil.copy2(DOCX_PATH, BACKUP_PATH)

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        files = {info.filename: zin.read(info.filename) for info in zin.infolist()}
    document_xml = files["word/document.xml"]
    register_existing_namespaces(document_xml)
    root = ET.fromstring(document_xml)
    rpr = body_rpr(root)

    counts: dict[str, int] = {}
    for paragraph in root.findall(".//w:p", NS):
        key = classify_response(paragraph_text(paragraph))
        if key is None:
            continue
        replace_paragraph(paragraph, RESPONSES[key], rpr)
        counts[key] = counts.get(key, 0) + 1

    missing = [key for key in RESPONSES if counts.get(key) != 1]
    if missing:
        raise RuntimeError(f"Expected one replacement per response, got counts={counts}")

    append_notes(root, rpr)
    files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=".") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name, data in files.items():
                zout.writestr(name, data)
        with zipfile.ZipFile(tmp_path, "r") as ztest:
            bad = ztest.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member: {bad}")
        tmp_path.replace(DOCX_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("Updated", DOCX_PATH)
    print("Backup", BACKUP_PATH)
    for key in RESPONSES:
        print(key, counts.get(key))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replace Yoonsung-marked reviewer responses while preserving DOCX styling."""

from __future__ import annotations

import copy
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


DOCX_PATH = Path("ResponsesToReviewersComments.docx")
BACKUP_PATH = Path("ResponsesToReviewersComments.before_yoon_edits.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


REPLACEMENTS = {
    "We agree with the reviewer. As shown in Table 2, the T-GCN improved precision only": (
        "[윤성] We agree with the reviewer. We revised the interpretation of the T-GCN "
        "results so that they are no longer described as an overall improvement. As shown "
        "in Table 2, T-GCN increased precision relative to the GCN baseline (0.865 vs. "
        "0.833), but it had lower accuracy (0.792 vs. 0.835), recall (0.629 vs. 0.785), "
        "and F1-score (0.729 vs. 0.808). We now describe this pattern as a "
        "precision-recall trade-off: the temporal model reduced false positives but missed "
        "more true treatment completions. This limitation is likely related to the "
        "structure of TEDS-D after conversion to a temporal format, because each episode "
        "provided only two time points (admission and discharge), which is not sufficient "
        "for the recurrent component to learn rich longitudinal dynamics."
    ),
    "State-level missingness was computed as the average proportion of not-reported": (
        "[윤성] We agree that partial reporting materially limited generalization "
        "performance, and we added a more explicit missingness summary to the manuscript. "
        'State-level missingness was computed as the average proportion of not-reported '
        '("-9") values across the 20 predictor variables within each state (STFIPS). '
        "Across states, the mean missingness rate was 0.080 (SD = 0.083). Scenario 01 "
        "classified states at or above mean + 1 SD (0.163) as partial-reporting states: "
        "Arizona (0.381), Wisconsin (0.285), Washington (0.268), Maryland (0.244), "
        "Louisiana (0.205), and California (0.164), comprising 308,822 episodes (22.2% "
        "of the sample). Scenario 02 used the stricter mean + 2 SD threshold (0.245), "
        "which selected Arizona, Wisconsin, and Washington, comprising 182,341 episodes "
        "(13.1%). The variables with the highest overall missingness were LIVARAG_D "
        "(24.6%), MARSTAT (24.3%), DSMCRIT (20.2%), ARRESTS_D (19.3%), EMPLOY_D "
        "(17.9%), ARRESTS (17.3%), EDUC (15.7%), and LIVARAG (15.6%). Consistent with "
        "the reviewer's concern, performance in partial-reporting states was often near "
        "random guessing and, in several cases, below the majority-class baseline of "
        "0.556. For example, under Scenario 01, GIN achieved accuracy 0.313, recall "
        "0.040, and F1-score 0.073; GCN achieved accuracy 0.447 and F1-score 0.413. "
        "GAT degraded less severely in Scenario 01 (accuracy 0.536, F1-score 0.611), "
        "but its performance still remained limited. We revised the discussion to "
        "emphasize that incomplete and uneven reporting across states limits cross-state "
        "generalizability and should be addressed before using these models for "
        "accountability decisions."
    ),
    "Hyperparameters differed across three separate experiments": (
        "[윤성] We revised the hyperparameter description to separate the pipelines rather "
        "than implying that all experiments used the same encoder. The original GAT "
        "baseline and the reviewer-requested graph-structure reruns used variable-specific "
        "entity embeddings projected to a shared seven-dimensional space, with a binary "
        "missingness indicator appended to each node representation. In the "
        "reviewer-requested reruns, GCN, GIN, GAT, and T-GCN used Adam (learning rate = "
        "0.001), batch size 32, up to 100 epochs, cross-entropy loss, and a "
        "ReduceLROnPlateau scheduler on validation loss (factor = 0.5, patience = 7). "
        "Early stopping used validation loss with patience 20 for GCN, GIN, and GAT and "
        "patience 10 for T-GCN. The GAT architecture used hidden dimension 64, dropout "
        "0.5, four attention heads in the first GAT layer, and one head in the final GAT "
        "layer. The Table 3 state-generalization GAT was implemented in a separate "
        "pipeline with a 64-dimensional ID embedding, hidden dimension 64, eight "
        "first-layer attention heads, one final-layer head, and dropout 0.1. We will "
        "report these settings separately in the manuscript to avoid suggesting that one "
        "hyperparameter configuration was shared by all analyses."
    ),
    "The final reported models were trained with cross-entropy loss": (
        "[윤성] We clarified this point because the original wording could imply that "
        "focal loss and threshold adjustment were part of the final reported analyses. "
        "They were not. The final reported GNN models used cross-entropy loss, and "
        "predicted classes were obtained with the standard argmax rule applied to the "
        "model logits. Sigmoid focal loss and post-hoc probability-threshold adjustment "
        "were not used in the final reported results. The outcome distribution was only "
        "mildly imbalanced in the analytic sample (55.6% non-completion vs. 44.4% "
        "completion), so no additional imbalance-handling procedure was applied in the "
        "final reported models. We revised the manuscript accordingly."
    ),
    "The seven-dimensional projection was used in the baseline": (
        "[윤성] We clarified the scope and rationale for the seven-dimensional projection. "
        "This encoder was used in the original GAT baseline and in the reviewer-requested "
        "graph-structure reruns; it was not used in the Table 3 state-generalization "
        "experiment, which used a separate 64-dimensional ID-embedding pipeline. In the "
        "seven-dimensional encoder, each categorical variable first received a "
        "variable-specific entity embedding. The initial embedding width followed the "
        "square-root heuristic, ceil(sqrt(number of categories)), and each embedding was "
        "then projected into a shared seven-dimensional space. A binary missingness "
        "indicator was appended to each node representation. The purpose of this "
        "projection was to place heterogeneous categorical variables into a comparable "
        "node-feature space while keeping the input much smaller than a one-hot "
        "representation."
    ),
    "Recall (sensitivity) is defined as TP / (TP + FN)": (
        "[윤성] Recall (sensitivity) is defined as TP / (TP + FN), where TP is the number "
        "of actual treatment completions (REASON = 1) correctly predicted as completions "
        "and FN is the number of actual treatment completions incorrectly predicted as "
        "non-completions. It measures the proportion of true treatment completions that "
        "the model correctly identifies."
    ),
    "Scenarios 01 and 02 differ in how strictly": (
        "[윤성] Scenarios 01 and 02 differ only in how strictly they define a "
        'partial-reporting state. For each state, average missingness was computed as '
        'the mean proportion of not-reported ("-9") values across the 20 predictor '
        "variables. The across-state distribution had a mean of 0.080 and an SD of "
        "0.083. Scenario 01 classified states as partial-reporting when their average "
        "missingness was at or above mean + 1 SD (0.163), resulting in six states: "
        "Arizona, Wisconsin, Washington, Maryland, Louisiana, and California. Scenario "
        "02 used the stricter mean + 2 SD threshold (0.245), resulting in three states: "
        "Arizona, Wisconsin, and Washington. In both scenarios, models were trained on "
        "comprehensive-reporting states and evaluated on held-out partial-reporting "
        "states to assess cross-state generalization. Here, standard deviation refers "
        "to the SD of the across-state average-missingness distribution, not to "
        "variation within a single variable."
    ),
}


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(t.text or "" for t in paragraph.findall(".//w:t", NS)).strip()


def find_body_rpr(paragraph: ET.Element) -> ET.Element:
    for run in paragraph.findall("w:r", NS):
        text = "".join(t.text or "" for t in run.findall(".//w:t", NS)).strip()
        if text and text not in {"[", "윤성", "]", "[윤성]"}:
            rpr = run.find("w:rPr", NS)
            if rpr is not None:
                return copy.deepcopy(rpr)
    rpr = ET.Element(f"{{{W_NS}}}rPr")
    rfonts = ET.SubElement(rpr, f"{{{W_NS}}}rFonts")
    for key in ("ascii", "eastAsia", "hAnsi", "cs"):
        rfonts.set(f"{{{W_NS}}}{key}", "Aptos")
    color = ET.SubElement(rpr, f"{{{W_NS}}}color")
    color.set(f"{{{W_NS}}}val", "000000")
    sz = ET.SubElement(rpr, f"{{{W_NS}}}sz")
    sz.set(f"{{{W_NS}}}val", "22")
    sz_cs = ET.SubElement(rpr, f"{{{W_NS}}}szCs")
    sz_cs.set(f"{{{W_NS}}}val", "22")
    return rpr


def replace_paragraph_text(paragraph: ET.Element, new_text: str) -> None:
    ppr = paragraph.find("w:pPr", NS)
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)

    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    run.append(copy.deepcopy(BODY_RPR))
    text = ET.SubElement(run, f"{{{W_NS}}}t")
    text.set(f"{{{XML_NS}}}space", "preserve")
    text.text = new_text


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)

    if not BACKUP_PATH.exists():
        shutil.copy2(DOCX_PATH, BACKUP_PATH)

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        document_xml = zin.read("word/document.xml")
        all_files = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    text = document_xml.decode("utf-8", errors="ignore")
    for prefix, uri in re.findall(r'xmlns(?::([^=]+))?="([^"]+)"', text):
        ET.register_namespace(prefix, uri)

    root = ET.fromstring(document_xml)

    global BODY_RPR
    BODY_RPR = None
    for paragraph in root.findall(".//w:p", NS):
        p_text = paragraph_text(paragraph)
        if "[윤성]" in p_text and len(p_text) > len("[윤성]"):
            for run in paragraph.findall("w:r", NS):
                run_text = "".join(t.text or "" for t in run.findall(".//w:t", NS)).strip()
                rpr = run.find("w:rPr", NS)
                if run_text.startswith("We ") or run_text.startswith("State-") or run_text.startswith("Hyper"):
                    if rpr is not None:
                        BODY_RPR = copy.deepcopy(rpr)
                        break
        if BODY_RPR is not None:
            break
    if BODY_RPR is None:
        BODY_RPR = find_body_rpr(root.find(".//w:p", NS))

    replaced = {}
    for paragraph in root.findall(".//w:p", NS):
        p_text = paragraph_text(paragraph)
        if "[윤성]" not in p_text or p_text.strip() == "[윤성]":
            continue
        normalized = p_text.replace("[윤성] ", "").replace("[윤성]", "").strip()
        for old_start, new_text in REPLACEMENTS.items():
            if normalized.startswith(old_start):
                replace_paragraph_text(paragraph, new_text)
                replaced[old_start] = replaced.get(old_start, 0) + 1
                break

    missing = [key for key in REPLACEMENTS if replaced.get(key, 0) != 1]
    if missing:
        raise RuntimeError(f"Expected one replacement for each key; problematic keys: {missing}; counts={replaced}")

    all_files["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=".") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name, data in all_files.items():
                zout.writestr(name, data)
        with zipfile.ZipFile(tmp_path, "r") as ztest:
            bad = ztest.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member after writing: {bad}")
        tmp_path.replace(DOCX_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("Updated", DOCX_PATH)
    print("Backup", BACKUP_PATH)
    for key in REPLACEMENTS:
        print("replaced:", key)


if __name__ == "__main__":
    main()

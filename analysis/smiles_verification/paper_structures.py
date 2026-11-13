"""Extract every drawn structure from source PDFs: MolDetect finds molecule and label boxes on each
page (with molecule-label pairing), MolScribe reads each molecule, EasyOCR reads each label.

Runs in the detection environment (MolDetect + MolScribe, torch 1.x), not in the project venv:
    /ai/workspace/tools/ocsr/detect-venv/bin/python analysis/smiles_verification/paper_structures.py \
        --pdf-dir <dir> --out-dir <dir> [--only <doi-stem> ...] [--dpi 200]

Writes one JSON per paper: {"pdf", "pages": [{"page", "molecules": [{"bbox", "smiles", "labels"}]}]}.
Existing outputs are skipped, so the run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download  # noqa: F401  (imported by rxnscribe; kept explicit)
from molscribe import MolScribe
from PIL import Image
from rxnscribe import MolDetect

MOLDETECT = "/ai/workspace/tools/ocsr/moldetect_coref_best_hf.ckpt"
MOLSCRIBE = "/ai/workspace/tools/ocsr/swin_base_char_aux_1m680k.pth"


class LocalMolDetect(MolDetect):
    """MolDetect that loads the local MolScribe checkpoint instead of downloading another one."""

    def get_molscribe(self):
        return MolScribe(MOLSCRIBE, device=self.device)


def page_images(pdf: Path, dpi: int, tmp: Path) -> list[Path]:
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(tmp / "p")], check=True, capture_output=True)
    return sorted(tmp.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1]))


def extract(model: LocalMolDetect, pdf: Path, dpi: int) -> dict:
    out = {"pdf": pdf.name, "dpi": dpi, "pages": []}
    with tempfile.TemporaryDirectory() as d:
        for img in page_images(pdf, dpi, Path(d)):
            page = int(img.stem.split("-")[-1])
            pred = model.predict_image(Image.open(img).convert("RGB"), molscribe=True, coref=True, ocr=True)
            boxes = pred["bboxes"]
            labels: dict[int, list[str]] = {}
            for mol_i, idt_i in pred.get("corefs", []):
                text = boxes[idt_i].get("text") if idt_i < len(boxes) else None
                if text:
                    labels.setdefault(mol_i, []).extend(text if isinstance(text, list) else [text])
            mols = [{"bbox": [round(v, 4) for v in b["bbox"]], "smiles": b.get("smiles"), "labels": labels.get(i, [])}
                    for i, b in enumerate(boxes) if b.get("category") == "[Mol]"]
            if mols:
                out["pages"].append({"page": page, "molecules": mols})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="PDF stems (DOI with / replaced by _)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model = LocalMolDetect(MOLDETECT, device=torch.device("cpu"), coref=True)
    pdfs = sorted(a.pdf_dir.glob("*.pdf"))
    if a.only:
        pdfs = [p for p in pdfs if p.stem in set(a.only)]
    for pdf in pdfs:
        target = a.out_dir / f"{pdf.stem}.json"
        if target.exists():
            continue
        started = time.time()
        try:
            result = extract(model, pdf, a.dpi)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next paper
            result = {"pdf": pdf.name, "error": f"{type(exc).__name__}: {exc}"}
        result["seconds"] = round(time.time() - started, 1)
        target.write_text(json.dumps(result))
        n = sum(len(p["molecules"]) for p in result.get("pages", []))
        print(f"{pdf.stem}: {n} molecules, {result['seconds']} s{'  ERROR ' + result['error'] if 'error' in result else ''}",
              flush=True)


if __name__ == "__main__":
    main()

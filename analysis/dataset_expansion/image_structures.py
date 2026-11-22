"""MolDetect + MolScribe over figure images of one paper directory -> JSON (same layout as paper_structures.py)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "smiles_verification"))
import torch
from PIL import Image
from paper_structures import LocalMolDetect, MOLDETECT, extract
torch.set_num_threads(48)
model = LocalMolDetect(MOLDETECT, device=torch.device("cpu"), coref=True)
for d in map(Path, sys.argv[2:]):
    out = {"source": d.name, "figures": []}
    t0 = time.time()
    for img in sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif")):
        if img.suffix.lower() == ".gif" and (d / (img.stem + ".jpg")).exists():
            continue
        pred = model.predict_image(Image.open(img).convert("RGB"), molscribe=True, coref=True, ocr=True)
        boxes = pred["bboxes"]; labels = {}
        for mi, ii in pred.get("corefs", []):
            t = boxes[ii].get("text") if ii < len(boxes) else None
            if t: labels.setdefault(mi, []).extend(t if isinstance(t, list) else [t])
        mols = [{"bbox": [round(v, 4) for v in b["bbox"]], "smiles": b.get("smiles"), "labels": labels.get(i, [])}
                for i, b in enumerate(boxes) if b.get("category") == "[Mol]"]
        if mols: out["figures"].append({"figure": img.name, "molecules": mols})
    for pdf in sorted(d.glob("*.pdf")):
        r = extract(model, pdf, 200)
        out.setdefault("si_pages", {})[pdf.name] = r["pages"]
    out["seconds"] = round(time.time() - t0, 1)
    (Path(sys.argv[1]) / f"{d.name}.json").write_text(json.dumps(out))
    print(d.name, sum(len(f["molecules"]) for f in out["figures"]), "figure molecules;",
          {k: sum(len(p["molecules"]) for p in v) for k, v in out.get("si_pages", {}).items()}, out["seconds"], "s", flush=True)

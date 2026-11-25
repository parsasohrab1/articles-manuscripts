"""Download the 15 UCI archives into <dir> and write manifest.json (URL, bytes, SHA-256)."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "17": "breast+cancer+wisconsin+diagnostic", "45": "heart+disease", "519": "heart+failure+clinical+records",
    "225": "ilpd+indian+liver+patient+dataset", "43": "haberman+s+survival", "176": "blood+transfusion+service+center",
    "161": "mammographic+mass", "212": "vertebral+column", "329": "diabetic+retinopathy+debrecen", "193": "cardiotocography",
    "529": "early+stage+diabetes+risk+prediction+dataset", "96": "spectf+heart", "15": "breast+cancer+wisconsin+original",
    "277": "thoracic+surgery+data", "46": "hepatitis"}

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, name in SOURCES.items():
        url = f"https://archive.ics.uci.edu/static/public/{i}/{name}.zip"
        f = out / f"{i}.zip"
        if not f.exists():
            urllib.request.urlretrieve(url, f)
        manifest[i] = {"url": url, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

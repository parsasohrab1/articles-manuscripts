"""Download the public data used by both studies and write a manifest (source, licence, size, SHA-256)."""
import hashlib, json, sys, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(sys.argv[1]); UA = {"User-Agent": "research-client/1.0"}
def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r: return r.read()
        except Exception as e:
            if k == tries - 1 or getattr(e, "code", 0) == 404: raise
            time.sleep(5 * (k + 1))
def save(folder, name, data, source, licence, man):
    p = ROOT / folder / name; p.write_bytes(data)
    man.append({"file": name, "source": source, "licence": licence, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
def nasa(folder, name, lat, lon, start, end, freq, man):
    url = f"https://power.larc.nasa.gov/api/temporal/{freq}/point?parameters=T2M&community=RE&latitude={lat}&longitude={lon}&start={start}&end={end}&format=CSV&time-standard=UTC"
    save(folder, name, get(url), url, "NASA POWER, open data", man)
what = sys.argv[2]; man = []
if what == "hvac":
    for rec in (23099351, 23099489):
        meta = json.loads(get(f"https://zenodo.org/api/records/{rec}"))
        (ROOT / "hvac" / str(rec)).mkdir(exist_ok=True)
        for f in meta["files"]:
            url = f"https://zenodo.org/api/records/{rec}/files/{urllib.parse.quote(f['key'])}/content"
            data = get(url); assert hashlib.md5(data).hexdigest() == f["checksum"].split(":")[1], f["key"]
            save("hvac", f"{rec}/{f['key']}", data, f"https://doi.org/10.5281/zenodo.{rec}", "CC-BY-4.0", man); time.sleep(0.7)
    nasa("hvac", "nasa_t2m_fornes_hourly_2025.csv", 36.94, -3.86, "20250101", "20251231", "hourly", man)
else:
    pts = {"NL_L": ("NL-TSO-0001", "DIS-00201", 52.10, 5.18), "BE_H": ("BE-TSO-0001", "DIS-00191", 50.85, 4.35), "IT": ("IT-TSO-0001", "DIS-00005", 45.46, 9.19)}
    for key, (op, pk, lat, lon) in pts.items():
        for y in range(2022, 2026):
            q = urllib.parse.urlencode({"indicator": "Physical Flow", "operatorKey": op, "pointKey": pk, "directionKey": "exit", "from": f"{y}-01-01", "to": f"{y}-12-31", "periodType": "day", "timezone": "CET", "limit": -1}, quote_via=urllib.parse.quote)
            url = "https://transparency.entsog.eu/api/v1/operationalData?" + q
            old = ROOT / "gas" / f"entsog_{key}_{y}.json"
            try: data = old.read_bytes() if old.exists() else get(url)
            except Exception as e: print("FAIL", key, y, e); continue
            save("gas", f"entsog_{key}_{y}.json", data, url, "ENTSOG Transparency Platform, public", man)
            print(key, y, len(json.loads(data).get("operationalData", [])), flush=True)
        nasa("gas", f"nasa_t2m_{key}_daily.csv", lat, lon, "20220101", "20251231", "daily", man)
(ROOT / what / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(what, len(man), "files", round(sum(m["bytes"] for m in man) / 1e6, 2), "MB")

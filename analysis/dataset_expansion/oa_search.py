import json, time, sys, urllib.parse, urllib.request, urllib.error
Q = {
 "ESIPT": '"excited-state intramolecular proton transfer" AND (DFT OR "TD-DFT" OR calculation)',
 "AGG": '(excimer OR "J-aggregate" OR "H-aggregate" OR "exciton coupling") AND (emission OR fluorescence) AND (DFT OR "TD-DFT" OR calculation)',
 "ICT": '("twisted intramolecular charge transfer" OR "intramolecular charge transfer") AND solvatochromism AND (DFT OR "TD-DFT")',
 "PET": '"photoinduced electron transfer" AND (fluorescent OR fluorescence) AND probe AND (DFT OR "TD-DFT" OR HOMO)',
 "RAD": '("dark state" OR "radiative rate" OR "radiative decay" OR "oscillator strength") AND fluorescence AND (aggregation OR solid) AND (DFT OR "TD-DFT")',
}
def get(url):
    for wait in (2, 5, 10, 20, 40):
        time.sleep(1.2)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(wait)
def abstract(inv):
    if not inv: return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)
out = {}
for fam, q in Q.items():
    f = f"title_and_abstract.search:{q},type:article,from_publication_date:2025-07-01,is_oa:true"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f, "per-page": 25, "sort": "relevance_score:desc"})
    res = get(url)["results"]
    out[fam] = [{"doi": (w.get("doi") or "").replace("https://doi.org/", ""), "title": w.get("title"),
                 "date": w.get("publication_date"), "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                 "oa_pdf": (w.get("best_oa_location") or {}).get("pdf_url"), "license": (w.get("best_oa_location") or {}).get("license"),
                 "abstract": abstract(w.get("abstract_inverted_index"))[:1500]} for w in res]
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
for fam, ws in out.items():
    print(fam, len(ws), sum(bool(w["oa_pdf"]) for w in ws))

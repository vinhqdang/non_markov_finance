"""Verify every reference in main.tex against Crossref (DOIs) and arXiv (preprints).

For each bibliography entry the script extracts the DOI or arXiv id, fetches the
authoritative record, and compares title, year, first author, journal, volume and
pages with what the manuscript claims.  Anything that does not match is printed
as a MISMATCH; anything that cannot be resolved is printed as UNRESOLVED.

Usage:  python verify_refs.py [path/to/main.tex]
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

MAILTO = "vinh.dang@buv.edu.vn"          # Crossref asks for a contact address

# Entries that predate DOIs, or whose publisher deposits none, are verified
# against zbMATH Open instead so that every reference has an authority.
ZBMATH = {
    "dyn61": 3181533,      # Dynkin (1961), Sel. Transl. Math. Stat. Probab. 1
    "nist10": 5765058,     # NIST Handbook of Mathematical Functions (2010)
}
S = requests.Session()
S.headers.update({"User-Agent": f"ref-verify/1.0 (mailto:{MAILTO})"})


def norm(s: str) -> str:
    """Lowercase, strip accents, drop LaTeX and punctuation for comparison."""
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s or "")
    s = re.sub(r"[{}$\\]", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def parse_bib(tex: str):
    """Split the thebibliography block into entries."""
    block = tex.split(r"\begin{thebibliography}")[1].split(r"\end{thebibliography}")[0]
    parts = re.split(r"\\bibitem", block)[1:]
    out = []
    for p in parts:
        m = re.match(r"\s*\[(.*?)\]\s*\{(.*?)\}(.*)", p, re.S)
        if not m:
            continue
        label, key, body = m.groups()
        body = " ".join(body.split())
        doi = None
        dm = re.search(r"doi\.org/(10\.[^\s},]+)", body)
        if dm:
            doi = dm.group(1).rstrip(".,;")
        arx = None
        am = re.search(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5})", body)
        if am:
            arx = am.group(1)
        ym = re.search(r"\((\d{4}[a-z]?)\)", body)
        year = ym.group(1)[:4] if ym else None
        # claimed title: text between the year-parenthesis and the journal/period
        after = body.split(")", 1)[1] if ")" in body else body
        after = re.sub(r"^\s*", "", after)
        tm = re.split(r",\s*\{\\it|\.\s*(?:In:|arXiv|[A-Z])", after, maxsplit=1)
        title = tm[0].strip(" ,.") if tm else ""
        jm = re.search(r"\{\\it\s+(.*?)\}", body)
        journal = jm.group(1) if jm else None
        vm = re.search(r"\{\\bf\s+(\d+)\}", body)
        vol = vm.group(1) if vm else None
        # Page range: strip parenthesised issue numbers such as "(1--2)" first,
        # otherwise the issue is mistaken for the pagination.
        depaged = re.sub(r"\(\s*\d+\s*--\s*\d+\s*\)", " ", body)
        depaged = re.sub(r"\(\s*\d+[a-z]?\s*\)", " ", depaged)
        pm = re.search(r"(\d+)--(\d+)", depaged)
        pages = (pm.group(1), pm.group(2)) if pm else None
        first_author = body.split("(")[0].strip()
        out.append(dict(key=key, label=label, body=body, doi=doi, arxiv=arx,
                        year=year, title=title, journal=journal, vol=vol,
                        pages=pages, first_author=first_author))
    return out


def crossref(doi):
    r = S.get(f"https://api.crossref.org/works/{doi}", timeout=30)
    if r.status_code != 200:
        return None
    return r.json()["message"]


def zbmath(zid):
    r = S.get(f"https://api.zbmath.org/v1/document/{zid}", timeout=30)
    if r.status_code != 200:
        return None
    d = r.json().get("result") or {}
    if isinstance(d, list):
        d = d[0] if d else {}
    if not d:
        return None
    src = d.get("source") or {}
    return dict(title=(d.get("title") or {}).get("title", ""),
                year=str(d.get("year") or ""),
                authors=[a.get("name", "")
                         for a in (d.get("contributors") or {}).get("authors", [])],
                pages=src.get("pages"),
                source=src.get("source", ""))


def arxiv(aid):
    r = S.get("http://export.arxiv.org/api/query",
              params={"id_list": aid, "max_results": 1}, timeout=30)
    if r.status_code != 200:
        return None
    x = r.text
    t = re.search(r"<entry>.*?<title>(.*?)</title>", x, re.S)
    a = re.findall(r"<author>\s*<name>(.*?)</name>", x, re.S)
    d = re.search(r"<published>(\d{4})", x)
    if not t:
        return None
    return dict(title=" ".join(t.group(1).split()), authors=a,
                year=d.group(1) if d else None)


def main():
    tex = Path(sys.argv[1] if len(sys.argv) > 1 else "main.tex").read_text()
    entries = parse_bib(tex)
    print(f"parsed {len(entries)} bibliography entries\n")
    problems, checked, unresolved = [], 0, []

    for e in entries:
        tag = f"[{e['key']}]"
        if e["doi"]:
            time.sleep(0.3)
            m = crossref(e["doi"])
            if m is None:
                print(f"{tag:12s} UNRESOLVED  DOI {e['doi']} not found in Crossref")
                problems.append((e["key"], f"DOI {e['doi']} not found"))
                continue
            checked += 1
            ctitle = (m.get("title") or [""])[0]
            cyear = None
            for f in ("published-print", "published-online", "issued"):
                if m.get(f, {}).get("date-parts", [[None]])[0][0]:
                    cyear = str(m[f]["date-parts"][0][0])
                    break
            cjournals = m.get("container-title") or []
            cvol = m.get("volume")
            cpage = m.get("page")
            cauth = m.get("author", [{}])[0].get("family", "")
            r = ratio(e["title"], ctitle)
            issues = []
            if r < 0.90:
                issues.append(f"title {r:.2f}\n      ours : {e['title'][:90]}"
                              f"\n      theirs: {ctitle[:90]}")
            if e["year"] and cyear and e["year"] != cyear:
                issues.append(f"year ours={e['year']} theirs={cyear}")
            if cauth and norm(cauth) not in norm(e["first_author"]):
                issues.append(f"first author ours='{e['first_author'][:40]}' "
                              f"theirs='{cauth}'")
            # every surname Crossref lists must appear in our author string
            ours_auth = norm(e["first_author"])
            # Crossref sometimes packs given names into `family`
            # (e.g. "Craig MacKinlay"), so match on the final surname token.
            missing = [a.get("family", "") for a in m.get("author", [])
                       if a.get("family")
                       and norm(a["family"]).split()[-1] not in ours_auth]
            if missing:
                issues.append(f"authors missing from our entry: {missing}")
            # A journal name may legitimately be abbreviated or be the series
            # title of a book chapter, so accept containment either way.
            if e["journal"] and cjournals:
                a = norm(e["journal"])
                ok = any(a in norm(c) or norm(c) in a or ratio(a, c) >= 0.60
                         for c in cjournals)
                if not ok:
                    issues.append(f"journal ours='{e['journal']}' "
                                  f"theirs={cjournals}")
            if e["vol"] and cvol and e["vol"] != str(cvol):
                issues.append(f"volume ours={e['vol']} theirs={cvol}")
            if e["pages"] and cpage:
                cp = cpage.replace("\u2013", "-").split("-")
                if cp[0].strip() != e["pages"][0]:
                    issues.append(f"pages ours={e['pages'][0]}-{e['pages'][1]} "
                                  f"theirs={cpage}")
            if issues:
                print(f"{tag:12s} MISMATCH  {e['doi']}")
                for i in issues:
                    print(f"    - {i}")
                problems.append((e["key"], "; ".join(i.split(chr(10))[0] for i in issues)))
            else:
                print(f"{tag:12s} OK  {cauth} ({cyear}) {ctitle[:62]}")
        elif e["arxiv"]:
            time.sleep(0.3)
            m = arxiv(e["arxiv"])
            if m is None:
                print(f"{tag:12s} UNRESOLVED  arXiv:{e['arxiv']} not found")
                problems.append((e["key"], f"arXiv {e['arxiv']} not found"))
                continue
            checked += 1
            r = ratio(e["title"], m["title"])
            issues = []
            if r < 0.90:
                issues.append(f"title {r:.2f}\n      ours : {e['title'][:90]}"
                              f"\n      theirs: {m['title'][:90]}")
            fam = m["authors"][0].split()[-1] if m["authors"] else ""
            if fam and norm(fam) not in norm(e["first_author"]):
                issues.append(f"first author ours='{e['first_author'][:40]}' "
                              f"theirs='{m['authors'][0]}'")
            ours_auth = norm(e["first_author"])
            missing = [a for a in m["authors"]
                       if norm(a.split()[-1]) not in ours_auth]
            if missing:
                issues.append(f"authors missing from our entry: {missing}")
            # initials must match too: compare "F. Surname" tokens
            for a in m["authors"]:
                parts = a.split()
                if len(parts) >= 2:
                    ini, sur = norm(parts[0])[:1], norm(parts[-1])
                    if sur in ours_auth and f"{ini} {sur}" not in ours_auth:
                        issues.append(f"initial for '{a}' does not match our entry")
            if issues:
                print(f"{tag:12s} MISMATCH  arXiv:{e['arxiv']}")
                for i in issues:
                    print(f"    - {i}")
                problems.append((e["key"], "; ".join(i.split(chr(10))[0] for i in issues)))
            else:
                print(f"{tag:12s} OK  arXiv:{e['arxiv']} {m['title'][:58]}")
        elif e["key"] in ZBMATH:
            time.sleep(0.3)
            m = zbmath(ZBMATH[e["key"]])
            if m is None:
                print(f"{tag:12s} UNRESOLVED  zbMATH {ZBMATH[e['key']]} not found")
                problems.append((e["key"], f"zbMATH {ZBMATH[e['key']]} not found"))
                continue
            checked += 1
            issues = []
            r = ratio(e["title"], m["title"])
            if r < 0.90:
                issues.append(f"title {r:.2f}\n      ours : {e['title'][:90]}"
                              f"\n      theirs: {m['title'][:90]}")
            if e["year"] and m["year"] and e["year"] != m["year"]:
                issues.append(f"year ours={e['year']} theirs={m['year']}")
            if e["pages"] and m["pages"]:
                if m["pages"].split("-")[0].strip() != e["pages"][0]:
                    issues.append(f"pages ours={e['pages'][0]}-{e['pages'][1]} "
                                  f"theirs={m['pages']}")
            if issues:
                print(f"{tag:12s} MISMATCH  zbMATH {ZBMATH[e['key']]}")
                for i in issues:
                    print(f"    - {i}")
                problems.append((e["key"], "; ".join(i.split(chr(10))[0] for i in issues)))
            else:
                print(f"{tag:12s} OK  zbMATH {ZBMATH[e['key']]} ({m['year']}) "
                      f"{m['title'][:52]}")
        else:
            print(f"{tag:12s} NO IDENTIFIER  (verify by hand)")
            unresolved.append(e["key"])

    print(f"\n{'='*72}")
    print(f"checked against an authority: {checked}/{len(entries)}")
    print(f"no identifier (manual check) : {len(unresolved)}  {unresolved}")
    print(f"problems                     : {len(problems)}")
    for k, why in problems:
        print(f"  - {k}: {why}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

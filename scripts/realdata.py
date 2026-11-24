import csv, gzip, io, json, os, random, resource, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("REALDATA_DIR", "realdata")).resolve(); ROOT.mkdir(exist_ok=True)
S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
CHEM = {
    "bace": ("bace.csv", "mol", "CID", "Model"),
    "bbbp": ("BBBP.csv", "smiles", None, None),
    "clintox": ("clintox.csv.gz", "smiles", None, None),
    "esol": ("delaney-processed.csv", "smiles", None, None),
    "lipo": ("Lipophilicity.csv", "smiles", "CMPD_CHEMBLID", None),
    "hiv": ("HIV.csv", "smiles", None, None),
}

def fetch(url, dest):
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if url.endswith(".gz"):
            data = gzip.decompress(data)
        dest.write_bytes(data)
    return dest

def prepare_chem(name, spec):
    fname, smiles_col, id_col, split_col = spec
    raw = fetch(S3 + fname, ROOT / (name + "-raw.csv"))
    rows = list(csv.DictReader(io.StringIO(raw.read_text(encoding="utf-8", errors="replace"))))
    out = ROOT / f"{name}.csv"
    rng = random.Random(1)
    with out.open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["id", "smiles", "fold_random"] + (["fold_given"] if split_col else []))
        seen = set()
        for i, row in enumerate(rows):
            ident = (row.get(id_col) or "").strip() if id_col else ""
            ident = ident or f"{name}{i:06d}"
            while ident in seen: ident += "_dup"
            seen.add(ident)
            fold = "test" if rng.random() < 0.2 else "train"
            given = []
            if split_col:
                g = row[split_col].strip().lower()
                given = ["test" if g in {"test", "valid"} else "train"]
            w.writerow([ident, row[smiles_col], fold] + given)
    return out, len(rows), bool(split_col)

def prepare_protein():
    url = "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,sequence,organism_id,date_created&query=reviewed:true+AND+organism_id:9606"
    raw = fetch(url, ROOT / "human-raw.tsv")
    rows = list(csv.DictReader(io.StringIO(raw.read_text()), delimiter="\t"))
    out = ROOT / "human.csv"; rng = random.Random(1)
    with out.open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["id", "sequence", "created", "fold_random"])
        for row in rows:
            w.writerow([row["Entry"], row["Sequence"], row["Date of creation"], "test" if rng.random() < 0.2 else "train"])
    return out, len(rows)

def policy(path, dataset, constraint, extra=""):
    p = ROOT / f"{path.stem}-{constraint['metric']}.yaml"
    body = {
        "splitproof": 1,
        "dataset": {"path": str(path), "id": "id", "similarity": {"s": dataset}},
        "scheme": {"kind": "holdout", "fractions": {"train": 0.8, "test": 0.2}},
        "constraints": [dict(kind="similarity", key="s", **constraint)],
        "seed": 1,
    }
    p.write_text(json.dumps(body))
    return p

def run(label, args, timeout=3600):
    start = time.time()
    cmd = [sys.executable, "-m", "splitproof.cli"] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT, check=False)
    except subprocess.TimeoutExpired:
        return {"label": label, "exit": "timeout", "seconds": timeout}
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    err = r.stderr
    out = {"label": label, "exit": r.returncode, "seconds": round(time.time() - start, 1), "rss_mb": rss // 1024,
           "traceback": "Traceback" in r.stderr or "UNEXPECTED" in r.stdout + r.stderr,
           "stderr": err[-1500:], "stdout": r.stdout[-1500:]}
    print(json.dumps({k: v for k, v in out.items() if k not in ("stdout",)}), flush=True)
    return out

results = []
which = sys.argv[1:] or list(CHEM) + ["human"]
for name in which:
    if name == "human":
        path, n = prepare_protein(); given = False
        metrics = [{"metric": "kmer_jaccard", "threshold": 0.3}, {"metric": "identity", "threshold": 0.3}]
        ds = {"column": "sequence", "kind": "protein"}
    else:
        path, n, given = prepare_chem(name, CHEM[name])
        metrics = [{"metric": "canonical_smiles", "threshold": 1.0}, {"metric": "ring_systems", "threshold": 1.0}, {"metric": "tanimoto", "threshold": 0.8}]
        ds = {"column": "smiles", "kind": "molecule"}
    print(f"### {name}: {n} rows", flush=True)
    for m in metrics:
        pol = policy(path, ds, m)
        tag = f"{name}/{m['metric']}"
        a = run(tag + " audit random", ["audit", str(pol), "--split-column", "fold_random", "--json", "--cert", f"{tag.replace('/','-')}-ar.json"])
        if given:
            run(tag + " audit given", ["audit", str(pol), "--split-column", "fold_given", "--json"])
        s = run(tag + " split", ["split", str(pol), "--out", f"{tag.replace('/','-')}.split.csv", "--cert", f"{tag.replace('/','-')}.cert.json", "--json", "--tolerance", "0.5"])
        if s.get("exit") in (0, 1):
            run(tag + " verify", ["verify", f"{tag.replace('/','-')}.cert.json", "--split", f"{tag.replace('/','-')}.split.csv", "--json"])
            run(tag + " verify audit cert", ["verify", f"{tag.replace('/','-')}-ar.json", "--split-column", "fold_random", "--json"]) if a.get("exit") in (0, 1) else None

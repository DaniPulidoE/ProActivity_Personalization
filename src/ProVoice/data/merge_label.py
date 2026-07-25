#   python data/merge_label.py --in data/with_segments.jsonl --labels data/labels.csv --out data/labeled_data.jsonl
import argparse, json, pathlib, csv

LEVELS = [f"Level_{i}" for i in range(1,6)]

def parse_labels_csv(path: str):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Strictly accept only the five Level_* columns (loa and label fields are no longer accepted)
        miss = [k for k in (["segment_id"]+LEVELS) if k not in r.fieldnames]
        if miss:
            raise ValueError(f"labels.csv is missing columns: {miss}, required columns: ['segment_id']+Level_1..Level_5")
        for row in r:
            sid = str(row.get("segment_id","")).strip()
            if not sid: continue
            vec = [int(float(row[k])) for k in LEVELS]
            # Normalise to 0/1
            vec = [1 if v>=1 else 0 for v in vec]
            m[sid] = vec
    if not m:
        raise SystemExit("labels.csv is empty.")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",     dest="in_jsonl", required=True)
    ap.add_argument("--labels", dest="labels_csv", required=True)
    ap.add_argument("--out",    dest="out_jsonl", required=True)
    args = ap.parse_args()

    seg2vec = parse_labels_csv(args.labels_csv)
    outp = pathlib.Path(args.out_jsonl); outp.parent.mkdir(parents=True, exist_ok=True)

    cnt_all = 0; cnt_labeled = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Error] Failed to parse JSON line: {e}")
                continue
            cnt_all += 1
            sid = str(obj.get("segment_id","")).strip()
            vec = seg2vec.get(sid)
            if vec:
                for i,k in enumerate(LEVELS):
                    obj[k] = int(vec[i])
                cnt_labeled += 1
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Output to {outp}；All frames {cnt_all}，Labeled frames {cnt_labeled}")

if __name__ == "__main__":
    main()
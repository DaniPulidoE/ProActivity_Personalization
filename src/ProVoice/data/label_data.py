
# =========================================
# file: make_labels_template.py
# Purpose: export annotation template CSV from a JSONL file that contains segment_id
# Usage:
#   python data/label_data.py --in data/with_segments.jsonl --out data/labels.csv
# You can fill in only loa (0..4), or fill in Level_1..Level_5 directly (0/1, multi-hot)
# =========================================
import argparse, json, pathlib, csv

LEVELS = [f"Level_{i}" for i in range(1,6)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_jsonl", required=True)
    ap.add_argument("--out", dest="out_csv",  required=True)
    args = ap.parse_args()

    segs = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except Exception as e:                
                print(f"[Error] Failed to parse JSON line: {e}")  
                continue
            sid = str(obj.get("segment_id","")).strip()
            if sid:
                segs.append(sid)
    segs = sorted(set(segs))
    if not segs:
        raise SystemExit("No segment_id。Run gen_segments.py first。")

    outp = pathlib.Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", *LEVELS])
        for sid in segs:
            w.writerow([sid, 0,0,0,0,0])
    print(f"[OK] Template written to：{outp}（Please fill in the 0/1 with your multi-level labels）")

if __name__ == "__main__":
    main()

#   python data/generate_id.py --in data/raw_data.jsonl --out data/with_segments.jsonl --chunk 500
#   (--chunk=0 disables frame-count-based segmentation; segment index only increments on composite-key changes)
import argparse, json, pathlib

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_jsonl", required=True)
    ap.add_argument("--out", dest="out_jsonl", required=True)
    ap.add_argument("--chunk", type=int, default=0, help="Maximum frames per segment; 0 = do not split by frame count")
    # Composite key fields (configurable)
    ap.add_argument("--keys", default="participantid,environment,secondary_task,functionname",
                    help="Comma-separated field names used to build the segment key")
    ap.add_argument("--sep", default="|", help="Composite key separator")
    return ap.parse_args()

def main():
    args = parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    outp = pathlib.Path(args.out_jsonl); outp.parent.mkdir(parents=True, exist_ok=True)

    cur_key = None
    seg_idx = 0
    count_in_seg = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Error] Failed to parse JSON line: {e}")
                continue

            # Composite key
            key_vals = [str(obj.get(k, "")).strip() for k in keys]
            key = args.sep.join(key_vals)

            # Segmentation condition: composite key changes or chunk size limit reached
            if key != cur_key or (args.chunk > 0 and count_in_seg >= args.chunk):
                cur_key = key
                seg_idx += 1
                count_in_seg = 0

            count_in_seg += 1
            seg_id = f"{key}{args.sep}seg{seg_idx:03d}"
            obj["segment_id"] = seg_id

            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote -> {outp}")

if __name__ == "__main__":
    main()


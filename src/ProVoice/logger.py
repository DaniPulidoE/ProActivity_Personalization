from __future__ import annotations
import json, csv, os
from typing import Any, Dict, List

# Fixed schema for decisions.csv. New keys from any strategy or data_collector
# are silently dropped (extrasaction='ignore'); missing keys are written as ''.
# Extend this list when genuinely new columns are added — never derive it from
# a live row, which caused the header/column misalignment bug (C6).
DECISION_COLUMNS: List[str] = [
    "timestamp", "session_id", "participantid",
    "functionname", "environment", "secondary_task",
    "modeltype", "state_model", "w_fcd",
    "action", "level", "LoA", "message",
    "probs", "profile", "fcd",
    "fallback", "fallback_reason", "sub",
    "emotion", "hr_delta", "rr_delta",
]


class Logger:
    def __init__(self, raw_data_file: str = "./data/raw_data.jsonl",
                 processed_data_file: str = "./data/decisions.csv") -> None:
        self.raw_data_file = raw_data_file
        self.processed_data_file = processed_data_file

    def log_raw(self, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.raw_data_file) or ".", exist_ok=True)
            with open(self.raw_data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data or {}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logger] failed: {e}")

    def _flatten_for_csv(self, result: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = dict(result or {})
        if isinstance(row.get("probs"), (list, tuple)):
            row["probs"] = ",".join(str(float(x)) for x in row["probs"])
        for k in list(row.keys()):
            if isinstance(row[k], (dict, list)):
                try:
                    row[k] = json.dumps(row[k], ensure_ascii=False)
                except Exception:
                    row[k] = str(row[k])
        return row

    def log_processed(self, result: Dict[str, Any] | Any) -> None:
        try:
            os.makedirs(os.path.dirname(self.processed_data_file) or ".", exist_ok=True)
            is_new = not os.path.exists(self.processed_data_file) or os.path.getsize(self.processed_data_file) == 0
            with open(self.processed_data_file, "a", newline="", encoding="utf-8") as f:
                if isinstance(result, dict):
                    row = self._flatten_for_csv(result)
                    writer = csv.DictWriter(
                        f, fieldnames=DECISION_COLUMNS,
                        extrasaction='ignore', restval='',
                    )
                    if is_new:
                        writer.writeheader()
                    writer.writerow(row)
                else:
                    csv.writer(f).writerow([str(result)])
        except Exception as e:
            print(f"[Logger] Failed to write processed data: {e}")

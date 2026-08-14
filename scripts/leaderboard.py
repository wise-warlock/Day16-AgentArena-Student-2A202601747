#!/usr/bin/env python3
"""Bảng xếp hạng Agent Arena — cột chính là KHOẢNG CÁCH, không phải tổng điểm.

    python3 scripts/leaderboard.py runs/*.json
    python3 scripts/leaderboard.py runs/ --baseline-file runs/blind-dump.json
    python3 scripts/leaderboard.py runs/ --require-gap 20

TẠI SAO KHOẢNG CÁCH, KHÔNG PHẢI TỔNG ĐIỂM
=========================================

Một vòng thi có thể chạy hoàn hảo và KHÔNG ĐO ĐƯỢC GÌ. Chuyện đó xảy ra
khi một harness KHÔNG SUY LUẬN GÌ — chỉ dán lại nguyên văn những dòng nó
vừa lấy về — đạt xấp xỉ điểm của một agent trung thực. Khi ấy mọi
assertion vẫn xanh: thang điểm vẫn đơn điệu, cổng trace vẫn qua, kiểm tra
conformance vẫn rỗng. Không có gì báo động, vì "brief khó đúng mức" và
"cuộc thi hỏng" trông giống hệt nhau trên bảng tổng điểm.

Thứ phân biệt được hai tình huống đó chỉ có một: KHOẢNG CÁCH giữa bài nộp
và mốc "dán mù" (blind dump) đó. Vì vậy bảng này luôn in GAP, và
`--require-gap` là điều kiện nghiệm thu, không phải tổng điểm.

Với bạn, GAP là công cụ tự kiểm tra: dựng mốc của chính mình bằng
`run_practice.py --layers none --tag baseline` rồi so bài có đủ năm lớp
với nó. Nếu năm lớp không kéo được khoảng cách nào so với baseline thì
chúng chưa thực sự làm gì, dù tổng điểm nhìn có cao đến đâu.

    GAP >= 20  ->  CÓ GRADIENT: công sức thật sự được trả công.
    10..20     ->  YẾU: xem lại brief trước khi tính điểm chính thức.
    < 10       ->  KHÔNG CÓ GRADIENT: vòng thi không đo được gì. DỪNG.

MỐC "DÁN MÙ" LẤY Ở ĐÂU
======================

Theo thứ tự ưu tiên: `--baseline-total`, rồi `--baseline-file`, rồi bất kỳ
file điểm nào tự đánh dấu `"baseline": true` (dùng
`run_practice.py --tag baseline`), rồi cuối cùng là hằng số đã đo trong
`MEASURED_BLIND_DUMP` — kèm cảnh báo, vì con số đó phụ thuộc vào bộ brief
và phải được đo lại trên đúng endpoint sẽ dùng.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#: Khoảng cách tối thiểu để coi là vòng thi có gradient.
GAP_TARGET = 20.0
GAP_WEAK = 10.0

#: Mốc "dán mù" ĐÃ ĐO trên BỘ BRIEF CÔNG KHAI. ĐÂY LÀ ƯỚC LƯỢNG, không
#: phải phép đo trên máy bạn — luôn ưu tiên một file baseline thật, dựng
#: bằng `run_practice.py --layers none --tag baseline`.
#:
#:   public  87.30  blind line-dump trên bộ brief công khai. Con số cao như
#:                  vậy CHÍNH LÀ lý do bảng luyện tập chỉ mang tính tham
#:                  khảo: đáp án của bộ công khai nằm ngay trong corpus,
#:                  nên dán mù cũng gần đầy điểm. Ở vòng chấm điểm (bộ
#:                  brief riêng, model thật) cùng đòn tấn công đó rơi
#:                  xuống sát sàn abstain — mốc ấy do giảng viên đo trên
#:                  đúng endpoint của vòng thi, không có sẵn ở đây.
MEASURED_BLIND_DUMP = {"public": 87.30}
DEFAULT_BLIND_DUMP = 87.30

#: Cờ đáng để giảng viên xem lại bằng mắt (không trừ điểm tự động).
REVIEW_FLAGS = {
    "single_model_call": "chỉ 1 lần gọi model",
    "no_final_output": "không có FINAL đọc được",
    "blocked_emits": "cố tự ghi model_call/tool_call",
    "synthesised_abstain": "runner phải tự sinh abstain",
    "aborted": "bị runner cắt (trần thời gian/số lời gọi)",
    "error": "code sinh viên ném lỗi",
    "no_submit_event": "không có sự kiện submit",
}


def load_entries(paths) -> list:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    entries = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠ bỏ qua {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict) or "runs" not in payload:
            print(f"⚠ bỏ qua {path}: không phải file điểm arena", file=sys.stderr)
            continue
        payload["_path"] = str(path)
        entries.append(payload)
    return entries


def entry_flags(entry: dict) -> list:
    seen = []
    for run in entry.get("runs", []):
        for flag in run.get("flags", []) or []:
            key = str(flag).split(":", 1)[0]
            if key in REVIEW_FLAGS and key not in seen:
                seen.append(key)
        if run.get("error") and "error" not in seen:
            seen.append("error")
        if not run.get("gate_passed", True) and "gate_failed" not in seen:
            seen.append("gate_failed")
    return seen


def mean_total(entry: dict) -> float:
    if isinstance(entry.get("mean_total"), (int, float)):
        return float(entry["mean_total"])
    totals = [
        run.get("total", 0.0)
        for run in entry.get("runs", [])
        if isinstance(run.get("total"), (int, float))
    ]
    return sum(totals) / len(totals) if totals else 0.0


def resolve_baseline(entries, args) -> tuple:
    """(giá trị, nguồn, đáng tin cậy)."""
    if args.baseline_total is not None:
        return float(args.baseline_total), "--baseline-total", True
    if args.baseline_file:
        loaded = load_entries([args.baseline_file])
        if not loaded:
            raise SystemExit(f"Không đọc được baseline: {args.baseline_file}")
        return mean_total(loaded[0]), f"file {args.baseline_file}", True
    declared = [e for e in entries if e.get("baseline") or e.get("kind") == "baseline"]
    if declared:
        best = max(declared, key=mean_total)
        return mean_total(best), f"file tự khai baseline ({best['_path']})", True
    brief_set = next(
        (e.get("brief_set") for e in entries if e.get("brief_set")), "public"
    )
    value = MEASURED_BLIND_DUMP.get(brief_set, DEFAULT_BLIND_DUMP)
    return value, f"hằng số đã đo cho bộ {brief_set!r}", False


def verdict(gap: float) -> tuple:
    if gap >= GAP_TARGET:
        return "CÓ GRADIENT", "✔"
    if gap >= GAP_WEAK:
        return "YẾU", "~"
    return "KHÔNG CÓ GRADIENT", "✘"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Xếp hạng các file điểm arena theo KHOẢNG CÁCH so với mốc dán mù.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="file điểm JSON hoặc thư mục chứa chúng")
    parser.add_argument("--baseline-total", type=float, default=None)
    parser.add_argument("--baseline-file", default=None)
    parser.add_argument("--require-gap", type=float, default=None,
                        help="thoát khác 0 nếu bài dẫn đầu không cách mốc đủ xa")
    parser.add_argument("--strict-baseline", action="store_true",
                        help="thoát khác 0 nếu phải dùng hằng số ước lượng")
    parser.add_argument("--json", action="store_true", help="in JSON thay vì bảng")
    args = parser.parse_args(argv)

    entries = load_entries(args.paths)
    if not entries:
        print("Không có file điểm nào để xếp hạng.", file=sys.stderr)
        return 2

    baseline, source, trusted = resolve_baseline(entries, args)
    ranked = sorted(entries, key=mean_total, reverse=True)

    rows = []
    for rank, entry in enumerate(ranked, start=1):
        total = mean_total(entry)
        gap = total - baseline
        label, mark = verdict(gap)
        rows.append(
            {
                "rank": rank,
                "entry_id": entry.get("entry_id") or Path(entry["_path"]).stem,
                "path": entry["_path"],
                "brief_set": entry.get("brief_set", "?"),
                "model": entry.get("model", "?"),
                "mean_total": round(total, 2),
                "baseline": round(baseline, 2),
                "gap": round(gap, 2),
                "verdict": label,
                "is_baseline": bool(entry.get("baseline") or entry.get("kind") == "baseline"),
                "advisory": bool(entry.get("advisory")),
                "flags": entry_flags(entry),
                "n_runs": entry.get("n_runs", len(entry.get("runs", []))),
                "mark": mark,
            }
        )

    payload = {
        "baseline_total": round(baseline, 2),
        "baseline_source": source,
        "baseline_trusted": trusted,
        "gap_target": GAP_TARGET,
        "entries": rows,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("=" * 96)
        print("AGENT ARENA — BẢNG XẾP HẠNG (cột quyết định là GAP, không phải TỔNG)")
        print(f"  mốc dán mù : {baseline:.2f}   (nguồn: {source})")
        if not trusted:
            print("  ⚠ MỐC NÀY LÀ ƯỚC LƯỢNG. Hãy chạy một harness dán-mù thật trên đúng")
            print("    endpoint sẽ dùng rồi truyền vào bằng --baseline-file.")
        print(f"  ngưỡng     : GAP >= {GAP_TARGET:.0f} là có gradient; < {GAP_WEAK:.0f} là vòng thi không đo được gì")
        print("=" * 96)
        print(f"{'#':>2}  {'bài nộp':<26}{'TỔNG':>8}{'GAP':>9}   {'kết luận':<20}cờ")
        print("-" * 96)
        for row in rows:
            flags = ", ".join(REVIEW_FLAGS.get(f, f) for f in row["flags"])
            name = row["entry_id"] + (" [mốc]" if row["is_baseline"] else "")
            print(
                f"{row['rank']:>2}  {name:<26}{row['mean_total']:>8.2f}"
                f"{row['gap']:>+9.2f} {row['mark']} {row['verdict']:<20}{flags}"
            )
        print("-" * 96)
        if any(row["advisory"] for row in rows):
            print("⚠ Có bài chạy trên BỘ CÔNG KHAI: bảng đó chỉ tham khảo, không tính điểm.")
        top = rows[0] if rows else None
        if top and top["gap"] < GAP_WEAK:
            print(
                "✘ KHÔNG CÓ GRADIENT. Bài tốt nhất không hơn một harness dán mù quá "
                f"{top['gap']:.2f} điểm. Vòng chạy này sạch nhưng không đo được gì: "
                "hãy xem lại các lớp trước khi tin vào tổng điểm."
            )
        print("=" * 96)

    if args.strict_baseline and not trusted:
        print("✘ --strict-baseline: chưa có mốc dán mù thật.", file=sys.stderr)
        return 3
    if args.require_gap is not None:
        top = max((row["gap"] for row in rows if not row["is_baseline"]), default=None)
        if top is None or top < args.require_gap:
            print(
                f"✘ --require-gap {args.require_gap}: khoảng cách lớn nhất là "
                f"{top if top is not None else float('nan'):.2f}.",
                file=sys.stderr,
            )
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

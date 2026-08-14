#!/usr/bin/env python3
"""Kiểm tra lab Agent Arena — danh sách đánh số, thoát khác 0 nếu có mục hỏng.

    python3 scripts/verify.py           # ~20 giây
    python3 scripts/verify.py --full    # thêm toàn bộ pytest

Mỗi mục là một câu hỏi CÓ/KHÔNG về một thứ có thể hỏng âm thầm. Ba mục
quan trọng nhất (7, 8, 9) kiểm tra ba điều khoản provenance của runner
đóng băng — nếu một trong ba hỏng, cả lớp sẽ bị chấm NOT_FROM_MODEL mà
KHÔNG có triệu chứng nào khác: cổng trace vẫn qua, bài chạy vẫn xong,
điểm chỉ đơn giản là sai với tất cả mọi người cùng lúc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#: Băm MD5 của năm file ĐÓNG BĂNG. Sinh viên không được sửa chúng; nếu
#: một dòng nào đó thay đổi thì mọi con số đã đo trong lab này hết hiệu
#: lực. Đây cũng là mẻ kiểm tra chống gian lận rẻ nhất có thể có.
FROZEN_MD5 = {
    "arena/trace.py": "6d457f6aaa49977fe1063154b810651a",
    "arena/corpus.py": "ce30f315620e78122f054f74a8e8654c",
    "arena/tools.py": "91eda60d8fc2855f7b5354216b352f94",
    "arena/model.py": "7e71ed083122dc4f68373a1df7ed4f75",
    "arena/scorer.py": "0263ec85c6ca16e30c68d4c14425e757",
}


class Checklist:
    def __init__(self) -> None:
        self.number = 0
        self.failures: list[str] = []
        self.started = time.time()

    def check(self, title: str, fn) -> None:
        self.number += 1
        try:
            detail = fn()
            ok, note = (True, detail) if not isinstance(detail, tuple) else detail
        except Exception as exc:
            ok, note = False, f"{type(exc).__name__}: {exc}"
        mark = "✔" if ok else "✘"
        print(f"{self.number:>3}. [{mark}] {title}")
        if note:
            print(f"          {note}")
        if not ok:
            self.failures.append(f"{self.number}. {title} — {note}")

    def finish(self) -> int:
        elapsed = time.time() - self.started
        print("-" * 78)
        if self.failures:
            print(f"✘ {len(self.failures)}/{self.number} mục KHÔNG đạt ({elapsed:.1f}s):")
            for failure in self.failures:
                print(f"    {failure}")
            return 1
        print(f"✔ {self.number}/{self.number} mục đạt ({elapsed:.1f}s).")
        return 0


# ---------------------------------------------------------------------------
# Các phép kiểm tra
# ---------------------------------------------------------------------------


def check_python():
    if sys.version_info < (3, 10):
        return False, f"cần Python >= 3.10, đang chạy {sys.version.split()[0]}"
    return f"Python {sys.version.split()[0]}"


def check_frozen():
    bad = []
    for relative, expected in FROZEN_MD5.items():
        path = LAB_ROOT / relative
        if not path.exists():
            bad.append(f"{relative}: thiếu file")
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != expected:
            bad.append(f"{relative}: {digest} != {expected}")
    if bad:
        return False, "; ".join(bad)
    return f"{len(FROZEN_MD5)} file đóng băng nguyên vẹn"


def check_imports():
    import arena.runner as runner

    required = (
        "run_brief", "run_session", "score_result", "preflight",
        "assert_provenance", "normalise_output", "strip_timing",
        "RunnerConfig", "RunResult", "PreflightError",
    )
    missing = [name for name in required if not hasattr(runner, name)]
    if missing:
        return False, f"arena.runner thiếu: {missing}"
    return f"arena.runner {runner.RUNNER_VERSION}, {len(required)} tên công khai"


def check_corpus():
    from arena.corpus import Corpus

    a = Corpus.generate(seed=42)
    b = Corpus.generate(seed=42)
    if [d.doc_id for d in a.docs] != [d.doc_id for d in b.docs]:
        return False, "Corpus.generate không tất định"
    if len(a.docs) < 100:
        return False, f"chỉ có {len(a.docs)} tài liệu"
    return f"{len(a.docs)} tài liệu, tất định"


def check_briefs():
    from arena.briefs import load_public_briefs, schema_problems

    public = load_public_briefs()
    problems = [p for b in public for p in schema_problems(b)]
    if problems:
        return False, f"bộ công khai sai schema: {problems[:3]}"
    return f"{len(public)} brief công khai, đúng schema"


def _fixture():
    from arena.briefs import load_public_briefs
    from arena.corpus import Corpus

    return load_public_briefs()[0], Corpus.generate(seed=42)


def check_baseline_run():
    from arena.model import MockModel
    from arena.runner import RunnerConfig, run_brief, score_result

    brief, corpus = _fixture()
    result = run_brief(
        brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus, seed=11,
        config=RunnerConfig(),
    )
    ok, reason = result.gate()
    if not ok:
        return False, f"cổng trace hỏng: {reason}"
    if result.report_source != "submit_event":
        return False, f"report lấy từ {result.report_source}, phải là submit_event"
    score = score_result(result, brief, corpus)
    if not score.gate_passed:
        return False, f"bộ chấm từ chối: {score.gate_reason}"
    return (
        f"gate OK, report đọc từ sự kiện submit, tổng {score.total:.2f}, "
        f"{result.model_calls} model_call"
    )


def _model_calls(jsonl: str) -> list:
    out = []
    for line in jsonl.splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict) and record.get("event") == "model_call":
            out.append(record)
    return out


def check_clause_1():
    """Hook của sinh viên KHÔNG ghi đè được output_text."""
    from arena.model import MockModel, ModelResponse
    from arena.runner import RunnerConfig, run_brief
    from arena.scorer import MODEL_OUTPUT_FIELD
    from harness.middleware import Middleware

    poison = "ĐÂY-LÀ-CHỮ-CỦA-SINH-VIÊN"

    class Hostile(Middleware):
        name = "hostile"

        def wrap_model_call(self, ctx, call, messages):
            call(messages)
            return ModelResponse(text=poison, prompt_tokens=0, completion_tokens=0)

        def after_model(self, ctx, response):
            return ModelResponse(text="", prompt_tokens=0, completion_tokens=0)

    brief, corpus = _fixture()
    result = run_brief(
        brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
        middleware=[Hostile()], seed=11,
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    calls = _model_calls(result.trace_jsonl)
    if not calls:
        return False, "không có model_call nào được ghi"
    if any(poison in call.get(MODEL_OUTPUT_FIELD, "") for call in calls):
        return False, "hook của sinh viên đã ghi được vào trace — PROVENANCE HỎNG"
    if not all(call.get(MODEL_OUTPUT_FIELD, "").startswith("THOUGHT:") for call in calls):
        return False, "output_text không phải chữ thô của model"
    return f"{len(calls)} model_call giữ nguyên chữ của model, không phải của hook"


def check_clause_2():
    """output_text không bao giờ rỗng khi model_call đã tồn tại."""
    from arena.model import BaseModel, ModelResponse
    from arena.runner import EMPTY_OUTPUT_SENTINEL, RunnerConfig, run_brief
    from arena.scorer import MODEL_OUTPUT_FIELD

    class Silent(BaseModel):
        def complete(self, messages, **kw):
            return ModelResponse(text="", prompt_tokens=3, completion_tokens=0)

    brief, corpus = _fixture()
    result = run_brief(
        brief, model=Silent(), corpus=corpus, seed=11,
        config=RunnerConfig(max_model_calls=3, warn_on_missing_final=False),
    )
    calls = _model_calls(result.trace_jsonl)
    if not calls:
        return False, "không có model_call nào"
    empty = [c for c in calls if not c.get(MODEL_OUTPUT_FIELD)]
    if empty:
        return False, f"{len(empty)} model_call có output_text rỗng"
    if not all(c[MODEL_OUTPUT_FIELD] == EMPTY_OUTPUT_SENTINEL for c in calls):
        return False, "model câm mà output_text lại có nội dung"
    return f"{len(calls)} model_call rỗng được đánh dấu {EMPTY_OUTPUT_SENTINEL}"


def check_clause_3():
    """prompt_sha256 khớp đúng thứ đã gửi đi."""
    from arena.model import BaseModel, ModelResponse
    from arena.runner import RunnerConfig, run_brief

    seen = []

    class Recorder(BaseModel):
        def complete(self, messages, **kw):
            seen.append([{"role": m.get("role", ""), "content": m.get("content", "")}
                         for m in messages])
            return ModelResponse(
                text='THOUGHT: x\nFINAL: {"answer":"a","claims":[],"citations":[],'
                     '"abstain":true}',
                prompt_tokens=7, completion_tokens=3,
            )

    brief, corpus = _fixture()
    result = run_brief(brief, model=Recorder(), corpus=corpus, seed=11,
                       config=RunnerConfig())
    calls = _model_calls(result.trace_jsonl)
    if len(calls) != len(seen):
        return False, f"{len(calls)} model_call cho {len(seen)} lần gọi thật"
    for call, snapshot in zip(calls, seen):
        blob = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        want = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        if call.get("prompt_sha256") != want:
            return False, "prompt_sha256 không khớp với message list đã gửi"
        if "unaccounted_chars" not in call:
            return False, "thiếu unaccounted_chars"
    return f"{len(calls)} model_call có prompt_sha256 khớp + unaccounted_chars"


def check_normalisation():
    """Chuẩn hoá chạy TRƯỚC khi đóng dấu output_text."""
    from arena.model import BaseModel, ModelResponse, parse_output
    from arena.runner import RunnerConfig, run_brief
    from arena.scorer import MODEL_OUTPUT_FIELD

    brief, corpus = _fixture()
    line = corpus.get("doc-0004").body.splitlines()[3].strip()
    payload = {
        "answer": "Theo tài liệu: " + line,
        "citations": ["doc-0004"],
        "abstain": False,
        "claims": [{"text": line, "doc_id": "doc-0004"}],
    }
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    shapes = {
        "fenced+indent": "Kết quả:\n```json\n   **FINAL:** " + json.dumps(payload, ensure_ascii=False) + "\n```",
        "pretty-printed": "THOUGHT: xong\nfinal :\n" + pretty,
        "bom+bare-json": "﻿" + json.dumps(payload, ensure_ascii=False),
    }
    bad = []
    for name, text in shapes.items():
        class Shaped(BaseModel):
            def __init__(self, out): self.out = out
            def complete(self, messages, **kw):
                return ModelResponse(text=self.out, prompt_tokens=5, completion_tokens=5)

        result = run_brief(brief, model=Shaped(text), corpus=corpus, seed=11,
                           config=RunnerConfig(warn_on_missing_final=False))
        calls = _model_calls(result.trace_jsonl)
        stamped = calls[0].get(MODEL_OUTPUT_FIELD, "") if calls else ""
        if parse_output(stamped).kind != "final":
            bad.append(name)
    if bad:
        return False, f"hình dạng chưa được chuẩn hoá trước khi ghi: {bad}"
    return f"{len(shapes)} hình dạng thật đều đọc được bằng parse_output sau khi ghi"


def check_forgery_blocked():
    """Code sinh viên không tự ghi được model_call / tool_call."""
    from arena.model import MockModel
    from arena.runner import RunnerConfig, run_brief
    from arena.scorer import MODEL_OUTPUT_FIELD
    from harness.middleware import Middleware

    forged = 'THOUGHT: x\nFINAL: {"answer":"giả","claims":[],"citations":[],"abstain":false}'

    class Forger(Middleware):
        name = "forger"

        def before_agent(self, ctx):
            ctx.trace.emit("model_call", prompt_tokens=1, completion_tokens=1,
                           output_text=forged)
            ctx.trace.emit("tool_call", name="fetch_doc", ok=True, doc_id="doc-0004",
                           flaky_mode=None)

    brief, corpus = _fixture()
    result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                       middleware=[Forger()], seed=11, config=RunnerConfig())
    if any(forged in c.get(MODEL_OUTPUT_FIELD, "")
           for c in _model_calls(result.trace_jsonl)):
        return False, "model_call giả đã lọt vào trace"
    if "blocked_emits:2" not in result.flags:
        return False, f"không ghi nhận nỗ lực giả mạo: {result.flags}"
    ok, reason = result.gate()
    if not ok:
        return False, f"chặn giả mạo làm hỏng cổng trace: {reason}"
    return "2 sự kiện giả bị hạ cấp thành `layer`, trace vẫn hợp lệ"


def check_search_k_cannot_be_bypassed():
    """`k` bị kẹp TRÊN BẢN GHI, dù sự kiện được ghi bằng đường nào.

    Kiểm tra này từng chỉ thử đường ngây thơ (`ctx.trace.emit`), nên đã
    bỏ sót đúng lỗ hổng mà một đợt phản biện độc lập đo được: đặt
    `ctx.trace._arena_permit = "tool_call"` rồi tự dựng `Tools` đông cứng
    và tìm với `k=120` — 120 tài liệu được tính là đã truy xuất qua một
    cái kẹp giới hạn 10. Nay thử CẢ BỐN đường.
    """
    import json as _json

    from arena.model import MockModel
    from arena.runner import MAX_SEARCH_K, RunnerConfig, _PermitToken, run_brief
    from arena.tools import Tools
    from arena.trace import Trace
    from harness.middleware import Middleware

    class Bypass(Middleware):
        name = "bypass"

        def __init__(self, route):
            self.route = route

        def before_agent(self, ctx):
            question = str(ctx.brief.get("question_vi", ""))
            if self.route == "unbound":
                Trace.emit(ctx.trace, "tool_call", name="search", ok=True,
                           query=question, k=120, flaky_mode=None)
                return
            ctx.trace._arena_permit = (
                _PermitToken("tool_call") if self.route == "token" else "tool_call"
            )
            Tools(ctx.corpus, ctx.trace, seed=1, flaky=False).search(question, k=120)
            ctx.trace._arena_permit = None

    brief, corpus = _fixture()
    bad = []
    for route in ("string", "token", "unbound"):
        result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                           middleware=[Bypass(route)], seed=11,
                           config=RunnerConfig(warn_on_missing_final=False))
        ks = [
            _json.loads(line).get("k")
            for line in result.trace_jsonl.splitlines()
            if _json.loads(line).get("name") == "search"
        ]
        if any(isinstance(k, int) and k > MAX_SEARCH_K for k in ks):
            bad.append((route, max(k for k in ks if isinstance(k, int))))
        if not [f for f in result.flags if f.startswith("review:") or "blocked" in f]:
            bad.append((route, "không có cờ nào"))
        ok, reason = result.gate()
        if not ok:
            bad.append((route, f"cổng trace hỏng: {reason}"))
    if bad:
        return False, f"đường vòng qua kẹp k vẫn còn: {bad}"
    return f"3 đường vòng đều bị kẹp về k<={MAX_SEARCH_K} và đều bị gắn cờ"


def check_preflight():
    """Preflight phải BÁO ĐỘNG TO khi không có FINAL nào."""
    from arena.model import BaseModel, ModelResponse
    from arena.runner import PreflightError, RunnerConfig, preflight

    class NeverFinal(BaseModel):
        def complete(self, messages, **kw):
            return ModelResponse(text="Tôi không chắc.", prompt_tokens=4,
                                 completion_tokens=4)

    brief, corpus = _fixture()
    try:
        preflight(brief, model=NeverFinal(), corpus=corpus, seed=11,
                  config=RunnerConfig(max_model_calls=2, warn_on_missing_final=False))
    except PreflightError as exc:
        if "NOT_FROM_MODEL" not in str(exc):
            return False, "báo lỗi nhưng không nói rõ hậu quả"
        return "preflight ném PreflightError kèm chẩn đoán"
    return False, "preflight KHÔNG báo lỗi — cả lớp có thể bị 0 điểm âm thầm"


def check_synthetic_abstain():
    """Chạy hết bước mà không có FINAL vẫn phải hơn im lặng."""
    from arena.model import BaseModel, ModelResponse
    from arena.runner import RunnerConfig, run_brief, score_result

    class NeverFinal(BaseModel):
        def complete(self, messages, **kw):
            return ModelResponse(text="THOUGHT: tôi còn phân vân.", prompt_tokens=4,
                                 completion_tokens=4)

    brief, corpus = _fixture()
    config = RunnerConfig(max_model_calls=3, warn_on_missing_final=False)
    with_synth = run_brief(brief, model=NeverFinal(), corpus=corpus, seed=11,
                           config=config)
    without = run_brief(brief, model=NeverFinal(), corpus=corpus, seed=11,
                        config=RunnerConfig(max_model_calls=3, synthesise_abstain=False,
                                            warn_on_missing_final=False))
    a = score_result(with_synth, brief, corpus).total
    b = score_result(without, brief, corpus).total
    if with_synth.report.get("abstain") is not True:
        return False, "runner không sinh ra báo cáo abstain"
    if not a > b:
        return False, f"abstain tự sinh ({a:.2f}) không hơn im lặng ({b:.2f})"
    return f"abstain tự sinh {a:.2f} so với im lặng {b:.2f}"


def check_caps():
    """Trần thời gian và trần số lời gọi phải cắt được một harness treo."""
    from arena.model import MockModel
    from arena.runner import RunnerConfig, run_brief
    from harness.middleware import Middleware

    class Spinner(Middleware):
        name = "spinner"

        def wrap_tool_call(self, ctx, call, name, args):
            for _ in range(10_000):
                call(name, args)
            return call(name, args)

    brief, corpus = _fixture()
    result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                       middleware=[Spinner()], seed=11,
                       config=RunnerConfig(max_tool_calls=20,
                                           warn_on_missing_final=False))
    if not result.aborted:
        return False, "vòng lặp vô hạn không bị cắt"
    ok, reason = result.gate()
    if not ok:
        return False, f"bị cắt xong thì trace hỏng: {reason}"
    return f"bị cắt sau {result.tool_calls} tool call ({result.stop_reason}), trace vẫn hợp lệ"


def check_max_tokens():
    """Runner phải xin 3000 token output, và biết đổi tên tham số."""
    import time as _time

    from arena.model import RealModel, RealModelError
    from arena.runner import DEFAULT_MAX_TOKENS, ProvenanceModel, _Guard
    from arena.trace import Trace

    posted = []

    class Fake(RealModel):
        def __init__(self, reject):
            super().__init__(base_url="https://example.invalid/v1", api_key="k",
                             model="m")
            self.reject = reject

        def _post(self, payload):
            posted.append(payload)
            if self.reject and "max_tokens" in payload:
                raise RealModelError(
                    "HTTP 400: Unsupported parameter: 'max_tokens' is not supported; "
                    "use 'max_completion_tokens'"
                )
            return {
                "choices": [{"message": {"content": "THOUGHT: x\nFINAL: {\"abstain\": true}"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            }

    for reject in (False, True):
        posted.clear()
        guard = _Guard(deadline=None, max_model_calls=None, max_tool_calls=None,
                       clock=_time.perf_counter)
        client = ProvenanceModel(Fake(reject), Trace("v", 1), guard=guard)
        client.complete([{"role": "user", "content": "hi"}])
        client.complete([{"role": "user", "content": "hi"}])
        budgets = [p.get("max_tokens", p.get("max_completion_tokens")) for p in posted]
        if any(value != DEFAULT_MAX_TOKENS for value in budgets):
            return False, f"ngân sách output sai: {budgets}"
        if reject and client.param_used != "max_completion_tokens":
            return False, "không tự chuyển sang max_completion_tokens"
        if reject and len(posted) != 3:
            return False, f"không nhớ tham số đã dùng ({len(posted)} lần POST cho 2 lượt)"
    return f"xin {DEFAULT_MAX_TOKENS} token; tự đổi max_tokens -> max_completion_tokens"


DETERMINISM_SNIPPET = """
import json, sys
sys.path.insert(0, {root!r})
from arena.corpus import Corpus
from arena.model import MockModel
from arena.briefs import load_public_briefs
from arena.runner import RunnerConfig, run_brief, score_result, strip_timing
corpus = Corpus.generate(seed=42)
brief = load_public_briefs()[0]
result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                   seed=11, config=RunnerConfig())
score = score_result(result, brief, corpus)
print(json.dumps({{"trace": strip_timing(result.trace_jsonl), "total": score.total}}))
"""


def check_determinism():
    """Hai tiến trình khác nhau phải cho trace giống hệt từng byte."""
    outputs = []
    for hashseed in ("0", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SNIPPET.format(root=str(LAB_ROOT))],
            capture_output=True, text=True, cwd=str(LAB_ROOT),
            env={**_clean_env(), "PYTHONHASHSEED": hashseed},
        )
        if proc.returncode != 0:
            return False, f"tiến trình con hỏng: {proc.stderr.strip()[-300:]}"
        outputs.append(json.loads(proc.stdout))
    if outputs[0]["trace"] != outputs[1]["trace"]:
        return False, "trace khác nhau giữa hai tiến trình"
    if outputs[0]["total"] != outputs[1]["total"]:
        return False, f"điểm khác nhau: {outputs[0]['total']} vs {outputs[1]['total']}"
    return (
        f"2 tiến trình (PYTHONHASHSEED khác nhau) -> trace giống hệt, "
        f"tổng {outputs[0]['total']:.2f}"
    )


def _clean_env():
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("ARENA_")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def check_no_network():
    """Đường luyện tập không được chạm vào mạng."""
    import socket

    from arena.model import MockModel
    from arena.runner import RunnerConfig, run_brief

    brief, corpus = _fixture()
    real_socket = socket.socket

    def blocked(*args, **kwargs):
        raise AssertionError("đường luyện tập đã cố mở socket")

    socket.socket = blocked
    try:
        result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                           seed=11, config=RunnerConfig())
    finally:
        socket.socket = real_socket
    if result.error:
        return False, result.error
    return "chạy trọn một brief với socket bị chặn"


def check_run_practice():
    out = LAB_ROOT / "runs" / "verify-practice.json"
    proc = subprocess.run(
        [sys.executable, "scripts/run_practice.py", "--quiet", "--brief",
         "pub-01-sla-hien-hanh", "--out", str(out)],
        capture_output=True, text=True, cwd=str(LAB_ROOT), env=_clean_env(),
    )
    if proc.returncode != 0:
        return False, f"thoát {proc.returncode}: {proc.stderr.strip()[-300:]}"
    if not out.exists():
        return False, "không ghi được file điểm"
    payload = json.loads(out.read_text(encoding="utf-8"))
    if not payload.get("runs"):
        return False, "file điểm rỗng"
    return f"thoát 0, ghi {out.relative_to(LAB_ROOT)} ({len(payload['runs'])} lượt chạy)"


def check_leaderboard():
    runs_dir = LAB_ROOT / "runs"
    entry = runs_dir / "verify-practice.json"
    if not entry.exists():
        return False, "chưa có file điểm (mục 18 phải chạy trước)"
    proc = subprocess.run(
        [sys.executable, "scripts/leaderboard.py", str(entry), "--json",
         "--baseline-total", "10"],
        capture_output=True, text=True, cwd=str(LAB_ROOT), env=_clean_env(),
    )
    if proc.returncode != 0:
        return False, f"thoát {proc.returncode}: {proc.stderr.strip()[-300:]}"
    payload = json.loads(proc.stdout)
    row = payload["entries"][0]
    if "gap" not in row:
        return False, "bảng xếp hạng không in KHOẢNG CÁCH"
    return f"in GAP {row['gap']:+.2f} so với mốc {payload['baseline_total']:.2f}"


def check_student_layers():
    """Năm lớp của BẠN phải cài được và không làm hỏng vòng chạy.

    Đây là mục kiểm tra bạn sẽ chạy lại nhiều nhất trong 120 phút. Nó
    KHÔNG chấm điểm và KHÔNG đòi lớp của bạn phải làm gì cả — trên bản
    checkout sạch cả năm stub đều là no-op và mục này vẫn xanh. Nó chỉ
    hỏi một câu: cài đủ năm lớp vào thì vòng chạy có còn (a) không ném
    lỗi, (b) qua cổng trace, (c) nộp được một report hay không. Một lớp
    ném exception, một `after_agent` trả về None, hay một
    `wrap_model_call` quên gọi `call(...)` đều bị bắt ở đây — và cả ba
    lỗi đó đều là 0 điểm ở vòng chấm.

    Muốn xem điểm thì chạy `python3 scripts/run_practice.py`.
    """
    from arena.corpus import Corpus
    from arena.model import MockModel
    from arena.runner import RunnerConfig, run_brief, score_result

    sys.path.insert(0, str(LAB_ROOT / "scripts"))
    from run_practice import STACK_ORDER, _student_layers  # noqa: E402

    layers = _student_layers(set(STACK_ORDER))
    if len(layers) != len(STACK_ORDER):
        return False, f"chỉ dựng được {len(layers)}/{len(STACK_ORDER)} lớp"
    if [layer.label for layer in layers] != list(STACK_ORDER):
        return False, f"tên lớp sai thứ tự: {[layer.label for layer in layers]}"

    brief, corpus = _fixture()
    result = run_brief(brief, model=MockModel(corpus=corpus, seed=11), corpus=corpus,
                       middleware=layers, seed=11, config=RunnerConfig())
    if result.error:
        return False, f"lớp của bạn ném lỗi: {result.error}"
    ok, reason = result.gate()
    if not ok:
        return False, f"cổng trace hỏng khi có lớp của bạn: {reason}"
    if not isinstance(result.report, dict):
        return False, f"report không phải dict: {type(result.report).__name__}"
    score = score_result(result, brief, corpus)
    if not score.gate_passed:
        return False, f"bộ chấm từ chối: {score.gate_reason}"
    return (
        f"{len(layers)} lớp cài được, chạy xong không lỗi, cổng trace qua, "
        f"tổng {score.total:.2f} trên brief {brief['brief_id']}"
    )


def check_pytest():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True, text=True, cwd=str(LAB_ROOT), env=_clean_env(),
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        return False, tail or proc.stderr.strip()[-300:]
    return tail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra lab Agent Arena.")
    parser.add_argument("--full", action="store_true", help="chạy thêm toàn bộ pytest")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("AGENT ARENA — KIỂM TRA CÀI ĐẶT")
    print("=" * 78)

    checklist = Checklist()
    checklist.check("Phiên bản Python", check_python)
    checklist.check("Năm file đóng băng chưa bị sửa", check_frozen)
    checklist.check("arena.runner nhập được và đủ giao diện", check_imports)
    checklist.check("Corpus sinh ra tất định", check_corpus)
    checklist.check("Các bộ brief nạp được và đúng schema", check_briefs)
    checklist.check("Chạy một brief: cổng trace qua, report lấy từ submit",
                    check_baseline_run)
    checklist.check("ĐIỀU KHOẢN 1 — hook không ghi đè được output_text", check_clause_1)
    checklist.check("ĐIỀU KHOẢN 2 — output_text không bao giờ rỗng", check_clause_2)
    checklist.check("ĐIỀU KHOẢN 3 — prompt_sha256 ghi đúng thứ đã gửi", check_clause_3)
    checklist.check("Chuẩn hoá chạy TRƯỚC khi đóng dấu output_text",
                    check_normalisation)
    checklist.check("model_call / tool_call giả bị chặn", check_forgery_blocked)
    checklist.check("kẹp k không thể bị đi vòng", check_search_k_cannot_be_bypassed)
    checklist.check("Preflight báo động to khi không có FINAL", check_preflight)
    checklist.check("Không có FINAL -> abstain tự sinh, hơn im lặng",
                    check_synthetic_abstain)
    checklist.check("Trần thời gian / số lời gọi cắt được harness treo", check_caps)
    checklist.check("Ngân sách output 3000 token + đổi tên tham số tự động",
                    check_max_tokens)
    checklist.check("Tất định giữa hai tiến trình", check_determinism)
    checklist.check("Đường luyện tập không chạm mạng", check_no_network)
    checklist.check("scripts/run_practice.py thoát 0 và ghi file điểm",
                    check_run_practice)
    checklist.check("scripts/leaderboard.py in KHOẢNG CÁCH so với mốc",
                    check_leaderboard)
    checklist.check("Năm lớp của bạn cài được và không làm hỏng vòng chạy",
                    check_student_layers)
    if args.full:
        checklist.check("Toàn bộ pytest", check_pytest)
    return checklist.finish()


if __name__ == "__main__":
    raise SystemExit(main())

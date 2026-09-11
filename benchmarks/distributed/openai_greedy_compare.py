#!/usr/bin/env python3
"""Greedy token-ID completion with logprobs, and a comparator for two runs.

run:      POST /v1/completions with an exact token-ID prompt, temperature 0,
          logprobs=K, return_token_ids, stream=false. Saves the raw response
          plus a compact summary (generated token ids / strings, per-token
          logprobs, first-token top-K) to --out.
compare:  load two summaries and report the first divergence index of the
          generated token ids and the first-token top-K logprob deltas.

Prompt specs (all lengths must be multiples of 2048 for VLLM_DSV4_CP2PP4):
  const:<token_id>:<len>            e.g. const:1000:32768
  rand:<seed>:<len>:<lo>:<hi>       pseudo-random ids in [lo, hi)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request


def build_prompt(spec: str) -> list[int]:
    parts = spec.split(":")
    if parts[0] == "const" and len(parts) == 3:
        return [int(parts[1])] * int(parts[2])
    if parts[0] == "rand" and len(parts) == 5:
        seed, length, lo, hi = (int(p) for p in parts[1:])
        rng = random.Random(seed)
        return [rng.randrange(lo, hi) for _ in range(length)]
    raise SystemExit(f"bad prompt spec: {spec!r}")


def cmd_run(args) -> int:
    prompt = build_prompt(args.prompt_spec)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "logprobs": args.logprobs,
        "return_token_ids": True,
        "stream": False,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    summary = {
        "prompt_spec": args.prompt_spec,
        "prompt_len": len(prompt),
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
    }
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        err = error.read().decode(errors="replace")
        summary.update({"ok": False, "http_status": error.code, "error": err})
        print(json.dumps(summary, indent=1))
        _dump(args.out, summary)
        return 1
    except urllib.error.URLError as error:
        summary.update({"ok": False, "error": str(error)})
        print(json.dumps(summary, indent=1))
        _dump(args.out, summary)
        return 1
    elapsed = time.perf_counter() - start
    data = json.loads(body)
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    top = lp.get("top_logprobs") or []
    summary.update(
        {
            "ok": True,
            "elapsed_s": elapsed,
            "text": choice.get("text"),
            "finish_reason": choice.get("finish_reason"),
            "token_ids": choice.get("token_ids"),
            "tokens": lp.get("tokens"),
            "token_logprobs": lp.get("token_logprobs"),
            "first_token_top": top[0] if top else None,
            "usage": data.get("usage"),
            "raw": data,
        }
    )
    _dump(args.out, summary)
    shown = {k: v for k, v in summary.items() if k != "raw"}
    print(json.dumps(shown, indent=1, ensure_ascii=False))
    return 0


def _dump(path: str | None, obj) -> None:
    if path:
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1, ensure_ascii=False)


def _load(path: str):
    with open(path) as fh:
        return json.load(fh)


def cmd_compare(args) -> int:
    a = _load(args.a)
    b = _load(args.b)
    out = {"a": args.a, "b": args.b}
    if not (a.get("ok") and b.get("ok")):
        out["result"] = "one or both runs failed"
        out["a_ok"] = a.get("ok")
        out["b_ok"] = b.get("ok")
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 2
    ids_a = a.get("token_ids") or a.get("tokens") or []
    ids_b = b.get("token_ids") or b.get("tokens") or []
    n = min(len(ids_a), len(ids_b))
    first_div = next((i for i in range(n) if ids_a[i] != ids_b[i]), None)
    if first_div is None and len(ids_a) != len(ids_b):
        first_div = n
    out.update(
        {
            "num_tokens_a": len(ids_a),
            "num_tokens_b": len(ids_b),
            "first_divergence_index": first_div,
            "all_tokens_match": first_div is None,
            "text_a": a.get("text"),
            "text_b": b.get("text"),
            "token_ids_a": ids_a,
            "token_ids_b": ids_b,
        }
    )
    lpa = a.get("token_logprobs") or []
    lpb = b.get("token_logprobs") or []
    if lpa and lpb:
        m = min(len(lpa), len(lpb))
        out["max_abs_chosen_logprob_diff"] = max(
            abs(lpa[i] - lpb[i]) for i in range(m)
        )
    ta = a.get("first_token_top") or {}
    tb = b.get("first_token_top") or {}
    keys_a = list(ta.keys())
    keys_b = list(tb.keys())
    common = [k for k in keys_a if k in tb]
    out["first_token_top_a"] = ta
    out["first_token_top_b"] = tb
    out["first_token_top_same_set"] = set(keys_a) == set(keys_b)
    out["first_token_top_same_order"] = keys_a == keys_b
    out["first_token_top_max_abs_diff_common"] = (
        max(abs(ta[k] - tb[k]) for k in common) if common else None
    )
    out["first_token_top_num_common"] = len(common)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-spec", default="const:1000:32768")
    run.add_argument("--max-tokens", type=int, default=1)
    run.add_argument("--logprobs", type=int, default=5)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--out", default=None)
    run.set_defaults(func=cmd_run)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

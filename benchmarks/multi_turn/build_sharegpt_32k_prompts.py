"""Build deterministic, exact-length ShareGPT prompts for CP2PP4 benchmarks."""

import argparse
import json
import random
from pathlib import Path

from vllm.tokenizers import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharegpt", type=Path, required=True)
    parser.add_argument("--model", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--target", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
    )
    with args.sharegpt.open() as source:
        data = json.load(source)

    rng = random.Random(args.seed)
    order = list(range(len(data)))
    rng.shuffle(order)

    texts = []
    for index in order:
        for turn in data[index].get("conversations") or []:
            value = (turn.get("value") or "").strip()
            if value:
                texts.append(value)

    prompts = []
    id_lists = []
    position = 0
    attempts = 0
    while len(prompts) < args.num and position < len(texts):
        attempts += 1
        parts = []
        token_ids = []
        while len(token_ids) < args.target + 512 and position < len(texts):
            parts.append(texts[position])
            position += 1
            token_ids = tokenizer.encode(
                "\n\n".join(parts),
                add_special_tokens=False,
            )

        token_ids = token_ids[: args.target]
        text = tokenizer.decode(token_ids)
        round_trip_ids = tokenizer.encode(text, add_special_tokens=False)
        adjustments = 0
        while len(round_trip_ids) != args.target and adjustments < 8:
            delta = len(round_trip_ids) - args.target
            if delta > 0:
                token_ids = token_ids[: len(token_ids) - delta]
            else:
                token_ids = round_trip_ids[: args.target]
            text = tokenizer.decode(token_ids)
            round_trip_ids = tokenizer.encode(text, add_special_tokens=False)
            adjustments += 1

        if len(round_trip_ids) == args.target:
            prompts.append(text)
            id_lists.append(round_trip_ids)
            distinct_ids = len(set(round_trip_ids))
            print(
                f"prompt {len(prompts)}: ok after {adjustments} adjustment(s), "
                f"distinct_ids={distinct_ids}"
            )
        else:
            print(
                f"attempt {attempts}: could not stabilize "
                f"({len(round_trip_ids)} tokens), skipping"
            )

    if len(prompts) != args.num:
        raise RuntimeError(f"only built {len(prompts)} prompts")

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "sharegpt32k.jsonl"
    with jsonl_path.open("w") as output:
        for prompt in prompts:
            output.write(
                json.dumps({"prompt": prompt, "output_tokens": 1}) + "\n"
            )

    ids_path = args.out / "sharegpt32k-ids.json"
    with ids_path.open("w") as output:
        json.dump(id_lists, output)
    print(f"wrote {jsonl_path} and {ids_path}")


if __name__ == "__main__":
    main()

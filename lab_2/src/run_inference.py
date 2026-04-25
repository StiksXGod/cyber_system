"""Run a batch of prompts against a local Ollama server and save the responses."""

from __future__ import annotations

import argparse

from ollama_client import check_server, generate_text
from utils import dump_json, load_json_list, now_iso


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Base URL of the local Ollama server.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:0.5b",
        help="Model name served by Ollama.",
    )
    parser.add_argument(
        "--prompts",
        default="data/prompts.json",
        help="Path to the JSON file with prompts.",
    )
    parser.add_argument(
        "--output",
        default="reports/inference_results.json",
        help="Path to the JSON file where results will be stored.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Read timeout in seconds for one Ollama response.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens for one answer.",
    )
    return parser.parse_args()


def build_results(
    base_url: str,
    model: str,
    prompts: list[str],
    timeout: float,
    max_tokens: int,
    output_path: str,
) -> list[dict[str, str]]:
    """Run inference for every prompt, save progress, and collect final results."""
    results: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts, start=1):
        response = generate_text(
            base_url=base_url,
            model=model,
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        results.append(
            {
                "id": str(index),
                "model": model,
                "generated_at": now_iso(),
                "prompt": prompt,
                "response": response,
            }
        )
        dump_json(results, output_path)
        print(f"[{index}/{len(prompts)}] completed", flush=True)
    return results


def main() -> None:
    """Check the server, run all prompts, and save the JSON report."""
    args = parse_args()
    check_server(args.base_url)
    prompts = [str(item) for item in load_json_list(args.prompts)]
    results = build_results(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )
    output_path = dump_json(results, args.output)
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()

"""Run DiffPlanner's output/post_processing.py (alignment + doors/windows) on a sample json
without touching the submodule (the original __main__ uses hard-coded relative paths and
overwrites its input). Usage: python postprocess.py <in b.json> <out json>
Run in the fpe-diffplanner env.
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "external", "methods", "diffplanner", "output"))
import post_processing  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        syn = json.load(f)
    outputs, failed = [], {}
    for d in syn:
        try:
            outputs.append(post_processing.main(d, True))
        except Exception as e:  # noqa
            failed[d["name"]] = f"{type(e).__name__}: {e}"
    print(f"post-processed {len(outputs)} / {len(syn)}, failed {len(failed)}")
    with open(dst, "w") as f:
        json.dump(outputs, f, ensure_ascii=False)
    with open(dst.replace(".json", "_failed.json"), "w") as f:
        json.dump(failed, f, indent=1)


if __name__ == "__main__":
    main()

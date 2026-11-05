# Holodeck (CVPR 2024): run notes

Checkout: `external/methods/holodeck` (allenai/Holodeck, commit 362b8ed, 2024-08-26; release LLM gpt-4o-2024-05-13).
Not modified: `scripts/holodeck/generate.py` imports its floor-plan, wall and door generators and only replaces
the LLM object (`self.llm(prompt) -> str`) with a client of a local vLLM server.

## LLMs (decided 2026-11-05: open-weight models instead of OpenAI)
Two rows, one per model, both served by vLLM on the H100 machine kw61498 (`ssh birsakm@10.67.24.157`; the DNS name
KW61498.KAUST.EDU.SA points to a different machine):

| key | model | GPU | port |
|---|---|---|---|
| qwen | Qwen/Qwen3.8-27B (bf16; weights in Fedor's HF cache) | 0 | 8201 |
| gemma | google/gemma-4-31B-it (bf16) | 1 | 8202 |

- Servers: `~/fpe-vllm/serve.sh qwen|gemma` on kw61498, in tmux session `fpe-vllm`; logs in `~/fpe-vllm/logs/`.
  They use Fedor's vLLM env (`/home/rodionfa/envs/vllm`, vLLM 0.31.0, torch 2.13+cu130), read-only, with his
  agreement. Localhost only. GPU 3 is Fedor's Strata server (Qwen3.8-Flash-Next, 3-bit); not used here because a
  quantized model would have to be disclosed and it serves one request at a time.
- GPUs are claimed with the group's `gpuctl` (installed on all nodes; status read from /datawaha/cggroup/gpu_data):
  `gpuctl claim kw61498 0 1 --purpose "..."` before starting the servers, `gpuctl release kw61498 0 1` after
  stopping them. `gpuctl status -b` / `gpuctl free` show the nodes. claim refuses GPUs that are busy; `--force` is
  only for GPUs busy with your own processes. A watchdog releases claims idle for more than 2 h.
- Context 8k tokens (prompts + replies stay below ~4k). Qwen3.8-27B is a hybrid Mamba model: `--max-num-seqs 256`.
  Gemma needs `--gpu-memory-utilization 0.95` to fit an 8k KV cache next to the bf16 weights on one H100.
- Sampling: as Holodeck, only max_tokens 2048 is set. langchain 0.0.171 turns Holodeck's `OpenAI(model_name=gpt-4o...)`
  into `OpenAIChat`, which sends no temperature, so the release ran gpt-4o at OpenAI's default (1.0). Here the
  models' own defaults apply (vLLM reads generation_config.json): temperature 1.0, top_p 0.95, top_k 20 (Qwen) /
  64 (Gemma). One user message per call, thinking off (gpt-4o is not a reasoning model and Holodeck parses the raw
  reply). Per-request seed, so a run is reproducible on the same server.
- flashinfer JIT-compiles kernels at startup and needs nvcc: serve.sh sets CUDA_HOME to the env's
  `nvidia/cu13` (the machine has no system CUDA).

## Environment
`/datawaha/cggroup/birsakm/venvs/fpe-holodeck` (python 3.11 from the fpe conda env, not system-site): torch
2.15 (open_clip_torch upgraded the CPU wheel; CLIP runs on CPU), open_clip_torch (CLIP ViT-L-14 laion2b, for Holodeck's material and door-style choice), langchain 0.0.171
(PromptTemplate), pydantic<2, shapely, compress_json/compress_pickle. The interpreter is copied (no symlinks on CIFS).
Works on kw61498 too (/datawaha is mounted there).

Base data: AI2's `holodeck/2023_09_23.tar` (77 MB, sha256 c5ff33c1...; the file objathor's
`download_holodeck_base_data` fetches) extracted to `data/holodeck/objathor/holodeck/2023_09_23` (materials, doors,
CLIP features). The Objaverse assets are not needed (no object stages).

## Run
```bash
# on kw61498
cd /datawaha/cggroup/birsakm/floorplan_eval
V=/datawaha/cggroup/birsakm/venvs/fpe-holodeck/bin/python
PYTHONPATH=. $V scripts/holodeck/generate.py --model qwen --limit 20      # test
PYTHONPATH=. $V scripts/holodeck/generate.py --model qwen                 # 1,000 inputs
```
Output `outputs/holodeck/rplan-g2p_text_test1000_realdoors_<model>/`: `samples/`, `raw/` (every prompt and
reply), `run_info.json` (failures, attempts, tokens).

## What is Holodeck's and what is ours
- **Ours (input):** the text query, written from the bubble+areas program (`query()` in generate.py): unique room
  names with net areas, the door connections, the entrance room. Holodeck takes free text only.
- **Holodeck's:** prompts, parsing, validity check, wall heights, door choice and placement, its prompt's
  constraints (rectangular rooms, sides 3-8 m, at most 48 m^2).
- **Retries:** Holodeck raises on an invalid floor plan; we retry with a new seed, at most 3 attempts per input.
- **Not run:** windows, objects, AI2-THOR (as for ProcTHOR and Infinigen).

## Runs (2026-11-05, Table 1(a), 1,000 RPLAN inputs)
| LLM | plans | attempts 1/2/3 | tokens | time | main failure |
|---|---|---|---|---|---|
| Qwen3.8-27B | 904 | 560/241/199 | 2.1M | 44 min | Holodeck's validity check (overlap, vertex inside room, angles) |
| Gemma-4-31B | 812 | 489/211/300 | 2.6M | 75 min | overlap; also replies with extra text ("Wait, I must recalculate...") that Holodeck cannot parse |

Evaluation: `fpeval.evaluate variants|fid <key>` with the fpe env, then `scripts/rplan/block_a.py` (logs/holodeck/).
Servers stopped and GPUs 0, 1 released (gpuctl) after the runs.

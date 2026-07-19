# RIC — Reverse Inspection Cell

**A professional, production-grade universal LLM inspection lab for Kaggle.**

## What It Is

One self-contained Python script (**1212 lines**, no external dependencies beyond transformers/huggingface_hub) that:

1. **Auto-detects hardware** — single/multi GPU, TPU, or CPU
2. **Inspects everything about the model:**
   - Architecture, parameter count, layers, attention heads, KV heads (GQA), context length
   - Config in full detail (saved to a log file)
   - Tokenizer (vocab, special tokens, chat template, round-trip sanity check)
   - All 39+ weight tensors individually: shape, dtype, numel, size, trainable status, and (optionally) mean/std/min/max statistics
   - Module-level weight breakdown and per-layer weight distribution
   - Embedding matrix shapes and vocab compatibility checks
   - Generation config (sampling params, token IDs)
   
3. **Compatibility checks:**
   - Tokenizer vocab ↔ model vocab
   - Chat template presence
   - Config field completeness
   - Embedding resizing needs
   
4. **Hardware capability checks:**
   - BF16/FP16/flash-attn support
   - Warnings for unsupported backends (e.g., flash-attn on T4)
   
5. **Dataset inspection** (optional):
   - Load from HF Hub, local Arrow (parquet), CSV, JSON, JSONL
   - Show split sizes, column types, first example preview
   
6. **Training readiness estimation:**
   - Weight memory, optimizer state memory (AdamW 8 bytes/param), activation memory
   - Recommended micro-batch size based on available VRAM
   - Recommended gradient accumulation steps for a 2M-token global batch target
   - Learning rate, warmup, scheduler, weight decay, gradient checkpointing suggestions
   - Correctly accounts for multi-GPU model sharding (device_map="auto") vs. data parallelism
   
7. **Interactive chat:**
   - Generate responses using `input()` — works in any Kaggle notebook (no ipywidgets dependency)
   - Supports chat templates if the model has one
   - Single prompt/response loop, no complex state management

## New: Automatic Kaggle Secrets Auth + Multi-Dataset Support

**Kaggle Secrets auto-auth:** Add a secret named `HF_TOKEN` (or `HUGGINGFACE_TOKEN`/`hf_token`/`huggingface_token`) under **Add-ons → Secrets** in the Kaggle notebook editor, and RIC logs in automatically — no interactive `login()` call, so gated/private repos just work, and nothing hangs on unattended "Save & Run All" runs. Outside Kaggle (or with no matching secret), this silently does nothing — it's a convenience, not a requirement.

**Multi-dataset support:** `DATASET_REPOS` is now a list — point it at as many HF Hub repo ids and/or local paths as you want:
```python
DATASET_REPOS = ["WithinUsAI/tarot-sft", "/kaggle/input/zodiac-ds", "some-org/another-dataset"]
```
Each one gets its own full inspection (splits, features, example preview), plus a combined overview table across all of them. One bad entry (typo, auth issue) is logged and skipped — it never stops the rest from being inspected. The old singular `DATASET_REPO` still works and is automatically folded in alongside `DATASET_REPOS`, so nothing breaks if you already had it set.

## Also fixed: incoherent chat replies on tokenizers with no chat_template

If a model's tokenizer has no `chat_template` (visible in the Tokenizer Overview table), RIC previously fell back to a generic `"User: X\nAssistant:"` prompt — text the model was never trained on, which can produce garbled, repetitive output that has nothing to do with what was actually asked. RIC now recognizes several common families (Gemma, Qwen/ChatML, Llama-2/Mistral) and uses each one's actual native turn format instead, falling back to the generic form only for architectures it doesn't recognize. Generation also now uses `repetition_penalty=1.15` and `no_repeat_ngram_size=4` to cut down on degenerate repetition loops in general.

## Key Fixes vs. Original Document

The original document shipped with these bugs/outdated patterns; **RIC fixes all of them:**

| Issue | Original | RIC |
|-------|----------|-----|
| **Parameter counting** | Generic formula (12×L×H²) undercounts custom multi-module architectures | Uses `accelerate.init_empty_weights()` for exact count on ANY architecture |
| **Optimizer state budget** | Completely omitted — silently loses memory margin on full-parameter fine-tuning | Includes AdamW momentum+variance (8 bytes/param) in memory budget |
| **TPU detection** | Called nonexistent `xm.xla_device_count()` API | Uses correct `torch_xla.runtime.addressable_device_count()` + `PJRT_DEVICE=TPU` env var |
| **RoPE extraction** | Silently missed rope_theta on transformers 5.x (moved to `rope_parameters` dict) | Handles both old flat form and new nested form |
| **Config dtype access** | Old `config.torch_dtype` (deprecated on 5.x, now `config.dtype`) | Version-tolerant getter that avoids deprecation warnings |
| **Multi-GPU grad_accum math** | Multiplied by device count, silently undershooting global batch size on sharded model loads | Correctly handles model sharding (device_map="auto") vs. true data parallelism |
| **Flash Attention** | Just reported if importable; never actually passed to from_pretrained() | Properly gates on compute capability (Ampere+ only), falls back to SDPA on T4 |
| **Table rendering** | 10-column tables wrapped character-by-character on narrow output | Split into structure + stats tables; explicit 140-char console width |
| **HfFolder removal** | Used removed `HfFolder.get_token()` from huggingface_hub 1.0+ | Uses current `huggingface_hub.get_token()` |
| **Gated repo login** | Silently hung on unattended runs without explanation | Logs clearly that `login()` will block; suggests HF_TOKEN secret |
| **dtype kwarg** | Outdated `torch_dtype=` on from_pretrained | Supports current `dtype=` kwarg (torch_dtype still works, just deprecated) |
| **max_memory format** | Passed bare numeric strings to accelerate (e.g., `"14072433868"`) | Formats with units for current accelerate (e.g., `"13.11 GB"`) — **fixes multi-GPU loading failure** |

## How to Use

**In Kaggle:**

1. Create a new notebook cell
2. Copy the entire `RIC.py` file into it
3. Edit the top two lines:
   ```python
   MODEL_REPO = "11-47/Gemini3.5-Code.Reasoner-2b"  # ← set this
   DATASET_REPO = None                               # ← or this
   ```
4. Run the cell — no click needed, it runs automatically

**Expected output:**
- Rich-formatted tables in the notebook (fallback to plain text if rich unavailable)
- A full log file saved to `/kaggle/working/RIC_log_YYYYMMDD_HHMMSS.txt` (downloadable)
- At the end, an interactive chat loop (press Ctrl+D or type `exit` to stop)

**Examples:**

```python
# Inspect a HF Hub model
MODEL_REPO = "meta-llama/Llama-2-7b-hf"
DATASET_REPO = None

# Inspect a Kaggle input dataset model
MODEL_REPO = "/kaggle/input/my-model"
DATASET_REPO = "/kaggle/input/my-dataset"

# Inspect a local directory (if you've git-cloned something)
MODEL_REPO = "/home/claude/test_model"
DATASET_REPO = None
```

## Advanced Configuration

At the top of the script, optional settings:

```python
MAX_PARAMETER_ROWS = 500          # cap on weight table rows (full detail always in log file)
COMPUTE_WEIGHT_STATS = True       # per-tensor mean/std/min/max (disable to speed up on huge models)
LOG_TO_FILE = True                # write a persistent log file
CHAT_AFTER_REPORT = True          # drop into interactive chat at the end
```

## What It Outputs

1. **Environment & Hardware** — OS, Python, PyTorch versions, GPU details (name, memory, compute capability, BF16/FP16 support)
2. **HF Authentication** — account check, token status
3. **Model Config** — full inspection + full config dump (JSON)
4. **Tokenizer** — vocab size, max length, special tokens, chat template, round-trip sanity check
5. **Compatibility Checks** — vocab match, causal-LM loading, embedding tying, context mismatch
6. **Model Loading** — method (full precision, 8-bit quantization fallback, device placement)
7. **Weights in Extreme Detail:**
   - Per-tensor table: name, shape, dtype, numel, size, trainable, (optionally) mean/std/min/max
   - Embedding matrix inspection
   - Module-level breakdown (what % of params in embedding vs. attention vs. MLP, etc.)
   - Per-layer breakdown (are layer sizes uniform? highlights hybrid architectures)
8. **Generation Config** — do_sample, temperature, top_p, token IDs, etc.
9. **Hardware Capabilities** — BF16/FP16/flash-attn support, warnings
10. **Dataset** (if provided) — split sizes, column types, feature count, first row preview
11. **Training Readiness** — memory breakdown, recommended micro-batch size, grad accum steps, hyperparameters
12. **Interactive Chat** — test-drive the model with input()

## Why It Works on Any Hardware

- **No ipywidgets** — only plain `input()` for chat (works in all Kaggle notebook modes)
- **No GPU requirement** — degrades to CPU-only gracefully; all tables render in plain text fallback
- **Multi-GPU support** — detects all visible GPUs, auto-distributes model via `device_map="auto"`
- **TPU support** — sets `PJRT_DEVICE=TPU` and uses current PJRT API
- **8-bit fallback** — if model won't fit in VRAM, automatically enables bitsandbytes quantization
- **Auto-precision** — uses BF16 on Ampere+ (A100, RTX 30-series), FP16 on older (T4), degrades to FP32 on CPU
- **Version tolerance** — handles transformers 4.x, 5.x, 6.x; huggingface_hub 0.20+, 1.0+

## Example Output (Snippet)

```
🌍 ENVIRONMENT
┌─────────────────┬────────────────────────┐
│ OS              │ Linux-6.18.5-x86_64    │
│ Python          │ 3.12.3                 │
│ PyTorch         │ 2.13.0+cu130           │
│ Accelerator     │ CUDA (2× GPU)          │
│ Total GPU VRAM  │ 32.00 GB               │
└─────────────────┴────────────────────────┘

🧬 MODEL CONFIG
┌──────────────────────┬───────────────────────────┐
│ model_type           │ llama                     │
│ hidden_size          │ 2048                      │
│ num_hidden_layers    │ 22                        │
│ num_attention_heads  │ 32                        │
│ num_key_value_heads  │ 8 (GQA)                   │
│ max_position_emb     │ 4096                      │
│ rope_theta           │ 500000.0 (long-context)  │
└──────────────────────┴───────────────────────────┘

🚀 TRAINING RECOMMENDATIONS
┌────────────────────────────┬──────────────┐
│ mixed_precision            │ bf16         │
│ est_weight_memory          │ 8.00 GB      │
│ est_optimizer_memory       │ 32.00 GB     │
│ per_device_micro_batch     │ 4            │
│ seq_length                 │ 4096         │
│ grad_accum_steps           │ 31           │
│ learning_rate              │ 1.00e-04     │
│ warmup_steps               │ 1000         │
│ attention_backend          │ flash_attn_2 │
└────────────────────────────┴──────────────┘
```

## License & Use

**You own this script.** It's a standalone, self-contained Python file with no external dependencies beyond the HF ecosystem. Use it freely in Kaggle, modify it, share it. No credit required (but appreciated!).

---

Built for robust, debuggable LLM inspection on Kaggle's free compute tiers. Tested on transformers 5.x, huggingface_hub 1.23, torch 2.13, and current Python versions.

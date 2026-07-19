#!/usr/bin/env python3
# =============================================================================
#  RIC — Reverse Inspection Cell
#  Universal LLM inspection lab: environment, hardware, HF auth, model config,
#  tokenizer, weights (in extreme detail), generation config, datasets, and
#  training-readiness / recommended-hyperparameter estimation — for ANY causal
#  LM on ANY Kaggle hardware (single/multi GPU, TPU, or CPU).
#
#  Paste this whole cell into Kaggle, set MODEL_REPO (and optionally
#  DATASET_REPOS) below, and run. No button, no widget click required — it
#  runs top to bottom the moment the cell executes.
#
#  Gated/private HF repos: add a Kaggle secret named HF_TOKEN (Add-ons ->
#  Secrets in the notebook editor) and RIC logs in with it automatically —
#  no interactive login() call, so nothing hangs on unattended runs.
# =============================================================================

# =============================================================================
# 0. USER CONFIG — the only lines you should need to touch
# =============================================================================
MODEL_REPO = "Plans11/gpt2-medium-sol-luna-tuned"   # HF repo id, /kaggle/input/... path, or local dir

# Datasets to inspect — a list, so you can point RIC at as many as you like.
# Each entry can be an HF Hub dataset repo id or a local/Kaggle-input path. Leave empty to skip.
DATASET_REPOS = []                                 # e.g. ["WithinUsAI/tarot-sft", "/kaggle/input/zodiac-ds"]
DATASET_REPO = None                                 # kept for backward compat: a single dataset (or None);
                                                     # automatically folded into DATASET_REPOS if set

# Kaggle Secrets name(s) to check for an HF token, in priority order. Add one under
# Add-ons -> Secrets in the Kaggle notebook editor. Auto-auth silently does nothing if
# kaggle_secrets isn't available (i.e. not running on Kaggle) or no matching secret exists.
HF_TOKEN_SECRET_NAMES = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "hf_token", "huggingface_token"]

# Advanced / optional overrides — safe to leave alone
MAX_PARAMETER_ROWS = 500          # cap on how many individual weight tensors get their own table row
COMPUTE_WEIGHT_STATS = True       # per-tensor mean/std/min/max (a bit slower on huge models; safe to disable)
LOG_TO_FILE = True                # also write a full log file you can download afterwards
CHAT_AFTER_REPORT = True          # drop into an interactive chat loop once the report is done

# =============================================================================
# 1. AUTO-INSTALL & IMPORTS
# =============================================================================
import sys, os, gc, io, json, re, time, math, logging, warnings, subprocess, tempfile, platform, datetime
import importlib
import importlib.metadata as _ilm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg])


for _pkg, _import_name in [
    ("transformers", "transformers"), ("accelerate", "accelerate"), ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"), ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"), ("rich", "rich"), ("psutil", "psutil"), ("tqdm", "tqdm"),
]:
    try:
        importlib.import_module(_import_name)
    except ImportError:
        try:
            _pip_install(_pkg)
        except Exception as _e:
            print(f"[RIC] Warning: could not auto-install {_pkg}: {_e}")

# bitsandbytes is CUDA-only and occasionally fails to build on odd platforms — optional, guarded.
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    try:
        _pip_install("bitsandbytes")
        import bitsandbytes as bnb  # noqa: F401
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False  # fine — 8-bit fallback just won't be offered

# TPU support (torch_xla) — optional, guarded. xm.xla_device_count() is NOT a real API (common bug);
# the current PJRT API lives in torch_xla.runtime.
TPU_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

import torch
import numpy as np
import psutil
from tqdm.auto import tqdm

import huggingface_hub
from huggingface_hub import HfApi, get_token, login, list_repo_files, whoami
try:
    # current (huggingface_hub >= ~0.20) canonical location
    from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError
except ImportError:
    # older releases exposed the same classes under .utils
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError

import transformers
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
)
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from accelerate import init_empty_weights
import accelerate

import datasets as hf_datasets

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# =============================================================================
# 2. LOGGING SETUP (professional logging: console + optional persisted file)
# =============================================================================
_RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/home/claude"
_LOG_PATH = os.path.join(_LOG_DIR, f"RIC_log_{_RUN_STAMP}.txt")

logger = logging.getLogger("RIC")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(logging.Formatter("[RIC] %(message)s"))
    logger.addHandler(_console_handler)
    if LOG_TO_FILE:
        try:
            _file_handler = logging.FileHandler(_LOG_PATH, mode="w", encoding="utf-8")
            _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(_file_handler)
        except Exception as _e:
            print(f"[RIC] Warning: could not open log file {_LOG_PATH}: {_e}")

console = Console(width=140) if RICH_AVAILABLE else None

try:
    from IPython.display import display, Markdown
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False
    def display(x):
        print(x)
    def Markdown(x):
        return x

# =============================================================================
# 3. UTILITIES — printing/formatting helpers. Everything here degrades to
#    plain print() if rich/IPython aren't available, so the core report NEVER
#    depends on frontend rendering support (unlike ipywidgets, which silently
#    fails to render in some Kaggle/Jupyter frontends).
# =============================================================================
def format_bytes(n: float) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def format_int(n: int) -> str:
    return f"{n:,}"


def section(title: str):
    logger.info("")
    if console is not None:
        console.rule(f"[bold cyan]{title}")
    else:
        bar = "=" * 78
        logger.info(bar)
        logger.info(title)
        logger.info(bar)


def print_md(text: str):
    logger.info(text.replace("**", "").replace("`", ""))
    if _IPYTHON_AVAILABLE:
        display(Markdown(text))


def render_table(title: str, headers: List[str], rows: List[List[Any]], max_rows: Optional[int] = None):
    """Render a table with rich if available; otherwise a clean fixed-width plain-text table.
    Always also logs a compact plain version, so the file log has a durable record too."""
    shown_rows = rows if max_rows is None else rows[:max_rows]
    if console is not None:
        table = Table(title=title, title_style="bold cyan", header_style="bold magenta", show_lines=False)
        for h in headers:
            # ellipsis (not fold) so a long dotted parameter path truncates cleanly on screen
            # instead of wrapping character-by-character on narrow output; the log file (written
            # below regardless) always keeps the full untruncated value.
            table.add_column(str(h), overflow="ellipsis", no_wrap=True, max_width=64)
        for r in shown_rows:
            table.add_row(*[str(x) for x in r])
        console.print(table)
        if max_rows is not None and len(rows) > max_rows:
            console.print(f"[dim]... {len(rows) - max_rows} more rows omitted (see log file for full detail)[/dim]")
    else:
        widths = [max(len(str(h)), *(len(str(r[i])) for r in shown_rows)) if shown_rows else len(str(h))
                  for i, h in enumerate(headers)]
        widths = [min(w, 60) for w in widths]
        line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
        print(f"\n{title}\n" + "-" * len(line))
        print(line)
        print("-" * len(line))
        for r in shown_rows:
            print(" | ".join(str(r[i])[:widths[i]].ljust(widths[i]) for i in range(len(headers))))
        if max_rows is not None and len(rows) > max_rows:
            print(f"... {len(rows) - max_rows} more rows omitted (see log file for full detail)")

    # Durable plain-text record in the log file regardless of rich/console state.
    try:
        logger.info(f"TABLE: {title}")
        for r in rows:
            logger.info("  " + " | ".join(str(x) for x in r))
    except Exception:
        pass


def safe_getattr(obj: Any, name: str, default: Any = "—") -> Any:
    try:
        val = getattr(obj, name)
        return val if val is not None else default
    except Exception:
        return default


def truncate(text: Any, n: int = 200) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def json_safe(obj: Any, _depth: int = 0) -> Any:
    """Recursively coerce an object (e.g. config.to_dict()) into something JSON/print safe."""
    if _depth > 6:
        return "…"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [json_safe(x, _depth + 1) for x in obj[:50]]
    if isinstance(obj, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in list(obj.items())[:200]}
    if isinstance(obj, torch.dtype):
        return str(obj)
    try:
        return str(obj)
    except Exception:
        return "<unrepr-able>"


def pretty_dict_block(d: dict, indent: int = 2) -> str:
    lines = []
    pad = " " * indent
    for k, v in d.items():
        vs = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        lines.append(f"{pad}{k}: {truncate(vs, 300)}")
    return "\n".join(lines)


def get_config_dtype(config) -> Optional[torch.dtype]:
    """transformers renamed config.torch_dtype -> config.dtype, AND changed its stored form from
    a plain string to an actual torch.dtype object (confirmed against a live transformers 5.x
    config: config.dtype returns torch.float32, not "float32"). A comparison like
    `config.torch_dtype in ("float16","bfloat16")` — which looks reasonable and is what you'd
    write against older transformers docs — silently never matches on current versions. This
    checks the current attribute first (no deprecation warning), falls back to the old one for
    configs loaded on older transformers, and normalizes either a torch.dtype or a string form."""
    raw = getattr(config, "dtype", None)
    if raw is None:
        raw = getattr(config, "torch_dtype", None)
    if raw is None:
        return None
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, None)
    return None


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        logger.info(f"⏱  {self.label}: {time.time() - self.t0:.2f}s")

# =============================================================================
# 4. HARDWARE & ENVIRONMENT INSPECTION
# =============================================================================
def detect_hardware() -> Dict[str, Any]:
    """Returns {'device_type': 'cuda'|'tpu'|'cpu', 'devices': [...], 'total_memory_bytes': int}."""
    hw = {"device_type": "cpu", "devices": [], "total_memory_bytes": 0}

    if torch.cuda.is_available():
        hw["device_type"] = "cuda"
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = prop.major >= 8
            hw["devices"].append({
                "id": i,
                "name": prop.name,
                "total_memory": prop.total_memory,
                "compute_capability": (prop.major, prop.minor),
                "bf16_supported": bool(bf16_ok),
                "fp16_supported": prop.major >= 6,
            })
            hw["total_memory_bytes"] += prop.total_memory
        return hw

    if TPU_AVAILABLE:
        try:
            # PJRT needs this set before it will find a TPU at all.
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            # xm.xla_device_count() is not a real API — a common source of "TPU never detected"
            # bugs. addressable_device_count() (torch_xla.runtime) is the current, correct call.
            tpu_devices = xr.addressable_device_count()
            if tpu_devices > 0:
                hw["device_type"] = "tpu"
                # A v3-8/v2-8 pod has 8 cores total, but a single un-spawned process only gets
                # 1 addressable core under PJRT — full pod usage needs torch_xla.launch()/xmp.spawn.
                # We report what THIS process can use, not the full pod, so memory math stays honest.
                hw["total_memory_bytes"] = tpu_devices * 16 * 1024 ** 3  # heuristic: ~16GB/core
                for i in range(tpu_devices):
                    hw["devices"].append({
                        "id": i, "name": f"TPU core {i} (XLA)", "total_memory": 16 * 1024 ** 3,
                        "bf16_supported": True, "fp16_supported": False,
                    })
                return hw
        except Exception as e:
            logger.warning(f"TPU detection failed: {e}")

    hw["total_memory_bytes"] = psutil.virtual_memory().total
    return hw


def get_package_versions() -> Dict[str, str]:
    pkgs = ["torch", "transformers", "accelerate", "huggingface_hub", "datasets", "tokenizers",
            "safetensors", "sentencepiece", "bitsandbytes", "flash_attn", "torch_xla", "numpy",
            "psutil", "rich"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = _ilm.version(p)
        except _ilm.PackageNotFoundError:
            versions[p] = "not installed"
    return versions


def get_cpu_brand() -> str:
    try:
        if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def inspect_environment() -> Dict[str, Any]:
    section("🌍 ENVIRONMENT")
    hw = detect_hardware()

    rows = [
        ["OS / Platform", platform.platform()],
        ["Python", platform.python_version()],
        ["PyTorch", torch.__version__],
        ["CPU", get_cpu_brand()],
        ["CPU cores (logical)", os.cpu_count()],
        ["RAM total", format_bytes(psutil.virtual_memory().total)],
        ["RAM available", format_bytes(psutil.virtual_memory().available)],
    ]
    try:
        disk = psutil.disk_usage("/")
        rows.append(["Disk free", format_bytes(disk.free)])
        rows.append(["Disk total", format_bytes(disk.total)])
    except Exception:
        pass
    rows.append(["Accelerator type", hw["device_type"].upper()])
    rows.append(["Accelerator count", len(hw["devices"])])
    rows.append(["Total accelerator memory",
                 format_bytes(hw["total_memory_bytes"]) if hw["devices"] else "N/A (CPU only)"])
    render_table("System Overview", ["Property", "Value"], rows)

    if hw["device_type"] == "cuda":
        gpu_rows = [[d["id"], d["name"], format_bytes(d["total_memory"]),
                     f"{d['compute_capability'][0]}.{d['compute_capability'][1]}",
                     "Yes" if d["bf16_supported"] else "No",
                     "Yes" if d["fp16_supported"] else "No"]
                    for d in hw["devices"]]
        render_table("GPU Details", ["ID", "Name", "Memory", "Compute Cap.", "BF16", "FP16"], gpu_rows)
        if len(hw["devices"]) > 1:
            names = {d["name"] for d in hw["devices"]}
            if len(names) > 1:
                logger.warning(f"Heterogeneous GPUs detected ({names}) — memory/dtype decisions below "
                                f"use device 0's capability, which may not be optimal for the others.")
    elif hw["device_type"] == "tpu":
        render_table("TPU Details", ["Core", "Name", "Assumed Memory"],
                     [[d["id"], d["name"], format_bytes(d["total_memory"])] for d in hw["devices"]])
        logger.info("NOTE: a single un-spawned Python process sees 1 addressable TPU core under PJRT. "
                     "Full multi-core pod parallelism needs torch_xla.launch()/xmp.spawn, which is out "
                     "of scope for this single-process inspection/chat cell.")
    else:
        logger.warning("No GPU/TPU detected — running on CPU. Loading and inference will be slow, "
                        "and training is impractical for anything beyond tiny models.")

    render_table("Installed Package Versions", ["Package", "Version"],
                 [[k, v] for k, v in get_package_versions().items()])

    return hw

# =============================================================================
# 5. HUGGING FACE AUTHENTICATION & REPO RESOLUTION
# =============================================================================
def auto_auth_from_kaggle_secrets(secret_names: Optional[List[str]] = None) -> Optional[str]:
    """Best-effort: pull an HF token out of Kaggle's Secrets (Add-ons -> Secrets in the notebook
    editor) and log in with it non-interactively, so gated/private repos work without ever
    calling bare login() (which prompts and BLOCKS — exactly what hangs unattended runs).
    Silently does nothing outside Kaggle, or if none of the given secret names are set — this
    is a convenience, not a requirement, and its absence should never be treated as an error."""
    secret_names = secret_names or HF_TOKEN_SECRET_NAMES
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None  # not running on Kaggle (or the Secrets add-on isn't available here)

    try:
        user_secrets = UserSecretsClient()
    except Exception as e:
        logger.warning(f"kaggle_secrets is importable but UserSecretsClient() failed: {e}")
        return None

    for name in secret_names:
        try:
            token = user_secrets.get_secret(name)
        except Exception:
            continue  # no secret under this name — try the next candidate
        if token:
            try:
                os.environ["HF_TOKEN"] = token
                login(token=token, add_to_git_credential=False)
                logger.info(f"Authenticated with Hugging Face automatically using Kaggle secret '{name}'.")
                return token
            except Exception as e:
                logger.warning(f"Found Kaggle secret '{name}' but login with it failed: {e}")
                return None
    return None  # kaggle_secrets available, but none of the candidate names are set as a secret


def check_hf_auth() -> Optional[str]:
    section("🔑 HUGGING FACE AUTHENTICATION")
    token = get_token()
    if not token:
        logger.warning("No HF token found (not logged in, and no matching Kaggle secret in "
                        f"{HF_TOKEN_SECRET_NAMES}). Public repos still work; gated/private repos "
                        "will fail until you add one of those names under Add-ons -> Secrets in "
                        "Kaggle, or call huggingface_hub.login() yourself.")
        return None
    try:
        info = whoami(token=token)
        username = info.get("name") or info.get("fullname") or "unknown"
        auth_type = "—"
        if isinstance(info.get("auth"), dict):
            auth_type = info["auth"].get("accessToken", {}).get("role", "—") if isinstance(
                info["auth"].get("accessToken"), dict) else info["auth"].get("type", "—")
        render_table("HF Account", ["Field", "Value"],
                     [["username", username], ["auth/token type", auth_type]])
        logger.info(f"Authenticated as: {username}")
        return username
    except Exception as e:
        logger.warning(f"A token is set but whoami() failed ({e}) — it may be invalid/expired.")
        return None


def resolve_model_path(repo: str) -> str:
    section(f"📦 RESOLVING MODEL PATH: {repo}")
    if os.path.isdir(repo):
        logger.info("Local directory — using it directly, no Hub lookup needed.")
        return repo
    if repo.startswith("/kaggle/input/"):
        if os.path.isdir(repo):
            logger.info("Kaggle input dataset path found.")
            return repo
        raise FileNotFoundError(f"Kaggle input path not found: {repo}")
    if repo.startswith("https://github.com") or repo.startswith("git@"):
        dest = tempfile.mkdtemp(prefix="ric_github_")
        logger.info(f"Cloning {repo} ...")
        subprocess.run(["git", "clone", "--depth", "1", repo, dest], check=True)
        for root, dirs, files in os.walk(dest):
            if "config.json" in files:
                logger.info(f"Found config.json at {root}")
                return root
        raise FileNotFoundError("No config.json found anywhere in the cloned repo.")

    # Otherwise: a Hugging Face Hub repo id.
    token = get_token()
    try:
        files = list_repo_files(repo, token=token)
        logger.info(f"Hub repo reachable — {len(files)} files.")
    except GatedRepoError:
        if token:
            logger.warning("Repo is gated and the current token doesn't have access to it. "
                            "Loading will likely fail until access is granted.")
        else:
            logger.warning("Repo is gated and no HF token is set. Calling login() now — this WILL "
                            "BLOCK waiting for input, so it hangs if this notebook runs unattended "
                            "(e.g. Kaggle 'Save & Run All'). Set an HF_TOKEN secret beforehand to avoid that.")
            login()
    except RepositoryNotFoundError:
        raise ValueError(f"Repo '{repo}' was not found on the Hub — check spelling and visibility.")
    except HfHubHTTPError as e:
        logger.warning(f"Hub reachability check failed with an HTTP error ({e}); attempting to load anyway.")
    except Exception as e:
        logger.warning(f"Hub reachability check raised {type(e).__name__}: {e}; attempting to load anyway.")
    return repo


# =============================================================================
# 6. MODEL CONFIG INSPECTION
# =============================================================================
def inspect_config(path: str) -> Tuple[Any, bool]:
    section("🧬 MODEL CONFIG")
    needs_trust_remote_code = False
    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    except Exception as e:
        logger.info(f"Standard config load failed ({type(e).__name__}); this repo likely ships "
                     f"custom modeling code — retrying with trust_remote_code=True.")
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        needs_trust_remote_code = True

    if getattr(config, "auto_map", None):
        needs_trust_remote_code = True

    key_fields = [
        "model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size", "max_position_embeddings",
        "tie_word_embeddings", "sliding_window", "dtype", "hidden_act",
        "attention_dropout", "initializer_range", "rms_norm_eps", "layer_norm_eps",
    ]
    rows = [[k, truncate(getattr(config, k), 120)] for k in key_fields if hasattr(config, k)]
    render_table("Key Config Fields", ["Field", "Value"], rows)

    # transformers >= 5.0 moved RoPE settings off the config entirely into a `rope_parameters`
    # dict (config.rope_theta no longer exists as a flat attribute — confirmed empirically, not
    # assumed). `rope_scaling` is kept as a backward-compat alias of the same dict on newer
    # versions, but on OLDER transformers it's a separate, differently-shaped dict (e.g. just
    # {"type": "yarn", "factor": 4.0} with no theta in it). setdefault() below merges all three
    # possible sources without duplicating or overwriting, regardless of which era produced this config.
    rope_info: Dict[str, Any] = {}
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        rope_info.update(rp)
    if getattr(config, "rope_theta", None) is not None:
        rope_info.setdefault("rope_theta", config.rope_theta)
    scaling = getattr(config, "rope_scaling", None)
    if isinstance(scaling, dict):
        for k, v in scaling.items():
            rope_info.setdefault(k, v)

    if rope_info:
        render_table("RoPE Settings", ["Field", "Value"], [[k, v] for k, v in rope_info.items()])
    else:
        logger.info("No RoPE fields on this config — likely absolute/learned positional embeddings.")

    logger.info(f"trust_remote_code required: {needs_trust_remote_code}")

    try:
        full_dict = json_safe(config.to_dict())
        logger.info("FULL CONFIG DUMP (also in log file):\n" + json.dumps(full_dict, indent=2, default=str))
    except Exception as e:
        logger.warning(f"Could not fully serialize config to JSON: {e}")

    return config, needs_trust_remote_code


# =============================================================================
# 7. TOKENIZER INSPECTION
# =============================================================================
def inspect_tokenizer(path: str, needs_trust_remote_code: bool) -> Any:
    section("🔤 TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=needs_trust_remote_code)

    rows = [
        ["vocab_size (base)", tokenizer.vocab_size],
        ["len(tokenizer) incl. added tokens", len(tokenizer)],
        ["model_max_length", tokenizer.model_max_length],
        ["is_fast (Rust tokenizers backend)", tokenizer.is_fast],
        ["bos_token", f"{tokenizer.bos_token!r} (id={tokenizer.bos_token_id})"],
        ["eos_token", f"{tokenizer.eos_token!r} (id={tokenizer.eos_token_id})"],
        ["pad_token", f"{tokenizer.pad_token!r} (id={tokenizer.pad_token_id})" if tokenizer.pad_token else "not set"],
        ["unk_token", f"{tokenizer.unk_token!r} (id={tokenizer.unk_token_id})" if tokenizer.unk_token else "—"],
        ["chat_template present", "Yes" if getattr(tokenizer, "chat_template", None) else "No"],
        ["padding_side", tokenizer.padding_side],
        ["truncation_side", tokenizer.truncation_side],
    ]
    render_table("Tokenizer Overview", ["Property", "Value"], rows)

    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    if added:
        logger.info(f"{len(added)} added/special tokens beyond the base BPE/unigram vocab.")

    # Quick round-trip sanity check — catches broken tokenizer files early.
    try:
        probe = "The quick brown fox."
        ids = tokenizer.encode(probe)
        back = tokenizer.decode(ids)
        logger.info(f"Round-trip sanity check OK — {len(ids)} tokens for a {len(probe)}-char probe string.")
    except Exception as e:
        logger.warning(f"Tokenizer round-trip probe failed: {e}")

    return tokenizer


# =============================================================================
# 8. COMPATIBILITY CHECKS
# =============================================================================
def run_compatibility_checks(config, tokenizer, needs_trust_remote_code: bool) -> List[Tuple[str, str, str]]:
    section("✅ COMPATIBILITY CHECKS")
    checks: List[Tuple[str, str, str]] = []  # (name, "PASS"/"FAIL"/"INFO", detail)

    tok_vocab = len(tokenizer)
    cfg_vocab = getattr(config, "vocab_size", None)
    if cfg_vocab is None:
        checks.append(("tokenizer ↔ config vocab", "FAIL", "config has no vocab_size field"))
    elif tok_vocab == cfg_vocab:
        checks.append(("tokenizer ↔ config vocab", "PASS", f"both {cfg_vocab}"))
    elif tok_vocab < cfg_vocab:
        checks.append(("tokenizer ↔ config vocab", "PASS",
                        f"tokenizer {tok_vocab} < config {cfg_vocab} — extra padding/reserved rows, fine"))
    else:
        checks.append(("tokenizer ↔ config vocab", "FAIL",
                        f"tokenizer {tok_vocab} > config {cfg_vocab} — call "
                        f"model.resize_token_embeddings({tok_vocab}) before training or you WILL index "
                        f"out of range"))

    architectures = getattr(config, "architectures", None) or []
    looks_causal = any(("CausalLM" in a) for a in architectures)
    if needs_trust_remote_code:
        causal_ok = looks_causal
        causal_detail = (f"architectures={architectures} names a CausalLM class" if looks_causal else
                          f"architectures={architectures} — name doesn't obviously say CausalLM; "
                          f"verify this repo is actually meant for AutoModelForCausalLM")
    else:
        causal_ok = type(config) in AutoModelForCausalLM._model_mapping.keys()
        causal_detail = ("registered in transformers' built-in causal-LM auto-mapping" if causal_ok else
                          "NOT in the standard causal-LM mapping — double check config.architectures")
    checks.append(("Loadable via AutoModelForCausalLM", "PASS" if causal_ok else "FAIL", causal_detail))

    checks.append(("trust_remote_code", "INFO",
                    "Required — repo ships custom modeling code (auto_map in config.json)"
                    if needs_trust_remote_code else "Not required — standard transformers architecture"))

    ctx = getattr(config, "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if ctx and tok_max and tok_max < 10 ** 8:  # filter out the "unset" sentinel (~1e30)
        if tok_max < ctx:
            checks.append(("tokenizer max_length vs model context", "FAIL",
                            f"tokenizer.model_max_length={tok_max} < model context {ctx} — tokenizer "
                            f"will silently truncate below what the model can actually use"))
        else:
            checks.append(("tokenizer max_length vs model context", "PASS", f"tokenizer {tok_max} ≥ model {ctx}"))

    tie_emb = getattr(config, "tie_word_embeddings", None)
    checks.append(("Embedding tying", "INFO",
                    f"tie_word_embeddings={tie_emb} — " +
                    ("input/output embeddings share one matrix (smaller, common for small models)"
                     if tie_emb else "input/output embeddings are separate matrices (more params, more capacity)")))

    rows = [[name, status, detail] for name, status, detail in checks]
    render_table("Compatibility Checks", ["Check", "Result", "Detail"], rows)

    failed = [c for c in checks if c[1] == "FAIL"]
    if failed:
        logger.warning(f"{len(failed)} compatibility check(s) FAILED — resolve these before training. See table above.")
    else:
        logger.info("All hard compatibility checks passed.")
    return checks

# =============================================================================
# 9. MODEL LOADING — auto-hardware, auto-precision, auto-quantization-fallback
# =============================================================================
def load_model_auto(path: str, config: Any, hw: Dict[str, Any], needs_trust_remote_code: bool):
    section("⚖️  MODEL LOADING")
    device_type = hw["device_type"]
    total_mem = hw["total_memory_bytes"]

    cfg_dtype = get_config_dtype(config)
    param_bytes = 2 if cfg_dtype in (torch.float16, torch.bfloat16) else 4

    # Exact parameter count via meta-device instantiation (accelerate.init_empty_weights) — works
    # for ANY architecture, including custom trust_remote_code multi-module designs, unlike a
    # generic vanilla-transformer formula (12*layers*hidden^2 + vocab*hidden) which undercounts
    # anything with extra fused submodules per layer. An undercount here can wrongly skip the
    # 8-bit fallback below and OOM on load instead of degrading gracefully.
    try:
        with init_empty_weights():
            meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=needs_trust_remote_code)
        estimated_params = sum(p.numel() for p in meta_model.parameters())
        del meta_model
        logger.info(f"Meta-device parameter estimate: {format_int(estimated_params)}")
    except Exception as e:
        logger.warning(f"Meta-device sizing failed ({type(e).__name__}: {e}); falling back to a "
                        f"rough vanilla-transformer formula estimate.")
        hidden = getattr(config, "hidden_size", 0) or 0
        layers = getattr(config, "num_hidden_layers", 0) or 0
        vocab = getattr(config, "vocab_size", 0) or 0
        estimated_params = 12 * layers * hidden ** 2 + vocab * hidden
    model_mem_est = estimated_params * param_bytes

    load_in_8bit = False
    if device_type == "cuda" and total_mem and total_mem < model_mem_est * 1.5:
        if BNB_AVAILABLE and BitsAndBytesConfig is not None:
            logger.info("Estimated size leaves little headroom for activations on this GPU → "
                        "enabling 8-bit quantization (bitsandbytes).")
            load_in_8bit = True
        else:
            logger.warning("Model may not fit comfortably in VRAM and bitsandbytes isn't available "
                            "— loading in full precision anyway; watch for an OOM.")
    elif device_type == "tpu":
        logger.info("TPU: loading in bfloat16 (XLA-native).")
    elif device_type == "cpu":
        logger.warning("CPU only — loading in float32; this will be slow for anything but a tiny model.")

    load_kwargs: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": needs_trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if device_type == "cuda":
        if load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if len(hw["devices"]) > 1:
                max_memory = {d["id"]: format_bytes(int(d['total_memory'] * 0.9)) for d in hw["devices"]}
                max_memory["cpu"] = "48GB"
                load_kwargs["max_memory"] = max_memory
            load_kwargs["device_map"] = "auto"
        else:
            d0 = hw["devices"][0]
            if d0["bf16_supported"]:
                load_kwargs["dtype"] = torch.bfloat16
            elif d0["fp16_supported"]:
                load_kwargs["dtype"] = torch.float16
            else:
                load_kwargs["dtype"] = torch.float32
            if len(hw["devices"]) > 1:
                # accelerate.get_max_memory() expects either raw bytes (int) or strings with units
                # ("5GB", "10.5GB"), NOT numeric strings like "14072433868". Format with units.
                max_memory = {d["id"]: format_bytes(int(d['total_memory'] * 0.9)) for d in hw["devices"]}
                max_memory["cpu"] = "48GB"
                load_kwargs["max_memory"] = max_memory
            load_kwargs["device_map"] = "auto"
    elif device_type == "tpu":
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = None
    else:  # cpu
        load_kwargs["dtype"] = torch.float32
        load_kwargs["device_map"] = None

    # Attention backend: importable flash_attn does NOT mean it's actually used — from_pretrained
    # defaults to eager/sdpa unless attn_implementation is passed explicitly. Also, upstream
    # flash-attn 2 still doesn't support Turing (T4) — only Ampere+ (compute capability >= 8) —
    # so requesting it on a T4 errors at generate() time. SDPA is the safe, fast default there.
    attn_backend_used = "default (model's own choice)"
    if device_type == "cuda":
        cc = hw["devices"][0]["compute_capability"]
        try:
            import flash_attn  # noqa: F401
            flash_attn_importable = True
        except ImportError:
            flash_attn_importable = False
        if flash_attn_importable and cc[0] >= 8:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            attn_backend_used = "flash_attention_2"
        else:
            load_kwargs["attn_implementation"] = "sdpa"
            attn_backend_used = "sdpa" + ("" if cc[0] >= 8 else " (flash-attn unsupported on this GPU generation)")

    logger.info(f"Loading with: { {k: v for k, v in load_kwargs.items() if k != 'config'} }")
    with Timer("Model load"):
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)

    if device_type == "tpu":
        model.to(xm.xla_device())
        logger.info("Model moved to TPU XLA device.")

    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"Total params: {format_int(total_p)} | Trainable: {format_int(trainable_p)} | "
                f"Actual weight memory: {format_bytes(weight_bytes)}")

    try:
        actual_dtype = next(model.parameters()).dtype
    except StopIteration:
        actual_dtype = load_kwargs.get("dtype", torch.float32)

    return model, total_p, trainable_p, actual_dtype, attn_backend_used

# =============================================================================
# 10. DEEP WEIGHT EXPOSITION — "extreme detail" on every tensor in the model
# =============================================================================
def extract_layer_index(param_name: str) -> Optional[int]:
    m = re.search(r"\.(\d+)\.", param_name)
    return int(m.group(1)) if m else None


def expose_weights_in_detail(model: Any, tokenizer: Any, config: Any) -> Dict[str, Any]:
    section("🔬 WEIGHTS — DEEP INSPECTION")

    try:
        arch_str = str(model)
        logger.info("FULL MODULE TREE (also in log file in full):\n" + arch_str)
        if console is not None:
            preview = arch_str if len(arch_str) < 4000 else arch_str[:4000] + \
                "\n... (truncated on screen — full tree is in the log file)"
            console.print(preview)
    except Exception as e:
        logger.warning(f"Could not stringify model architecture: {e}")

    all_params = list(model.named_parameters())

    struct_rows, stat_rows = [], []
    for name, p in all_params:
        struct_rows.append([name, str(list(p.shape)), str(p.dtype).replace("torch.", ""),
                             format_int(p.numel()), f"{p.numel() * p.element_size() / 1024 ** 2:.3f} MB",
                             p.requires_grad])
        if COMPUTE_WEIGHT_STATS:
            try:
                flat = p.detach().float()
                stat_rows.append([name, f"{flat.mean().item():.4g}", f"{flat.std().item():.4g}",
                                   f"{flat.min().item():.4g}", f"{flat.max().item():.4g}"])
            except Exception:
                stat_rows.append([name, "—", "—", "—", "—"])

    order = {name: i for i, (name, p) in enumerate(
        sorted(all_params, key=lambda np_: -np_[1].numel()))}
    struct_rows.sort(key=lambda r: order[r[0]])
    stat_rows.sort(key=lambda r: order[r[0]])

    render_table(f"Per-Tensor Structure ({len(struct_rows)} tensors total, largest first)",
                 ["Name", "Shape", "Dtype", "Numel", "Size", "Trainable"],
                 struct_rows, max_rows=MAX_PARAMETER_ROWS)
    if COMPUTE_WEIGHT_STATS:
        render_table(f"Per-Tensor Statistics ({len(stat_rows)} tensors total, largest first)",
                     ["Name", "Mean", "Std", "Min", "Max"],
                     stat_rows, max_rows=MAX_PARAMETER_ROWS)

    try:
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        emb_rows = []
        if in_emb is not None and hasattr(in_emb, "weight"):
            emb_rows.append(["Input embedding shape", str(tuple(in_emb.weight.shape))])
        if out_emb is not None and hasattr(out_emb, "weight"):
            emb_rows.append(["Output embedding (lm_head) shape", str(tuple(out_emb.weight.shape))])
            if in_emb is not None and hasattr(in_emb, "weight"):
                try:
                    tied = out_emb.weight.data_ptr() == in_emb.weight.data_ptr()
                except Exception:
                    tied = "unknown"
                emb_rows.append(["Tied in memory (same tensor)", tied])
        if in_emb is not None and hasattr(in_emb, "weight") and tokenizer is not None:
            tok_vocab = len(tokenizer)
            rows_match = in_emb.weight.shape[0]
            emb_rows.append(["Input embedding rows vs tokenizer vocab",
                              f"{rows_match} vs {tok_vocab} — {'OK' if rows_match >= tok_vocab else 'TOO SMALL, resize needed'}"])
        if emb_rows:
            render_table("Embedding Matrices", ["Property", "Value"], emb_rows)
    except Exception as e:
        logger.warning(f"Embedding inspection failed: {e}")

    mod_sizes: Dict[str, int] = {}
    mod_counts: Dict[str, int] = {}
    for name, p in all_params:
        top = name.split(".")[0]
        mod_sizes[top] = mod_sizes.get(top, 0) + p.numel() * p.element_size()
        mod_counts[top] = mod_counts.get(top, 0) + p.numel()
    render_table("Top-Level Module Breakdown", ["Module", "Params", "Size"],
                 [[m, format_int(mod_counts[m]), format_bytes(s)]
                  for m, s in sorted(mod_sizes.items(), key=lambda x: -x[1])])

    layer_sizes: Dict[int, int] = {}
    layer_counts: Dict[int, int] = {}
    for name, p in all_params:
        idx = extract_layer_index(name)
        if idx is not None:
            layer_sizes[idx] = layer_sizes.get(idx, 0) + p.numel() * p.element_size()
            layer_counts[idx] = layer_counts.get(idx, 0) + p.numel()
    if layer_sizes:
        render_table("Per-Layer Breakdown", ["Layer #", "Params", "Size"],
                     [[i, format_int(layer_counts[i]), format_bytes(layer_sizes[i])]
                      for i in sorted(layer_sizes.keys())])
        sizes = list(layer_sizes.values())
        if sizes and max(sizes) > 0 and (max(sizes) - min(sizes)) / max(sizes) > 0.05:
            logger.info("Layer sizes aren't uniform — likely a heterogeneous/hybrid-module "
                         "architecture rather than a repeated identical block.")

    total_bytes = sum(p.numel() * p.element_size() for _, p in all_params)
    logger.info(f"TOTAL: {len(all_params)} tensors, {format_bytes(total_bytes)}")
    return {"total_tensors": len(all_params), "total_weight_bytes": total_bytes,
            "module_breakdown": mod_sizes, "layer_breakdown": layer_sizes}

# =============================================================================
# 11. GENERATION CONFIG INSPECTION
# =============================================================================
def inspect_generation_config(model: Any) -> Optional[Any]:
    section("🎛️  GENERATION CONFIG")
    gen_config = getattr(model, "generation_config", None)
    if gen_config is None:
        logger.info("No generation_config attached to this model.")
        return None
    fields = ["do_sample", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty",
              "max_new_tokens", "max_length", "min_new_tokens", "no_repeat_ngram_size",
              "bos_token_id", "eos_token_id", "pad_token_id", "renormalize_logits", "length_penalty"]
    rows = [[f, getattr(gen_config, f)] for f in fields
            if hasattr(gen_config, f) and getattr(gen_config, f) is not None]
    if rows:
        render_table("Generation Config", ["Field", "Value"], rows)
    else:
        logger.info("generation_config present but has no notable non-default fields set.")
    return gen_config


# =============================================================================
# 12. HARDWARE CAPABILITY CHECKS
# =============================================================================
def check_hardware_capabilities(hw: Dict[str, Any]) -> Dict[str, Any]:
    section("🧪 HARDWARE CAPABILITY CHECKS")
    caps: Dict[str, Any] = {}
    device_type = hw["device_type"]

    if device_type == "cuda":
        d0 = hw["devices"][0]
        caps["bf16_supported"] = d0["bf16_supported"]
        caps["fp16_supported"] = d0["fp16_supported"]
        try:
            import flash_attn
            caps["flash_attn_importable"] = True
            caps["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
        except ImportError:
            caps["flash_attn_importable"] = False
        caps["flash_attn_hw_supported"] = caps["flash_attn_importable"] and d0["compute_capability"][0] >= 8
        caps["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        rows = [
            ["BF16 supported", caps["bf16_supported"]],
            ["FP16 supported", caps["fp16_supported"]],
            ["flash-attn installed", caps["flash_attn_importable"]],
            ["flash-attn usable on this GPU", caps["flash_attn_hw_supported"]],
            ["SDPA (PyTorch native attention) available", caps["sdpa_available"]],
        ]
        if caps["flash_attn_importable"] and not caps["flash_attn_hw_supported"]:
            rows.append(["Note", f"flash-attn is installed but compute capability "
                                  f"{d0['compute_capability']} is below Ampere (8.0) — upstream "
                                  f"FlashAttention-2 doesn't support this GPU generation "
                                  f"(e.g. T4/Turing); SDPA is used instead and is the right call here"])
    elif device_type == "tpu":
        caps["bf16_supported"], caps["fp16_supported"] = True, False
        rows = [["BF16 (native on TPU)", True], ["FP16", "not used on TPU — bf16 is native"]]
    else:
        caps["bf16_supported"], caps["fp16_supported"] = False, False
        rows = [["Note", "CPU — dtype/attention-backend distinctions aren't meaningful; "
                          "everything runs float32 eager"]]

    render_table("Hardware Capabilities", ["Check", "Result"], rows)
    return caps


# =============================================================================
# 13. DATASET INSPECTION (optional — set DATASET_REPOS to enable, supports any
#     number of datasets: HF Hub repo ids and/or local/Kaggle-input paths)
# =============================================================================
def inspect_dataset(repo_or_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not repo_or_path:
        return None
    section(f"📚 DATASET: {repo_or_path}")

    ds = None
    try:
        if os.path.isdir(repo_or_path):
            try:
                ds = hf_datasets.load_from_disk(repo_or_path)
                logger.info("Loaded via load_from_disk (Arrow-format saved dataset).")
            except Exception:
                exts: Dict[str, List[str]] = {"csv": [], "json": [], "jsonl": [], "parquet": []}
                for f in sorted(os.listdir(repo_or_path)):
                    for ext in exts:
                        if f.endswith("." + ext):
                            exts[ext].append(os.path.join(repo_or_path, f))
                for fmt, files in exts.items():
                    if files:
                        loader_fmt = "json" if fmt in ("json", "jsonl") else fmt
                        ds = hf_datasets.load_dataset(loader_fmt, data_files=files)
                        logger.info(f"Loaded {len(files)} .{fmt} file(s) as a '{loader_fmt}' dataset.")
                        break
                if ds is None:
                    raise FileNotFoundError(f"No Arrow/csv/json/parquet data found in {repo_or_path}")
        else:
            ds = hf_datasets.load_dataset(repo_or_path)
            logger.info("Loaded from the Hugging Face Hub.")
    except Exception as e:
        logger.warning(f"Dataset inspection failed: {type(e).__name__}: {e}")
        return None

    splits = dict(ds) if isinstance(ds, hf_datasets.DatasetDict) else {"train": ds}

    rows = [[name, format_int(len(split)), ", ".join(split.column_names)] for name, split in splits.items()]
    render_table("Dataset Splits", ["Split", "Rows", "Columns"], rows)

    first_name, first_split = next(iter(splits.items()))
    feat_rows = [[name, str(feat)] for name, feat in first_split.features.items()]
    render_table(f"Features (from '{first_name}' split)", ["Column", "Type"], feat_rows)

    if len(first_split) > 0:
        example = first_split[0]
        preview_rows = [[k, truncate(v, 200)] for k, v in example.items()]
        render_table("First Example Preview", ["Column", "Value (truncated)"], preview_rows)

    total_rows = sum(len(s) for s in splits.values())
    logger.info(f"Total examples across all splits: {format_int(total_rows)}")
    return {"splits": {k: len(v) for k, v in splits.items()}, "columns": first_split.column_names,
            "total_rows": total_rows}


def inspect_all_datasets(targets: List[str]) -> Dict[str, Dict[str, Any]]:
    """Inspect any number of datasets — each gets its own full detail section (splits, features,
    example preview), plus a combined overview table across all of them when there's more than
    one. A failure on any single dataset (bad repo id, auth, network) is logged and skipped;
    it never stops the rest of the list from being inspected."""
    results: Dict[str, Dict[str, Any]] = {}
    if not targets:
        logger.info("No datasets configured (DATASET_REPOS / DATASET_REPO) — skipping.")
        return results

    section(f"📚 DATASETS — {len(targets)} target(s) configured")
    for target in targets:
        try:
            summary = inspect_dataset(target)
        except Exception as e:
            logger.warning(f"Unexpected error inspecting dataset '{target}': "
                            f"{type(e).__name__}: {e} — skipping it, continuing with the rest.")
            summary = None
        if summary is not None:
            results[target] = summary

    if len(targets) > 1:
        section("📚 DATASETS — OVERVIEW (all targets)")
        overview_rows = [[name, format_int(r["total_rows"]), len(r["splits"]), len(r["columns"])]
                          for name, r in results.items()]
        failed = [t for t in targets if t not in results]
        if overview_rows:
            render_table("Datasets Overview", ["Dataset", "Total Rows", "Splits", "Columns"], overview_rows)
        grand_total = sum(r["total_rows"] for r in results.values())
        logger.info(f"{len(results)}/{len(targets)} dataset(s) loaded successfully — "
                    f"{format_int(grand_total)} rows combined.")
        if failed:
            logger.warning(f"Failed to load: {failed} — see the warnings above for why.")
    return results

# =============================================================================
# 14. TRAINING READINESS & RECOMMENDED HYPERPARAMETERS
# =============================================================================
def training_recommendations(config: Any, total_p: int, hw: Dict[str, Any],
                              actual_dtype: torch.dtype, attn_backend_used: str) -> Dict[str, Any]:
    section("🚀 TRAINING READINESS & RECOMMENDATIONS")
    device_type = hw["device_type"]
    if device_type == "cpu":
        logger.warning("No accelerator detected — training is impractical here beyond a toy model.")
        return {}

    rec: Dict[str, Any] = {}
    if actual_dtype == torch.bfloat16:
        rec["mixed_precision"] = "bf16"
    elif actual_dtype == torch.float16:
        rec["mixed_precision"] = "fp16"
    else:
        rec["mixed_precision"] = "fp32"
    if device_type == "tpu":
        rec["mixed_precision"] = "bf16"
        rec["mixed_precision_note"] = "bfloat16 (TPU native)"

    max_len = min(getattr(config, "max_position_embeddings", 4096) or 4096, 4096)
    hidden = getattr(config, "hidden_size", 0) or 0
    layers = getattr(config, "num_hidden_layers", 0) or 0
    act_per_token = 2 * hidden * layers * 34  # rough Megatron-style bytes/token/layer estimate

    if device_type == "cuda":
        effective_mem = sum(d["total_memory"] for d in hw["devices"]) * 0.9
    elif device_type == "tpu":
        effective_mem = hw["total_memory_bytes"] * 0.9
    else:
        effective_mem = 0

    weight_mem = total_p * 2 if rec["mixed_precision"] in ("fp16", "bf16") else total_p * 4
    # AdamW keeps fp32 momentum + variance ≈ 8 bytes/param — easy to leave out of a memory budget,
    # and on a T4 doing full-parameter fine-tuning this is usually the actual thing that OOMs, not
    # activations. (LoRA/PEFT only pays this on the much smaller adapter params — scale down accordingly.)
    optimizer_mem = total_p * 8
    rec["est_weight_memory"] = format_bytes(weight_mem)
    rec["est_optimizer_memory_full_finetune"] = format_bytes(optimizer_mem)
    available = effective_mem - weight_mem - optimizer_mem - 2e9  # 2GB buffer
    micro_batch = (max(1, int(available / (act_per_token * max_len) * 0.7))
                   if available > 0 and act_per_token > 0 else 1)

    rec["per_device_micro_batch_size"] = micro_batch
    rec["seq_length"] = max_len
    global_batch_tokens = 2_000_000
    # NOTE: load_model_auto() loads multi-GPU via device_map="auto" — one model instance SHARDED
    # across devices (naive model parallelism), not one replica per device — so device count does
    # NOT multiply per-step throughput here. If you run true multi-GPU data parallelism elsewhere
    # (e.g. Trainer with each GPU holding a full replica, no device_map), multiply by device count there.
    rec["grad_accum_steps"] = max(1, int(global_batch_tokens / (micro_batch * max_len)))
    rec["learning_rate"] = 3e-4 if total_p < 1e9 else 1e-4
    rec["warmup_steps"] = 1000
    rec["weight_decay"] = 0.1
    rec["lr_scheduler"] = "cosine"
    rec["gradient_checkpointing"] = True
    rec["attention_backend"] = attn_backend_used if device_type == "cuda" else "N/A"
    if device_type == "tpu":
        rec["training_framework"] = "torch_xla (single-core in this cell; xmp.spawn needed for full pod)"

    if available <= 0:
        logger.warning("Even a micro-batch of 1 may not comfortably fit at this precision — consider "
                        "8-bit loading, LoRA/PEFT instead of full-parameter fine-tuning, or a shorter "
                        "sequence length.")

    render_table("Recommended Training Settings", ["Setting", "Value"], [[k, v] for k, v in rec.items()])
    return rec


# =============================================================================
# 15. FINAL REPORT
# =============================================================================
def final_report(model_repo: str, config: Any, tokenizer: Any, total_p: int, trainable_p: int,
                  hw: Dict[str, Any], compat_checks: List[Tuple[str, str, str]],
                  weight_summary: Dict[str, Any], training_rec: Dict[str, Any],
                  dataset_summaries: Dict[str, Dict[str, Any]], elapsed: float) -> None:
    section("📋 FINAL REPORT")
    lines = []
    archs = getattr(config, "architectures", None) or [getattr(config, "model_type", "unknown")]
    lines.append(f"**Model:** `{model_repo}`")
    lines.append(f"**Architecture:** {', '.join(archs)}")
    lines.append(f"**Total parameters:** {format_int(total_p)} (trainable: {format_int(trainable_p)})")
    lines.append(f"**Hidden size:** {getattr(config, 'hidden_size', '—')} | "
                 f"**Layers:** {getattr(config, 'num_hidden_layers', '—')} | "
                 f"**Attention heads:** {getattr(config, 'num_attention_heads', '—')}"
                 + (f" ({getattr(config, 'num_key_value_heads')} KV heads, GQA)"
                    if getattr(config, "num_key_value_heads", None) else ""))
    lines.append(f"**Context length:** {getattr(config, 'max_position_embeddings', '—')} tokens")
    lines.append(f"**Vocab size:** {getattr(config, 'vocab_size', '—')} | "
                 f"**Tokenizer length:** {len(tokenizer) if tokenizer else '—'}")
    lines.append(f"**Hardware:** {hw['device_type'].upper()}"
                 + (f" × {len(hw['devices'])} ({format_bytes(hw['total_memory_bytes'])})"
                    if hw["devices"] else " (no accelerator)"))
    lines.append(f"**Total weight memory (actual):** {format_bytes(weight_summary.get('total_weight_bytes', 0))}")

    failed = [c for c in compat_checks if c[1] == "FAIL"]
    passed = [c for c in compat_checks if c[1] == "PASS"]
    lines.append(f"**Compatibility:** {len(passed)}/{len(passed) + len(failed)} checks passed"
                 + (f" — **{len(failed)} FAILED, see Compatibility Checks above**" if failed else " ✅"))

    if dataset_summaries:
        lines.append(f"**Datasets:** {len(dataset_summaries)} inspected")
        for name, s in dataset_summaries.items():
            lines.append(f"  - `{name}`: {format_int(s['total_rows'])} rows across {len(s['splits'])} split(s)")

    if training_rec:
        lines.append("\n**Recommended pretraining / fine-tuning settings:**")
        for k, v in training_rec.items():
            lines.append(f"  - {k}: {v}")

    lines.append(f"\n_Inspection completed in {elapsed:.1f}s."
                 + (f" Full detail logged to `{_LOG_PATH}`._" if LOG_TO_FILE else "_"))
    print_md("\n".join(lines))


# =============================================================================
# 16. INTERACTIVE CHAT — plain input()-loop by design. This does NOT depend on
#     ipywidgets, which can silently fail to render in some Kaggle/Jupyter
#     frontends (no error, just nothing on screen). input() always works in a
#     normal (non-batch) Kaggle notebook cell.
# =============================================================================
def _build_fallback_chat_text(model_type: str, tokenizer: Any,
                               history: List[Tuple[str, str]], prompt: str) -> str:
    """When the tokenizer ships no chat_template, a generic 'User: X\\nAssistant:' string is
    text the model was never trained on — for a model like Gemma, whose real turn format uses
    <start_of_turn>/<end_of_turn> tokens, that mismatch can push generation far outside its
    training distribution (garbled, repetitive output unrelated to what was asked). Use each
    known family's actual native format when we can detect it; fall back to the generic form
    only for unrecognized architectures."""
    mt = (model_type or "").lower()
    special = set(getattr(tokenizer, "additional_special_tokens", None) or [])

    if mt.startswith("gemma") and "<start_of_turn>" in special:
        text = "".join(f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n"
                        for u, a in history)
        return text + f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    if mt in ("qwen2", "qwen3") and "<|im_start|>" in special:
        text = "".join(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
                        for u, a in history)
        return text + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if mt in ("llama", "mistral"):
        text = "".join(f"[INST] {u} [/INST] {a} " for u, a in history)
        return text + f"[INST] {prompt} [/INST]"

    return "".join(f"User: {u}\nAssistant: {a}\n" for u, a in history) + f"User: {prompt}\nAssistant:"


def interactive_chat(model: Any, tokenizer: Any, hw: Dict[str, Any], config: Any = None) -> None:
    section("💬 INTERACTIVE CHAT")
    if not getattr(tokenizer, "chat_template", None):
        model_type = getattr(config, "model_type", "") if config is not None else ""
        print(f"(No chat_template on this tokenizer — using a best-effort '{model_type or 'generic'}' "
              f"native-format fallback. If replies look incoherent/repetitive, that's the model's own "
              f"output on this format, not a RIC bug — verify against the model's actual documented "
              f"prompt format if unsure.)")
    print("Type a message and press Enter. Type 'exit' or 'quit' to stop.")
    history: List[Tuple[str, str]] = []
    try:
        device = xm.xla_device() if hw["device_type"] == "tpu" else next(model.parameters()).device
    except StopIteration:
        device = "cpu"

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat ended]")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("[chat ended]")
            break

        try:
            if getattr(tokenizer, "chat_template", None):
                messages = []
                for u, a in history:
                    messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                messages.append({"role": "user", "content": prompt})
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                                            add_generation_prompt=True)
            else:
                model_type = getattr(config, "model_type", "") if config is not None else ""
                text = _build_fallback_chat_text(model_type, tokenizer, history, prompt)
                input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=256,
                    repetition_penalty=1.15, no_repeat_ngram_size=4,
                    pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[generation error: {type(e).__name__}: {e}]")
            continue

        print(f"Assistant: {response}")
        history.append((prompt, response))



# =============================================================================
# 17. MAIN ORCHESTRATION
# =============================================================================
def run_ric(model_repo: str, dataset_repos: Optional[List[str]] = None):
    start = time.time()
    dataset_repos = dataset_repos or []
    section(f"🔬 RIC — REVERSE INSPECTION CELL")
    logger.info(f"Target model: {model_repo}")
    if dataset_repos:
        logger.info(f"Target dataset(s): {dataset_repos}")
    if LOG_TO_FILE:
        logger.info(f"Full log will be written to: {_LOG_PATH}")

    auto_auth_from_kaggle_secrets()  # before anything else touches the Hub
    hw = inspect_environment()
    check_hf_auth()
    model_path = resolve_model_path(model_repo)
    config, needs_trc = inspect_config(model_path)
    tokenizer = inspect_tokenizer(model_path, needs_trc)
    compat_checks = run_compatibility_checks(config, tokenizer, needs_trc)
    model, total_p, trainable_p, actual_dtype, attn_backend = load_model_auto(
        model_path, config, hw, needs_trc)
    weight_summary = expose_weights_in_detail(model, tokenizer, config)
    inspect_generation_config(model)
    check_hardware_capabilities(hw)
    dataset_summaries = inspect_all_datasets(dataset_repos)
    training_rec = training_recommendations(config, total_p, hw, actual_dtype, attn_backend)

    elapsed = time.time() - start
    final_report(model_repo, config, tokenizer, total_p, trainable_p, hw, compat_checks,
                 weight_summary, training_rec, dataset_summaries, elapsed)

    if CHAT_AFTER_REPORT:
        interactive_chat(model, tokenizer, hw, config)

    return model, tokenizer, config


# =============================================================================
# 18. AUTO-RUN — executes the moment this cell runs. No button, no widget
#     click required. (__name__ == "__main__" is true in a normal notebook
#     cell's execution namespace, so this fires automatically in Kaggle.)
# =============================================================================
if __name__ == "__main__":
    _dataset_targets = list(DATASET_REPOS) if DATASET_REPOS else []
    if DATASET_REPO and DATASET_REPO not in _dataset_targets:
        _dataset_targets.append(DATASET_REPO)
    _model, _tokenizer, _config = run_ric(MODEL_REPO, _dataset_targets)

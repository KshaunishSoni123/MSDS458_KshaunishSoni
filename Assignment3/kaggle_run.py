"""Run the whole Assignment 3 workflow on Kaggle (or any Linux machine with 1+ NVIDIA GPUs).

What it does
    1. Finds the code: clones the GitHub repo (--repo), or copies it from an attached
       Kaggle dataset under /kaggle/input. Work happens in /kaggle/working, the only
       folder Kaggle keeps after a session.
    2. Seeds results/ with any earlier results found in /kaggle/input (e.g. a dataset of
       your Mac's results/ folder), so finished models are not re-run.
    3. Installs the missing packages (peft, sacrebleu) without touching Kaggle's torch.
    4. Logs in to Hugging Face if a Kaggle secret named HF_TOKEN exists (optional).
    5. Splits the jobs across the GPUs and runs them in parallel, one queue per GPU:
         - LoRA fine-tuning + hyperparameter search, one job per pretrained model,
           balanced across GPUs by model size (largest first)
         - few-shot evaluation of pretrained models without cached generations
         - the from-scratch transformers' hyperparameter search
       Each queue's output goes to logs/gpu<N>.log; progress is printed every few minutes.
    6. Rebuilds the combined summaries and the comparison plot from everything saved,
       and zips results/ to /kaggle/working/assignment3_results.zip.

Reruns are safe: tuned adapters and generations are cached in results/, so after a
timeout, rerunning the notebook only does the unfinished work.

Kaggle setup
    Settings: Accelerator "GPU T4 x2" (or P100), Internet "On".
    Optional: Add-ons > Secrets > HF_TOKEN (avoids Hub rate limits).
    Optional: upload your local Assignment3/results/ folder as a Kaggle dataset and
              attach it, so the 6 few-shot models already run on your Mac are reused.
    Notebook cell (safe to rerun: clones once, then pulls; always absolute paths):
        !cd /kaggle/working && ( [ -d repo ] && git -C repo pull -q || git clone -q https://github.com/KshaunishSoni123/MSDS458_KshaunishSoni.git repo )
        !python /kaggle/working/repo/Assignment3/kaggle_run.py
    For a run of several hours use "Save Version > Save & Run All (Commit)": it runs in
    the background (up to 12 h) and keeps /kaggle/working as the version's output.

Options
    --quick              smoke test (20 problems, tiny grids), ~10 min
    --models A B ...     pretrained models (default: MODELS in Assignment03_KshaunishSoni.py)
    --skip finetune scratch fewshot   leave out job types
    --gpus N             use only the first N GPUs
    --dry-run            print the plan and commands without running anything
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/KshaunishSoni123/MSDS458_KshaunishSoni.git"
WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
MAIN = "Assignment03_KshaunishSoni.py"
PROGRESS_EVERY_S = 300


def sh(cmd, **kwargs):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def find_code_dir(repo):
    """Folder that holds Assignment03_KshaunishSoni.py, under /kaggle/working."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, MAIN)) and here.startswith(WORKING):
        return here                                         # running from a clone already
    hits = sorted(glob.glob(os.path.join(WORKING, "**", MAIN), recursive=True), key=len)
    if hits:
        return os.path.dirname(hits[0])     # shallowest copy if it was cloned more than once
    hits = glob.glob(os.path.join("/kaggle/input", "**", MAIN), recursive=True)
    if hits:                                                # read-only dataset: copy it
        dest = os.path.join(WORKING, "Assignment3")
        shutil.copytree(os.path.dirname(hits[0]), dest, dirs_exist_ok=True)
        return dest
    sh(f"git clone -q {repo} {os.path.join(WORKING, 'repo')}")
    return os.path.dirname(glob.glob(os.path.join(WORKING, "repo", "**", MAIN), recursive=True)[0])


def seed_results(results_dir):
    """Copy earlier results from attached datasets; never overwrite newer local files."""
    sources = {os.path.dirname(p) for p in
               glob.glob("/kaggle/input/**/generations_*.jsonl", recursive=True)}
    for src in sorted(sources):
        if os.path.realpath(src) == os.path.realpath(results_dir):
            continue
        copied = 0
        for path in glob.glob(os.path.join(src, "**", "*"), recursive=True):
            dest = os.path.join(results_dir, os.path.relpath(path, src))
            if os.path.isfile(path) and not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(path, dest)
                copied += 1
        print(f"Seeded {copied} result files from {src}")


def install():
    sh(f"{sys.executable} -m pip install -q 'peft>=0.13' 'sacrebleu>=2.4'")
    # Kaggle's image ships torchao 0.10; recent peft refuses to build LoRA layers when an
    # old torchao is installed ("only versions above 0.16.0 are supported"). Nothing here
    # uses torchao (it is for quantized training), so remove it.
    sh(f"{sys.executable} -m pip uninstall -y -q torchao || true")
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 45):
        sh(f"{sys.executable} -m pip install -q -U 'transformers>=4.45' 'accelerate>=0.30'")


def hf_login():
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("Using the HF_TOKEN Kaggle secret")
    except Exception:
        print("No HF_TOKEN secret (fine for the default models)")


def gpu_names():
    out = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


# ----------------------------------------------------------------------------
# Plan: one queue of commands per GPU
# ----------------------------------------------------------------------------
def model_billions(model_id):
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", model_id.split("/")[-1])
    return float(m.group(1)) if m else 3.0


def plan(models, n_gpus, skip, quick, code_dir):
    """Longest-job-first assignment of jobs to GPUs; cost ~ model size.
    Fine-tuning a model costs ~4x its few-shot evaluation."""
    sys.path.insert(0, code_dir)
    from Assignment03_KshaunishSoni import generations_path, CONFIG   # noqa: E402
    q = ["--quick"] if quick else []
    n_eval = 20 if quick else 500
    jobs = []      # (cost, label, command)
    if "finetune" not in skip:
        for m in models:
            jobs.append((4 * model_billions(m), f"LoRA {m}",
                         ["Assignment03_finetune.py", "--models", m, *q]))
    if "fewshot" not in skip:
        CONFIG["eval_split"] = "test"
        # all models are passed so pretrained_metrics_summary.csv covers every one;
        # those with cached generations are only rescored (seconds)
        todo = [m for m in models if not os.path.exists(generations_path(m, n_eval))]
        print(f"Few-shot: {len(todo)} of {len(models)} models need generating: {todo}")
        jobs.append((max(0.1, sum(model_billions(m) for m in todo)),
                     f"few-shot ({len(todo)} to generate)",
                     [MAIN, "--only", "pretrained", "--models", *models, *q]))
    if "scratch" not in skip:
        jobs.append((1.5, "from-scratch grid search", ["Assignment03_transformer_models.py", *q]))

    queues = [{"cost": 0.0, "jobs": []} for _ in range(n_gpus)]
    for job in sorted(jobs, key=lambda j: -j[0]):
        target = min(queues, key=lambda qu: qu["cost"])
        target["cost"] += job[0]
        target["jobs"].append(job)
    return queues


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def launch(queues, code_dir, log_dir):
    procs = []
    for gpu, queue in enumerate(queues):
        if not queue["jobs"]:
            continue
        # ';' not '&&': one failed job does not stop the rest of the queue
        script = " ; ".join(
            f"echo '=== START {label} ===' ; {sys.executable} -u "
            + " ".join(f"'{a}'" for a in cmd) + f" ; echo \"=== END {label} (exit $?) ===\""
            for _, label, cmd in queue["jobs"])
        log = open(os.path.join(log_dir, f"gpu{gpu}.log"), "a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        procs.append((gpu, subprocess.Popen(["bash", "-c", script], cwd=code_dir, env=env,
                                            stdout=log, stderr=subprocess.STDOUT), log))
    return procs


def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            lines = [l for l in f.read().decode(errors="replace").replace("\r", "\n").split("\n")
                     if l.strip()]
        return lines[-1][:150] if lines else ""
    except OSError:
        return ""


def wait(procs, log_dir, start):
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(PROGRESS_EVERY_S)
        print(f"\n[{(time.time() - start) / 60:.0f} min]", flush=True)
        for gpu, p, _ in procs:
            state = "running" if p.poll() is None else f"done (exit {p.returncode})"
            print(f"  GPU {gpu} {state}: {last_line(os.path.join(log_dir, f'gpu{gpu}.log'))}",
                  flush=True)
    for gpu, _, log in procs:
        log.close()
        with open(os.path.join(log_dir, f"gpu{gpu}.log"), errors="replace") as f:
            ends = [l.strip() for l in f if l.startswith("=== END")]
        print(f"GPU {gpu}: " + "; ".join(ends))


def finalize(models, quick, skip, code_dir):
    """Rebuild the combined summaries from the cached results (no training)."""
    # --only finetuned: every fine-tuned model is cached now, so this only rescores them
    # (a model whose job failed is retried here), writes the combined fine-tuning CSV and
    # heatmap, reuses the saved few-shot / from-scratch summaries, and writes
    # model_comparison_summary.csv and the comparison plot
    flags = ["--only", "finetuned", "--models", *models] + (["--quick"] if quick else [])
    if "finetune" in skip:
        flags.append("--no-tune")
    try:
        sh(" ".join([sys.executable, MAIN, *flags]), cwd=code_dir)
    except subprocess.CalledProcessError as err:   # still zip whatever finished
        print(f"\nFinal summary step failed (exit {err.returncode}); see the error above "
              f"and the logs. Finished models are cached: rerun to complete.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", default=REPO_URL)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--skip", nargs="+", default=[], choices=["finetune", "scratch", "fewshot"])
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    start = time.time()

    code_dir = find_code_dir(args.repo)
    results_dir = os.path.join(code_dir, "results")
    log_dir = os.path.join(WORKING, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Code: {code_dir}\nResults: {results_dir}\nLogs: {log_dir}")
    seed_results(results_dir)

    gpus = gpu_names()
    n_gpus = min(len(gpus), args.gpus or len(gpus))
    print(f"GPUs: {gpus or 'none'}")
    if not n_gpus and not args.dry_run:
        sys.exit("No GPU found: in Kaggle set Settings > Accelerator to 'GPU T4 x2'.")
    if not args.dry_run:
        install()
        hf_login()

    sys.path.insert(0, code_dir)
    from Assignment03_KshaunishSoni import MODELS                       # noqa: E402
    models = args.models or MODELS
    queues = plan(models, max(n_gpus, 1), set(args.skip), args.quick, code_dir)
    for gpu, queue in enumerate(queues):
        print(f"\nGPU {gpu} (estimated cost {queue['cost']:.1f}):")
        for _, label, cmd in queue["jobs"]:
            print(f"  {label}:  python {' '.join(cmd)}")
    if args.dry_run:
        return

    wait(launch(queues, code_dir, log_dir), log_dir, start)
    finalize(models, args.quick, set(args.skip), code_dir)

    archive = shutil.make_archive(os.path.join(WORKING, "assignment3_results"), "zip", results_dir)
    print(f"\nDone in {(time.time() - start) / 60:.0f} min. Results: {results_dir}\n"
          f"Download: {archive} (Output tab, or the version's Output after a commit run)")


if __name__ == "__main__":
    main()

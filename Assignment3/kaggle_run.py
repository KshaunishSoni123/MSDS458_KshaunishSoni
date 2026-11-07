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
    7. Pushes results/ (and the run logs) to GitHub, on the branch kaggle-results, if a
       Kaggle secret named GITHUB_TOKEN exists. Kaggle deletes /kaggle/working when an
       interactive session ends, so this is the reliable way to keep the results.

GitHub push setup (once)
    1. github.com > Settings > Developer settings > Fine-grained tokens > Generate:
       Repository access "Only select repositories" > MSDS458_KshaunishSoni,
       Permissions > Contents "Read and write". Copy the token.
    2. Kaggle notebook > Add-ons > Secrets > Add: label GITHUB_TOKEN, value the token,
       and tick it for this notebook.
    3. On your Mac, after a run:
           git fetch origin kaggle-results
           git checkout origin/kaggle-results -- Assignment3/results
    The push goes to its own branch (not main) so it never conflicts with your local
    work. LoRA adapters are left out by default (a 3B model's adapters can be over
    GitHub's 100 MB file limit); --push-adapters includes those under the limit.

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
    --push-only          skip all work; just push the current results/ to GitHub (use it
                         from a still-open session whose results were not pushed)
    --push-branch NAME   branch to push to (default kaggle-results)
    --no-push            do not push to GitHub
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


def kaggle_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Push results to GitHub
# ----------------------------------------------------------------------------
GITHUB_FILE_LIMIT = 95 * 2**20      # GitHub rejects files over 100 MB


def git(args, cwd, token, check=True):
    """Run git without ever printing the token (it is in the remote URL)."""
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    text = (out.stdout + out.stderr).replace(token, "***")
    if check and out.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {text.strip()}")
    return out.returncode, text


def push_results(code_dir, repo_url, branch, log_dir, include_adapters, quick):
    """Copy results/ into a fresh clone of the repo, commit, push to `branch`."""
    token = kaggle_secret("GITHUB_TOKEN")
    if not token:
        print("\nNot pushing to GitHub: no GITHUB_TOKEN Kaggle secret (see the setup notes at "
              "the top of kaggle_run.py). Download assignment3_results.zip instead.")
        return False
    m = re.match(r"https://github\.com/(.+?)(?:\.git)?$", repo_url)
    if not m:
        print(f"\nNot pushing: --repo must be an https://github.com/... URL, got {repo_url}")
        return False
    auth_url = f"https://x-access-token:{token}@github.com/{m.group(1)}.git"

    # where results/ lives inside the repo (e.g. Assignment3/results)
    rc, top = git(["rev-parse", "--show-toplevel"], code_dir, token, check=False)
    sub = os.path.relpath(code_dir, top.strip()) if rc == 0 else os.path.basename(code_dir)
    clone = os.path.join("/tmp" if os.path.isdir("/tmp") else WORKING, "results_push")
    shutil.rmtree(clone, ignore_errors=True)
    try:
        rc, _ = git(["clone", "-q", "--depth", "1", "--branch", branch, auth_url, clone],
                    WORKING, token, check=False)
        if rc != 0:                                   # branch does not exist yet
            git(["clone", "-q", "--depth", "1", auth_url, clone], WORKING, token)
            git(["checkout", "-q", "-b", branch], clone, token)

        dest = os.path.join(clone, sub, "results")
        skipped = []
        for path in glob.glob(os.path.join(code_dir, "results", "**", "*"), recursive=True):
            rel = os.path.relpath(path, os.path.join(code_dir, "results"))
            if not os.path.isfile(path):
                continue
            if rel.startswith("adapters" + os.sep) and not include_adapters:
                continue
            if os.path.getsize(path) > GITHUB_FILE_LIMIT:
                skipped.append(rel)
                continue
            os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
            shutil.copy2(path, os.path.join(dest, rel))
        if os.path.isdir(log_dir):
            shutil.copytree(log_dir, os.path.join(dest, "kaggle_logs"), dirs_exist_ok=True)
        if skipped:
            print(f"Not pushed (over GitHub's 100 MB limit): {skipped}")

        name = os.environ.get("GIT_AUTHOR_NAME", "Kaggle run")
        email = os.environ.get("GIT_AUTHOR_EMAIL", "kaggle-run@users.noreply.github.com")
        git(["config", "user.name", name], clone, token)
        git(["config", "user.email", email], clone, token)
        git(["add", "-A", os.path.join(sub, "results")], clone, token)
        rc, _ = git(["diff", "--cached", "--quiet"], clone, token, check=False)
        if rc == 0:
            print(f"\nGitHub: results unchanged on branch {branch}; nothing to push")
            return True
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        git(["commit", "-q", "-m", f"Kaggle results{' (quick)' if quick else ''}, {stamp}"],
            clone, token)
        git(["push", "-q", "origin", branch], clone, token)
        print(f"\nPushed results to https://github.com/{m.group(1)}/tree/{branch}/{sub}/results\n"
              f"On your Mac:  git fetch origin {branch} && "
              f"git checkout origin/{branch} -- {sub}/results")
        return True
    except RuntimeError as err:
        print(f"\nGitHub push failed: {err}\nCheck that the token has Contents: Read and write "
              f"on this repository. The zip in /kaggle/working is still there.")
        return False
    finally:
        shutil.rmtree(clone, ignore_errors=True)


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
    p.add_argument("--push-only", action="store_true")
    p.add_argument("--push-branch", default="kaggle-results")
    p.add_argument("--push-adapters", action="store_true")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    start = time.time()

    code_dir = find_code_dir(args.repo)
    results_dir = os.path.join(code_dir, "results")
    log_dir = os.path.join(WORKING, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Code: {code_dir}\nResults: {results_dir}\nLogs: {log_dir}")
    if args.push_only:
        push_results(code_dir, args.repo, args.push_branch, log_dir, args.push_adapters,
                     args.quick)
        return
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
    if not args.no_push:
        push_results(code_dir, args.repo, args.push_branch, log_dir, args.push_adapters,
                     args.quick)


if __name__ == "__main__":
    main()

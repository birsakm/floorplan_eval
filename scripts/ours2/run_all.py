"""Evaluate one Mason2 generator version end to end: fetch, convert, baselines on the same programs, all metrics,
comparison table and the paper's Table 1 block (b) numbers.

  python scripts/ours2/run_all.py list [--min-homes 500]       # versions on noface, newest first
  python scripts/ours2/run_all.py <version>                    # full run (about 4-5 h for ~1,000 homes)
  python scripts/ours2/run_all.py <version> --redo procthor    # re-run a step and everything after it
  python scripts/ours2/run_all.py <version> --status           # which steps are done
  python scripts/ours2/run_all.py <version> --no-paper         # leave paper/sec/results_auto.tex alone
  python scripts/ours2/run_all.py <version> --benchmark rplan   # a version generated from the Table 1(a) RPLAN
                                                                # programs (scripts/rplan/export_mason_programs.py)

Run it with the fpe python (any python with the repo's requirements):
  PYTHONPATH=. /datawaha/cggroup/birsakm/miniconda3/envs/fpe/bin/python scripts/ours2/run_all.py <version>

Resumable: each finished step leaves outputs/ours/mason2_<version>/pipeline/<step>.done, so after a crash the same
command continues where it stopped. Logs: logs/mason2/<version>/<step>.log.

Steps (two lanes run in parallel after convert; only one metric evaluation runs at a time to bound memory):
  fetch      tunnel to noface (opened if needed) + scripts/ours2/fetch.py (serial API requests, read-only)
  convert    scripts/ours2/convert.py (fails if < 95% of the homes convert, e.g. after a format change;
             multi-storey homes are rejected, the converter handles one storey)
  ours       fpeval.evaluate variants + scripts/ours/evaluate_ours.py on our layouts
  procthor   ProcTHOR on our programs, then the same two evaluations          (lane A, after ours)
  infinigen  Infinigen on our programs (nice 10), convert, the two evaluations (lane B)
  report     scripts/report/compare_methods.py --version + scripts/ours2/paper_numbers.py --version
The paper itself is not committed or pushed.

--benchmark rplan (Table 1 block (a), APM row): fetch, convert (homes matched to their RPLAN input by input_id; the
real layouts of the same inputs as FID floor), ours (fpeval.evaluate variants + fid against RPLAN), report
(scripts/rplan/block_a.py --apm <version> + paper_numbers.py). No baseline lanes: ProcTHOR and Infinigen already ran
on these programs.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FPE_PY = os.environ.get("FPE_PY", "/datawaha/cggroup/birsakm/miniconda3/envs/fpe/bin/python")
PROCTHOR_PY = os.environ.get("PROCTHOR_PY", "/datawaha/cggroup/birsakm/venvs/fpe-procthor/bin/python")
INFINIGEN_PY = os.environ.get("INFINIGEN_PY", "/datawaha/cggroup/birsakm/miniconda3/envs/infinigen/bin/python")
NOFACE = os.environ.get("NOFACE", "michael@10.68.74.7")
PORT = 13021
STEPS = ["fetch", "convert", "ours", "procthor", "infinigen", "report"]
AFTER = {"fetch": ["convert"], "convert": ["ours", "procthor", "infinigen"], "ours": ["report"],
         "procthor": ["report"], "infinigen": ["report"], "report": []}  # steps that use a step's output
BENCH = {"rplan": ("rplan_g2p_test1000", "rplan_graph2plan:test")}  # --benchmark -> (programs, FID reference)
EVAL_LOCK = threading.Lock()  # one metric evaluation at a time (each uses --workers processes)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def variants(version):
    from fpeval.eval_config import mason2_variants
    return mason2_variants(version)


# --- noface ---------------------------------------------------------------------------------------------------------
def port_open():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def open_tunnel():
    """Open an SSH tunnel to the noface API unless one is already listening; returns the process (or None)."""
    if port_open():
        return None
    p = subprocess.Popen(["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                          "-L", f"{PORT}:127.0.0.1:3021", NOFACE])
    for _ in range(30):
        if port_open():
            return p
        if p.poll() is not None:
            break
        time.sleep(1)
    p.kill()
    raise SystemExit(f"could not open the tunnel to {NOFACE} (port {PORT})")


def api(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=120) as r:
        return json.load(r)


def cmd_list(min_homes):
    t = open_tunnel()
    try:
        rows = api("/api/generator/versions")["versions"]
    finally:
        if t:
            t.terminate()
    rows = sorted((r for r in rows if (r.get("homes") or 0) >= min_homes), key=lambda r: r["created_at"], reverse=True)
    done = {d.name[len("mason2_"):] for d in (ROOT / "outputs" / "ours").glob("mason2_*")}
    print(f"{'created':20} {'homes':>6} {'passing':>7} {'status':10}  version  (note; * = already here)")
    for r in rows:
        print(f"{r['created_at']:20} {r['homes']:>6} {r.get('passing') or 0:>7} {str(r.get('publication_status')):10}  "
              f"{r['id']}{' *' if r['id'] in done else ''}  ({r.get('note') or ''})")


# --- steps ----------------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, version, workers, reference, paper=True, benchmark=None):
        self.v, self.workers, self.reference, self.paper = version, workers, reference, paper
        self.bench = BENCH[benchmark][0] if benchmark else None
        if benchmark:
            self.reference = BENCH[benchmark][1]
        self.keys = variants(version)
        self.pdir = ROOT / "outputs" / "ours" / f"mason2_{version}" / "pipeline"
        self.ldir = ROOT / "logs" / "mason2" / version
        self.ldir.mkdir(parents=True, exist_ok=True)

    def done(self, step):
        return (self.pdir / f"{step}.done").exists()

    def mark(self, step):
        self.pdir.mkdir(parents=True, exist_ok=True)
        (self.pdir / f"{step}.done").write_text(datetime.now().isoformat(timespec="seconds"))

    def sh(self, step, args, nice=0):
        """Run one command, appending to the step's log; raise on failure."""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1")
        cmd = (["nice", "-n", str(nice)] if nice else []) + [str(a) for a in args]
        with open(self.ldir / f"{step}.log", "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"{step}: exit {rc}, see {self.ldir / f'{step}.log'}")

    def evaluate(self, step, key):
        with EVAL_LOCK:
            log(f"{step}: evaluating {key}")
            self.sh(step, [FPE_PY, "-m", "fpeval.evaluate", "variants", key, "--workers", self.workers])
            self.sh(step, [FPE_PY, "scripts/ours/evaluate_ours.py", "--kind", "apartment", "--variant", key,
                           "--workers", self.workers])

    def fetch(self):
        t = open_tunnel()
        try:
            self.sh("fetch", [FPE_PY, "scripts/ours2/fetch.py", "--version", self.v,
                              "--base", f"http://127.0.0.1:{PORT}"])
        finally:
            if t:
                t.terminate()
        d = ROOT / "data" / "ours2" / self.v
        total = json.load(open(d / "index.json"))["total"]
        got = len(list((d / "homes").glob("*.json.gz")))
        if got < total:
            raise RuntimeError(f"fetch: {got} of {total} homes")
        log(f"fetch: {got} homes")

    def convert(self):
        self.sh("convert", [FPE_PY, "scripts/ours2/convert.py", "--version", self.v, "--reference", self.reference]
                + (["--benchmark", self.bench] if self.bench else []))
        info = json.load(open(ROOT / "outputs" / "ours" / f"mason2_{self.v}" / "run_info.json"))
        log(f"convert: {info['n_converted']}/{info['n_homes']} converted; room types {info['native_room_types']}")
        if info["failures"]:
            log(f"convert: {len(info['failures'])} FAILED, e.g. {next(iter(info['failures'].items()))}")
        if info["n_converted"] < 0.95 * info["n_homes"]:
            raise RuntimeError("convert: fewer than 95% of the homes converted (format change?), see run_info.json")

    def ours(self):
        if self.bench:  # rule metrics + FID against RPLAN with the real layouts of the same inputs as floor
            key = self.keys["ours"]
            self.sh("ours", [FPE_PY, "-m", "fpeval.evaluate", "variants", key, "--workers", self.workers])
            self.sh("ours", [FPE_PY, "-m", "fpeval.evaluate", "fid", key, "--workers", self.workers])
            return
        self.evaluate("ours", self.keys["ours"])

    def procthor(self):
        name = self.keys["procthor"].split("/", 1)[1]
        log("procthor: generating")
        self.sh("procthor", [PROCTHOR_PY, "scripts/procthor/generate.py", "--kind", "apartment", "--source",
                             self.keys["ours"], "--name", name, "--workers", self.workers])
        self.evaluate("procthor", self.keys["procthor"])

    def infinigen(self):
        name = self.keys["infinigen"].split("/", 1)[1]
        log("infinigen: generating (the long step)")
        common = ["--kind", "apartment", "--source", self.keys["ours"], "--name", name]
        self.sh("infinigen", [INFINIGEN_PY, "scripts/infinigen/generate.py", *common, "--workers", self.workers],
                nice=10)
        self.sh("infinigen", [FPE_PY, "scripts/infinigen/convert.py", *common])
        self.evaluate("infinigen", self.keys["infinigen"])

    def report(self):
        if self.bench:
            self.sh("report", [FPE_PY, "scripts/rplan/block_a.py", "--apm", self.v, "--workers", self.workers])
            if self.paper:
                self.sh("report", [FPE_PY, "scripts/ours2/paper_numbers.py"])
            print((ROOT / "results" / "rplan" / "block_a.md").read_text(), flush=True)
            return
        self.sh("report", [FPE_PY, "scripts/report/compare_methods.py", "--version", self.v])
        if self.paper:
            self.sh("report", [FPE_PY, "scripts/ours2/paper_numbers.py", "--version", self.v])
        md = (ROOT / "results" / "comparison.md").read_text()
        i = md.index("## On the same programs")
        print(md[i:md.index("\n## ", i + 3)], flush=True)

    # --- driver ---------------------------------------------------------------------------------------------------
    def step(self, name):
        if self.done(name):
            log(f"{name}: done before, skipped")
            return
        t0 = time.time()
        getattr(self, name)()
        self.mark(name)
        log(f"{name}: finished in {(time.time() - t0) / 60:.0f} min")

    def redo(self, name):
        todo, stack = set(), [name]
        while stack:
            s = stack.pop()
            if s not in todo:
                todo.add(s)
                stack += AFTER[s]
        for s in todo:
            (self.pdir / f"{s}.done").unlink(missing_ok=True)
        # the generators skip existing outputs, so a redo starts from empty output folders
        for m in ("procthor", "infinigen"):
            if m in todo:
                shutil.rmtree(ROOT / "outputs" / self.keys[m], ignore_errors=True)
        log(f"redo: {', '.join(s for s in STEPS if s in todo)}")

    def status(self):
        for s in STEPS:
            f = self.pdir / f"{s}.done"
            print(f"{s:10} {'done ' + f.read_text().strip() if f.exists() else '-'}")

    def all(self):
        self.step("fetch")
        self.step("convert")
        if self.bench:
            self.step("ours")
            self.step("report")
            log("all done. Results: results/rplan/block_a.md; paper numbers: paper/sec/results_auto.tex "
                "(compile, then commit and push the paper submodule by hand)")
            return
        errors = []

        def lane(*names):
            try:
                for n in names:
                    self.step(n)
            except Exception as e:  # noqa: BLE001  reported after both lanes
                errors.append(e)

        lanes = [threading.Thread(target=lane, args=("ours", "procthor")),
                 threading.Thread(target=lane, args=("infinigen",))]
        for t in lanes:
            t.start()
        for t in lanes:
            t.join()
        if errors:
            raise SystemExit("FAILED: " + "; ".join(map(str, errors)) + "\n(fix, then run the same command again)")
        self.step("report")
        log(f"all done. Results: results/comparison.md; paper numbers: paper/sec/results_auto.tex "
            f"(compile, then commit and push the paper submodule by hand)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="generator version id, or 'list'")
    ap.add_argument("--min-homes", type=int, default=500, help="list: hide versions with fewer homes")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reference", default="swiss_dwellings:all")
    ap.add_argument("--redo", choices=STEPS, help="re-run this step and all later ones")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-paper", action="store_true", help="do not rewrite paper/sec/results_auto.tex (test runs)")
    ap.add_argument("--benchmark", choices=sorted(BENCH), default=None,
                    help="the version was generated from the Table 1(a) programs (APM row)")
    a = ap.parse_args()
    if a.version == "list":
        return cmd_list(a.min_homes)
    r = Run(a.version, a.workers, a.reference, paper=not a.no_paper, benchmark=a.benchmark)
    if a.status:
        return r.status()
    if a.redo:
        r.redo(a.redo)
    log(f"Mason2 {a.version}: logs in {r.ldir}")
    r.all()


if __name__ == "__main__":
    main()

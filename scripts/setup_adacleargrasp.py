"""Initialize pinned Ada sources and apply the recorded native push correction."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "third_party" / "AdaClearGrasp"
REVISION = "2d9561a0995c38cd3305fce707ebf7a9090252e0"
PATCH = ROOT / "third_party" / "patches" / "adacleargrasp-push-pose.patch"


def setup():
    subprocess.run(["git", "submodule", "update", "--init", "third_party/AdaClearGrasp"], cwd=ROOT, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=NATIVE, text=True).strip()
    if revision != REVISION:
        raise RuntimeError("Ada submodule revision differs from the audited native baseline")
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=NATIVE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if reverse.returncode == 0:
        print("Ada source and push-pose correction are ready")
        return
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=NATIVE, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=NATIVE, check=True)
    print("Applied the tracked native push-pose correction")


if __name__ == "__main__":
    setup()

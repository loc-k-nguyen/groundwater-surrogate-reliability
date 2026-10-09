"""Run explicit fast or full CPU suites with a bounded thread budget."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fast", "full"), default="fast")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and (args.report.exists() or ROOT in args.report.resolve().parents):
        raise FileExistsError("Use a new report outside the package")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    if args.suite == "full":
        suite = loader.discover(str(ROOT / "tests"))
    else:
        suite = unittest.TestSuite(loader.discover(str(ROOT / "tests"), pattern=name) for name in
            ("test_conformal_scale_v6.py", "test_synthetic_v6.py", "test_release_notices.py"))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"suite": args.suite, "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "seconds": time.perf_counter() - started, "cpu_threads": 1,
              "passed": result.wasSuccessful()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    print(json.dumps(record))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import path, symlink
from typing import Any

import git
import pandas as pd
import yaml
from tabulate import tabulate

from xl2times import utils
from xl2times.__main__ import parse_args, run
from xl2times.dd_to_csv import main
from xl2times.utils import max_workers

logger = utils.get_logger()


def parse_result(output: str) -> tuple[float, int, int]:
    # find pattern in multiline string
    m = re.findall(
        r"(\d+\.\d)% of ground truth rows present in output \((\d+)/(\d+)\), (\d+) additional rows",
        output,
        flags=re.MULTILINE,
    )
    if not m or len(m) == 0:
        logger.error(f"could not parse output of run:\n{output}")
        sys.exit(2)
    # return (accuracy, num_correct_rows, num_additional_rows)
    m = m[0]
    return (float(m[0]), int(m[1]), int(m[3]))


def run_gams_gdxdiff(
    benchmark: Any,
    times_folder: str,
    dd_folder: str,
    out_folder: str,
    verbose: bool = False,
) -> str:
    if "dd_files" not in benchmark:
        return "Error: dd_files not in benchmark"

    # Copy GAMS scaffolding
    scaffolding_folder = path.join(
        path.dirname(path.realpath(__file__)), "..", "xl2times", "gams_scaffold"
    )
    shutil.copytree(scaffolding_folder, out_folder, dirs_exist_ok=True)
    # Create link to TIMES source
    if not path.exists(path.join(out_folder, "source")):
        symlink(times_folder, path.join(out_folder, "source"), True)

    # Run GAMS
    res = subprocess.run(
        ["gams", "runmodel"],
        cwd=out_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        logger.info(res.stdout)
        logger.info(res.stderr if res.stderr is not None else "")
        logger.error(f"GAMS failed on {benchmark['name']}")
        sys.exit(3)
    if "error" in res.stdout.lower():
        logger.info(res.stdout)
        logger.error(f"GAMS errored on {benchmark['name']}")
        return "Error running GAMS"

    # Run GAMS on ground truth:
    shutil.copytree(scaffolding_folder, dd_folder, dirs_exist_ok=True)
    # Modify batinclude files according to benchmarks.yml's `dd_files`
    scenario_run = open(path.join(dd_folder, "scenario.run")).readlines()
    with open(path.join(dd_folder, "scenario.run"), "w") as f:
        for line in scenario_run:
            if line.strip() == "$BATINCLUDE output.dd":
                for file in benchmark["dd_files"]:
                    f.write(f"$BATINCLUDE {file}.dd\n")
                continue
            f.write(line)
    # Create link to TIMES source
    if not path.exists(path.join(dd_folder, "source")):
        symlink(times_folder, path.join(dd_folder, "source"), True)
    res = subprocess.run(
        ["gams", "runmodel"],
        cwd=dd_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        logger.info(res.stdout)
        logger.info(res.stderr if res.stderr is not None else "")
        logger.error(f"GAMS failed on {benchmark['name']} ground truth")
        sys.exit(4)
    if "error" in res.stdout.lower():
        logger.info(res.stdout)
        logger.error(f"GAMS errored on {benchmark['name']}")
        return "Error running GAMS on ground truth"

    # Run gdxdiff to compare
    res = subprocess.run(
        [
            "gdxdiff",
            path.join(dd_folder, "scenario.gdx"),
            path.join(out_folder, "scenario.gdx"),
            path.join(out_folder, "diffile.gdx"),
            "Eps=0.000001",
            "RelEps=0.000001",
            "setDesc=N",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if verbose:
        logger.info(res.stdout)
        logger.info(res.stderr if res.stderr is not None else "")
    if res.returncode != 0:
        # Report the number of lines in the gdxdump of the difffile
        res = subprocess.run(
            ["gdxdump", path.join(out_folder, "diffile.gdx")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if verbose or res.returncode != 0:
            logger.info(res.stdout)
            logger.info(res.stderr if res.stderr is not None else "")
        if res.returncode != 0:
            logger.error(f"GAMS gdxdiff failed on {benchmark['name']}")
            sys.exit(4)
        return str(len(res.stdout.splitlines()))

    return "OK"


def run_benchmark(
    benchmark: Any,
    benchmarks_folder: str,
    times_folder: str,
    run_gams: bool = False,
    skip_csv: bool = False,
    out_folder: str = "out",
    verbose: bool = False,
    debug: bool = False,
) -> tuple[str, float, str, float, int, int]:
    xl_folder = path.join(benchmarks_folder, "xlsx", benchmark["input_folder"])
    dd_folder = path.join(benchmarks_folder, "dd", benchmark["dd_folder"])
    csv_folder = path.join(benchmarks_folder, "csv", benchmark["name"])
    out_folder = path.join(benchmarks_folder, out_folder, benchmark["name"])

    # First convert ground truth DD to csv
    if not skip_csv:
        shutil.rmtree(csv_folder, ignore_errors=True)
        if not debug:
            # run as subprocess if not in --debug mode
            res = subprocess.run(
                [
                    "python",
                    "xl2times/dd_to_csv.py",
                    dd_folder,
                    csv_folder,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True if os.name == "nt" else False,
                check=False,
                encoding="utf-8",
            )
            if res.returncode != 0:
                # Remove partial outputs
                shutil.rmtree(csv_folder, ignore_errors=True)
                logger.info(res.stdout)
                logger.error(f"dd_to_csv failed on {benchmark['name']}")
                sys.exit(5)
        else:
            # If debug option is set, run as a function call to allow stepping with a debugger.
            try:
                main([dd_folder, csv_folder])
            except Exception:
                logger.exception(f"dd_to_csv failed on {benchmark['name']}")
                shutil.rmtree(csv_folder, ignore_errors=True)
                sys.exit(5)

    elif not path.exists(csv_folder):
        logger.error(f"--skip_csv is true but {csv_folder} does not exist")
        sys.exit(6)

    # Then run the tool
    args = [
        "--output_dir",
        out_folder,
        "--ground_truth_dir",
        csv_folder,
    ]
    args += ["--dd"] if run_gams else []
    if "regions" in benchmark:
        args.extend(["--regions", benchmark["regions"]])
    if "inputs" in benchmark:
        args.extend(path.join(xl_folder, b) for b in benchmark["inputs"])
    else:
        args.append(xl_folder)
    start = time.time()

    res = None
    if not debug:
        res = subprocess.run(
            ["xl2times"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            encoding="utf-8",
        )
    else:
        # If debug option is set, run as a function call to allow stepping with a debugger.
        summary = run(parse_args(args))
        # pack the results into a namedtuple pretending to be a return value from a subprocess call (as above).
        res = namedtuple("stdout", ["stdout", "stderr", "returncode"])(summary, "", 0)

    runtime = time.time() - start

    if verbose:
        line = "-" * 80
        logger.info(f"\n{line}\n{benchmark['name']}\n{line}\n\n{res.stdout}")
        logger.info(res.stderr if res.stderr is not None else "")
    else:
        logger.info(".", end="", flush=True)

    if res.returncode != 0:
        logger.info(res.stdout)
        logger.error(f"tool failed on {benchmark['name']}")
        sys.exit(7)
    with open(path.join(out_folder, "stdout"), "w") as f:
        f.write(res.stdout)

    (accuracy, num_correct, num_additional) = parse_result(res.stdout)

    if run_gams:
        dd_res = run_gams_gdxdiff(benchmark, times_folder, dd_folder, out_folder)
    else:
        dd_res = "--"

    return (benchmark["name"], runtime, dd_res, accuracy, num_correct, num_additional)


def run_all_benchmarks(
    benchmarks_folder: str,
    benchmarks: list,
    times_folder: str,
    run_gams=False,
    skip_csv=False,
    skip_main=False,
    skip_regression=False,
    seq: bool = False,
    verbose: bool = False,
    debug: bool = False,
):
    logger.info("Running benchmarks", end="", flush=True)
    headers = ["Benchmark", "Time (s)", "GDX Diff", "Accuracy", "Correct", "Additional"]
    run_a_benchmark = partial(
        run_benchmark,
        benchmarks_folder=benchmarks_folder,
        times_folder=times_folder,
        skip_csv=skip_csv,
        run_gams=run_gams,
        verbose=verbose,
        debug=debug,
    )

    if debug or seq:
        # bypass process pool and call benchmarks directly if --debug is set.
        results = [run_a_benchmark(b) for b in benchmarks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_a_benchmark, benchmarks))

    logger.info("\n\n" + tabulate(results, headers, floatfmt=".1f") + "\n")

    if skip_regression:
        logger.info("Skipping regression tests.")
        sys.exit(0)

    # The rest of this script checks regressions against main
    # so skip it if we're already on main
    repo = git.Repo(".")  # pyright: ignore
    origin = (
        repo.remotes.origin if "origin" in repo.remotes else repo.remotes[0]
    )  # don't assume remote is called 'origin'
    origin.fetch("main")
    if "main" not in repo.heads:
        repo.create_head("main", origin.refs.main).set_tracking_branch(origin.refs.main)
    try:
        mybranch = repo.active_branch
    except TypeError:  # If we're not on a branch (like on CI), create one:
        mybranch = repo.create_head("mybranch")

    if mybranch.name == "main":
        logger.info("Skipping regression tests as we're on main branch. Goodbye!")
        sys.exit(0)

    if skip_main:
        results_main = []
        for benchmark in benchmarks:
            with open(
                path.join(benchmarks_folder, "out-main", benchmark["name"], "stdout"),
            ) as f:
                result = parse_result(f.readlines()[-1])
            # Use a fake runtime and GAMS result
            results_main.append((benchmark["name"], 999, "--", *result))
        logger.info(
            f"Skipped running on main. Using results from {path.join(benchmarks_folder, 'out-main')}"
        )

    else:
        if repo.is_dirty():
            logger.info(
                "Your working directory is not clean. Skipping regression tests."
            )
            sys.exit(8)

        # Re-run benchmarks on main - check it out and pull
        repo.heads.main.checkout()
        origin.pull("main")  # if main already exists, make sure it's up to date
        logger.info("Running benchmarks on main")
        run_a_benchmark = partial(
            run_benchmark,
            benchmarks_folder=benchmarks_folder,
            times_folder=times_folder,
            skip_csv=True,
            run_gams=run_gams,
            out_folder="out-main",
            verbose=verbose,
            debug=debug,
        )

        if debug or seq:
            results_main = [run_a_benchmark(b) for b in benchmarks]
        else:
            with ProcessPoolExecutor(max_workers) as executor:
                results_main = list(executor.map(run_a_benchmark, benchmarks))

    # Print table with combined results to make comparison easier
    trunc = lambda s: s[:10] + "\u2026" if len(s) > 10 else s  # noqa

    combined_results = [
        (
            f"{b:<20}",
            f"{t0:5.1f} {t:5.1f}",
            f"{trunc(f0):<10} {trunc(f):<10}",
            f"{a0:5.1f} {a:5.1f}",
            f"{c0:6d} {c:6d}",
            f"{d0:6d} {d:6d}",
        )
        for ((b, t, f, a, c, d), (_, t0, f0, a0, c0, d0)) in zip(results, results_main)
    ]
    logger.info("\n\n" + tabulate(combined_results, headers, stralign="right") + "\n")

    # Checkout back to branch
    mybranch.checkout()

    # Compare results
    main_headers = ["Benchmark"] + ["M " + h for h in headers[1:]]
    df = pd.merge(
        pd.DataFrame(results_main, columns=main_headers),
        pd.DataFrame(results, columns=headers),
        on="Benchmark",
        how="outer",
    )
    if df.isna().values.any():
        logger.error(f"number of benchmarks changed:\n{df}")
        sys.exit(9)
    accu_regressions = df[df["Correct"] < df["M Correct"]]["Benchmark"]
    addi_regressions = df[df["Additional"] > df["M Additional"]]["Benchmark"]
    time_regressions = df[df["Time (s)"] > 2 * df["M Time (s)"]]["Benchmark"]

    our_time = df["Time (s)"].sum()
    main_time = df["M Time (s)"].sum()
    runtime_change = our_time - main_time

    logger.info(f"Total runtime: {our_time:.2f}s (main: {main_time:.2f}s)")
    logger.info(
        f"Change in runtime (negative == faster): {runtime_change:+.2f}s ({100 * runtime_change / main_time:+.1f}%)"
    )

    our_correct = df["Correct"].sum()
    main_correct = df["M Correct"].sum()
    correct_change = our_correct - main_correct
    logger.info(
        f"Change in correct rows (higher == better): {correct_change:+d} ({100 * correct_change / main_correct:+.1f}%)"
    )

    our_additional_rows = df["Additional"].sum()
    main_additional_rows = df["M Additional"].sum()
    additional_change = our_additional_rows - main_additional_rows
    logger.info(
        f"Change in additional rows: {additional_change:+d} ({100 * additional_change / main_additional_rows:+.1f}%)"
    )

    if len(accu_regressions) + len(addi_regressions) + len(time_regressions) > 0:
        if not accu_regressions.empty:
            logger.error(f"correct rows regressed on: {', '.join(accu_regressions)}")
        if not addi_regressions.empty:
            logger.error(f"additional rows regressed on: {', '.join(addi_regressions)}")
        if not time_regressions.empty:
            logger.error(f"runtime regressed on: {', '.join(time_regressions)}")
        sys.exit(10)
    # TODO also check if any new tables are missing?

    logger.success("No regressions. You're awesome!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "benchmarks_yaml",
        type=str,
        help="Benchmarks specification file",
    )
    args_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run a single benchmark instead of all benchmarks",
    )
    args_parser.add_argument(
        "--dd",
        action="store_true",
        default=False,
        help="Generate DD files, and use GAMS to compare with ground truth",
    )
    args_parser.add_argument(
        "--times_dir",
        type=str,
        default=None,
        help="Absolute path to TIMES_model. Required if using --dd",
    )
    args_parser.add_argument(
        "--skip_csv",
        action="store_true",
        default=False,
        help="Skip generating csv versions of ground truth DD files",
    )
    args_parser.add_argument(
        "--skip_main",
        action="store_true",
        default=False,
        help="Skip running tool on main and reuse existing result files",
    )
    args_parser.add_argument(
        "--skip_regression",
        action="store_true",
        default=False,
        help="Skip regression testing against main branch",
    )
    args_parser.add_argument(
        "--seq",
        action="store_true",
        default=False,
        help="Run benchmarks sequentially, instead of in parallel",
    )
    args_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print output of run on each benchmark",
    )
    args_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run each benchmark as a direct function call (disables subprocesses) to allow a debugger to stop at breakpoints "
        "in benchmark runs.",
    )
    args = args_parser.parse_args()

    spec = yaml.safe_load(open(args.benchmarks_yaml))
    benchmarks_folder = spec["benchmarks_folder"]
    benchmark_names = [b["name"] for b in spec["benchmarks"]]
    if len(set(benchmark_names)) != len(benchmark_names):
        logger.error("Found duplicate name in benchmarks YAML file")
        sys.exit(11)

    if args.dd and args.times_dir is None:
        logger.error("--times_dir is required when using --dd")
        sys.exit(12)

    if args.run is not None:
        benchmark = next((b for b in spec["benchmarks"] if b["name"] == args.run), None)
        if benchmark is None:
            logger.error(f"could not find {args.run} in {args.benchmarks_yaml}")
            sys.exit(13)

        _, runtime, gms, acc, cor, add = run_benchmark(
            benchmark,
            benchmarks_folder,
            times_folder=args.times_dir,
            run_gams=args.dd,
            skip_csv=args.skip_csv,
            verbose=args.verbose,
            debug=args.debug,
        )
        logger.info(
            f"Ran {args.run} in {runtime:.2f}s. {acc}% ({cor} correct, {add} additional).\n"
            f"GAMS: {gms}"
        )
    else:
        run_all_benchmarks(
            benchmarks_folder,
            benchmarks=spec["benchmarks"],
            times_folder=args.times_dir,
            run_gams=args.dd,
            skip_csv=args.skip_csv,
            skip_main=args.skip_main,
            skip_regression=args.skip_regression,
            seq=args.seq,
            verbose=args.verbose,
            debug=args.debug,
        )

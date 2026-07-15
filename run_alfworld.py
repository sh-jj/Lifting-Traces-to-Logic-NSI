"""Run the released NSI pipeline on ALFWorld."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from runner_utils import (
    fail_on_state_errors,
    requested_state_errors,
    trial_directory,
    validate_common_args,
)

ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate released NSI on ALFWorld")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trial_name", default="release")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--eval_split",
        default="eval_out_of_distribution",
        choices=["eval_out_of_distribution", "eval_in_distribution"],
    )
    parser.add_argument(
        "--log_folder",
        default=str(Path.cwd() / "outputs" / "alfworld"),
        help="Parent directory for trial outputs.",
    )
    parser.add_argument(
        "--alfworld_data",
        default=os.environ.get("ALFWORLD_DATA"),
        help="ALFWorld data root; defaults to ALFWORLD_DATA.",
    )
    parser.add_argument("--force_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load_skill", action="store_true")
    parser.add_argument("--load_call", action="store_true")
    parser.add_argument("--all_ready", action="store_true")
    parser.add_argument("--online_mode", action="store_true")
    parser.add_argument(
        "--sub_task",
        default=None,
        choices=["put", "clean", "heat", "cool", "examine", "puttwo"],
    )
    parser.add_argument(
        "--induction_mode",
        default="ilp",
        choices=["single", "ilp"],
        help="Offline induction pipeline (default: structural 'ilp').",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_common_args(parser, args, online_flag="online_mode")
    if not args.alfworld_data:
        parser.error("set ALFWORLD_DATA or pass --alfworld_data.")
    data_root = Path(args.alfworld_data).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"ALFWorld data directory does not exist: {data_root}")
    required_data = (
        data_root / "json_2.1.1",
        data_root / "logic" / "alfred.pddl",
    )
    missing_data = [str(path) for path in required_data if not path.exists()]
    if missing_data:
        parser.error("ALFWorld data root is incomplete; missing:\n  - " + "\n  - ".join(missing_data))
    os.environ["ALFWORLD_DATA"] = str(data_root)

    examples_dir = ROOT / "alfworld_runs" / "grounding_examples"
    config_path = ROOT / "alfworld_runs" / "base_config.yaml"
    if not examples_dir.is_dir() or not config_path.is_file():
        parser.error("the installed ALFWorld examples or base_config.yaml are missing.")

    try:
        log_folder = trial_directory(args.log_folder, args.llm, args.trial_name)
    except ValueError as exc:
        parser.error(str(exc))
    workflow_dir_name = "ilp_workflow" if args.induction_mode == "ilp" else "induced_workflow"
    fail_on_state_errors(
        parser,
        requested_state_errors(
            log_folder,
            workflow_dir_name=workflow_dir_name,
            examples_dir=examples_dir,
            load_skill=args.load_skill,
            load_call=args.load_call,
            all_ready=args.all_ready,
        ),
    )
    log_folder.mkdir(parents=True, exist_ok=True)

    # ALFWorld reads ALFWORLD_DATA while importing its environment modules.
    from alfworld_runs.alfworld_test import AlfworldTest
    from alfworld_runs.agent import NesyMemorySkillAgent

    agent = NesyMemorySkillAgent(
        example_dir=str(examples_dir),
        log_folder=str(log_folder),
        llm=args.llm,
        temperature=args.temperature,
        load_skill=args.load_skill,
        load_call=args.load_call,
        all_ready=args.all_ready,
        online_mode=args.online_mode,
        induction_mode=args.induction_mode,
        verbose=args.verbose,
    )
    evaluator = AlfworldTest(
        config_path=str(config_path),
        split=args.eval_split,
        correction=True,
    )
    evaluator.test(
        agent,
        start_env_idx=args.start_index,
        num_envs=args.num_envs,
        log_folder=str(log_folder),
        verbose=args.verbose,
        force_run=args.force_run,
        recover_history_of_skill=True,
        sub_task=args.sub_task,
    )


if __name__ == "__main__":
    main()

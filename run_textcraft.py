"""Run the released NSI pipeline on TextCraft."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner_utils import (
    fail_on_state_errors,
    requested_state_errors,
    trial_directory,
    validate_common_args,
)

ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate released NSI on TextCraft")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trial_name", default="release")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--log_folder",
        default=str(Path.cwd() / "outputs" / "textcraft"),
        help="Parent directory for trial outputs.",
    )
    parser.add_argument("--force_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load_skill", action="store_true")
    parser.add_argument("--load_call", action="store_true")
    parser.add_argument("--all_ready", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--depth", type=int, default=None, choices=[2, 3, 4])
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_common_args(parser, args, online_flag="online")

    examples_dir = ROOT / "textcraft_runs" / "examples"
    if not examples_dir.is_dir():
        parser.error("the installed TextCraft examples are missing.")
    try:
        log_folder = trial_directory(args.log_folder, args.llm, args.trial_name)
    except ValueError as exc:
        parser.error(str(exc))
    fail_on_state_errors(
        parser,
        requested_state_errors(
            log_folder,
            workflow_dir_name="ilp_workflow",
            examples_dir=examples_dir,
            load_skill=args.load_skill,
            load_call=args.load_call,
            all_ready=args.all_ready,
        ),
    )

    from textcraft_runs.textcraft_test import TextcraftTest

    evaluator = TextcraftTest(split=args.split)
    available_envs = len(evaluator.env.goal_list)
    requested_end = args.start_index + args.num_envs
    if requested_end > available_envs:
        parser.error(
            f"requested environment range [{args.start_index}, {requested_end}) exceeds "
            f"the {args.split} split size ({available_envs})."
        )

    log_folder.mkdir(parents=True, exist_ok=True)
    from textcraft_runs.agent import NesyMemorySkillAgent

    agent = NesyMemorySkillAgent(
        example_path=None,
        example_dir=str(examples_dir),
        log_folder=str(log_folder),
        llm=args.llm,
        temperature=args.temperature,
        load_skill=args.load_skill,
        load_call=args.load_call,
        all_ready=args.all_ready,
        online_mode=args.online,
        verbose=args.verbose,
    )
    evaluator.test(
        agent,
        start_env_idx=args.start_index,
        num_envs=args.num_envs,
        log_folder=str(log_folder),
        verbose=args.verbose,
        force_run=args.force_run,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()

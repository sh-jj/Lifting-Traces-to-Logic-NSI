"""Dependency-free validation helpers shared by the public runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


READY_STATE_FILES = (
    "trajectory_memory_graph.json",
    "sub_goals.json",
    "skill_calling_memory_graph.json",
    "skill_json_file_for_run.json",
    "skill_memory_blocks_with_task_type.json",
    "revised_calling_memory.json",
    "task_level_guidelines.json",
)


def interaction_lines(skill_interaction: Any) -> List[str]:
    """Normalize an optional skill trace without inventing environment actions."""
    if skill_interaction is None:
        return []
    return str(skill_interaction).splitlines()


def validate_common_args(parser: Any, args: Any, *, online_flag: str) -> None:
    """Reject invalid or internally inconsistent public-runner arguments."""
    if args.start_index < 0:
        parser.error("--start_index must be non-negative.")
    if args.num_envs <= 0:
        parser.error("--num_envs must be greater than zero.")
    if not str(args.trial_name).strip():
        parser.error("--trial_name must not be empty.")

    online_enabled = bool(getattr(args, online_flag))
    if online_enabled and not args.all_ready:
        option = "--online_mode" if online_flag == "online_mode" else "--online"
        parser.error(
            f"{option} requires an existing state: also pass "
            "--load_skill --load_call --all_ready."
        )
    if args.all_ready and not (args.load_skill and args.load_call):
        parser.error("--all_ready requires both --load_skill and --load_call.")


def trial_directory(log_root: str, llm: str, trial_name: str) -> Path:
    """Return a trial directory without allowing user values to alter its depth."""

    def checked_component(value: str, option: str) -> str:
        component = str(value).strip()
        if not component:
            raise ValueError(f"{option} must not be empty.")
        if component in {".", ".."} or "/" in component or "\\" in component or "\0" in component:
            raise ValueError(f"{option} must be a single path component (no '/' or '\\').")
        return component

    model = checked_component(llm, "--llm")
    trial = checked_component(trial_name, "--trial_name")
    return Path(log_root).expanduser().resolve() / f"nsms_{model}_{trial}"


def requested_state_errors(
    trial_dir: Path,
    *,
    workflow_dir_name: str,
    examples_dir: Path,
    load_skill: bool,
    load_call: bool,
    all_ready: bool,
) -> List[str]:
    """Describe missing/corrupt state before a load-mode agent is constructed."""
    errors: List[str] = []
    if not (load_skill or load_call or all_ready):
        return errors
    if not trial_dir.is_dir():
        return [f"trial directory does not exist: {trial_dir}"]

    workflow_dir = trial_dir / workflow_dir_name
    if load_skill:
        if not workflow_dir.is_dir():
            errors.append(f"workflow directory is missing: {workflow_dir}")
        elif not any(workflow_dir.glob("*_valid.json")):
            errors.append(f"no *_valid.json workflows found in: {workflow_dir}")

    calling_graph = trial_dir / "skill_calling_memory_graph.json"
    if load_call and not calling_graph.is_file():
        errors.append(f"state file is missing: {calling_graph}")

    if not all_ready:
        return errors

    loaded_json = {}
    for filename in READY_STATE_FILES:
        path = trial_dir / filename
        if not path.is_file():
            errors.append(f"state file is missing: {path}")
            continue
        try:
            loaded_json[filename] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"state file is not valid JSON: {path} ({exc})")

    grounding_dir = trial_dir / "grounding_examples"
    for source in sorted(examples_dir.rglob("*.txt")):
        relative = source.relative_to(examples_dir)
        expected = grounding_dir / relative.parent / f"{source.stem}_grounding.txt"
        if not expected.is_file():
            errors.append(f"grounded demonstration is missing: {expected}")

    skill_state = loaded_json.get("skill_json_file_for_run.json")
    if isinstance(skill_state, dict):
        runnable_skills = [name for name, value in skill_state.items() if value is not None]
        if not runnable_skills:
            errors.append("skill_json_file_for_run.json contains no runnable skills")
        for skill_name in runnable_skills:
            for suffix in ("json", "mmd"):
                path = workflow_dir / f"{skill_name}_valid.{suffix}"
                if not path.is_file():
                    errors.append(f"workflow file is missing: {path}")

    return errors


def fail_on_state_errors(parser: Any, errors: List[str]) -> None:
    """Render state preflight errors through argparse's normal error path."""
    if errors:
        parser.error("incomplete load state:\n  - " + "\n  - ".join(errors))

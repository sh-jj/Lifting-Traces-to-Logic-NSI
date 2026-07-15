import os
import csv
import traceback
from typing import Any, List, Set, Tuple

from tqdm import tqdm
import re

# try:
#     from textcraft import TextCraft
# except ImportError as exc:
#     raise ImportError("textcraft package is required to run TextcraftTest.") from exc

from .env import WrappedTextCraft
from runner_utils import interaction_lines

CRAFT_LINE_PATTERN = re.compile(
    r"^craft\s+(?:(?P<count>\d+)\s+)?(?P<item>.+?)\s+using\s+(?P<ingredients>.+)$",
    flags=re.IGNORECASE,
)
INGREDIENT_PATTERN = re.compile(r"(?P<count>\d+)\s+(?P<item>.+)$")


def _normalize_item_name(name: str) -> str:
    cleaned = (name or "").strip().lower().rstrip(".")
    if not cleaned:
        return ""
    return re.sub(r"\s+", " ", cleaned)


def _extract_item_names_from_commands(commands: str) -> List[str]:
    items: List[str] = []
    seen: Set[str] = set()
    for line in (commands or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.lower().startswith("crafting commands"):
            continue
        match = CRAFT_LINE_PATTERN.match(text)
        if not match:
            continue
        target_item = _normalize_item_name(match.group("item"))
        if target_item and target_item not in seen:
            items.append(target_item)
            seen.add(target_item)
        ingredients = match.group("ingredients") or ""
        for chunk in ingredients.split(","):
            token = chunk.strip()
            if not token:
                continue
            ingredient_match = INGREDIENT_PATTERN.match(token)
            ingredient_name = ingredient_match.group("item") if ingredient_match else token
            ingredient_item = _normalize_item_name(ingredient_name)
            if ingredient_item and ingredient_item not in seen:
                items.append(ingredient_item)
                seen.add(ingredient_item)
    return items

def write_line_to_main_log_csv(path: str, data: Any) -> None:
    """Write one line of output into the main CSV."""
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        if isinstance(data, list):
            writer.writerow(data)
        elif isinstance(data, dict):
            writer.writerow([val for _, val in data.items()])


def build_log_csv(log_folder: str, csv_file_name: str, rewrite_csv: bool = False) -> str:
    """Create the CSV file (with header) for logging results."""
    os.makedirs(log_folder, exist_ok=True)
    csv_path = os.path.join(log_folder, csv_file_name + ".csv")
    header = [
        "env_idx",
        "success",
        "done",
        "total_reward",
        "goal", 
        "depth", 
        "num_steps",
        "error",
    ]

    if rewrite_csv and os.path.exists(csv_path):
        os.remove(csv_path)

    if not os.path.exists(csv_path):
        write_line_to_main_log_csv(csv_path, header)

    return csv_path


class TextcraftTest:
    """Simple evaluator for TextCraft environments (mirrors alfworld_test style)."""

    def __init__(self, split, test_max_step: int = 40) -> None:
        self.env = WrappedTextCraft(split=split)
        self.test_max_step = test_max_step
        self.limit_action_repetitions = 5

    def test(
        self,
        agent: Any,
        start_env_idx: int = 0,
        num_envs: int = 10,
        verbose: bool = True,
        log_folder: str = "outputs/textcraft",
        force_run: bool = False,
        depth: int = None,
    ) -> None:
        csv_file = build_log_csv(log_folder, "textcraft_results")
        traj_dir = os.path.join(log_folder, "traj_data")
        os.makedirs(traj_dir, exist_ok=True)

        agent_log_dir = os.path.join(log_folder, "agent_log")
        os.makedirs(agent_log_dir, exist_ok=True)

        successes, counts = 0, 0

        with tqdm(total=num_envs, desc="TextCraft Episodes") as pbar:
            for env_idx in range(num_envs):
                observation, info = self.env.reset(env_index=start_env_idx + env_idx, verbose=verbose)

                if depth is not None and self.env.depth != depth:
                    pbar.update(1)
                    continue

                
                commands, task = observation.split("Goal: ", maxsplit=1)

                observation = "Your task is to: " + task.strip()

                item_name_domain = _extract_item_names_from_commands(commands)

                self.env.set_item_domain(item_name_domain)

                # print(commands)
                # print(item_name_domain)
                # exit(0)
                if hasattr(agent, "item_name_domain"):
                    agent.item_name_domain = item_name_domain

                task_name = f"{start_env_idx + env_idx}_goal_{self.env.goal}_depth_{self.env.depth}"
                safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", task_name)
                traj_path = os.path.join(traj_dir, f"{safe_name}.txt")
                if os.path.exists(traj_path):
                    os.remove(traj_path)
                
                with open(traj_path, "a", encoding="utf-8") as handle:
                    handle.write(commands.strip() + "\n")

                agent_log_path = os.path.join(agent_log_dir, f"{safe_name}.txt")
                if os.path.exists(agent_log_path):
                    os.remove(agent_log_path)
                
                with open(agent_log_path, "a", encoding="utf-8") as handle:
                    handle.write(commands.strip() + "\n")
                
                # Allow agents to reset any internal state.
                agent.reset(commands=commands, task=task, craft_depth=self.env.depth, 
                            agent_log_path=agent_log_path,
                            traj_path=traj_path)
                
                if verbose:
                    print(commands)
                
                num_current_repetitions, num_nothing_happens, num_repetitions = 0, 0, 0        
                num_correction = 0

                prev_action = "xxx"
                error_message = ""

                total_reward = 0
                success_flag = False
                done_flag = False
                error_message = ""
                
                

                try:
                    for step_i in range(self.test_max_step):

                        continue_flag = False

                        if verbose:
                            print("obs: ", observation.strip())

                        with open(traj_path, "a", encoding="utf-8") as handle:
                            handle.write("obs: " + observation.strip() + "\n")
                        
                        if observation.strip() == "OK.":
                            num_nothing_happens += 1
                        else:
                            num_nothing_happens = 0

                        action, response_type = self._get_action(agent, observation)

                        # if agent.name == "nsms-agent":
                        #     with open(traj_path, "a", encoding="utf-8") as handle:
                        #         handle.write("[sub-goal]: " + agent.sub_goal_instruction.strip() + "\n")


                        if verbose:
                            print(f"action [{response_type}]: {action}")

                        if response_type =="textual action":

                            with open(traj_path, "a", encoding="utf-8") as handle:
                                handle.write("> " + action.strip() + "\n")

                            if 'task failed!' in action.lower(): 
                                done_flag = True
                                success_flag = False
                                break

                            if action.startswith("think:"): 
                                observation = "OK.\n"
                                continue_flag = True
                            
                        elif response_type == "skill action":
                            action_json = action
                            if action_json["skill"] in ("none", None):
                                action = action_json.get("action") or "think: no action produced"
                                response_type = "textual action"
                                if action.startswith("think:"):
                                    observation = "OK.\n"
                                    continue_flag = True

                                with open(traj_path, "a", encoding="utf-8") as handle:
                                    handle.write("> " + action.strip() + "\n")
                            
                            
                            else:
                                action = str(action_json["skill"]) + "(" + ", ".join([f"{s_i[0]}={s_i[1]}" for s_i in action_json["parameter_bindings"].items()]) + ")"
                            

                                with open(traj_path, "a", encoding="utf-8") as handle:
                                    handle.write(">> " + action.strip() + "\n")
                            
                            continue_flag = True
                        
                        # Breaking early in case thoughts were repeated as well.
                        if min(num_nothing_happens, num_current_repetitions) == self.limit_action_repetitions:
                            early_stop = f"ENV_ERROR: Too many ({num_current_repetitions}) consecutive repetitions and nothing happens."
                            if verbose:
                                print("EARLY STOP: ", early_stop)
                            break
                        
                        
                        

                        if verbose:
                            if response_type == "textual action":
                                print("> "+action.strip())
                            else:
                                print(">> "+action.strip())
                            
                        # if action.startswith("skill: "): 
                        # check if action is a high-level skill
                        if response_type == "skill action":
                            

                            last_observation, skill_interaction, env_done_flag, task_success_flag = agent.execute_skill_online(action_json, self.env, observation)
                            
                            # print("skill_interaction: ")
                            # print(skill_interaction)
                            # print("---")
                            with open(traj_path, "a", encoding="utf-8") as handle:

                                interactions_list = interaction_lines(skill_interaction)
                                if not interactions_list:
                                    print("Warning: skill execution returned no interaction trace.")
                                for idx_inter, interaction in enumerate(interactions_list):
                                    if idx_inter % 2 == 0:
                                        handle.write("> " + interaction.strip() + "\n")

                                        if verbose:
                                            print("> " + interaction.strip())

                                        if not "think:" in interaction:
                                            step_i += 1
                                    else:
                                        if idx_inter == len(interactions_list) - 1:
                                            continue
                                        handle.write("obs: " + interaction.strip() + "\n")
                                        if verbose:
                                            print("obs: " + interaction.strip())
                            

                            # exit(0)
                            if last_observation is not None:
                                observation = str(last_observation)
                            elif interactions_list:
                                observation = interactions_list[-1]
                            
                            if verbose:
                                print("last observation: ", last_observation)

                            total_reward += task_success_flag
                            if env_done_flag:
                                success_flag = task_success_flag
                                done_flag = env_done_flag
                                break

                                         
                        else:
                            agent.update_history("> "+action) #Adding '>' symbol back in as this is how the example prompt is presented to LLMs.

                        if continue_flag:
                            continue


                        if action == prev_action:
                            num_current_repetitions += 1
                            num_repetitions += 1
                        else:
                            num_current_repetitions = 0
                            prev_action = action

                        # print("executing action in the environment: ", action)
                        observation, reward, done, _, step_info = self.env.step(action)
                        total_reward += reward


                        if reward > 0:
                            success_flag = True
                        if done:
                            done_flag = done
                            break

                    # end for step_i
                except Exception as exc:  # pylint: disable=broad-except
                    error_message = str(exc)
                    print(f"Error during test at env index {start_env_idx + env_idx}: {exc}")
                    traceback.print_exc()
                    if not force_run:
                        raise

                logging_row = {
                    "env_idx": start_env_idx + env_idx,
                    "success": success_flag,
                    "done": done_flag,
                    "total_reward": total_reward,
                    "goal": self.env.goal,
                    "depth": self.env.depth,
                    "num_steps": step_i + 1 if "step_i" in locals() else 0,
                    "error": error_message,
                }
                write_line_to_main_log_csv(csv_file, logging_row)

                # Post-episode online evolution analysis
                try:
                    agent.reveiew_episode(
                        success_flag=success_flag,
                        episode_idx=start_env_idx + env_idx,
                        goal=self.env.goal,
                        depth=self.env.depth,
                    )
                except Exception as evo_exc:
                    print(f"[EVOLUTION] Post-episode analysis failed: {evo_exc}")
                    traceback.print_exc()
                    if not force_run:
                        raise

                if hasattr(agent, "agent_log"):
                    with open(agent_log_path, "w", encoding="utf-8") as handle:
                        handle.write(agent.agent_log)

                successes += int(success_flag)
                counts += 1

                pbar.set_postfix({"success rate": f"{successes / counts:.4f}"})
                pbar.update(1)

        if counts:
            print(f"Final result: success {successes} / {counts} = {successes / counts:.4f}")
        else:
            print("No TextCraft episodes matched the requested filters.")

    @staticmethod
    def _get_action(agent: Any, obs: str) -> Tuple[str, str]:
        """Normalize the agent.act output to (action, response_type)."""
        act_out, response_type = agent.act(obs=obs)
        
        if response_type == "textual action":
            return act_out.strip(), response_type

        else:
            return act_out, response_type

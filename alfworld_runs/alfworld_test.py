
import os
import yaml
from alfworld.agents.environment import get_environment
from tqdm import tqdm
import re
import csv
import traceback
from runner_utils import interaction_lines

ENV_TYPES = {
'pick_and_place': 'put',
'pick_clean_then_place': 'clean',
'pick_heat_then_place': 'heat',
'pick_cool_then_place': 'cool',
'look_at_obj': 'examine',
'pick_two_obj': 'puttwo'
}

import json

def get_env_type(env_name):
    """ Extracts which type of env it is"""
    env_type = ""
    for key, value in ENV_TYPES.items():
        if env_name.startswith(key):
            env_type = value
    return env_type

# Receptacle/item names emitted by the planner often use underscores
# (e.g., 'go to drawer_1') because they were copied from FOL facts which use the
# `<type>_<idx>` form. ALFWorld's text env only accepts spaced names ('go to drawer 1'),
# so the underscore form silently 'Nothing happens'. Skill-execution path already
# normalizes via `inverse_all_items` (in infoflow2graph), but the raw-action path
# bypasses it. We apply the same regex here.
_ALFWORLD_NAME_RE = re.compile(r'\b([A-Za-z]+)_(\d+)\b')

def normalize_action_names(action: str) -> str:
    """Convert internal 'drawer_1' tokens back to env's 'drawer 1' form."""
    return _ALFWORLD_NAME_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", action or "")

def nojson_action_cleaning(action):
    """Action cleaning for no json."""
    action = action.strip()
    action+="\n"
    return action

def transform_put_action(action, transform_to_move_syntax=True):
    """ Put action grammar correction. """
    put_regex_1 = """put(?:\s\w+)(?:\s\w+)?(?:\s\d+)\son(?:\s\w+)(?:\s\w+)?(?:\s\d+)"""
    put_regex_2 = """put(?:\s\w+)(?:\s\w+)?(?:\s\d+)\sin(?:\s\w+)(?:\s\w+)?(?:\s\d+)"""
    correction_happened = False

    if action.startswith("put"):
        answer = re.match(put_regex_1,action)
        if answer:
            action = action.replace(" on "," in/on ")
            correction_happened = True
        else:
            answer = re.match(put_regex_2,action)
            if answer:
                action = action.replace(" in "," in/on ")
                correction_happened = True
    
    if transform_to_move_syntax:
        action = action.replace("put","move")
        action = action.replace("in/on", "to")
        
    return action, correction_happened

def process_ob(ob, track_nothing_happens=False):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]

    nothing_happens = False

    if ob == "Nothing happens.":
        nothing_happens = True
        # ob = "Invalid or Impossible Action. Try again."

    if not ob.endswith("\n"):
        ob += "\n"

    if track_nothing_happens:
        return ob, nothing_happens
    else:
        return ob
def write_line_to_main_log_csv(name, data):
    """Writes one line of output into the main CSV"""
    with open(name, 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if type(data) == list:
            wr.writerow(data)
        elif type(data) == dict:
            data_list = [x for _,x in data.items()]
            wr.writerow(data_list)

def build_log_csv(log_folder, csv_file_name, rewrite_csv=False):
    MAIN_CSV_FILEPATH = os.path.join(log_folder, csv_file_name+".csv")
    CSV_HEADER = [
        # GENERAL AGENT/ENV Settings
        "env_idx",
        "env_type",
        "temperature",
        # PER GAME METRICS
        "success",
        "done",
        "total_reward",
        "early_stop",
        "error",
        # PER STEP METRICS
        "num_of_steps",
        "num_nothing_happens",
        "num_repetitions",
        "num_action_correction",
        # TOKEN COUNT
        "total_in_tokens", #Total number of in tokens accumulated
        "total_out_tokens", #Total number of tokens of the entire history (measured at the end)
        "total_price_estimated"
    ]

    # Writing the Header
    os.makedirs(log_folder, exist_ok=True)
    if rewrite_csv:
        file_counter = 0
        while os.path.exists(MAIN_CSV_FILEPATH):
            MAIN_CSV_FILEPATH = os.path.join(log_folder, csv_file_name+str(file_counter)+".csv")
            file_counter+=1
        # Now that we have a new unique csv file name create it and write the header.
        write_line_to_main_log_csv(MAIN_CSV_FILEPATH, CSV_HEADER)
    else:
        if not os.path.exists(MAIN_CSV_FILEPATH): #Only write header once
            write_line_to_main_log_csv(MAIN_CSV_FILEPATH, CSV_HEADER)
    
    print("build CSV log >>> ", MAIN_CSV_FILEPATH)
    return MAIN_CSV_FILEPATH


class AlfworldTest:
    def __init__(self, config_path="", test_max_step=100, split="eval_out_of_distribution", correction=True):
        
        with open(config_path) as reader:
            config = yaml.safe_load(reader)
        env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

        self.config = config
        self.env = get_environment(env_type)(config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
        self.test_max_step = test_max_step

        self.limit_action_repetitions = 5
        self.correction = correction


    def test(self, agent, start_env_idx=0, num_envs=134, verbose=True,
             log_folder="outputs/alfworld", force_run=False,
             recover_history_of_skill=False, 
             sub_task=None):

        overall_price = 0
        csv_file_name = "alfworld_results"
        self.log_csv_file = build_log_csv(log_folder, csv_file_name)

        self.traj_data_dir = os.path.join(log_folder, "traj_data")
        os.makedirs(self.traj_data_dir, exist_ok=True)

        self.agent_log_dir = os.path.join(log_folder, "agent_log")
        os.makedirs(self.agent_log_dir, exist_ok=True)

        if hasattr(agent, "set_log_dir"):
            agent.set_log_dir(log_folder)
        

        # Skipping Envs
        for i in range(start_env_idx):
            observation, info = self.env.reset()

        cnts = [0] * 6
        rs = [0] * 6
        with tqdm(total=num_envs, desc="ALFWorld Episodes") as pbar:
            for env_idx in range(0, num_envs):

                observation, info = self.env.reset()
                observation = '\n'.join(observation[0].split('\n\n')[1:])
                task_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
                observation += "\n"

                env_type = get_env_type(task_name)
                if sub_task is not None and env_type != sub_task:
                    pbar.update(1)
                    continue
                

                for task_type_idx, (k, v) in enumerate(ENV_TYPES.items()):
                    if task_name.startswith(k):
                        # save_task_dir = os.path.join(save_dir, f"{k}")
                        # os.makedirs(save_task_dir, exist_ok=True)
                        break
                safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", task_name)

                env_episode_name = f"index_{start_env_idx + env_idx}_{safe_name}"
                traj_data_file = os.path.join(self.traj_data_dir, f"{env_episode_name}.txt")
                if os.path.exists(traj_data_file):
                    os.remove(traj_data_file)

                agent_log_file = os.path.join(self.agent_log_dir, f"{env_episode_name}.txt")
                if os.path.exists(agent_log_file):
                    os.remove(agent_log_file)

                agent.reset(env_type=env_type, env_id = env_episode_name)

                if hasattr(agent, 'set_episode_info'):
                    agent.set_episode_info(
                        traj_path=traj_data_file,
                        task_name=task_name,
                        env_type=env_type,
                    )

                num_current_repetitions, num_nothing_happens, num_repetitions = 0, 0, 0
                num_correction = 0

                prev_action = "xxx"
                error_message = ""

                total_reward = 0

                success_flag, done_flag = False, False

                global completion_tokens, prompt_tokens
                completion_tokens, prompt_tokens = 0, 0
                agent.reset_token_counts()

                if verbose:
                    print(f"Starting Env with Index: {env_idx+start_env_idx} of type: {env_type}")
                


                try:
                    for step_i in range(self.test_max_step):


                        continue_flag = False

                        with open(traj_data_file, "a") as f:
                            f.write("obs: " + observation.strip() + "\n")

                        if verbose:
                            print(observation.strip())

                        action, response_type = agent.act(obs=observation)

                        if verbose:
                            print(f"action [{response_type}]: {action}")

                        if response_type =="textual action":
                            action = nojson_action_cleaning(action) 
                            if action.startswith("think:"): 
                                observation = "OK.\n"
                                continue_flag = True
                        elif response_type == "skill action":
                            action_json = action
                            if action_json["skill"] == "none":
                                action = action_json["action"]
                                response_type = "textual action"
                                if action.startswith("think:"): 
                                    observation = "OK.\n"
                                    continue_flag = True
                            else:
                                action = str(action_json["skill"]) + "(" + ", ".join([f"{s_i[0]}={s_i[1]}" for s_i in action_json["parameter_bindings"].items()]) + ")"
                            continue_flag = True



                        if action == prev_action:
                            num_current_repetitions += 1
                            num_repetitions += 1
                        else:
                            num_current_repetitions = 0
                            prev_action = action
                        
                        # Breaking early in case thoughts were repeated as well.
                        if num_current_repetitions == self.limit_action_repetitions:
                            early_stop = f"ENV_ERROR: Too many ({num_current_repetitions}) consecutive repetitions."
                            if verbose:
                                print("EARLY STOP: ", early_stop)
                            break
                        
                        with open(traj_data_file, "a") as f:
                            f.write("> "+action.strip() + "\n")

                        if verbose:
                            print("> "+action.strip())
                            
                        # if action.startswith("skill: "): 
                        # check if action is a high-level skill
                        if response_type == "skill action":
                            

                            last_observation, skill_interaction, env_done_flag, task_success_flag = agent.execute_skill_online(action_json, self.env, process_ob)
                            
                            with open(traj_data_file, "a") as f:

                                interactions_list = interaction_lines(skill_interaction)
                                if not interactions_list:
                                    print("Warning: skill execution returned no interaction trace.")
                                for idx_inter, interaction in enumerate(interactions_list):
                                    if idx_inter % 2:
                                        f.write("> " + interaction.strip() + "\n")
                                    else:
                                        f.write("obs: " + interaction.strip() + "\n")
                            
                            observation = last_observation
                            
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

                        if self.correction:
                            # NOTE!!! Since PR 94 into alfworld put ... in/on ... to move ... to ...
                            # https://github.com/alfworld/alfworld/pull/94
                            actual_action, correction_happened = transform_put_action(action, transform_to_move_syntax=True)
                            if correction_happened:
                                num_correction += 1
                                if verbose:
                                    print(f"transformed action: {actual_action.strip()} <- {action}")
                            
                            action = actual_action

                        # Normalize 'drawer_1' -> 'drawer 1' before sending to the environment.
                        normalized_action = normalize_action_names(action)
                        if normalized_action != action and verbose:
                            print(f"normalized action: {normalized_action.strip()} <- {action.strip()}")
                        action = normalized_action

                        observation, reward, done, info = self.env.step([action])
                        total_reward += reward[0]


                        observation, is_nothing_happens = process_ob(observation[0], track_nothing_happens=True)

                        # print("observation: ", observation)
                        # print(info)

                        if is_nothing_happens:
                            num_nothing_happens += 1
                            CURRENT_NOTHING_HAPPENS = True
                        else:
                            CURRENT_NOTHING_HAPPENS = False

                        if done[0] or info["won"][0]:
                            if info["won"][0]:
                                success_flag=True
                            done_flag = done[0]
                            break
                        
                except Exception as e:
                    
                    print(f"Error during test the agent {agent.name} at env index{env_idx+start_env_idx}: ", e)
                    error_message = str(e)

                    traceback.print_exc()
                    if not force_run:
                        raise

                    
                try:
                    agent.reveiew_episode(
                        success_flag=success_flag,
                        episode_idx=start_env_idx + env_idx,
                        task_name=task_name,
                        env_type=env_type,
                    )
                except Exception as evo_exc:
                    print(f"[EVOLUTION] Post-episode analysis failed: {evo_exc}")
                    traceback.print_exc()
                    if not force_run:
                        raise

                logging_dict = {}

                logging_dict["env_idx"] = env_idx+start_env_idx
                logging_dict["env_type"] = env_type
                logging_dict["temperature"] = agent.temperature

                logging_dict["success"] = success_flag
                logging_dict["done"] = done_flag
                logging_dict["total_reward"] = total_reward
                logging_dict["early_stop"] = step_i >= self.test_max_step
                logging_dict["error"] = error_message

                # Step logging & per step logging
                logging_dict["num_of_steps"] = step_i

                logging_dict["num_nothing_happens"] = num_nothing_happens
                logging_dict["num_repetitions"] = num_repetitions

                # CORRECTION & per step correction logging
                logging_dict["num_action_correction"] = num_correction

                
                # Token Count
                logging_dict["total_in_tokens"] = prompt_tokens
                logging_dict["total_out_tokens"] = completion_tokens
                logging_dict["total_price_estimated"] = agent.get_price()

                overall_price += logging_dict["total_price_estimated"]

                write_line_to_main_log_csv(self.log_csv_file, logging_dict)

                if hasattr(agent, 'agent_log'):
                    with open(agent_log_file, "w", encoding="utf-8") as f:
                        f.write(agent.agent_log)

                cnts[task_type_idx] += 1
                rs[task_type_idx] += int(success_flag)

                pbar.set_postfix({
                    # "r": r,
                    # "rs": rs,
                    # "cnts": cnts,
                    "success rate": f"{sum(rs)/sum(cnts):.4f}"
                })
                pbar.update(1)
        
        completed = sum(cnts)
        if completed:
            print(f"Final result: rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / completed}")
        else:
            print("No ALFWorld episodes matched the requested filters.")
        print(f"Overall price: {overall_price}")
        
        if hasattr(agent, 'induced_skill_set'):
            for skill_id, skill_dict in agent.induced_skill_set.items():
                with open(f"{log_folder}/induction_skills/{skill_id}.json", "w", encoding="utf-8") as f:
                    json.dump(skill_dict, f, ensure_ascii=False, indent=4)

                print(f"skill {skill_id} success rate: {skill_dict['success_count']}/{skill_dict['use_count']}", 
                      skill_dict["success_count"] / (skill_dict["use_count"] + 1e-5))
                print(f"skill {skill_id} success rate (reflection): {skill_dict['success_count_re']}/{skill_dict['use_count_re']}", 
                      skill_dict["success_count_re"] / (skill_dict["use_count_re"] + 1e-5))
                
                print("Encountering an exception: ", skill_dict["exception"])
        
        return (sum(rs) / completed if completed else 0.0), overall_price

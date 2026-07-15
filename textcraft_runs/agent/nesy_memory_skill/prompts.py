OBSERVATION_TO_PREDICATES_PROMPT = """You are an AI Embodied agent. Convert each observation (`observation:` …), together with the most recent command (`action:` …), into updated symbolic facts.

For every observation:
1) Identify which predicate instances become true.
2) Identify which previous predicate instances are contradicted or outdated.
3) Output two blocks:
   - `add_facts:` facts certainly true after this observation.
   - `remove_facts:` facts that became false/obsolete. Use `none` only if nothing is removed.

Predicates (use exactly these names/argument orders):
- `goal(item_id, count)`: the task requires at least `count` of `item_id`. Derive from sentences like “Your task is to: craft 2 oak planks.”
- `inventory(item_id, count)`: the agent currently holds `count` units of the item. Counts come from `Inventory:` lines or can be updated from action/observation pairs (e.g., `get 2 oak logs` + “Got 2 oak logs” increases the known count; crafting consumes the inputs stated in the action and adds the crafted output count).
- `unavailable(item_id)`: the world or recipe search failed for the item, indicated by messages like “Could not find X” or “Could not find enough items to craft Y.” Remove this fact only when that same item shows up in inventory or is crafted/fetched successfully; otherwise leave it unchanged.

Normalization and update rules:
- Convert item names to snake_case; drop prefixes like `minecraft:` (e.g., `minecraft:oak_planks` → `oak_planks`, `dark oak log` → `dark_oak_log`).
- For item naming, strictly follow the observation's item text (including singular/plural). Do not singularize or "normalize" plurals.
  - e.g., `Inventory: [logs] (1)` → `inventory(logs, 1)`, `Inventory: [stick] (6)` → `inventory(stick, 6)`.
- Use integers for counts: `inventory(oak_planks, 4)`.
- action `inventory` -> obs: `Inventory: ...` should refresh the inventory facts, remove all existing `inventory(...)` facts if it is not matched with this observation, add `inventory(item_id, count)` for each item in the inventory.
- When an observation describes a delta (e.g., action: "get 2 oak logs" -> observation:“Got 2 oak logs”, action "craft 3 dark oak sign using 6 dark oak planks, 1 stick"-> observation: “Crafted 3 minecraft:dark_oak_sign”), adjust counts using the action parameters and prior facts. Replace the corresponding `inventory(...)` facts for that item with the updated count; if an input is fully consumed, remove its `inventory` fact. 
- Do not emit duplicate facts. If a fact is already present and unchanged, leave it out of `add_facts`.
- If nothing changes, return empty lists.

Examples:

Input:
facts: []
action: 
observation: Your task is to: craft 2 dark oak logs.
Output:
```json
{{
  "remove_facts": [], 
  "add_facts": ["goal(dark_oak_logs, 2)"],
}}
```

Input:
facts: ["goal(dark_oak_logs, 2)"]
action: inventory
observation: Inventory: [stick] (1) [dark oak planks] (6)
Output:
```json
{{
  "remove_facts": [], 
  "add_facts": [
    "inventory(stick, 1)",
    "inventory(dark_oak_planks, 6)"
  ]
}}
```

Input:
facts: ["goal(dark_oak_logs, 2)", "inventory(stick, 1)"]
action: get 2 dark oak logs
observation: Got 2 dark oak logs
Output:
```json
{{
  "remove_facts": [], 
  "add_facts": ["inventory(dark_oak_logs, 2)"]
}}
```

Input:
facts: ["inventory(dark_oak_planks, 8)", "inventory(stick, 2)"]
action: craft 3 dark oak sign using 6 dark oak planks, 1 stick
observation: Crafted 3 minecraft:dark_oak_sign
Output:
```json
{{
  "remove_facts": [
    "inventory(dark_oak_planks, 8)",
    "inventory(stick, 2)"
  ], 
  "add_facts": [
    "inventory(dark_oak_sign, 3)", 
    "inventory(dark_oak_planks, 2)", 
    "inventory(stick, 1)"
    ]
}}
```

Follow exactly this schema for each new observation.

facts: {facts}
action: {new_action}
observation: {new_observation}
"""


TASK_LEVEL_SUMMARY_PROMPT = """
You are compiling task-level execution guidelines for a TextCraft crafting agent.
In this task, an item can be obtained either by fetching it directly from the environment or by crafting it
with the provided craft commands and required ingredients. 

Task type: {task_type}

Inputs:
- `Skill Info`: structured skill schemas and successful call snippets for this task type. Use them to infer what each skill does, when it is invoked, and how its parameters are bound (goal items, counts, ingredients, inventory checks).
- `Raw Trajectory`: raw action sub-trajectories where skills were not used. Use them to understand fallback or bridging behaviors (e.g., direct `inventory`, `get`, `craft` calls).

Goal (per task type):
- Produce a topology-aware summary of sub-goals and their dependencies.
- For each skill: cover all observed scenarios in which it is invoked (phases and triggers), the typical pre-skill context (key world facts like goals, inventory gaps, unavailable markers; recent intents; initiation rationale), and the typical post-skill outcomes (facts/follow-up intents).
- Summarize all observed "raw action" patterns (spans without skill calls) from Raw Trajectory with the same context/outcome framing.

Output format (STRICT JSON, no prose, no code fences):
{{
  "task_type": "{task_type}",
  "subgoal_topology": {{
    "subgoals": ["<sub-goal description; natural language; must not be empty>"],
    "subgoal_success_conditions": {{
      "<sub-goal description>": ["<observable success condition(s); use goal/inventory/unavailable facts>"]
    }},
    "ordering_constraints": ["<subgoal A> -> <subgoal B> (A before B); can be empty if no constraints>"]
  }},
  "skill_guidelines": [
    {{
      "skill": "<skill name>",
      "applicable_scenarios": [
        {{
          "phase": "<sub-goal phase; must match one entry in subgoal_topology.subgoals; add the reason>",
          "when_to_invoke": ["<all observed triggers/start-condition cues; each under 20 words>"],
          "typical_pre_context": {{
            "key_world_facts": ["<concise fact patterns (goal/inventory/unavailable/recipe); under 20 words>"],
            "recent_intents": ["<what the agent was trying to do right before; under 20 words>"],
            "initiation_rationale": "<self-contained reason to start the skill now; under 25 words>"
          }},
          "typical_post_outcome": {{
            "expected_state_change": ["<state changes or confirmations (inventory/goal progress/unavailable cleared); under 20 words>"],
            "follow_up_intents": ["<next intents the agent usually pursues; under 20 words>"]
          }},
          "record_indices": <Python List of Int, indices of success records (from `Skill Info`) that support this scenario; index from 1; e.g. [1, 3]; use [] if none>
        }},
        {{
          "phase": ...
          "when_to_invoke": ...
          "typical_pre_context": ...
          "typical_post_outcome": ...
          "record_indices": ...
        }}, ...
      ]
    }}
  ],
  "raw_action_guidelines": [
    {{
      "action": "<raw-action name>",
      "applicable_scenarios": [
        {{
          "phase": "<sub-goal phase; must match one entry in subgoal_topology.subgoals; add the reason>",
          "when_to_invoke": ["<all observed triggers/start-condition cues; each under 20 words>"],
          "typical_pre_context": {{
            "key_world_facts": ["<concise fact patterns; under 20 words>"],
            "recent_intents": ["<what the agent was trying to do right before; under 20 words>"],
            "initiation_rationale": "<self-contained reason to start the raw actions; under 25 words>"
          }},
          "typical_post_outcome": {{
            "expected_state_change": ["<state changes or confirmations; under 20 words>"],
            "follow_up_intents": ["<next intents the agent usually pursues; under 20 words>"]
          }},
        }}, 
        {{
          "phase": ...
          "when_to_invoke": ...
          "typical_pre_context": ...
          "typical_post_outcome": ...
        }}, ...
      ]
    }}
  ],
}}

Population rules:
- Cover every skill present in Skill Info; do not invent new skills.
- Cover every distinct raw-action pattern appearing in Raw Trajectory; do not leave raw_action_guidelines empty when raw actions exist.
- Merge near-duplicate scenarios; keep 1-3 concise scenarios per skill/action that reflect distinct phases or triggers.
- Prefer ordering by frequency and earliest appearance in examples; subgoal_topology.subgoals must always be provided (best-effort).
- Keep strings concise; follow the length cues above. Return valid JSON only; fill missing sections with empty arrays instead of prose.

# Skill Info:
{skill_info}


# Raw actions: 
inventory
get: <item_count> <item_name>
craft: <item_count> <item_name> using <ingredients_count_list> <ingredients_name_list>

# Raw Trajectory:
{raw_traj}
"""


PLANNING_PROMPT = """You are a planner for TextCraft. Given the goal, current state, and history, decide the ONE next action.

# Inputs
Goal: {task_goal}
Facts: {facts}
Crafting commands:
{crafting_commands}
Recent events:
{history_memory}
Failure memory (do NOT retry these):
{failure_memory}
Available skills:
{skill_summary}
Learned routing rules:
{routing_rules}
{hints}

# Rules
0. [HARD CONSTRAINT — apply BEFORE all other rules] Every entry in `Failure memory` with `count >= 2` is BLACKLISTED. Your `approach.name` MUST NOT match any blacklisted entry's subgoal/action string (covers both skill calls like `check_and_fetch_item(X)` and raw commands like `get N X` / `craft N X using ...`). When the natural next step is blacklisted, pivot to ONE of:
   (a) the recipe-craft path for that item if `Crafting commands` lists `craft <N> <item> using ...` whose output name exactly matches — switch to crafting the missing INGREDIENTS of that recipe;
   (b) the deepest unsatisfied ingredient in the recipe chain that has NEVER been attempted (i.e. NOT present in `Failure memory`).
   If both (a) and (b) lead to blacklisted items, recurse one layer deeper. Never re-emit a blacklisted approach to "see if it works this time".
1. To obtain an item (when not blacklisted): try fetch (`check_and_fetch_item`) first. Craft only if fetch fails AND a recipe exists whose output name EXACTLY matches the needed item.
2. Some recipe ingredients are generic tags (e.g. `planks`, `wool`). If you cannot fetch or craft the generic name, substitute a specific variant you have (e.g. replace `3 planks` with `3 oak planks` in the craft command).
3. A recipe is valid ONLY if its output name exactly matches what you need (no substring matches — `granite` does NOT match `granite_wall`). If no match exists, find a different path.
4. An action is only viable if its preconditions are met in current facts. If not, resolve the precondition first.
5. Follow learned routing rules — they encode strategies discovered from past episodes.
6. CRITICAL: `next_subgoal` may be "none" ONLY if the goal item appears in Facts with sufficient count. If you are stuck but the goal is NOT yet in inventory, you MUST still output a subgoal targeting the deepest unsatisfied dependency in the recipe chain.

# Output (strict JSON, no prose)
{{{{
  "reasoning": "1) Blacklist check: list every Failure-memory entry with count >= 2 and confirm my chosen approach is NOT one of them. 2) Goal check: does Facts already contain the goal item with enough count? 3) If not: trace the recipe chain from goal → current state. What do I need? What do I have? Which dependencies are blacklisted vs untried? What is the deepest unsatisfied UNTRIED dependency I can act on now?",
  "next_subgoal": "inventory(item, count) — ONLY 'none' if goal check confirms goal is in inventory",
  "approach": {{{{
    "type": "skill|action",
    "name": "check_and_fetch_item or exact craft command — must NOT match any blacklisted entry",
    "why": "short rationale; if pivoting away from a blacklisted entry, name the entry"
  }}}}
}}}}
"""


REGRESSION_TOPOLOGY_PROMPT = """
You are a regression planner for TextCraft.
Your job is to propose the next sub-goal (or revise the current sub-goal) by backward planning.
Regression planning here means repeatedly regressing from the final goal to earlier sub-goals until
you obtain a sub-goal path that is executable from the current state, where each sub-goal's
preconditions are satisfiable and the sequence can reach the final goal.

Inputs:
- Final task goal: {task_goal}
- Current facts: {facts}
- Known crafting commands: {crafting_commands}
- Sub-goal history (success/failure + reasons): 
{history_memory}

Referred Sub-Goal Decomposition on the related Tasks:
{subgoal_traces}

Existing subgoal decomposition (incremental update it):
{subgoals_decomposition}


Representation rules:
- Use the predicate form `inventory(item, count)` for subgoals.
- Ground `final_goal`, every `subgoal`, and `subgoal_success_conditions` with concrete item names and counts from the task, facts, or crafting commands (no placeholders like item/count).
- For item names in goal, subgoals and any condtions, strictly follow the exact names in the provided crafting commands' recipes, regardless of singular/plural. (e.g., use `acacia_logs` for "1 acacia logs" if "1 acacia logs" appears in the given receipes).
- Important: singular/plural variations are not allowed (e.g., `logs` and `log` are different items).
- Some recipe ingredients are generic tags (e.g. `planks`, `wool`). If you cannot fetch or craft the generic name, substitute a specific variant in the craft command (e.g. `3 planks` → `3 oak planks`).
- Distinguish missing ingredients vs unavailable items:
  - Use `not inventory(item, count)` to express missing ingredients or insufficient inventory.
  - Use `unavailable(item)` only when there is explicit failure evidence (e.g., direct fetch or craft failed for that item).
- When referencing a craft method, the target count must exactly match the recipe output count from the provided crafting commands (no partial batches, no multiples).
- Express topology only through:
  1) `subgoals`: each subgoal lists its `solutions` (OR) based on different options (e.g., different crafting recipes) to achieve the subgoal.
  2) `solutions` for each subgoal list the `preconditions` of each solution to be satisfied.
- Status update:
  - Based on the given current facts, update the `unsatisfied|satisfied` for each sub-goal and the preconditions of solutions.
  - If a subgoal/solution is not `blocked`, its status must be re-evaluated from Current facts each time (it can flip between `unsatisfied` and `satisfied`) because ingredients may be consumed by later crafts.
  - If a subgoal or solution is known to fail (from failure_memory), mark it as "blocked" with evidence.
- If an existing subgoal decomposition is provided, extend it incrementally rather than rewriting from scratch.
- If the final goal is already satisfied by current facts, set `decision.next_subgoal` to "none" and
  `decision.next_subgoal_description` to a short note like "final goal already satisfied".
{hints}

Output format (STRICT JSON, update the provided subgoal decomposition accordingly):
{{
  "final_goal": "{task_goal}",
  "subgoals_decomposition": [
    {{
      "subgoal": "..., e.g., inventory(item, count)",
      "status": "unsatisfied|satisfied|blocked",
      "solutions": [
        {{
          "name": "e.g., direct_get",
          "preconditions": [..., e.g., "not inventory(x, n)", "not unavailable(x)", ...],
          "status": "unsatisfied|satisfied|blocked",
          "evidence": "short reason or failure trace"
        }},
        {{
          "name": "e.g., one specific crafting recipe",
          "preconditions": [..., e.g., "inventory(x_1, n_1)", ...],
          "status": "unsatisfied|satisfied|blocked",
          "evidence": "short reason or failure trace"
        }},
        {{
          "name": "e.g., another crafting recipe",
          "preconditions": [..., e.g., "inventory(x_1, n_1)", ...],
          "status": "unsatisfied|satisfied|blocked",
          "evidence": "short reason or failure trace"
        }},...
      ]
    }}, 
    ...
  ],
  "draft_sequence": ["<subgoal A> -> <subgoal B> -> <subgoal C> -> ..."]
  "decision": {{
    "reasoning": "how current facts satisfy a certain subgoal's preconditions and why it is critical for reaching the final goal",
    "next_subgoal": ..., 
    "next_subgoal_description": a short description of how to achieve the given next_subgoal, includes a specific solution, as a sub-task instruction,    
    "decision_type": "keep|revise|replan",
    "replan_signal": "none|subgoal_unreachable|repeated_failure|missing_recipe"
  }}
}}
"""




MEMORY_CONSTRUCTION_PREDICATE_ANALYSIS_PROMPT = """You are in a crafting environment solving a task.
{interaction_history}

Current world state (facts): {facts_current}
Current action: {new_action}

Select only the facts that directly ground or parameterize this action (e.g., required goal item/count, relevant inventory counts, known unavailability of an item). Ignore unrelated context.

Output requirements (strict):
- Return ONLY a Python list of strings (no prose, no keys, no code fences).
- Copy each selected fact verbatim from the Current world state.
- Do not invent or rephrase facts; do not include duplicates.
- If none apply, return an empty list: []
"""


MEMORY_CONSTRUCTION_INTENT_PROMPT = """You are in a crafting environment solving a task.
{interaction_history}

Current action: {new_action}
Write one sentence stating the intent: the goal-conditioned purpose of the current action, explaining why it is appropriate.
Output requirements (strict):
- Single sentence (use "because" if helpful).
- Do not merely restate the action; explain the underlying purpose relative to the goal.
- Be clear and concise.
Intent: """


MEMORY_CONSTRUCTION_INTENT_WITH_PREDICATE_PROMPT = """You are in a crafting environment solving a task.
{interaction_history}

Critical world state facts: {important_predicates_for_action}
Current action: {new_action}
Write one sentence stating the intent: the goal-conditioned, fact-grounded purpose of the current action, explaining why it is appropriate.
Output requirements (strict):
- Single sentence (use "because" if helpful).
- Do not merely restate the action; explain the underlying purpose relative to the goal.
- Ground the rationale in the Critical facts and the task goal.
- Wrap every item identifier or type in $...$ (e.g., $oak log$, $oak_planks$, $dark oak sign$, $stick$, $oak_planks$).
- Be clear and concise.
Intent: """


SUBGOAL_PROMPT = """
You are an AI planner for TextCraft (Minecraft-style crafting).
Summarize **Parameterized Sub-goals** (Reusable Skills) from the provided interaction trajectories. Each sub-goal must cover a multi-step intent (not a single primitive command).

# Data
Here are some interaction trajectories:
{examples}

# Detailed Instructions for Sub-Goal Generation:
- Complexity: Each sub-goal must require 2–5 primitive actions; do NOT emit single-action or trivial one-check-one-action patterns.
- Generality: Sub-goals should transfer to similar crafting tasks (e.g., different item types or counts), not memorize a single trace.
- Frequency: Only emit sub-goals that are supported by multiple trajectory snippets; ignore patterns observed just once. Each emitted sub-goal must appear at least twice across the provided examples.
- When two sub-goals are similar, prefer **merging** them into a more abstract parameterized template (but do not over-generalize into a meaningless catch-all).
- Parameterize the use of actions (get and craft) so that they can be reused in different scenarios instead of simply memorizing them. 

## Primitive Actions
- inventory: Check the current inventory state.
- get <item_count> <item_name>: Pick up items from the ground or containers.
- craft <item_count> <item_name> using <ingredients_count_list> <ingredients_name_list>: Use items in the inventory to create new items.


# Output Requirements for Sub-Goals:
- Parameters and Types: List the input parameters and their types (choose only from): ItemName, Count, List_ItemName, List_Count. Use clear placeholder names (e.g., target_item: ItemName, need_count: Count).
- Start Conditions: Preconditions that must hold before starting (e.g., “know crafting recipe for {{target_item}}”, “inventory has at least {{need_count}} of {{target_item}}”, “lack of {{target_type}} in inventory”).
- Success Conditions: Observable state after completion (e.g., “inventory has >= {{need_count}} of {{target_item}}”, “unavailable({{target_item}}) cleared”).
- Naming: Use concise lowercase with underscores. Do NOT directly reuse primitive action names (inventory, get, craft, think).
- Steps Analysis: Provide 2–5 ordered steps using primitive actions (inventory, get, craft) and simple conditionals like “If insufficient {{target_type}} in inventory, then get more”.
- Applicable Tasks: Copy task names verbatim from the source data where this sub-goal applies (no paraphrasing).

# Output Format:
```json
[
  {{
    "name": "<concise_sub_goal_name>",
    "parameters": ["param_name_1: param_type_1", ...],
    "start_conditions": ["<condition 1>", ...],
    "success_conditions": ["<condition 1>", ...],
    "description": "<one-line summary of the multi-step intent>",
    "steps": ["<step 1>", "<step 2>", ...],
    "applicable_tasks": ["<task 1: copy verbatim>", ...]
  }},
  ...
]
```
"""


SUB_GOAL_WORKFLOW_PROMPT = """
You are a workflow synthesizer for TextCraft (Minecraft-style crafting).

Your task: 
- Read the sub-goal and the interaction trajectories, then instantiate the scaffold below to solve ONLY that sub-goal with TextCraft commands.
- For each sub-goal, output ONE Mermaid flowchart style (node-bound inputs, edge-only control). No prose.

# Sub-goal to implement
{sub_goal}

# Data trajectories
Here are some interaction trajectories:
{examples}

# Overall Guideline (must follow exactly):
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_item vs TARGET_ITEM).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Edge mid-labels can be comma-separated. 
- For any edge that enters a LoopControl node: use `Start_Loop` to indicate entering/resetting from outside (restart enumeration), and `Continue_Loop` to indicate continuing the current loop. Internal back-edges should carry `Continue_Loop`; outer re-entries should carry `Start_Loop`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node’s `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = {{CURRENT_RECEPTACLE}})`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Loop helpers `for` can be used only inside LoopControl nodes; they are not allowed in Check/PrimitiveAction/DataOp/Interface nodes.
- Selection helpers `select_one` and `select_all` can be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.
- Node IDs define unique instances. When the same type of node is needed multiple times with different successors, duplicate the node by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2; C_IS_CLOSED_1, C_IS_CLOSED_2; LOOP_FOR_LOCATIONS_1, LOOP_FOR_LOCATIONS_2; D_SELECT_ONE_1, D_SELECT_ONE_2).
- Item names stay in snake_case; counts must be integers.

Domain predicates (allowed inside DataOp/Check nodes):
- goal({{item_name}}, {{goal_count}})
- inventory({{item_name}}, {{have_count}})
- unavailable({{item_name}})
- current_count_of_item({{item_name}}) --> Count

Object Selection (ONLY allowed inside DataOp nodes):
- select_one('Item', an expression over `x` using domain predicates)  
  * select the first item that satisfies the expression, return None if no such item exists  
  * e.g., select_one('Item', not unavailable(x))
- select_all('Item', an expression over `x` using domain predicates)  
  * select all items satisfying the expression, return an empty list if no such item exists  
  * e.g., select_all('Item', unavailable(x)))

List helpers (ONLY allowed inside DataOp nodes):
- select_list_index(list_value, index_value)  
  * return the element at position index_value of list_value (0-based to align with LoopForControl)  
  * return None if index_value is out of bounds


# Current Mermaid-code Workflow

%% Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):
%%{{init: {{'theme': 'default', 'themeVariables': {{'background': '#ffffff'}} }} }}%%
flowchart TD
    %% ===================== Class Definitions =====================
    %% Guideline: Strictly follow the class definitions.
    classDef Spec fill:#f4f4ff,stroke:#6a6ab2,stroke-dasharray: 4 3,stroke-width:2px;
    classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;
    classDef LoopControl fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
    classDef PrimitiveAction fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
    classDef Check fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
    classDef DataOp fill:#f0f0f0,stroke:#888888,stroke-width:2px;

    %% ===================== Type Alias Legend =====================
    %% Guideline: Strictly follow the type alias legend.
    LEGEND["Type Legend:<br>ItemName := str<br>Count := int<br>Bool := bool<br>List_ItemName := sequence of ItemName<br>List_Count := sequence of Count"]:::Spec
    
    %% ===================== Spec =====================
    %% Guideline: Strictly follow the Spec.
{FLOW_SPEC_NODE}

    %% ===================== Interface & Spec =====================
    %% Guideline: START is entry; *_END are exits. No computations here. Line breaks use <br>. 
    %% Guideline: Strictly follow the interface definition.
{START_INTERFACE_NODE}
    SUCCESS_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=True<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=True)"]):::Interface
    FAILURE_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=False<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=False)"]):::Interface

    %% ===================== Primitive Actions =====================
{ACTION_NODE}  
    %% Guideline: Select the PrimitiveActions Nodes that workflow needs. Remove any nodes that are not needed. 
    %% Note that any revision on the setting of the PrimitiveActions Nodes is STRICTLY FORBIDDEN. Directly copy which node the workflow needs. 
    %% Guideline: When needed, instantiate multiple PrimitiveAction nodes by assigning unique IDs (e.g., A_GET_1, A_GET_2); each node id is a separate instance that can connect to different successors.
    A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction
    A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_count: ItemName = {{CURRENT_INGREDIENT}}, action_item_name: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS_LIST}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction
  

    %% ===================== LoopControl (use only if needed) =====================
    %% Guideline: Only foreach loops via LoopControl with edges 'body' and 'done'.
    %% Guideline: LoopControl nodes only operate on the loop variable (e.g., LOOP_FOR_INGREDIENTS["LoopControl: <br>For idx, ingredient_i in enumerate({{loop_items}})<br>writes GLOBAL: (CURRENT_INGREDIENT: ItemName:=ingredient_i, LOOP_INDEX: Count:=idx)<br>local in: (loop_items: List_ItemName = {{TARGET_ITEMS}})"]:::LoopControl, assign globals for the needed ingredients).
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple LoopControl nodes with distinct IDs (e.g., LOOP_FOR_1, LOOP_FOR_2) to represent separate loop sites.
{LOOP_FOR_NODE}
    %% Guideline: Include a new LoopControl node if you must use a loop.


    %% ===================== Checks =====================
    %% Guideline: Checks branch on Yes/No only; inputs resolved from GLOBALS. 
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: The check nodes only operate with the defined predicates, local_in variables, the basic boolean operators (and, or, not) and numerical operators (+, -, *, /, ==, !=, <, >, <=, >=). **Other operators are not allowed.**
    %% Guideline: List-level predicates are NOT supported; when you must check a list, decompose into a LoopControl that iterates the list, Check nodes that apply object-level predicates per element, and DataOp nodes that record whether the list-level condition holds.
    %% Guideline: When needed, duplicate Check nodes with unique IDs (e.g., C_INGREDIENT_READY_1, C_INGREDIENT_READY_2) to use the same predicate at different decision points.
{CHECK_NODE}
    %% Design new Check nodes with the domain predicates if needed
    %% Here are some examples. Note these are examples—adjust the check condition and parameters binding to fit the workflow.
    C_INGREDIENT_READY["Check: <br> (inventory({{check_ingredient_name}}, {{check_available_ingredient_count}}) and {{check_available_ingredient_count}} >= {{check_target_count}})<br>local in: (check_ingredient_name: ItemName = {{CURRENT_INGREDIENT}}, check_target_count: Count = {{TARGET_COUNT}}, check_available_ingredient_count: Count = {{AVAILABLE_COUNT}})"]:::Check
    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node below. 
{D_INIT_NODE}
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple DataOp nodes by assigning unique IDs (e.g., D_FIND_MISSING_1, D_FIND_MISSING_2) so each instance can feed different downstream consumers.
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    %% Example helpers for TextCraft (edit/remove as needed):
    D_SELECT_TARGET_COUNT["DataOp: <br>writes GLOBAL: (TARGET_COUNT: Count = select_list_index({{local_counts_list}}, {{local_index_arg}}))<br>local in: (local_counts_list: List_Count = {{TARGET_COUNTS}}, local_index_arg: Count = {{LOOP_INDEX}})"]:::DataOp
    D_FETCH_AVAILABLE["DataOp: <br>writes GLOBAL: (AVAILABLE_COUNT: Count = current_count_of_item({{local_fetch_ingredient_name}})) <br>local in: (local_fetch_ingredient_name: ItemName = {{CURRENT_INGREDIENT}})"]:::DataOp
    D_COMPUTE_MISSING["DataOp: <br>writes GLOBAL: (COUNT_TO_GET: Count = max({{local_compute_target_count}} - {{local_compute_available_count}}, 0)) <br>local in: (local_compute_target_count: Count = {{TARGET_COUNT}}, local_compute_available_count: Count = {{AVAILABLE_COUNT}})"]:::DataOp
    
    %% ===================== Node Class Assignments =====================
    %% Guideline: Assign each node to a class (e.g., `PrimitiveAction`, `Check`, `DataOp`, `LoopControl`) based on its role.
{CLASS_ASSIGNMENTS}

    %% ===================== Legend links =====================
    %% Guideline: Strictly follow the legend links.
    START -.- FLOW_SPEC
    START -.- LEGEND

    %% ===================== Control-Flow Edges =====================
    %% Guideline: Keep edges control-only (no parameters); label branches with Yes/No/body/done.
    %% Guideline: Improve the control-flow edges if needed. 
    %% Guideline: Ensure every referenced global is assigned earlier in the workflow or provided by START before the consuming node runs.
    %% Guideline: Prepare required globals (with same type) before any PrimitiveAction node that reads from them. 
    %% Guideline: For outgoing edges, LoopControl nodes must have both `body` and `done`; Check nodes must have `Yes` and `No`; DataOp/PrimitiveAction nodes must emit exactly one edge.
    %% Guideline: Label every edge entering a LoopControl node with Start_Loop (for external entry/reset) or Continue_Loop (for in-loop back edges); ensure at least one Start_Loop edge targets each LoopControl; do not use these labels on edges targeting non-loop nodes.
    %% Guideline: Use `,` to split the multiple labels (if needed) on the control-flow edges. e.g. |Yes, Continue_Loop|.

{CONTROL_FLOW_EDGE}

# Error Analysis of the Current Mermaid-code Workflow
{HINTS}


# Output Format
- Output ONLY the Mermaid code block(s). No explanations or extra text. 
- Output `No Solution` whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow under the given constraints.
"""


SUB_GOAL_WORKFLOW_TRACE2CODE_PROMPT_V2 = """
You are a workflow synthesizer for TextCraft (Minecraft-style crafting).

Goal:
- Given a single Skill Calling record and its aligned sub-trajectory, synthesize ONE Mermaid workflow that **exactly** replays that sub-trajectory.
- Keep the workflow compliant with the scaffold and dataflow rules.
- Preserve all provided comments/annotations and section headers (especially lines starting with `%%`); do NOT delete or rewrite them—only fill in node definitions/edges while keeping the scaffold comments intact.
- Prefer a straight-line sequence (DataOp → PrimitiveAction → DataOp → PrimitiveAction …) that simply mirrors the observed actions; avoid loops/checks unless the trajectory explicitly repeats structure.
- Favor a **simple linear chain** that mirrors the trajectory: DataOp (prepare params if needed) → PrimitiveAction → DataOp → PrimitiveAction → ... in the exact order of the expert actions; no extra branches/loops unless the trajectory itself repeats.

# Skill Calling (single trajectory to fit)
{skill_call_dict}

# Task Summary of the following trajectory:
{trajectory_summary}

# Sub-trajectory to fit (chronological; already segmented for this skill)
{sub_traj_info}

# Overall Guideline (must follow exactly):
- Fit to the provided sub-trajectory: include every expert action in order as PrimitiveAction nodes; prefer a straight-line sequence (DataOp → Action → DataOp → Action) that wires the needed globals into each action. Do **not** add Check or LoopControl nodes unless the trajectory explicitly shows repetition you cannot unroll.
- Bind START interface inputs from `parameter_bindings` inside the Skill Calling JSON; propagate them through D_INIT/DataOp so every node's `local in` can resolve without inventing new external parameters.
- Derive DataOp writes from the world states in `sub_traj_info` so downstream checks/actions have required globals; keep values consistent with the observed states or the provided parameter bindings (use placeholders when the value comes from bindings).
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_item vs TARGET_ITEM).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Edge mid-labels can be comma-separated. 
- For any edge that enters a LoopControl node: use `Start_Loop` to indicate entering/resetting from outside (restart enumeration), and `Continue_Loop` to indicate continuing the current loop. Internal back-edges should carry `Continue_Loop`; outer re-entries should carry `Start_Loop`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node’s `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = {{CURRENT_RECEPTACLE}})`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Loop helpers `for` can be used only inside LoopControl nodes; they are not allowed in Check/PrimitiveAction/DataOp/Interface nodes.
- Selection helpers `select_one` and `select_all` can be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.
- Node IDs define unique instances. When the same type of node is needed multiple times with different successors, duplicate the node by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2; C_IS_CLOSED_1, C_IS_CLOSED_2; LOOP_FOR_LOCATIONS_1, LOOP_FOR_LOCATIONS_2; D_SELECT_ONE_1, D_SELECT_ONE_2).
- Item names stay in snake_case; counts must be integers.

Domain predicates (allowed inside DataOp/Check nodes):
- goal({{item_name}}, {{goal_count}})
- inventory({{item_name}}, {{have_count}})
- unavailable({{item_name}})
- current_count_of_item({{item_name}}) --> Count

Object Selection (ONLY allowed inside DataOp nodes):
- select_one('Item', an expression over `x` using domain predicates)  
  * select the first item that satisfies the expression, return None if no such item exists  
  * e.g., select_one('Item', not unavailable(x))
- select_all('Item', an expression over `x` using domain predicates)  
  * select all items satisfying the expression, return an empty list if no such item exists  
  * e.g., select_all('Item', unavailable(x)))

List helpers (ONLY allowed inside DataOp nodes):
- select_list_index(list_value, index_value)  
  * return the element at position index_value of list_value (0-based to align with LoopForControl)  
  * return None if index_value is out of bounds


# Current Mermaid-code Workflow

%% Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):
%%{{init: {{'theme': 'default', 'themeVariables': {{'background': '#ffffff'}} }} }}%%
flowchart TD
    %% ===================== Class Definitions =====================
    %% Guideline: Strictly follow the class definitions.
    classDef Spec fill:#f4f4ff,stroke:#6a6ab2,stroke-dasharray: 4 3,stroke-width:2px;
    classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;
    classDef LoopControl fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
    classDef PrimitiveAction fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
    classDef Check fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
    classDef DataOp fill:#f0f0f0,stroke:#888888,stroke-width:2px;

    %% ===================== Type Alias Legend =====================
    %% Guideline: Strictly follow the type alias legend.
    LEGEND["Type Legend:<br>ItemName := str<br>Count := int<br>Bool := bool<br>List_ItemName := sequence of ItemName<br>List_Count := sequence of Count"]:::Spec
    
    %% ===================== Spec =====================
    %% Guideline: Strictly follow the Spec.
{FLOW_SPEC_NODE}

    %% ===================== Interface & Spec =====================
    %% Guideline: START is entry; *_END are exits. No computations here. Line breaks use <br>. 
    %% Guideline: Strictly follow the interface definition.
{START_INTERFACE_NODE}
    SUCCESS_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=True<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=True)"]):::Interface
    FAILURE_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=False<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=False)"]):::Interface

    %% ===================== Primitive Actions =====================
{ACTION_NODE}  
    %% Guideline: Select the PrimitiveActions Nodes that workflow needs. Remove any nodes that are not needed. 
    %% Note that any revision on the setting of the PrimitiveActions Nodes is STRICTLY FORBIDDEN. Directly copy which node the workflow needs. 
    %% Guideline: When needed, instantiate multiple PrimitiveAction nodes by assigning unique IDs (e.g., A_GET_1, A_GET_2); each node id is a separate instance that can connect to different successors.
{AVAILABLE_ACTIONS}

    %% ===================== LoopControl (omit unless absolutely necessary) =====================
    %% Guideline: For linear replay, leave this section empty. Use LoopControl only if the sub-trajectory explicitly repeats and cannot be unrolled.
{LOOP_FOR_NODE}
    %% Guideline: If no loops in the trajectory, do not add LoopControl nodes.


    %% ===================== Checks (omit for linear replay) =====================
    %% Guideline: For pure imitation, leave this section empty. Only add checks if the trajectory explicitly branches on observations.
{CHECK_NODE}
    %% Guideline: Do not add Check nodes unless strictly required by repeated branching in the trajectory.
    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node below. 
{D_INIT_NODE}
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple DataOp nodes by assigning unique IDs (e.g., D_FIND_MISSING_1, D_FIND_MISSING_2) so each instance can feed different downstream consumers.
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    %% Example helpers for TextCraft (edit/remove as needed):
    D_SELECT_TARGET_COUNT["DataOp: <br>writes GLOBAL: (TARGET_COUNT: Count = select_list_index({{local_counts_list}}, {{local_index_arg}}))<br>local in: (local_counts_list: List_Count = {{TARGET_COUNTS}}, local_index_arg: Count = {{LOOP_INDEX}})"]:::DataOp
    D_FETCH_AVAILABLE["DataOp: <br>writes GLOBAL: (AVAILABLE_COUNT: Count = current_count_of_item({{local_fetch_ingredient_name}})) <br>local in: (local_fetch_ingredient_name: ItemName = {{CURRENT_INGREDIENT}})"]:::DataOp
    D_COMPUTE_MISSING["DataOp: <br>writes GLOBAL: (COUNT_TO_GET: Count = max({{local_compute_target_count}} - {{local_compute_available_count}}, 0)) <br>local in: (local_compute_target_count: Count = {{TARGET_COUNT}}, local_compute_available_count: Count = {{AVAILABLE_COUNT}})"]:::DataOp
    
    %% ===================== Node Class Assignments =====================
    %% Guideline: Assign each node to a class (e.g., `PrimitiveAction`, `Check`, `DataOp`, `LoopControl`) based on its role.
{CLASS_ASSIGNMENTS}

    %% ===================== Legend links =====================
    %% Guideline: Strictly follow the legend links.
    START -.- FLOW_SPEC
    START -.- LEGEND

    %% ===================== Control-Flow Edges =====================
    %% Guideline: Keep edges control-only (no parameters); label branches with Yes/No/body/done.
    %% Guideline: Improve the control-flow edges if needed. 
    %% Guideline: Ensure every referenced global is assigned earlier in the workflow or provided by START before the consuming node runs.
    %% Guideline: Prepare required globals (with same type) before any PrimitiveAction node that reads from them. 
    %% Guideline: For outgoing edges, LoopControl nodes must have both `body` and `done`; Check nodes must have `Yes` and `No`; DataOp/PrimitiveAction nodes must emit exactly one edge.
    %% Guideline: Label every edge entering a LoopControl node with Start_Loop (for external entry/reset) or Continue_Loop (for in-loop back edges); ensure at least one Start_Loop edge targets each LoopControl; do not use these labels on edges targeting non-loop nodes.
    %% Guideline: Use `,` to split the multiple labels (if needed) on the control-flow edges. e.g. |Yes, Continue_Loop|.

{CONTROL_FLOW_EDGE}

# Error Analysis of the Current Mermaid-code Workflow
{HINTS}


# Output Format
- Output ONLY the Mermaid code block(s). No explanations or extra text. 
- Output `No Solution` whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow under the given constraints.
"""


SUB_GOAL_WORKFLOW_TRACE2CODE_PROMPT = """
You are a workflow synthesizer for TextCraft (Minecraft-style crafting).

Goal:
- Given a single Skill Calling record and its aligned sub-trajectory, synthesize ONE Mermaid workflow that **exactly** replays that sub-trajectory.
- Keep the workflow compliant with the scaffold and dataflow rules.
- Preserve all provided comments/annotations and section headers (especially lines starting with `%%`); do NOT delete or rewrite them—only fill in node definitions/edges while keeping the scaffold comments intact.

# Skill Calling (single trajectory to fit)
{skill_call_dict}

# Task Summary of the following trajectory:
{trajectory_summary}

# Sub-trajectory to fit (chronological; already segmented for this skill)
{sub_traj_info}

# Overall Guideline (must follow exactly):
- Fit to the provided sub-trajectory: include every expert action in order as PrimitiveAction nodes; add branches/loops only if the sub-trajectory shows repeated structure that cannot be safely unrolled.
- Bind START interface inputs from `parameter_bindings` inside the Skill Calling JSON; propagate them through D_INIT/DataOp so every node's `local in` can resolve without inventing new external parameters.
- Derive DataOp writes from the world states in `sub_traj_info` so downstream checks/actions have required globals; keep values consistent with the observed states or the provided parameter bindings (use placeholders when the value comes from bindings).
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_item vs TARGET_ITEM).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Edge mid-labels can be comma-separated. 
- For any edge that enters a LoopControl node: use `Start_Loop` to indicate entering/resetting from outside (restart enumeration), and `Continue_Loop` to indicate continuing the current loop. Internal back-edges should carry `Continue_Loop`; outer re-entries should carry `Start_Loop`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node’s `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = {{CURRENT_RECEPTACLE}})`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Loop helpers `for` can be used only inside LoopControl nodes; they are not allowed in Check/PrimitiveAction/DataOp/Interface nodes.
- Selection helpers `select_one` and `select_all` can be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.
- Node IDs define unique instances. When the same type of node is needed multiple times with different successors, duplicate the node by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2; C_IS_CLOSED_1, C_IS_CLOSED_2; LOOP_FOR_LOCATIONS_1, LOOP_FOR_LOCATIONS_2; D_SELECT_ONE_1, D_SELECT_ONE_2).
- Item names stay in snake_case; counts must be integers.

Domain predicates (allowed inside DataOp/Check nodes):
- goal({{item_name}}, {{goal_count}})
- inventory({{item_name}}, {{have_count}})
- unavailable({{item_name}})
- current_count_of_item({{item_name}}) --> Count

Object Selection (ONLY allowed inside DataOp nodes):
- select_one('Item', an expression over `x` using domain predicates)  
  * select the first item that satisfies the expression, return None if no such item exists  
  * e.g., select_one('Item', not unavailable(x))
- select_all('Item', an expression over `x` using domain predicates)  
  * select all items satisfying the expression, return an empty list if no such item exists  
  * e.g., select_all('Item', unavailable(x)))

List helpers (ONLY allowed inside DataOp nodes):
- select_list_index(list_value, index_value)  
  * return the element at position index_value of list_value (0-based to align with LoopForControl)  
  * return None if index_value is out of bounds


# Current Mermaid-code Workflow

%% Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):
%%{{init: {{'theme': 'default', 'themeVariables': {{'background': '#ffffff'}} }} }}%%
flowchart TD
    %% ===================== Class Definitions =====================
    %% Guideline: Strictly follow the class definitions.
    classDef Spec fill:#f4f4ff,stroke:#6a6ab2,stroke-dasharray: 4 3,stroke-width:2px;
    classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;
    classDef LoopControl fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
    classDef PrimitiveAction fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
    classDef Check fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
    classDef DataOp fill:#f0f0f0,stroke:#888888,stroke-width:2px;

    %% ===================== Type Alias Legend =====================
    %% Guideline: Strictly follow the type alias legend.
    LEGEND["Type Legend:<br>ItemName := str<br>Count := int<br>Bool := bool<br>List_ItemName := sequence of ItemName<br>List_Count := sequence of Count"]:::Spec
    
    %% ===================== Spec =====================
    %% Guideline: Strictly follow the Spec.
{FLOW_SPEC_NODE}

    %% ===================== Interface & Spec =====================
    %% Guideline: START is entry; *_END are exits. No computations here. Line breaks use <br>. 
    %% Guideline: Strictly follow the interface definition.
{START_INTERFACE_NODE}
    SUCCESS_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=True<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=True)"]):::Interface
    FAILURE_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=False<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=False)"]):::Interface

    %% ===================== Primitive Actions =====================
{ACTION_NODE}  
    %% Guideline: Select the PrimitiveActions Nodes that workflow needs. Remove any nodes that are not needed. 
    %% Note that any revision on the setting of the PrimitiveActions Nodes is STRICTLY FORBIDDEN. Directly copy which node the workflow needs. 
    %% Guideline: When needed, instantiate multiple PrimitiveAction nodes by assigning unique IDs (e.g., A_GET_1, A_GET_2); each node id is a separate instance that can connect to different successors.
{AVAILABLE_ACTIONS}

    %% ===================== LoopControl (use only if needed) =====================
    %% Guideline: Only foreach loops via LoopControl with edges 'body' and 'done'.
    %% Guideline: LoopControl nodes only operate on the loop variable (e.g., LOOP_FOR_INGREDIENTS["LoopControl: <br>For idx, ingredient_i in enumerate({{loop_items}})<br>writes GLOBAL: (CURRENT_INGREDIENT: ItemName:=ingredient_i, LOOP_INDEX: Count:=idx)<br>local in: (loop_items: List_ItemName = {{TARGET_ITEMS}})"]:::LoopControl, assign globals for the needed ingredients).
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple LoopControl nodes with distinct IDs (e.g., LOOP_FOR_1, LOOP_FOR_2) to represent separate loop sites.
{LOOP_FOR_NODE}
    %% Guideline: Include a new LoopControl node if you must use a loop.


    %% ===================== Checks =====================
    %% Guideline: Checks branch on Yes/No only; inputs resolved from GLOBALS. 
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: The check nodes only operate with the defined predicates, local_in variables, the basic boolean operators (and, or, not) and numerical operators (+, -, *, /, ==, !=, <, >, <=, >=). **Other operators are not allowed.**
    %% Guideline: List-level predicates are NOT supported; when you must check a list, decompose into a LoopControl that iterates the list, Check nodes that apply object-level predicates per element, and DataOp nodes that record whether the list-level condition holds.
    %% Guideline: When needed, duplicate Check nodes with unique IDs (e.g., C_INGREDIENT_READY_1, C_INGREDIENT_READY_2) to use the same predicate at different decision points.
{CHECK_NODE}
    %% Design new Check nodes with the domain predicates if needed
    %% Here are some examples. Note these are examples—adjust the check condition and parameters binding to fit the workflow.
    C_INGREDIENT_READY["Check: <br> (inventory({{check_ingredient_name}}, {{check_available_ingredient_count}}) and {{check_available_ingredient_count}} >= {{check_target_count}})<br>local in: (check_ingredient_name: ItemName = {{CURRENT_INGREDIENT}}, check_target_count: Count = {{TARGET_COUNT}}, check_available_ingredient_count: Count = {{AVAILABLE_COUNT}})"]:::Check
    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node below. 
{D_INIT_NODE}
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple DataOp nodes by assigning unique IDs (e.g., D_FIND_MISSING_1, D_FIND_MISSING_2) so each instance can feed different downstream consumers.
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    %% Example helpers for TextCraft (edit/remove as needed):
    D_SELECT_TARGET_COUNT["DataOp: <br>writes GLOBAL: (TARGET_COUNT: Count = select_list_index({{local_counts_list}}, {{local_index_arg}}))<br>local in: (local_counts_list: List_Count = {{TARGET_COUNTS}}, local_index_arg: Count = {{LOOP_INDEX}})"]:::DataOp
    D_FETCH_AVAILABLE["DataOp: <br>writes GLOBAL: (AVAILABLE_COUNT: Count = current_count_of_item({{local_fetch_ingredient_name}})) <br>local in: (local_fetch_ingredient_name: ItemName = {{CURRENT_INGREDIENT}})"]:::DataOp
    D_COMPUTE_MISSING["DataOp: <br>writes GLOBAL: (COUNT_TO_GET: Count = max({{local_compute_target_count}} - {{local_compute_available_count}}, 0)) <br>local in: (local_compute_target_count: Count = {{TARGET_COUNT}}, local_compute_available_count: Count = {{AVAILABLE_COUNT}})"]:::DataOp
    
    %% ===================== Node Class Assignments =====================
    %% Guideline: Assign each node to a class (e.g., `PrimitiveAction`, `Check`, `DataOp`, `LoopControl`) based on its role.
{CLASS_ASSIGNMENTS}

    %% ===================== Legend links =====================
    %% Guideline: Strictly follow the legend links.
    START -.- FLOW_SPEC
    START -.- LEGEND

    %% ===================== Control-Flow Edges =====================
    %% Guideline: Keep edges control-only (no parameters); label branches with Yes/No/body/done.
    %% Guideline: Improve the control-flow edges if needed. 
    %% Guideline: Ensure every referenced global is assigned earlier in the workflow or provided by START before the consuming node runs.
    %% Guideline: Prepare required globals (with same type) before any PrimitiveAction node that reads from them. 
    %% Guideline: For outgoing edges, LoopControl nodes must have both `body` and `done`; Check nodes must have `Yes` and `No`; DataOp/PrimitiveAction nodes must emit exactly one edge.
    %% Guideline: Label every edge entering a LoopControl node with Start_Loop (for external entry/reset) or Continue_Loop (for in-loop back edges); ensure at least one Start_Loop edge targets each LoopControl; do not use these labels on edges targeting non-loop nodes.
    %% Guideline: Use `,` to split the multiple labels (if needed) on the control-flow edges. e.g. |Yes, Continue_Loop|.

{CONTROL_FLOW_EDGE}

# Error Analysis of the Current Mermaid-code Workflow
{HINTS}


# Output Format
- Output ONLY the Mermaid code block(s). No explanations or extra text. 
- Output `No Solution` whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow under the given constraints.
"""




SUB_GOAL_WORKFLOW_GENERALIZER_PROMPT = """
You are a workflow synthesizer for TextCraft (Minecraft-style crafting).

Your task:
- Read the sub-goal, two candidate workflows, and their matched trajectories.
- Synthesize ONE new workflow that generalizes the candidates to match ALL provided trajectories for this sub-goal.
- Output ONE Mermaid flowchart style (node-bound inputs, edge-only control). No prose.

# Sub-goal to implement
{sub_goal}

# Candidate workflows and matched trajectories
Below are two workflows and the trajectories they already match. One may be more general and the other more specialized.
{workflow_records}

# Generalization Guidelines (use when helpful)
- Align the two program graphs to find their shared structure. For divergent paths, invent a CheckOp node with a discriminative guard predicate to separate contexts.
- Reuse useful subgraphs from one workflow in the other by rebinding input parameters to the current execution context.
- Lift hardcoded entities into type-based parameters by replacing concrete constant with a skill-level variable derived from shared selection criteria.
- Detect repeated sub-structures and compress them into LoopControl with a termination invariant to handle variable-sized workloads. 

# Mermaid Code Guideline (must follow exactly):
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_item vs TARGET_ITEM).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Edge mid-labels can be comma-separated. 
- For any edge that enters a LoopControl node: use `Start_Loop` to indicate entering/resetting from outside (restart enumeration), and `Continue_Loop` to indicate continuing the current loop. Internal back-edges should carry `Continue_Loop`; outer re-entries should carry `Start_Loop`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node’s `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = {{CURRENT_RECEPTACLE}})`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Loop helpers `for` can be used only inside LoopControl nodes; they are not allowed in Check/PrimitiveAction/DataOp/Interface nodes.
- Selection helpers `select_one` and `select_all` can be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.
- Node IDs define unique instances. When the same type of node is needed multiple times with different successors, duplicate the node by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2; C_IS_CLOSED_1, C_IS_CLOSED_2; LOOP_FOR_LOCATIONS_1, LOOP_FOR_LOCATIONS_2; D_SELECT_ONE_1, D_SELECT_ONE_2).
- Item names stay in snake_case; counts must be integers.

Domain predicates (allowed inside DataOp/Check nodes):
- goal({{item_name}}, {{goal_count}})
- inventory({{item_name}}, {{have_count}})
- unavailable({{item_name}})
- current_count_of_item({{item_name}}) --> Count

Object Selection (ONLY allowed inside DataOp nodes):
- select_one('Item', an expression over `x` using domain predicates)  
  * select the first item that satisfies the expression, return None if no such item exists  
  * e.g., select_one('Item', not unavailable(x))
- select_all('Item', an expression over `x` using domain predicates)  
  * select all items satisfying the expression, return an empty list if no such item exists  
  * e.g., select_all('Item', unavailable(x)))

List helpers (ONLY allowed inside DataOp nodes):
- select_list_index(list_value, index_value)  
  * return the element at position index_value of list_value (0-based to align with LoopForControl)  
  * return None if index_value is out of bounds


# Current Mermaid-code Workflow

%% Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):
%%{{init: {{'theme': 'default', 'themeVariables': {{'background': '#ffffff'}} }} }}%%
flowchart TD
    %% ===================== Class Definitions =====================
    %% Guideline: Strictly follow the class definitions.
    classDef Spec fill:#f4f4ff,stroke:#6a6ab2,stroke-dasharray: 4 3,stroke-width:2px;
    classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;
    classDef LoopControl fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
    classDef PrimitiveAction fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
    classDef Check fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
    classDef DataOp fill:#f0f0f0,stroke:#888888,stroke-width:2px;

    %% ===================== Type Alias Legend =====================
    %% Guideline: Strictly follow the type alias legend.
    LEGEND["Type Legend:<br>ItemName := str<br>Count := int<br>Bool := bool<br>List_ItemName := sequence of ItemName<br>List_Count := sequence of Count"]:::Spec
    
    %% ===================== Spec =====================
    %% Guideline: Strictly follow the Spec.
{FLOW_SPEC_NODE}

    %% ===================== Interface & Spec =====================
    %% Guideline: START is entry; *_END are exits. No computations here. Line breaks use <br>. 
    %% Guideline: Strictly follow the interface definition.
{START_INTERFACE_NODE}
    SUCCESS_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=True<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=True)"]):::Interface
    FAILURE_END(["Interface: <br> Return: <br>SUCCESS_FLAG: Bool:=False<br>writes GLOBAL: (SUCCESS_FLAG: Bool:=False)"]):::Interface

    %% ===================== Primitive Actions =====================
{ACTION_NODE}  
    %% Guideline: Select the PrimitiveActions Nodes that workflow needs. Remove any nodes that are not needed. 
    %% Note that any revision on the setting of the PrimitiveActions Nodes is STRICTLY FORBIDDEN. Directly copy which node the workflow needs. 
    %% Guideline: When needed, instantiate multiple PrimitiveAction nodes by assigning unique IDs (e.g., A_GET_1, A_GET_2); each node id is a separate instance that can connect to different successors.
{AVAILABLE_ACTIONS}

    %% ===================== LoopControl (use only if needed) =====================
    %% Guideline: Only foreach loops via LoopControl with edges 'body' and 'done'.
    %% Guideline: LoopControl nodes only operate on the loop variable (e.g., LOOP_FOR_INGREDIENTS["LoopControl: <br>For idx, ingredient_i in enumerate({{loop_items}})<br>writes GLOBAL: (CURRENT_INGREDIENT: ItemName:=ingredient_i, LOOP_INDEX: Count:=idx)<br>local in: (loop_items: List_ItemName = {{TARGET_ITEMS}})"]:::LoopControl, assign globals for the needed ingredients).
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple LoopControl nodes with distinct IDs (e.g., LOOP_FOR_1, LOOP_FOR_2) to represent separate loop sites.
{LOOP_FOR_NODE}
    %% Guideline: Include a new LoopControl node if you must use a loop.


    %% ===================== Checks =====================
    %% Guideline: Checks branch on Yes/No only; inputs resolved from GLOBALS. 
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: The check nodes only operate with the defined predicates, local_in variables, the basic boolean operators (and, or, not) and numerical operators (+, -, *, /, ==, !=, <, >, <=, >=). **Other operators are not allowed.**
    %% Guideline: List-level predicates are NOT supported; when you must check a list, decompose into a LoopControl that iterates the list, Check nodes that apply object-level predicates per element, and DataOp nodes that record whether the list-level condition holds.
    %% Guideline: When needed, duplicate Check nodes with unique IDs (e.g., C_INGREDIENT_READY_1, C_INGREDIENT_READY_2) to use the same predicate at different decision points.
{CHECK_NODE}
    %% Design new Check nodes with the domain predicates if needed
    %% Here are some examples. Note these are examples—adjust the check condition and parameters binding to fit the workflow.
    C_INGREDIENT_READY["Check: <br> (inventory({{check_ingredient_name}}, {{check_available_ingredient_count}}) and {{check_available_ingredient_count}} >= {{check_target_count}})<br>local in: (check_ingredient_name: ItemName = {{CURRENT_INGREDIENT}}, check_target_count: Count = {{TARGET_COUNT}}, check_available_ingredient_count: Count = {{AVAILABLE_COUNT}})"]:::Check
    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node below. 
{D_INIT_NODE}
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple DataOp nodes by assigning unique IDs (e.g., D_FIND_MISSING_1, D_FIND_MISSING_2) so each instance can feed different downstream consumers.
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    %% Example helpers for TextCraft (edit/remove as needed):
    D_SELECT_TARGET_COUNT["DataOp: <br>writes GLOBAL: (TARGET_COUNT: Count = select_list_index({{local_counts_list}}, {{local_index_arg}}))<br>local in: (local_counts_list: List_Count = {{TARGET_COUNTS}}, local_index_arg: Count = {{LOOP_INDEX}})"]:::DataOp
    D_FETCH_AVAILABLE["DataOp: <br>writes GLOBAL: (AVAILABLE_COUNT: Count = current_count_of_item({{local_fetch_ingredient_name}})) <br>local in: (local_fetch_ingredient_name: ItemName = {{CURRENT_INGREDIENT}})"]:::DataOp
    D_COMPUTE_MISSING["DataOp: <br>writes GLOBAL: (COUNT_TO_GET: Count = max({{local_compute_target_count}} - {{local_compute_available_count}}, 0)) <br>local in: (local_compute_target_count: Count = {{TARGET_COUNT}}, local_compute_available_count: Count = {{AVAILABLE_COUNT}})"]:::DataOp
    
    %% ===================== Node Class Assignments =====================
    %% Guideline: Assign each node to a class (e.g., `PrimitiveAction`, `Check`, `DataOp`, `LoopControl`) based on its role.
{CLASS_ASSIGNMENTS}

    %% ===================== Legend links =====================
    %% Guideline: Strictly follow the legend links.
    START -.- FLOW_SPEC
    START -.- LEGEND

    %% ===================== Control-Flow Edges =====================
    %% Guideline: Keep edges control-only (no parameters); label branches with Yes/No/body/done.
    %% Guideline: Improve the control-flow edges if needed. 
    %% Guideline: Ensure every referenced global is assigned earlier in the workflow or provided by START before the consuming node runs.
    %% Guideline: Prepare required globals (with same type) before any PrimitiveAction node that reads from them. 
    %% Guideline: For outgoing edges, LoopControl nodes must have both `body` and `done`; Check nodes must have `Yes` and `No`; DataOp/PrimitiveAction nodes must emit exactly one edge.
    %% Guideline: Label every edge entering a LoopControl node with Start_Loop (for external entry/reset) or Continue_Loop (for in-loop back edges); ensure at least one Start_Loop edge targets each LoopControl; do not use these labels on edges targeting non-loop nodes.
    %% Guideline: Use `,` to split the multiple labels (if needed) on the control-flow edges. e.g. |Yes, Continue_Loop|.

{CONTROL_FLOW_EDGE}

# Error Analysis of the Current Mermaid-code Workflow
{HINTS}


# Output Format
- Output ONLY the Mermaid code block(s). No explanations or extra text. 
- Output `No Solution` whether the current sub-goal definition, parameter contract, or provided workflows/trajectories (including world states) prevent you from producing a compliant workflow under the given constraints.
"""



SKILL_PARAMETER_ANALYSIS_PROMPT = """
You are a skill-parameter analyst for a TextCraft crafting agent.

Goal: 
 - Given a skill specification and several successful past usages, derive durable rules that maximize the probability of success in future calls. 
 - Ground on the abstract world state and the task goal. Do not assume any specific item names or counts from the previous environment context. 
 - Focus on generalizable guidance inferred from the data.

# Inputs
- skill_schema: a JSON object with fields `skill_name`, `description`, `parameters`, `start_conditions`, `success_conditions`.
- success_records: a list of past successful calls for this skill. Each record may include `pre_skill_interaction` (with observations and actions), `skill_call_rationale`, and `parameter_bindings` actually used.

# Output (strict JSON only; no prose, no code fences)
{{
  "parameter_roles": {{"<param_name>": "<role summary and hard constraints extracted from the schema>", ...}},
  "selection_guidelines": [
    "<data-driven rule grounded in start/success conditions and priors>",
    "<ordering rules for list-type parameters (e.g., prefer goal-critical items first, then missing inventory items)>",
    "<deduplication, tie-breaks (e.g., lowest numeric suffix, stable order from trajectories), and truncation policy>",
    "<fallback rules when evidence is weak (e.g., default to goal items, then recipe ingredients)>"
  ],
  "anti_patterns": [
    "<what to avoid when binding parameters (e.g., violating start conditions, including duplicates, using unseen/nonexistent names or counts)>"
  ],
  "checklist_before_call": [
    "<checks derived from start_conditions and success_conditions to validate a proposed binding before calling the skill>"
  ],
  "fail_reason": "<if the skill call failed, explain how parameter settings should be improved; otherwise return null>"
}}

# Rules
- Ground your analysis in `success_records`: extract goal vs. output count relationships, ingredient → product patterns, inventory prerequisites, and ordering observed across records; prefer generalizable rules that appear more often and earlier.
- Respect `start_conditions` and `success_conditions`; do not propose guidance that would violate them.
- For list parameters (e.g., `List_ItemName`, `List_Count`), produce ordered lists encoding priority; remove duplicates; document tie-breaks and truncation strategy.
- Use item identifiers exactly as they appear in observations/records (usually snake_case) when giving examples; do not fabricate new names.
- Be concise and avoid narrative; return only the JSON object described above.

skill_schema: 
{skill_schema}
success_records: 
{success_records}
"""


SKILL_PRECONDITION_SYNTH_PROMPT = """
You are a Concepts-DSL engineer that derives symbolic pre-condition checks for TextCraft skills.

Inputs:
- `skill_schema`: JSON with `name`, `description`, `parameters`, `start_conditions`, `success_conditions`, and `steps`.
- `success_records`: A markdown-style list (`## Record i`) of successful calls. Each record may include:
  * Pre-skill interaction traces (observations/actions)
  * Skill-call rationale
  * `Parameter Bindings`
  * `Current World Facts` (a python list of predicates true at call time)

Goal:
- Infer a predicate expression (Python-evaluable Concepts DSL syntax) that must be true before invoking the skill for a *new* parameter instantiation.
- Expression should generalize across records and align with the skill description/start conditions.

Representation rules:
- Refer to skill parameters via `{{paramName}}` exactly as they appear in `skill_schema.parameters`.
- Convert item names to `snake_case` (e.g., `dark oak log` -> `dark_oak_log`).
- List parameters (e.g., `List_ItemName`) should be treated as python lists.
- Every predicate argument or free variable must map directly to a declared skill parameter (or be bound by a lambda that iterates over elements originating from that parameter); never invent new identifiers.
- Prefer conjunctions of minimal necessary clauses; avoid redundant checks already implied by others.
- Express the final `pre_condition_expression` strictly as a conjunction: output it as a JSON list of clause strings, and assume the overall condition is the logical `and` over that list.
- Keep the clause order identical between `pre_condition_expression` and `clause_breakdown`.
- When unsure, state assumptions explicitly.
- Type violations are strictly forbidden: each predicate argument must respect the parameter's declared type.
- Use only the allowed domain predicates:
  `goal({{ItemName}}, {{Count}})`, `inventory({{ItemName}}, {{Count}})`, `unavailable({{ItemName}})`,
  `current_count_of_item({{ItemName}})`, `count({{ItemName}})`, `is_holding({{ItemName}})`,
  `has_enough({{ItemName}}, {{Count}})`, `needs_item({{ItemName}}, {{Count}})`,
  `can_get({{ItemName}}, {{Count}})`, `can_craft({{ItemName}}, {{Count}})`, `can_inventory()`
  - Use boolean operators: not, and, or.
  - Use numeric comparisons (e.g., `current_count_of_item({{ItemName}}) >= {{Count}}`) when needed.
  - For list parameters, membership tests can use only:
    - `exists_item_in_list({{List_ItemName}}, lambda x: is_holding(x))`
    - `forall_item_in_list({{List_ItemName}}, lambda x: current_count_of_item(x) >= 1)`

Output format (MUST be valid JSON, no prose outside the JSON):
{{ 
  "pre_condition_expression": ["<clause_1>", "<clause_2>", "..."],
  "clause_breakdown": [
    {{
      "clause": "<sub-expression or predicate>",
      "why_necessary": "<1 sentence referencing skill schema or records>",
      "evidence_records": ["Record 1", "Record 4"]
    }}
  ],
  "text_checklist": [
    "<imperative checklist item phrased so an engineer can manually verify it>",
    "..."
  ],
  "assumptions": [
    "<explicit assumption or 'none'>"
  ]
}}

skill_schema:
{skill_schema}
success_records:
{success_records}

{hints}
"""


TRAJ_SEGMENTATION_PROMPT = """You are an expert analyst that maps sub-goal schemas onto embodied-agent trajectories. 
Input
------


## Subgoal specification: 
{subgoal_spec}


## Overall Task Instruction of the following trajectory:
{task_instruction}

## Trajectory steps: 
{traj_info}

## Parameter type reference (only these types are valid):
- `ItemName`: use the precise item instance name from the trajectory (e.g., `oak planks.`, `dark oak sign`).
- `Count`: use the precise count from the trajectory. 
- `List_<T>` (e.g., `List_ItemName`, `List_Count`): output an ordered list whose elements each satisfy the guidance for the element type `T`; preserve the order induced by the subgoal execution.

Task
-----
1. Restate the subgoal intent in your own words.
2. Scan the trajectory in order and identify the **minimal contiguous** span of steps that corresponds to executing this subgoal. Include `think` steps only if they contribute to deciding or executing the search. The span must satisfy all start conditions before it begins and all success conditions by its end.
3. For every step inside that span, note why it is relevant (e.g., "retrieves the item", "crafts the items with ingredients").
4. Using that span, instantiate every parameter listed under `parameters` in the subgoal specification. Respect each parameter's declared type when formatting its value, following the type reference above.
  - For sub-goal with crafting the parameters bindings should follow the provided crafting recipe. 
  - {CRAFT_COMMANDS}
  - Craft operation with different parameters (e.g., different item names, counts) is **not allowed** in this task.
  - If the span uses a craft command, bind all count-related parameters (e.g., target outputs and ingredient counts) to exactly match that command's recipe counts. Do not bind smaller or larger counts than the craft action you cite.
5. Cite one or more steps that demonstrate each start condition held before the span and each success condition held at the end.
6. Summarize in natural language the key events that occurred **before** the selected span begins (i.e., steps with index before `start_index`) that are relevant for understanding the agent's situation. Avoid assumptions not grounded in the trajectory. The summary must be self-contained, and avoid citing supporting indexes. 
7. Explain, referencing the overall task and the pre-span context, why the agent initiates this subgoal at that specific moment (the intent behind starting the subgoal when it did). Ground the explanation in explicit evidence from the trajectory. The explanation must be self-contained, and avoid citing supporting indexes.

Output
-------
Return **only** valid JSON with this structure and double-quoted keys/strings:

{{
  "subgoal_name": string,
  "pre_span_summary": string,
  "start_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "success_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "subtrajectory_span": {{"start_index": int, "end_index": int}},
  "subtrajectory_steps": [
    {{"index": int, "observation": string, "action": string, "why_relevant": string}}
  ],
  "related_crafting_commands": [<string: from the provided crafting commands>, ...],
  "parameter_bindings": {{"<parameter_name>": <value>, ...}},
  "subgoal_initiation_intent": string,
  "notes": string
}}

Keep `why_relevant`, `explanation`, and `notes` concise; use "none" if no additional notes. 
`pre_span_summary` and `subgoal_initiation_intent` must stay self-contained, and avoid assumptions not grounded in the trajectory.
"""


TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT = """
You are the Textual-Gradient analyst for workflow-based household skills. 
Your mission is to read the current skill definition, the compiled Mermaid workflow, and the failed interaction trace, then extract actionable guidance that tells the workflow author exactly how to edit the graph to prevent the failure from recurring.

## Inputs
- Skill Calling (JSON with description, start/success conditions, steps, parameters): 
{skill_schema}
- Workflow (Mermaid code block with %% annotations, node definitions, and control-flow edges): 
{mermaid_workflow}
- Failed interaction transcript (chronological actions, observations): 
{failure_interaction}
- Execution trace (chronological actions, observations, world-states): 
{execution_trace}

## Objective
Infer why the workflow failed in the environment, then provide multi-perspective modification directions (e.g., start-condition gating, parameter repair, control-flow rewrites) that point the author to concrete edits. Always cite evidence from the failure trace or world facts and respect the existing skill intent—propose refinements, not new goals.

## Failure-mode perspective
Embodied skill failures usually fall into five buckets. For each bucket, analyze the listed sub-angles and tie findings back to the skill schema (start/success conditions, parameter declarations/types), the Skill Calling record, or the Mermaid workflow (DataOp/Check/PrimitiveAction nodes).

1. **Skill applicability** — the workflow fires even though its start conditions (e.g., `holding(item)` before a move) are false, meaning another skill should have been used earlier.
   - `skill_applicability`: are the start/success conditions and invariants implied by the skill schema satisfied at invocation time? If not, explain how to gate/adjust the skill or defer to prerequisite skills.

2. **Skill-call parameter contract** — the skill schema may be missing, mis-typed, or over-specified parameters relative to what the workflow actually needs from the outside world.
   - `parameter_contract_coverage`: do the declared parameters expose every external input referenced by start/success conditions or workflow nodes (DataOp/Check/PrimitiveAction), without leaving required globals implicit?
   - `parameter_type_alignment`: are parameter names/types scoped correctly (no type mismatch, overload, or unused parameters) so callers can bind them unambiguously?

3. **Skill-call parameter bindings** — the chosen skill is reasonable, but the `parameter_bindings` supplied in the Skill Calling JSON are missing, malformed, or mismatched to the schema, so the skill executes with bad arguments.
   - `parameter_schema_compliance`: do the provided `parameter_bindings` respect the schema's declarations (name/type/arity) and the start/success conditions that reference them? Highlight any mismatch between required parameter types and the actual selection.
   - `parameter_choice_quality`: assuming the schema is respected, would alternative parameter values (different receptacle/item selection, ordering, truncation) better satisfy the goal? Compare against the failure trace to pinpoint stronger binding heuristics.

4. **Workflow data-flow integrity** — the Mermaid workflow mishandles globals or data propagation, causing otherwise-correct parameters to be lost, stale, or misrouted inside the graph.
   - `data_wiring_and_scope`: are DataOp/LoopControl nodes assigning required globals before use, scoping them correctly across branches/loops, and resetting them to avoid leakage?
   - `selection_and_refresh`: do selection helpers populate complete/deduped lists and refresh when observations change? Are missing/None/empty cases handled so downstream nodes do not consume invalid data?
   - `state_validation`: are there checks after critical writes (e.g., verifying globals exist/reachable/contains) so downstream steps do not run on invalid data?

5. **Control-flow soundness** — the workflow logic itself has missing/incorrect branching, guards, retries, or action ordering (data wiring is covered above).
   - `control_flow_coverage`: do the Check / LoopControl nodes guard all necessary branches, early exits, and retries to keep the workflow aligned with the execution trace?
   - `primitive_sequence_correctness`: does the ordering of PrimitiveAction nodes (open/take/move/etc.) contradict the observed environment state or omit required actions seen in the failure?
   - `recovery_and_observability`: does the workflow provide fallback probes, alternate branches, or state validation after high-risk steps so the agent can react when observations contradict expectations?

## Output Format
Return a strict JSON object (no code fences) with:
{{
  "thought": "<2-3 sentences summarizing which failure buckets were implicated and why.>",
  "angle_reports": [
    {{
      "angle": "skill_applicability",
      "status": "issue | ok",
      "why": "<If ok, explain why this bucket is satisfied; if issue, summarize the root cause and cite start/success conditions.>",
      "subfindings": [
        {{
          "sub_angle": "skill_applicability",
          "finding": "<specific observation for this sub-angle>",
          "impact": "<how it affected the skill outcome>",
          "evidence": ["cite failure_interaction steps / world facts"],
          "edit_hint": "<pointer to node IDs, predicates, or schema clauses to edit>"
        }}
      ],
      "optimization_directions": ["<if status=issue, list concrete fixes such as gating conditions; otherwise explain why no change is needed>"]
    }},
    {{
      "angle": "skill_parameter_contract",
      "status": "issue | ok",
      "why": "<Explain whether the parameter list itself omits/overloads inputs needed by the workflow or schema clauses.>",
      "subfindings": [
        {{
          "sub_angle": "parameter_contract_coverage | parameter_type_alignment",
          "finding": "<detail for each applicable sub-angle>",
          "impact": "<how it affected the skill outcome>",
          "evidence": ["cite relevant steps / workflow nodes / schema clauses"],
          "edit_hint": "<reference missing/extra parameter names or type fixes in the Skill Calling schema>"
        }},
        ...
      ],
      "optimization_directions": ["<add/remove/retarget parameters so every external input used in workflow/start/success is explicit and correctly typed>"]
    }},
    {{
      "angle": "skill_call_parameter_bindings",
      "status": "issue | ok",
      "why": "<Summarize whether the Skill Calling parameter_bindings violated the schema or simply picked poor candidates.>",
      "subfindings": [
        {{
          "sub_angle": "parameter_schema_compliance | parameter_choice_quality",
          "finding": "<detail for each applicable sub-angle>",
          "impact": "<how it affected execution>",
          "evidence": ["cite relevant steps"],
          "edit_hint": "<reference parameter names or binding choices in Skill Calling>"
        }},
        ...
      ],
      "optimization_directions": ["<suggest parameter contract fixes, ordering heuristics, or binding defaults>"]
    }},
    {{
      "angle": "workflow_data_flow_integrity",
      "status": "issue | ok",
      "why": "<Explain whether DataOp/LoopControl/global writes kept data fresh and in scope.>",
      "subfindings": [
        {{
          "sub_angle": "data_wiring_and_scope | selection_and_refresh | state_validation",
          "finding": "<detail for each relevant sub-angle>",
          "impact": "<how it affected execution>",
          "evidence": ["cite relevant steps"],
          "edit_hint": "<reference DataOp/LoopControl nodes, globals, or selection helpers>"
        }},
        ...
      ],
      "optimization_directions": ["<if issue, list data-flow fixes; if ok, note why no change needed>"]
    }},
    {{
      "angle": "workflow_control_flow_soundness",
      "status": "issue | ok",
      "why": "<Explain whether the Mermaid workflow structure caused or avoided the failure (excluding data wiring).>",
      "subfindings": [
        {{
          "sub_angle": "control_flow_coverage | primitive_sequence_correctness | recovery_and_observability",
          "finding": "<detail for each relevant sub-angle>",
          "impact": "<how it affected execution>",
          "evidence": ["cite relevant steps"],
          "edit_hint": "<reference Check/LoopControl/PrimitiveAction node IDs>"
        }},
        ...
      ],
      "optimization_directions": ["<if issue, list structural workflow fixes; if ok, note why no change needed>"]
    }}
  ],
  "cross_angle_edits": [
    {{
      "priority": 1,
      "change": "<single actionable modification (add Check, move DataOp, widen receptacle list, etc.)>",
      "linked_angles": ["workflow_data_flow_integrity"],
      "success_metric": "<observable condition confirming the fix>"
    }},
    ...
  ],
  "open_questions": ["<clarifications the workflow author must gather>", ...]
}}

Rules:
- In `thought`, explicitly name which of the five failure-mode buckets (skill applicability, skill-call parameter contract, skill-call parameter bindings, workflow data-flow integrity, workflow control-flow soundness) were implicated.
- `angle_reports` must contain exactly five entries in the order shown above.
- Every `subfindings` entry must cite evidence (trajectory step numbers, observation quotes, or facts). If a bucket is `ok`, include a `subfindings` entry explaining how you validated it.
- When `status` is `issue`, provide at least one optimization direction grounded in the corresponding sub-angles (e.g., adjust start conditions, refine parameter selection heuristics, add new DataOp checks or control guards). When `status` is `ok`, use `optimization_directions` to state “no change; already satisfied because …”.
- `cross_angle_edits` should contain 2–4 high-impact modifications; include all relevant `linked_angles` and measurable success metrics.
- Prefer concise, surgical instructions tied to specific Mermaid nodes, parameter names, or predicate expressions.
- Keep `open_questions` empty (`[]`) only if no outstanding uncertainties remain; otherwise ask for precise missing data.

"""


SUBGOAL_IDENTIFICATION_PROMPT = """
# Role Definition
You are an expert Neuro-Symbolic Planner and Task Decomposition Specialist. Your task is to analyze interaction trajectories to identify **Parameterized Sub-goals** (Reusable Skills). 

# Input Data
## 1. Symbolic Predicate Definitions
The world state is described by the following predicates:
### A) State Facts
- `goal(item_id, count)`: The task target.
- `inventory(item_id, count)`: The agent holds `count` units of `item_id`.
- `unavailable(item_id)`: The item cannot be directly obtained from the environment and must be crafted.

### B) Naming Conventions
- `item_id`: snake_case strings (e.g., `dark_oak_log`).
- `count`: integer.

## 2. Interaction Trajectories
Here are multiple trajectories. Each step includes the Action and the resulting Symbolic State Change (Diff).
{examples}

# Task Instructions
You need to perform a **"Generalization Process"** to convert specific trajectory steps into abstract reusable sub-goal definitions, and then use those definitions to **fully segment every trajectory** into an ordered concatenation of sub-goal instances.

## Core Optimization Objective (Very Important)
Your primary goal is to build a **reusable** sub-goal library:
- Prefer sub-goal definitions that can be instantiated **many times across different trajectories**.
- Minimize the number of distinct sub-goals while still being faithful to the logs.
- Only create a sub-goal that appears in a single trajectory if it is **necessary** to achieve full coverage/segmentation.
- When two sub-goals are similar, prefer **merging** them into a more abstract parameterized template (but do not over-generalize into a meaningless catch-all).

## The Process:
1.  **Scan for Pivot States:** Look for steps where the **State Diff** shows a significant achievement (Mutation/Enablement). These pivot steps will typically be the **end step** of a sub-goal instance.
2.  **Extract Concrete Predicate:** Copy the *exact* logical expression that became true at that pivot step.
    *   *Example:* If the log says `Diff: +inventory(oak_planks, 4)`, the concrete predicate is `inventory(oak_planks, 4)`.
3.  **Derive Abstract Predicate:** Replace the specific values (constants) with generic variable placeholders `<name>`.
    *   *Transformation:* `inventory(oak_planks, 4)`  -->  `inventory(<item_id>, <quantity>)`.
4.  **Map Parameters:** Identify which concrete value corresponds to which abstract variable.
5.  **Cluster + Merge for Reuse:** Group pivot predicates across **all trajectories** into a small set of parameterized sub-goals that cover many instances.
6.  **Trajectory Segmentation (Coverage Requirement):**
    - For each trajectory, produce an ordered list of sub-goal instances (segments).
    - Each segment must be a **contiguous step range** `[start_step_id, end_step_id]` ending at a pivot step where the `concrete_outcome` becomes true.
    - Segments must cover the entire trajectory: start at step `1`, end at the last step, with **no gaps and no overlaps**.
    - **Hard constraint:** for segment `k>1`, `segments[k].start_step_id == segments[k-1].end_step_id + 1`.
    - **Hard constraint:** `segments[0].start_step_id == 1` and `segments[-1].end_step_id == overall_step_count`.
    - **Hard constraint:** each step index belongs to **exactly one** segment.
    - If you think multiple sub-goals “complete” on the same step, you **must not** create overlapping segments; instead, choose the single best primary `sub_goal_name` for that segment and list additional achieved predicates under `aux_outcomes`.
    - Steps that are “just navigation” should be absorbed into the nearest segment that they enable (usually the segment whose pivot happens next). Do not create a separate navigation-only sub-goal unless it is a genuinely reusable key-facility attainment and appears across trajectories.

## Rules for Predicates:
*   **Variable Naming:** Use angle brackets `<...>` for variables. Use meaningful names like `<target_item>`, `<tool_type>`, `<min_count>`.


# Constraints
*   **Ignore Navigation:** Do not label simple moves unless they reach a key facility (e.g., `Near(crafting_table)`).
*   **High-Level Semantics:** Focus on *what* was achieved (State), not just *how* (Action).

# Output Format
Output a single JSON object with two parts:
1) a **sub-goal library** (goal clusters shared across trajectories), and
2) a **trajectory decomposition** (full segmentation) that references the library.

JSON Structure:
```json
{{
  "sub_goal_library": [
    {{
      "sub_goal_name": "Name of the parameterized sub_goal (e.g., craft_material_batch, obtain_material, craft_tool)",
      "abstract_goal_expression": "The predicate pattern with placeholders (e.g., inventory(<target_item>, <target_count>))",
      "variable_definitions": {{
        "<target_item>": "The item ID being obtained/crafted",
        "<target_count>": "The (resulting) quantity"
      }},
      "reasoning": "Why this is a valid reusable sub-goal pattern and what it semantically captures",
      "instances": [
        {{
          "trajectory_id": 1,
          "end_step_id": 3,
          "concrete_outcome": "inventory(oak_planks, 4)",
          "parameters": {{ "item_id": "oak_planks", "count": 4 }}
        }},
        {{
          "trajectory_id": 2,
          "end_step_id": 5,
          "concrete_outcome": "inventory(stick, 2)",
          "parameters": {{ "item_id": "stick", "count": 2 }}
        }}
      ]
    }}
  ],
  "trajectory_decompositions": [
    {{
      "trajectory_id": 1,
      "overall_step_count": 5,
      "segments": [
        {{
          "segment_id": 1,
          "sub_goal_name": "sub_goal name",
          "start_step_id": 1,
          "end_step_id": 3,
          "concrete_outcome": "inventory(oak_planks, 4)",
          "aux_outcomes": [],
          "parameters": {{ "item_id": "oak_planks", "count": 4 }}
        }}, 
        {{
          "segment_id": 2,
          "sub_goal_name": "sub_goal name",
          "start_step_id": 4,
          "end_step_id": 5,
          "concrete_outcome": ...
          "parameters": ...
        }}, 
        ...
      ]
    }}
  ],
  "validation": {{
    "coverage_ok": true,
    "notes": "Briefly state how you ensured full coverage (no gaps/overlaps) and reuse (which sub-goals appear across multiple trajectories)."
  }}
}}
"""


SKILL_ACTION_ROUTER_PROMPT = """
You are a skill/action router for a TextCraft crafting agent. Decide which skill or raw action should be executed next.

# Task and history
{task_message}

# Current world facts (set semantics)
{facts_current}


# Skill/Action Information on the related Tasks
{selection_guideline}

# Current Sub-Goal (only one)
{sub_goal_guideline}

Decide the status of the current sub-goal:
1) doing: keep working on this sub-goal
2) completed: the sub-goal is already satisfied
3) failed: attempted steps but cannot reach it; needs a new plan

Field rules:
- "active_subtask": must be one of "doing", "completed", "failed".
- "subtask_reason":
  - doing: your next-step plan.
  - completed: success evidence from facts.
  - failed: brief summary of the blocking interaction(s).
- Do not retry the same skill with the same parameter bindings. If that would fail, explicitly recommend one of: different parameter bindings for the same skill, a different skill, a fallback raw-action sequence, or a re-analysis of the sub-goal.
- Follow item names from crafting commands. Some ingredients are generic tags (e.g. `planks`, `wool`) — if the generic name fails, substitute a specific variant from your inventory (e.g. `3 planks` → `3 oak planks`).

Routing steps:
1) Choose the current/next sub-goal from ordered_subtasks using history and facts.
2) Match that sub-goal against skill_guidelines and raw_action_guidelines using triggers and context.
3) Propose up to 3 candidates (prefer skills) with evidence and expected outcome.
4) Recommend one; if none apply, return type "none" and list missing info.

Craft Operations:
**Use crafting commands as templates. If an ingredient is a generic tag that you cannot obtain, replace it with a specific variant from your inventory.**
{crafting_commands}

Strict JSON ONLY (no prose, no code fences):
{{
  "active_subtask": "doing|completed|failed",
  "subtask_reason": "<why you chose this status>",
  "candidates": [
    {{
      "type": "skill|action",
      "name": "<skill name or raw action name>",
      "phase": "<phase/sub-task it aligns to>",
      "trigger_match": ["<triggers or pre-context cues that match>"],
      "supporting_facts": ["<facts that justify applicability>"],
      "expected_outcome": "<state change or follow-up intent>",
      "confidence": 0.0
    }}
  ],
  "recommendation": {{
    "type": "skill|action|none",
    "name": "<recommended name or none>",
    "why": "<short rationale>"
  }},
  "missing_info": ["<what evidence is needed if nothing applies>"]
}}
"""

EXECUTION_WITH_ROUTING_PROMPT = """
Interact with TextCraft to execute the routed skill/action.

# Here are two examples:
{examples}

# Here is the task.
{task_message}

# Current world facts:
{facts_current}

# Available actions: 
think: <thought about the overall task goal, sub-goals decomposition, current state, and the next sub-goal>
inventory
get: <item_count> <item_name>
craft: <item_count> <item_name> using <ingredients_count_list> <ingredients_name_list>
**Use crafting commands as templates. If an ingredient is a generic tag (e.g. `planks`) you cannot obtain, replace it with a specific variant from inventory (e.g. `oak planks`).**
{crafting_commands}

# Available skills: 
{skill_info}

# Current Sub-Goal
{sub_goal_guideline}

{action_check_error}


# Routing recommendation (follow unless impossible)
{routing_recommendation}

- Honor the routing recommendation. If recommendation.type is "skill", set `skill` to that name and `action` to null. If recommendation.type is "action", set `action` to the raw action string and `skill` to null. Only deviate if impossible, and explain in reasoning.
- Do not repeat the same action/skill if nothing happened or failed. 
- **Prioritize using the provided skills**.
- For the selected skills, follow the **Guidelines for your Parameter Bindings** to ensure the skill is called correctly to achieve the intended effect.
- **Craft commands**: use recipe counts exactly; for ingredient names, use the specific variant from your inventory if the generic tag name fails (e.g. `planks` → `oak planks`).
- If none of the skills are applicable, **then** select the action from the available actions.

Please enter your action in a json format.
```json {{
    "reasoning": <thought on action selection based on overall goals, current states, and sub-goals; note if deviating from recommendation>,
    "skill": <The selected skill. Return null if none of the skills are applicable or if a raw action is recommended.>,
    "start_condition_evidence": <thought on the start conditions of the selected skill>,
    "pre_span_summary": <summary in natural language the key events that occurred **before** the current skill/action begins that are relevant for understanding the agent's situation. The summary must be self-contained.>,
    "intent": <the intent of the current skill/action, must stay self-contained>,
    "parameter_bindings": {{
      <skill-param-1> : <value-1>, 
      ... 
    }},
    "action": <The selected action if using a raw action. If a skill is selected, set this field to null.>
}}
```
"""


TRAJ_SEGMENTATION_REVISION_PROMPT = """
You are an expert analyst that maps sub-goal schemas onto embodied-agent trajectories. You can leverage previously verified matches to stay consistent with how the subgoal has been satisfied before, while still grounding all claims in the new trajectory.
Input
------

## Subgoal specification: 
{subgoal_spec}

## Previously matched exemplars (JSON array; may be empty):
{matched_examples}

Note: Each exemplar entry may contain fields such as `overall_task_goal`, `pre_span_summary`, `subgoal_initiation_intent`, `detailed_subtrajectory`, `parameter_bindings`, `start_condition_evidence`, and `success_condition_evidence`. Use these exemplars to understand typical intent phrasing, parameter bindings, and evidence styles, but never contradict the observations in the current trajectory.

## Overall Task Instruction of the following trajectory:
{task_instruction}

## Trajectory steps: 
{traj_info}

## Parameter type reference (only these types are valid):
- `ItemName`: use the precise item instance name from the trajectory (e.g., `oak planks.`, `dark oak sign`).
- `Count`: use the precise count from the trajectory. 
- `List_<T>` (e.g., `List_ItemName`, `List_Count`): output an ordered list whose elements each satisfy the guidance for the element type `T`; preserve the order induced by the subgoal execution.

Task
-----
1. Review the matched exemplars to extract representative intent wording, parameter semantics, and evidence patterns that should remain consistent when possible.
2. Restate the subgoal intent in your own words, adapting terminology so it aligns with both the exemplars and the current trajectory.
3. Scan the trajectory in order and identify the **minimal contiguous** span of steps that corresponds to executing this subgoal. Include `think` steps only if they contribute to deciding or executing the search. The span must satisfy all start conditions before it begins and all success conditions by its end.
4. For every step inside that span, note why it is relevant (e.g., "retrieves the item", "crafts the items with ingredients").
5. Using that span, instantiate every parameter listed under `parameters` in the subgoal specification. Respect each parameter's declared type when formatting its value, following the type reference above. When multiple options are possible, prefer values consistent with the exemplars as long as they are supported by the current span.
  - For sub-goal with crafting the parameters bindings should follow the provided crafting recipe. 
  - {CRAFT_COMMANDS}
  - Craft operation with different parameters (e.g., different item names, counts) is **not allowed** in this task.
  - If the span uses a craft command, bind all count-related parameters (e.g., target outputs and ingredient counts) to exactly match that command's recipe counts. Do not bind smaller or larger counts than the craft action you cite.
6. Cite one or more steps that demonstrate each start condition held before the span and each success condition held at the end. Mirror the level of specificity used in the exemplars when it improves clarity, but keep the explanation grounded in the new trajectory.
7. Summarize in natural language the key events that occurred **before** the selected span begins (i.e., steps with index before `start_index`) that are relevant for understanding the agent's situation. Avoid assumptions not grounded in the trajectory. The summary must be self-contained, and avoid citing supporting indexes. 
8. Explain, referencing the overall task, the exemplars (if helpful), and the pre-span context, why the agent initiates this subgoal at that specific moment. Ground the explanation in explicit evidence from the trajectory. The explanation must be self-contained, and avoid citing supporting indexes.
9. Use the `notes` field to briefly mention whether any exemplar informed your decisions (reference them by short description or index if available); use "none" only when no exemplar insight was needed.

Output
-------
Return **only** valid JSON with this structure and double-quoted keys/strings:

{{
  "subgoal_name": string,
  "pre_span_summary": string,
  "start_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "success_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "subtrajectory_span": {{"start_index": int, "end_index": int}},
  "subtrajectory_steps": [
    {{"index": int, "observation": string, "action": string, "why_relevant": string}}
  ],
  "related_crafting_commands": [<string: from the provided crafting commands>, ...],
  "parameter_bindings": {{"<parameter_name>": <value>, ...}},
  "subgoal_initiation_intent": string,
  "notes": string
}}

Keep `why_relevant`, `explanation`, and `notes` concise. If no exemplar insight is required, set `notes` to "none".
`pre_span_summary` and `subgoal_initiation_intent` must stay self-contained, and avoid assumptions not grounded in the trajectory.
"""


SUBGOAL_JUDGEMENT_PROMPT = """
# Subgoal Description:
{subgoal}
# Skill Calling:
{skill_calling}
# Interaction history before skill execution:
{interaction_history_before_skill_execution}

# Skill execution record:
{skill_execution_record}

# Task:
1) Bind placeholders in conditions using `parameter_bindings` from the subgoal description.
2) Determine whether each success condition is satisfied at the end of the skill execution, grounded in the execution record (and pre-skill history if needed).
3) Provide a concise but explicit analysis of which conditions are met and which are not. Treat missing or contradictory evidence as not satisfied.
4) Set `success_flag` to true only if ALL success conditions are satisfied.

# Failure handling:
- If `success_flag` is true, set `failure_analysis` to "none" and return `updated_start_conditions` and `updated_success_conditions` identical to the input lists, and `updated_parameter_bindings_guidelines` identical to the input `parameter_bindings_guidelines`.
- If `success_flag` is false, write a short `failure_analysis` explaining the failure and propose `updated_start_conditions` and/or `updated_success_conditions` to prevent similar failing calls. Also propose `updated_parameter_bindings_guidelines` by minimally refining the input `parameter_bindings_guidelines` to avoid repeating the same binding mistake.
- Updates must be minimal, checkable, and grounded in the failure evidence. Prefer adding missing prerequisites to `start_conditions`; only change `success_conditions` if they were too weak or mismatched. For parameter binding guidelines, prefer appending new checks/anti-patterns instead of rewriting.
- Preserve the original wording style and keep parameter placeholders (e.g., {{target_item}}) intact for updated conditions. Do not add new parameters unless they already appear in the subgoal description.
- `failure_analysis` must be grounded with concrete bound parameters (exact item names + counts) and be concise.
- Use exact item names from the recipes/crafting commands, regardless of singular/plural (e.g., use `acacia_logs` for "1 acacia logs" if "1 acacia logs" appears in the given receipes). Keep it terse (no long paragraphs).
- `parameter_bindings_guidelines`: 
  - If `success_flag` is false AND start_conditions/pre_conditions are satisfied, treat the failure as likely parameter-binding error; explicitly update binding guidelines to prevent the same binding (e.g., output-count mismatch, invalid batching). 
  - Prefer appending new checks/anti-patterns/fail_reason instead of rewriting.

# Following Suggestions:
- Always provide `following_suggestions` as a short, executable plan (1-2 steps). 
- The current skill and its parameter bindings **have been evaluated as not viable**. State a concrete alternative path to achieve the same sub-goal using the task's actual parameters (include exact item names and counts).
- Use exact item names from the recipes/crafting commands, regardless of singular/plural (e.g., use `acacia_logs` for "1 acacia logs" if "1 acacia logs" appears in the given receipes). Keep it terse (no long paragraphs).


# Output Format (strict JSON only):
{{
  "success_conditons_analysis": string,
  "success_flag": Bool,
  "failure_analysis": string,
  "following_suggestions": string,
  "updated_start_conditions": [string, ...],
  "updated_success_conditions": [string, ...],
  "updated_parameter_bindings_guidelines": dict,
}}
"""

SUBGOAL_EXECUTION_RETHINK_PROMPT = """
# You are analyzing a failed skill execution. The skill workflow has already failed.
# Use the pre-skill interaction history, the execution record, and the execution trace
# to explain why the state before calling the skill caused the failure, and how to
# prevent it next time.

# Subgoal Description:
{subgoal}
# Skill Calling:
{skill_calling}
# Interaction history before skill execution:
{interaction_history_before_skill_execution}

# Skill execution record:
{skill_execution_record}

# Mermaid Workflow:
{mermaid_workflow}
# Execution trace:
{execution_trace}

# Task:
1) Identify the critical pre-skill facts/assumptions from the interaction history that should have held for success but did not.
2) Use the execution record and trace to pinpoint where the workflow failed and which preconditions/assumptions were violated.
3) Propose minimal, checkable updates to `start_conditions` and/or `success_conditions` to block similar failing calls in the future.
4) Propose minimal updates to `parameter_bindings_guidelines` to avoid repeating the same binding mistake. If start_conditions/pre_conditions are satisfied but execution still failed, treat it as a likely parameter-binding error and update binding guidelines explicitly (e.g., output-count mismatch, invalid batching).

# Guidelines:
- The failure is already confirmed; focus on *why the pre-skill state led to failure*.
- Keep updates grounded in evidence from the history/trace; prefer adding missing prerequisites to `start_conditions`.
- Preserve placeholders (e.g., {{target_item}}) and do not add new parameters or concrete item names unless they already appear in the subgoal description.

# Following Suggestions:
- Always provide `following_suggestions` as a short, executable plan (1-2 steps). 
- The current skill and its parameter bindings **have been evaluated as not viable**, state a concrete alternative path to achieve the same sub-goal using the task's actual parameters (e.g., name the specific target_item, counts, or ingredients), and make the steps directly actionable for the current task.
- Use the task's crafting commands and execution evidence to suggest missing ingredients and how to obtain/craft them.


# Output Format (strict JSON only):
{{
  "failure_analysis": string,
  "updated_start_conditions": [string, ...],
  "updated_success_conditions": [string, ...],
  "updated_parameter_bindings_guidelines": object,
  "following_suggestions": string
}}
"""


# ===================== Online Skill Evolution Prompts =====================

EPISODE_RECOVERY_ANALYSIS_PROMPT = """You are analyzing a TextCraft agent's episode trajectory to identify valuable failure-to-recovery patterns that can improve the agent's skill system.

The agent has these skills (mermaid flowchart graphs):
{skill_descriptions}

When a skill graph reaches FAILURE_END, the planner recovers by issuing actions or calling other skills. We want to identify recovery patterns and classify HOW they should improve the system.

# Task Goal
{task_goal}

# Episode Success
{episode_success}

# Crafting Commands Available
{crafting_commands}

# Skill Execution Log (structured)
{skill_execution_log}

# Full Episode Trajectory
{trajectory}

# Instructions

Analyze the trajectory and identify valuable failure-then-recovery patterns.

## Pattern Types
**Multiplicity**: The skill produced insufficient quantity. Recovery: re-fetch ingredients and re-craft.
**Depth Composition**: A skill tried to fetch/get an item but it was unavailable. Recovery: the planner used a different strategy (e.g., called another skill or crafted instead of fetching).

## Evolution Type Classification (CRITICAL)
For each pattern, decide WHERE the fix belongs:

**`skill_graft`** — The recovery uses only operations that are within the failing skill's own responsibility.
- Example: `craft_item_from_ingredients` fails because an ingredient is missing → recovery fetches the ingredient then re-enters the craft loop. Fetching ingredients IS part of "ensure ingredients are ready before crafting", so this is an intra-skill improvement.
- Rule: The recovery actions are the SAME type of operations the skill already performs (e.g., inventory, get, craft within a craft skill).

**`planner_routing`** — The recovery works by switching to a DIFFERENT skill or a fundamentally different strategy that is OUTSIDE the failing skill's responsibility.
- Example: `check_and_fetch_item` fails (can't fetch "red nether bricks") → planner recovers by looking up a recipe and calling `craft_item_from_ingredients`. Crafting is NOT the fetch skill's job — the planner should learn to route differently.
- Rule: The failing skill has a clear, narrow responsibility (e.g., "fetch items"). If the recovery requires a capability outside that scope (e.g., "craft items"), it's a planner routing issue, NOT a skill deficiency.

**`new_skill`** — The recovery pattern is reusable across multiple skills and represents a capability not covered by any existing skill.
- Rule: Use sparingly. Only if the pattern clearly doesn't fit as a graft or routing rule.

Only select patterns where:
1. The recovery was actually successful (the agent obtained what was missing)
2. The pattern is **structurally generalizable** — not a one-off fix for a specific item
3. The failure node in the skill graph is identifiable from the diagnostic

# Output Format (strict JSON only):
{{
  "analysis": "brief reasoning about what happened in this episode",
  "patterns": [
    {{
      "pattern_type": "multiplicity" or "depth_composition",
      "evolution_type": "skill_graft" or "planner_routing" or "new_skill",
      "failing_skill": "skill name that failed",
      "failure_diagnostic": "the diagnostic message from the failure",
      "recovery_segment": "the relevant portion of the trajectory showing the recovery actions (copy verbatim)",
      "recovery_skill_used": "name of the other skill used in recovery, or 'none' if only primitive actions",
      "value_assessment": "high" or "low",
      "reason": "why this pattern is/isn't valuable, and WHY this evolution_type was chosen"
    }}
  ]
}}
"""

ROUTING_RULE_EXTRACTION_PROMPT = """Extract a concise planner routing rule from the following failure-recovery pattern.

# Failing Skill
{failing_skill}: {failing_skill_description}

# Failure Diagnostic
{failure_diagnostic}

# Recovery Strategy (what the planner did to recover)
{recovery_segment}

# Recovery Skill Used
{recovery_skill_used}

# Available Skills
{skill_descriptions}

# Instructions
The planner failed by calling `{failing_skill}` in a situation where a different skill would have been appropriate. Extract a routing rule that tells the planner WHEN to use the alternative approach instead.

The rule should be:
- **General**: Apply to any item/ingredient, not just the specific ones in this episode
- **Actionable**: Tell the planner exactly what condition to check and what to do instead
- **Concise**: One or two sentences

# Output Format (strict JSON only):
{{
  "condition": "When [specific condition that indicates the failing skill won't work]",
  "action": "Use [alternative skill/approach] instead, because [reason]",
  "rule_summary": "One-line rule: if X then Y instead of Z"
}}
"""

TRAJECTORY_CLEANING_PROMPT = """Extract and clean a recovery sub-trajectory from the following episode segment.

# Context
The skill "{skill_name}" failed at node "{failure_node}" because: {diagnostic}
The planner then recovered by executing the following actions.

# Raw Recovery Segment
{recovery_segment}

# Crafting Commands Available
{crafting_commands}

# Instructions

Clean the trajectory by:
1. **Remove** failed retry attempts (e.g., repeated "get X" that all fail — keep only the last failed attempt if needed for context, but remove pure duplicates)
2. **Remove** irrelevant actions: "think: The final goal has been achieved" or other hallucinated think actions that don't contribute
3. **Remove** actions unrelated to resolving the specific failure
4. **Preserve** the essential causal chain: each action's output feeds the next action's input
5. **Preserve** completeness: the cleaned trajectory must still achieve the recovery from start to finish
6. **Verify** coherence: check that the sequence makes logical sense (you have item A before using it in craft)

Format each step as: action → observation (one per line).

# Output Format (strict JSON only):
{{
  "reasoning": "brief explanation of what was removed and why",
  "cleaned_trajectory": "action1 → observation1\\naction2 → observation2\\n..."
}}
"""

PARAMETER_GUIDELINE_REFINEMENT_PROMPT = """You are refining a skill's parameter binding guidelines based on accumulated execution failures.

# Skill
Name: {skill_name}
Parameters: {parameters}
Description: {description}

# Current Parameter Roles
{parameter_roles}

# Accumulated Anti-Patterns (raw failure logs)
{anti_patterns}

# Crafting Commands (reference for valid recipes)
{crafting_commands_sample}

# Instructions

The anti-patterns above are raw failure logs from repeated executions. Your job is to:

1. **Identify the root causes**: What general rules do these failures reveal? (e.g., "need_count must match recipe output count, not total quantity needed")
2. **Distill concise rules**: Replace N individual failure dumps with 1-3 general rules that prevent ALL of them
3. **Fix parameter_roles**: If any parameter role description is misleading (e.g., says "meet or exceed the task goal" when it should say "match the recipe output count"), correct it
4. **Keep useful specific anti-patterns**: If a failure reveals something unique (not covered by a general rule), keep it

# Output Format (strict JSON only):
{{
  "analysis": "What root cause(s) explain the majority of failures",
  "refined_parameter_roles": {{
    "param_name": "corrected role description"
  }},
  "refined_anti_patterns": [
    "concise general rule 1",
    "concise general rule 2"
  ],
  "refined_checklist": [
    "updated checklist item 1"
  ]
}}
"""

GRAFT_INDUCTION_PROMPT = """You are synthesizing a sub-graph fragment to be grafted onto a skill's failure point, based on a cleaned recovery trajectory.

# Original Skill Graph (Mermaid)
{original_mermaid}

# Referenced Skill Graphs
The recovery trajectory used the following other skills. Their full graph structures are provided below — you should **inline their logic** (adapt node IDs and variable bindings) rather than inventing abstract function calls.

{referenced_skill_graphs}

# Failure Point
The graph currently has an edge leading to {failure_node}. The diagnostic at failure was:
{failure_diagnostic}

# Cleaned Recovery Trajectory
{cleaned_trajectory}

# Pattern Type
{pattern_type}

# Available Scope Variables at Graft Point
{scope_variables}

# Available Predicate Functions (CLOSED list — do NOT invent others)
- `inventory(item: ItemName, count: Count)` → bool: agent holds ≥ count of item
- `goal(item: ItemName, count: Count)` → bool: task requires count of item
- `unavailable(item: ItemName)` → bool: item cannot be found/fetched
- `current_count_of_item(item: ItemName)` → Count: current inventory count
- `numerical_greater_equal(a: Count, b: Count)` → bool
- `numerical_less_than(a: Count, b: Count)` → bool
- `max(a: Count, b: Count)` → Count

# Crafting Commands
{crafting_commands}

# Instructions

## Critical: Generalization Principle
The trajectory above is ONE concrete example. Your graft must be a **general capability upgrade** to the skill, NOT a replay of this specific episode. Think: "What ability is this skill structurally missing?" rather than "How did the agent fix this particular case?"

- NEVER hardcode specific item names, counts, or recipe details from the episode.
- ALL values must come from scope variables (`{{CURRENT_INGREDIENT}}`, `{{COUNT_TO_GET}}`, `{{TARGET_ITEM}}`, etc.).
- The graft should work for ANY item/ingredient that could trigger the same failure mode.
- ONLY use predicate functions from the closed list above. Do NOT invent functions like `can_craft`, `determine_missing_ingredient`, etc.

## Compositional Inlining
When the recovery trajectory called another skill (e.g. `check_and_fetch_item`), do NOT create an abstract "call skill" node. Instead, **inline that skill's graph structure** directly into your graft:
- Copy the relevant nodes (DataOp, Check, PrimitiveAction) from the referenced skill graph
- Rename node IDs with a graft-local prefix to avoid collisions (e.g. `GRAFT_INVENTORY`, `GRAFT_GET`, `GRAFT_CHECK`)
- Rebind the referenced skill's input variables to the graft's scope variables (e.g. the referenced skill's `TARGET_ITEM_INPUT` maps to `{{CURRENT_INGREDIENT}}` in your graft scope)

## Strict Mermaid Syntax (CRITICAL — violations cause runtime crashes)
Every variable reference MUST use DOUBLE braces `{{{{VAR}}}}`. Single braces `{{VAR}}` will crash the Python template engine.

- In `local in:` clauses: `(local_name: Type = {{{{GLOBAL_VAR}}}})`
- In `writes GLOBAL:` clauses: `(VAR: Type:={{{{source_var}}}})` or `(VAR: Type = expr({{{{arg}}}}))`
- In action templates: `(action: 'get {{{{act_count}}}} {{{{act_item}}}}')`

### Correct examples (copy this syntax EXACTLY):
- DataOp: `GRAFT_FETCH["DataOp: <br>writes GLOBAL: (G_COUNT: Count = current_count_of_item({{{{g_item}}}})) <br>local in: (g_item: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::DataOp`
- Check: `GRAFT_CHECK["Check: <br> (inventory({{{{g_item}}}}, {{{{g_count}}}}) and {{{{g_count}}}} >= {{{{g_need}}}})<br>local in: (g_item: ItemName = {{{{CURRENT_INGREDIENT}}}}, g_count: Count = {{{{G_AVAILABLE}}}}, g_need: Count = {{{{TARGET_COUNT}}}})"]:::Check`
- PrimitiveAction: `GRAFT_GET["PrimitiveAction: <br>(action: 'get {{{{g_act_count}}}} {{{{g_act_item}}}}')<br>local in: (g_act_count: Count = {{{{G_MISSING}}}}, g_act_item: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::PrimitiveAction`

## Variable Isolation (CRITICAL — violations corrupt parent skill state)
The graft runs INSIDE the parent skill's variable scope. You MUST NOT overwrite parent variables that are still needed after the graft returns.

- **DO NOT** write to parent variables like TARGET_ITEM, NEED_COUNT, INGREDIENTS, INGREDIENTS_COUNT — the parent skill needs these intact when the graft returns via GRAFT_CONTINUE.
- Instead, use graft-scoped variable names with a `G_` prefix for any new variables the graft needs.
  - Example: instead of `writes GLOBAL: (TARGET_ITEM: ItemName:=...)`, use `writes GLOBAL: (G_FETCH_TARGET: ItemName:=...)`
- You MAY read parent variables (e.g., `{{{{CURRENT_INGREDIENT}}}}`, `{{{{COUNT_TO_GET}}}}`).
- You MAY write to variables the parent will re-compute anyway (e.g., `AVAILABLE_COUNT` if the parent re-fetches it in the next loop iteration).

## Edge Labels for PrimitiveAction
PrimitiveAction nodes can fail (e.g., `get X` returns "Could not find X"). Since PrimitiveAction is NOT a Check node, you CANNOT branch directly from it. Instead:
- After a PrimitiveAction, insert a **Check node** to test whether the action succeeded.
- Example: `GRAFT_GET --> GRAFT_RECHECK_COUNT` (DataOp to re-read inventory) → `GRAFT_CHECK_SUCCESS` (Check node that branches Yes/No).

## Sub-graph Requirements
Synthesize a mermaid sub-graph fragment that:
1. Starts with a node named `GRAFT_ENTRY` (this will be connected from the graft point source)
2. Uses ONLY double-brace `{{{{variable}}}}` references — NO single braces
3. Uses graft-scoped variable names (`G_` prefix) for any new globals to avoid overwriting parent state
4. Ends with either:
   - `GRAFT_CONTINUE` (success — feeds back to the main graph's continue point)
   - `GRAFT_FAIL` (failure — falls through to the original FAILURE_END)
5. Uses only these node types: DataOp, Check, PrimitiveAction (see syntax above)
6. After every PrimitiveAction, uses a Check node (not a bare edge) to decide success/failure

For **depth_composition** pattern: The item cannot be fetched directly (`get` failed) but CAN be crafted from sub-ingredients. The graft must inline the referenced craft skill's logic: check each sub-ingredient in inventory, fetch any missing ones (inline the fetch logic), then execute the craft action. If the skill's current parameters don't include recipe information (ingredients list, counts), propose adding them via `new_parameters`. The graft MUST contain a craft action node — a retry of `get` alone is NOT depth_composition.

For **multiplicity** pattern: After detecting insufficient count, attempt to fetch the missing amount (inline the referenced fetch logic), recheck, and either continue or fail.

Output ONLY the sub-graph nodes and edges (no flowchart TD header, no class definitions — these exist in the base graph).

# Output Format (strict JSON only):
{{
  "reasoning": "brief explanation of the induced sub-graph logic",
  "graft_mermaid": "the mermaid node definitions and edges as a string",
  "graft_point_src": "source node ID where graft connects (e.g. D_COMPUTE_MISSING)",
  "graft_point_dst": "original destination node being replaced (e.g. FAILURE_END)",
  "continue_target": "node ID where GRAFT_CONTINUE should connect (e.g. LOOP_FOR_INGREDIENTS)",
  "new_parameters": ["list of new parameter names if the skill signature needs extending, or empty list"]
}}
"""

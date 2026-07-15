

OBSERVATION_TO_PREDICATES_PROMPT = """You are an AI embodied agent. You convert each observation (`observation:` …) into ground facts for the predicates defined as following, using the literal semantics of each predicate. For every observation:

1. Read the text and determine which predicate instances are explicitly supported or refuted.
2. Output two blocks:
   • `add_facts:` list every predicate instance that is certainly true after reading the observation.
   • `remove_facts:` list predicate instances that the observation makes false (e.g., something was previously on a table but is now described elsewhere). If none, write `none`.

Use the canonical predicate names and argument order:
- `locate(receptacle_id)`: true when the observation states (explicitly or implicitly) that the agent is at/on/inside the receptacle. Accept cues like “You are at …” or “On the X you see …” (implies the agent is at X). If the agent leaves, emit a contradiction for the previous location.
- `reachable(receptacle_id)`: true when the observation says the agent can reach/go to the receptacle or when the environment lists nearby surfaces (e.g., “you see a drawer 1 nearby”). Treat co-listed surfaces as mutually reachable unless text says otherwise.
- `is_open(receptacle_id)`: true only if the observation states that the receptacle is open. False if it states the receptacle is closed/shut.
- `is_closed(receptacle_id)`: true if the observation states that the receptacle is closed/shut, false otherwise. 
- `contains(receptacle_id, item_id)`: true when the observation enumerates items on/in the receptacle (phrases like “On the table you see …” or “Inside the drawer there is …”). Replace the entire contents set with exactly the listed items. If the observation states an item is removed, emit a contradiction for the previous `contains`.
- `holding(item_id)`: true if the observation says “You pick up …,” “You are holding …,” or similar. False if the observation says you dropped/placed it elsewhere.
- `is_cleaned(item_id)`: true when the observation states that the item is cleaned, false othereises.
- `is_cooled(item_id)`: true when the observation states that the item is cooled, false othereises.
- `is_heated(item_id)`: true when the observation states that the item is heated, false othereises.
- `is_turned_on(item_id)`: true when the observation states that an item (lamp) is turned on, false othereises.


Formatting rules:
- Convert names to snake_case with numeric suffixes (e.g., `sofa 1` → `sofa_1`, `cellphone 3` → `cellphone_3`).
- Use exact predicate syntax, e.g., `locate(sofa_1)`.
- If a fact persists and the observation doesn't contradict it, you do not need to repeat it; list only facts newly entailed or contradicted by THIS observation.
- If nothing is entailed/contradicted, write `add_facts: []` and/or `remove_facts: []`.

Example:

Input:
facts: []
action: 
observation: You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1. Your task is to: put two cellphone in sofa.
Output: 
```json
{{
  "add_facts": [
    "locate(middle_of_room)",
    "reachable(coffeetable_1)",
    "reachable(diningtable_1)",
    "reachable(drawer_4)",
    "reachable(drawer_3)",
    "reachable(drawer_2)",
    "reachable(drawer_1)",
    "reachable(dresser_1)",
    "reachable(garbagecan_1)",
    "reachable(sidetable_2)",
    "reachable(sidetable_1)",
    "reachable(sofa_1)",
  ],
  "remove_facts": []
}}
```

Input:
facts: [
    "locate(middle_of_room)",
    "reachable(coffeetable_1)",
    "reachable(diningtable_1)",
    "reachable(drawer_4)",
    "reachable(drawer_3)",
    "reachable(drawer_2)",
    "reachable(drawer_1)",
    "reachable(dresser_1)",
    "reachable(garbagecan_1)",
    "reachable(sidetable_2)",
    "reachable(sidetable_1)",
    "reachable(sofa_1)",
  ]
action: goto dresser 1
observation: On the dresser 1, you see a book 2 and a mug 1.

Output:
```json
{{
  "add_facts": [
    "locate(dresser_1)",
    "contains(dresser_1, book_2)",
    "contains(dresser_1, mug_1)"
  ],
  "remove_facts": ["locate(middle_of_room)"]
}}
```

Second input:
facts: [
    "reachable(coffeetable_1)",
    "reachable(diningtable_1)",
    "reachable(drawer_4)",
    "reachable(drawer_3)",
    "reachable(drawer_2)",
    "reachable(drawer_1)",
    "reachable(dresser_1)",
    "reachable(garbagecan_1)",
    "reachable(sidetable_2)",
    "reachable(sidetable_1)",
    "reachable(sofa_1)",
    "locate(dresser_1)",
    "contains(dresser_1, book_2)",
    "contains(dresser_1, mug_1)"
  ]
action: take mug 1 from dresser 1
observation: You pick up the mug 1 from the dresser 1.

Output:
```json
{{
  "add_facts": [
    "holding(mug_1)"
  ],
  "remove_facts": [
    "contains(dresser_1, mug_1)"
  ]
}}
```
Follow exactly this schema for each new observation.

facts: {facts}
action: {new_action}
observation: {new_observation}
"""



MEMORY_CONSTRUCTION_INTENT_PROMPT = """You are in a household environment solving a task.
{interaction_history}

Current action: {new_action}
Write one sentence stating the intent: the goal-conditioned purpose of the current action, explaining why it is appropriate. 
Output requirements (strict): 
- Single sentence (use "because" if helpful).
- Do not merely restate the action; express the purpose behind the command text.
- Be clear and concise. 
Intent: """

MEMORY_CONSTRUCTION_INTENT_WITH_PREDICATE_PROMPT = """You are in a household environment solving a task.
{interaction_history}

Critical world state facts: {important_predicates_for_action}
Current action: {new_action}
Write one sentence stating the intent: the goal-conditioned, fact-grounded purpose of the current action, explaining why it is appropriate. 
Output requirements (strict): 
- Single sentence (use "because" if helpful).
- Do not merely restate the action; express the purpose behind the command text.
- Ground the rationale in the Critical facts and the task goal.
- Wrap every (pickable) item identifier or type in $...$ (e.g., $apple 3$, $mug$, $book 2$, $knife 1$).
- Wrap every (openable) receptacle/location identifier or type in &...& (e.g., &sofa 1&, &sofa&, &diningtable 4&, &diningtable&, &drawer 4&, &drawer 3&, &dresser 1&, &garbagecan 1&, &sidetable&, &sidetable 1&).
- Be clear and concise. 
Intent: """

MEMORY_CONSTRUCTION_PREDICATE_ANALYSIS_PROMPT = """You are in a household environment solving a task.
{interaction_history}

Current world state (facts): {facts_current}
Current action: {new_action}

Select the critical set of facts from the Current world state that directly determine the specific parameters of the Current action (e.g., which item and which receptacle).

Output requirements (strict):
- Return ONLY a Python list of strings (no prose, no keys, no code fences).
- Copy each selected fact verbatim from the Current world state.
- Do not invent or rephrase facts; do not include duplicates.
- If none apply, return an empty list: []
"""

RAW_ACTIONS = """
go to: <ReceptacleName>
open: <ReceptacleName>
take: <ItemName> from <ReceptacleName>
move: <ItemName> to <ReceptacleName>
clean: <ItemName> with <ReceptacleName>
heat: <ItemName> with <ReceptacleName>
cool: <ItemName> with <ReceptacleName>
use: <ItemName>
"""

REACT_WITH_STATE_GROUNDING_PROMPT = """
Interact with a household to solve a task.

# Here are two examples:
{examples}

# Here is the task.
{task_message}

# Available actions: 
think: <thought about the overall task goal, sub-goals decomposition, current state, and the next sub-goal>
go to: <ReceptacleName>
open: <ReceptacleName>
take: <ItemName> from <ReceptacleName>
move: <ItemName> to <ReceptacleName>
clean: <ItemName> with <ReceptacleName>
heat: <ItemName> with <ReceptacleName>
cool: <ItemName> with <ReceptacleName>
use: <ItemName>


# Available skills: 
{skill_info}

# Skill/Action Selection Guideline:
{task_guideline}

{action_check_error}

- Do not repeat the same action/skill if nothing happened or failed. 
- **Prioritize using the provided skills**.
- For the selected skills, follow the **Guidelines for your Parameter Bindings** to ensure the skill is called correctly to achieve the intended effect.
- If none of the skills are applicable, **then** select the action from the available actions.

Please enter your action in a json format.
```json {{
    "reasoning": <thought on action selection based on overall goals, current states, and sub-goals>
    "skill": <The selected skill. Return none if none of the skills are applicable.>
    "start_condition_evidence": <thought on the start conditions of the selected skill>
    "pre_span_summary": <summary in natural language the key events that occurred **before** the current skill/action begins that are relevant for understanding the agent's situation. The summary must be self-contained.>
    "intent": <the intent of the current skill/action, must stay self-contained>
    "parameter_bindings": {{
      <skill-param-1> : <value-1>, 
      ... 
    }}
    "action": <The selected action if none of the skills are applicable. If the skill is applicable, this field should be none.>
}}
"""

SKILL_ACTION_ROUTER_PROMPT = """
You are a skill/action router for an embodied household agent. Decide which skill or raw action should be executed next.

# Task and history
{task_message}

# Current world facts (set semantics)
{facts_current}

# Task guideline
{task_guideline}

# Learned routing rules (from past experience)
{routing_rules}

# Repeat-call diagnostic for planner anti-stuck handling.
{repeat_warning}

Routing steps:
1) Pick the active or next sub-task from the ordered_subtasks in the task guideline using the history and facts.
2) For that sub-task, compare triggers/pre/post contexts in skill_guidelines and raw_action_guidelines against the current facts/intents.
3) Propose up to 3 candidates (skills preferred) with evidence and expected outcome.
4) Recommend one candidate; if none apply, return type "none" and list missing info.

Strict JSON ONLY (no prose, no code fences):
{{
  "active_subtask": "<sub-task name from ordered_subtasks>",
  "subtask_reason": "<why this sub-task is active/next>",
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
Interact with a household to execute the routed skill/action.

# Routing recommendation (follow unless impossible)
{routing_recommendation}

# Here are two examples:
{examples}

# Here is the task.
{task_message}

# Current world facts:
{facts_current}

# Available actions: 
think: <thought about the overall task goal, sub-goals decomposition, current state, and the next sub-goal>
go to: <ReceptacleName>
open: <ReceptacleName>
take: <ItemName> from <ReceptacleName>
move: <ItemName> to <ReceptacleName>
clean: <ItemName> with <ReceptacleName>
heat: <ItemName> with <ReceptacleName>
cool: <ItemName> with <ReceptacleName>
use: <ItemName>


# Available skills:
{skill_info}

# Skill/Action Selection Guideline:
{task_guideline}

# Repeat-call diagnostic for anti-stuck handling.
{repeat_warning}

{action_check_error}

- Honor the routing recommendation. If recommendation.type is "skill", set `skill` to that name and `action` to null. If recommendation.type is "action", set `action` to the raw action string and `skill` to null. Only deviate if impossible, and explain in reasoning.
- Do not repeat the same action/skill if nothing happened or failed. 
- **Prioritize using the provided skills**.
- For the selected skills, follow the **Guidelines for your Parameter Bindings** to ensure the skill is called correctly to achieve the intended effect.
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

SKILL_PARAMETER_ANALYSIS_PROMPT = """
You are a skill-parameter analyst for an embodied LLM agent.

Goal: 
 - Given a skill specification and several successful past usages, derive durable rules that maximize the probability of success in future calls. 
 - Ground on the abstract world state and the task goal. Do not assume any specific ItemName/ItemType/ReceptacleName/ReceptacleType, from the previous environment context. 
 - Focus on generalizable guidance inferred from the data.

# Inputs
- skill_schema: a JSON object with fields `skill_name`, `description`, `parameters`, `start_conditions`, `success_conditions`.
- success_records: a list of past successful calls for this skill. Each record may include `pre_skill_interaction` (with observations and actions), `skill_call_rationale`, and `parameter_bindings` actually used.

# Output (strict JSON only; no prose, no code fences)
{{
  "parameter_roles": {{"<param_name>": "<role summary and hard constraints extracted from the schema>", ...}},
  "selection_guidelines": [
    "<data-driven rule grounded in start/success conditions and priors>",
    "<ordering rules for list-type parameters (e.g., highest prior first, then surfaces, then containers)>",
    "<deduplication, tie-breaks (e.g., lowest numeric suffix), and truncation policy>",
    "<fallback rules when evidence is weak (e.g., default to most frequent receptacle types across items)>"
  ],
  "anti_patterns": [
    "<what to avoid when binding parameters (e.g., violating start conditions, including duplicates, using unseen/nonexistent names)>"
  ],
  "checklist_before_call": [
    "<checks derived from start_conditions and success_conditions to validate a proposed binding before calling the skill>"
  ]
  "fail_reason": <thought on the reason for the skill call failure. If the skill failed, how should the parameter settings be improved?>
}}

# Rules
- Ground your analysis in `success_records`: extract item–location co-occurrences, ordering patterns, and counts; prefer generalizable rules that appear more often and earlier.
- Respect `start_conditions` and `success_conditions`; do not propose guidance that would violate them.
- For list parameters (e.g., `List_*`), produce ordered lists encoding search priority; remove duplicates; document tie-breaks and truncation strategy.
- Use exact surface names as they appear in observations (e.g., "fridge 1", "diningtable 2") when giving examples; do not convert to snake_case.
- Be concise and avoid narrative; return only the JSON object described above.

skill_schema: 
{skill_schema}
success_records: 
{success_records}
"""

SKILL_PRECONDITION_SYNTH_PROMPT = """
You are a Concepts-DSL engineer that derives symbolic pre-condition checks for high-level skills.

Inputs:
- `skill_schema`: JSON with `name`, `description`, `parameters`, `start_conditions`, `success_conditions`, and `steps`.
- `success_records`: A markdown-style list (`## Record i`) of successful calls. Each record may include:
  * Pre-skill interaction traces (observations/actions)
  * Skill-call rationale
  * `Parameter Bindings`
  * `Current World Facts` (a python list of predicates true at call time)

Goal:
- Infer a predicate expression (Concepts DSL syntax) that must be true before invoking the skill for a *new* parameter instantiation.
- Expression should generalize across records and align with the skill description/start conditions.

Representation rules:
- Refer to skill parameters via `{{paramName}}` exactly as they appear in `skill_schema.parameters`.
- Convert object/receptacle names to `snake_case` with numeric suffixes (e.g., `fridge 1` → `fridge_1`).
- List parameters (e.g., `List_ReceptacleName`) should be treated as python lists.
- Every predicate argument or free variable must map directly to a declared skill parameter (or be bound by a quantifier that iterates over elements originating from that parameter); never invent new identifiers.
- Prefer conjunctions of minimal necessary clauses; avoid redundant checks already implied by others.
- Express the final `pre_condition_expression` strictly as a conjunction: output it as a JSON list of clause strings, and assume the overall condition is the logical `and` over that list.
- Keep the clause order identical between `pre_condition_expression` and `clause_breakdown`.
- When unsure, state assumptions explicitly.
- Type violations are strictly forbidden: each predicate argument must respect the parameter's declared type (e.g., never feed `List_ReceptacleName` into `exists_item_in_list`); any clause with mismatched types is not allowed.
- Use only the allowed domain predicates:
  `locate({{ReceptacleName}})`, `reachable({{ReceptacleName}})`, `contains({{ReceptacleName}}, {{ItemName}})`, `holding({{ItemName}})`,
  `is_open({{ReceptacleName}})`, `is_closed({{ReceptacleName}})`, `is_cleaned({{ItemName}})`, `is_heated({{ItemName}})`, `is_cooled({{ItemName}})`,
  `is_turned_on({{ItemName}})`, `is_item_of_type({{ItemName}}, {{ItemType}})`, `is_receptacle_of_type({{ReceptacleName}}, {{ItemType}})`,
  - could use boolean operators: not, and, or.
  - could use quantifiers to express constraints over all items/receptacles in the world:
    - `exists("Item", lambda x: holding(x) or contains({{ReceptacleName}}, x))`
    - `forall("Receptacle", lambda x: ...)`
  - for the specific list, membership tests can use only the following predicates: exists_in_list and forall_in_list, e.g., 
    - exists_item_in_list({{List_ItemName}}, lambda x: holding(x)) 
    - forall_item_in_list({{List_ItemName}}, lambda x: is_item_of_type(x, {{ItemType}})) 
    - exists_receptacle_in_list({{List_ReceptacleName}}, lambda x: is_receptacle_of_type(x, {{ReceptacleType}})) 
    - forall_receptacle_in_list({{List_ReceptacleName}}, lambda x: reachable(x)) 


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

TASK_LEVEL_SUMMARY_PROMPT = """
You are compiling task-level execution guidelines for an embodied household agent.

Task type: {task_type}

Inputs:
- `Skill Info`: structured skill schemas and successful call snippets for this task type. Use them to infer what each skill does, when it is invoked, and how its parameters are bound.
- `Raw Trajectory`: raw action sub-trajectories where skills were not used. Use them to understand fallback or bridging behaviors.

Goal (per task type):
- Produce an ordered, high-level sequence of executable sub-tasks typically used to solve the task.
- For each skill: cover all observed scenarios in which it is invoked (phases and triggers), the typical pre-skill context (key world facts, recent intents, initiation rationale), and the typical post-skill outcomes (facts/follow-up intents).
- Summarize all observed "raw action" patterns (spans without skill calls) from Raw Trajectory with the same context/outcome framing.

Output format (STRICT JSON, no prose, no code fences):
{{
  "task_type": "{task_type}",
  "ordered_subtasks": ["<sub-task description in likely execution order; base on frequency/earliest appearance; expressed in natural language, precise and easy to understand; must not be empty>"],
  "skill_guidelines": [
    {{
      "skill": "<skill name>",
      "applicable_scenarios": [
        {{
          "phase": "<sub-task phase; must match one entry in ordered_subtasks; add the reason>",
          "when_to_invoke": ["<all observed triggers/start-condition cues; each under 20 words>"],
          "typical_pre_context": {{
            "key_world_facts": ["<concise fact patterns; under 20 words>"],
            "recent_intents": ["<what the agent was trying to do right before; under 20 words>"],
            "initiation_rationale": "<self-contained reason to start the skill now; under 25 words>"
          }},
          "typical_post_outcome": {{
            "expected_state_change": ["<state changes or confirmations; under 20 words>"],
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
          "phase": "<sub-task phase; must match one entry in ordered_subtasks; add the reason>",
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
- Prefer ordering by frequency and earliest appearance in examples; ordered_subtasks must always be provided (best-effort).
- Keep strings concise; follow the length cues above. Return valid JSON only; fill missing sections with empty arrays instead of prose.

# Skill Info:
{skill_info}


# Raw actions: 
go to: <ReceptacleName>
open: <ReceptacleName>
take: <ItemName> from <ReceptacleName>
move: <ItemName> to <ReceptacleName>
clean: <ItemName> with <ReceptacleName>
heat: <ItemName> with <ReceptacleName>
cool: <ItemName> with <ReceptacleName>
use: <ItemName>

# Raw Trajectory:
{raw_traj}
"""


SUBGOAL_PROMPT = """
You are an AI planner for an embodied environment.
Summarize the sub-goals from the provided interaction trajectories. These sub-goals should capture the overall task goal and the intermediate steps to achieve it.

# Data
Here are some interaction trajectories:
{examples}

# Detailed Instructions for Sub-Goal Generation:
- **Complexity**: Sub-goals should have mediocre complexity:
  - Must include at least two primitive actions; sub-goals implementable by a single action OR one condition check followed by one primitive action are NOT allowed.
  - Steps list length must be between 2 and 5 inclusive (reject 1-step designs).
- **Generality**: Sub-goals must be applicable to other similar tasks, not just the provided examples.
- **Arguments**: Use common variable types (e.g., strings, lists). Avoid complex inputs like another function.

- **Parameters and Types**: For each sub-goal, explicitly list the input parameters and their types.
  - Allowed parameter types (choose only from): ReceptacleName, ItemName, ReceptacleType, ItemType, List_T (e.g. List_ItemName, List_ReceptacleType).
  - Typed placeholder examples (primitive actions for illustration):
    - go to {{gotoRec: ReceptacleName}}
    - open {{openRec: ReceptacleName}}
    - take {{takeItem: ItemName}} from {{fromRec: ReceptacleName}}
    - cool {{coolItem: ItemName}} with {{recep: ReceptacleName}}
    - heat {{heatItem: ItemName}} with {{recep: ReceptacleName}}
    - clean {{cleanItem: ItemName}} with {{recep: ReceptacleName}}
    - move {{moveItem: ItemName}} to {{toRec: ReceptacleName}}
    - use {{desklamp: ItemName}}
  - Typed placeholder examples (overall goal for illustration):
    - 'find an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put an item of type {{itemType: ItemType}} on a receptacle of type {{RecType: ReceptacleType}}'
    - 'clean an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a clean item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a hot item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'heat an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a cool item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'cool an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'examine an item of type {{itemType: ItemType}} with the desklamp',
    - 'look at an item of type {{itemType: ItemType}} under the desklamp', 
    - 'put two items of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
  - In the JSON output, provide parameters as strings in the form "param_name: param_type".

- **Start Conditions (Preconditions)**: For each sub-goal, specify clear, checkable preconditions under which the sub-goal can begin. If these are not satisfied, this sub-goal MUST NOT be selected.
  - Express conditions succinctly with parameter references (e.g., "the agent does not locate at {{gotoRec}}, and {{gotoRec}} is reachable", "the agent already locates at {{openRec}} and {{openRec}} is current closed", "the agent already holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open").
  - Prefer condition forms that can be grounded in observations or known state (e.g., reachable, is open/is closed, contains, holding, exists).
  - Keep preconditions minimal and necessary; avoid hidden assumptions. 

- **Success Conditions (Postconditions)**: For each sub-goal, specify the observable state that must hold after successful completion. If these are not satisfied, the sub-goal is incomplete/failed.
  - Use the same expression style as start_conditions with parameter references (e.g., "the agent locates at {{gotoRec}}", "{{openRec}} is current open", "{{toRec}} contains {{moveItem}}", "the agent holds a {{moveItem}}").
  - Keep postconditions minimal, necessary, and directly verifiable from observations or world facts.

- **Naming**: Name each sub-goal to summarize a multi-step intent, not a single primitive action.
  - Be concise and descriptive; use lowercase with underscores.
  - Do NOT reuse primitive action names or their near variants: go_to, open, take, put, move, clean, heat, cool, use.
  - Prefer names like: locate_and_open_receptacle, retrieve_item_from_container, transfer_item_to_surface, clean_item_and_place, heat_item_then_place.

- **Steps Analysis**: For each sub-goal, describe the sequence of actions required to achieve it.
  - Use concise, descriptive parameter names aligned with the **Parameters and Types** of the current sub-goal.
  - Ground each step in actual primitive actions: go to, open, take, put, move, clean, heat, cool, use.
  - Express branches and loops using production rules in the form: "If <condition>, then <primitive action>". You may use "Else if" and a final default action when appropriate.
  - Represent loops compactly as: "For each <X> in <LIST>: <primitive action or rule>" (keep total steps ≤ 5).
  - Express condition in natural language with the parameters declared in "parameters".
  - Minimum steps policy: ensure the "steps" array contains 2-5 items and includes at least two non-redundant primitive actions across all rules; do not output single-step sub-goals.
  - Examples:
    - If the agent does not already locate at {{gotoRec}}, and {{gotoRec}} is reachable, then go to {{gotoRec}}
    - If the agent locates at {{openRec}} and {{openRec}} is current closed, then open {{openRec}}
    - If the agent holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open, then move {{moveItem}} to {{toRec}}

- **Applicable Tasks**: For each sub-goal, provide tasks where it applies.
  - Copy task names verbatim from the source data (no paraphrasing, no reformatting, preserve original casing and punctuation).


# Output Format:
```json {{
[
    {{
    "name": <sub_goal_1_name, concise multi-step summary; use '_' instead of ' '; MUST NOT equal or directly mirror a primitive action name such as 'go_to', 'open', 'take', 'put', 'move', 'clean', 'heat', 'cool', 'use'>,
    "parameters": [
        "param_name_1: param_type_1",
        "param_name_2: param_type_2",
        ...
    ],
    "start_conditions": [
        "<condition 1>",
        "<condition 2>",
        ...
    ],
    "success_conditions": [
        "<condition 1>",
        "<condition 2>",
        ...
    ],
    "description": "<sub-goal 1 description>"
    "steps": [
        "<step 1>",
        "<step 2>",
        ...
    ],
    "applicable_tasks": [
        "<task 1: copy verbatim>",
        "<task 2: copy verbatim>",
        ...
    ],
    }}, 
    {{
    "name": <sub_goal_2_name, concise multi-step summary; use '_' instead of ' '; MUST NOT equal or directly mirror a primitive action name>,
    "parameters": [
        "param_name_1: param_type_1",
        ...
    ],
    "start_conditions": [
        "<condition 1>",
        ...
    ],
    "success_conditions": [
        "<condition 1>",
        ...
    ],
    "description": "<sub-goal 2 description>"
    "steps": [
        "<step 1>",
        "<step 2>",
        ...
    ],
    "applicable_tasks": [
        "<task 1: copy verbatim>",
        "<task 2: copy verbatim>",
        ...
    ],
    }}, ...
]
}}
"""



SUB_GOAL_WORKFLOW_PROMPT = """
You are a workflow synthesizer. 

Your task: 
- Read the sub-goal and the interaction trajectories, then instantiate the scaffold above to solve ONLY that sub-goal.
- For each sub-goal, output ONE Mermaid flowchart style (node-bound inputs, edge-only control). No prose.

# Sub-goal to implement
{sub_goal}

# Data trajectories
Here are some interaction trajectories:
{examples}

Overall Guideline (must follow exactly):
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_rec vs TARGET_RECEPTACLES).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Edge mid-labels can be comma-separated. 
- For any edge that enters a LoopControl node: use `Start_Loop` to indicate entering/resetting from outside (restart enumeration), and `Continue_Loop` to indicate continuing the current loop. Internal back-edges should carry `Continue_Loop`; outer re-entries should carry `Start_Loop`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node’s `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = {{CURRENT_RECEPTACLE}})`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Selection helpers `select_one` and `select_all` can be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.
- Node IDs define unique instances. When the same type of node is needed multiple times with different successors, duplicate the node by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2; C_IS_CLOSED_1, C_IS_CLOSED_2; LOOP_FOR_LOCATIONS_1, LOOP_FOR_LOCATIONS_2; D_SELECT_ONE_1, D_SELECT_ONE_2).

Domain predicates (allowed inside DataOp/Check nodes):
- locate({{receptacle_name}}); reachable({{receptacle_name}})
- contains({{receptacle_name}}, {{item_name}}); holding({{item_name}})
- is_open({{receptacle_name}}); is_closed({{receptacle_name}})
- is_cleaned({{item_name}}); is_heated({{item_name}}); is_cooled({{item_name}}); is_turned_on({{item_name}})
- type checks: is_item_of_type({{item_name}}, {{target_type}}); is_receptacle_of_type({{receptacle_name}}, {{target_type}})
- Quantifiers, e.g. `exists("Item", lambda x: contains({{receptacle_name}}, x) and is_item_of_type(x, {{target_type}}))`

Object Selection (ONLY allowed inside DataOp nodes):
- select_one(\'Item\' or \'Receptacle\', a FOL expression with domain predicates conditioned on `x`)
  * select the first Item/Receptacle that satisfies the expression, return None if no such object exists
  * e.g., select_one(\'Item\', is_item_of_type(x, {{target_type}})))
- select_all(\'Item\' or \'Receptacle\', a FOL expression with domain predicates conditioned on `x`)
  * select all Items/Receptacles that satisfies the expression, return an empty list if no such object exists
  * e.g., select_all(\'Item\', is_item_of_type(x, {{target_type}})))


Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):

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
    LEGEND["Type Legend:<br>ReceptacleName := str<br>ItemName := str<br>ObjectType := str<br>ReceptacleType := str<br>Bool := bool<br>List_T := sequence of T<br>Optional_T := T or None"]:::Spec
    
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
    %% Guideline: Here are more allowed PrimitiveActions Nodes. 
    %% Guideline: Select the PrimitiveActions Nodes that workflow needs. Remove any nodes that are not needed. 
    %% Note that any revision on the setting of the PrimitiveActions Nodes is STRICTLY FORBIDDEN. Directly copy which node the workflow needs. 
    %% Guideline: When needed, instantiate multiple PrimitiveAction nodes by assigning unique IDs (e.g., A_OPEN_1, A_OPEN_2); each node id is a separate instance that can connect to different successors.
    A_GOTO["PrimitiveAction: <br>(action: 'go to {{goto_rec}}')<br>local in: (goto_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_OPEN["PrimitiveAction: <br>(action: 'open {{open_rec}}')<br>local in: (open_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_TAKE["PrimitiveAction: <br>(action: 'take {{take_item}} from {{from_rec}}')<br>local in: (take_item: ItemName = {{ITEM_TO_TAKE}}, from_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_MOVE["PrimitiveAction: <br>(action: 'move {{move_item}} to {{to_rec}}')<br>local in: (move_item: ItemName = {{ITEM_TO_MOVE}}, to_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_CLEAN["PrimitiveAction: <br>(action: 'clean {{clean_item}} with {{clean_rec}}')<br>local in: (clean_item: ItemName = {{ITEM_TO_CLEAN}}, clean_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_HEAT["PrimitiveAction: <br>(action: 'heat {{heat_item}} with {{heat_rec}}')<br>local in: (heat_item: ItemName = {{ITEM_TO_HEAT}}, heat_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_COOL["PrimitiveAction: <br>(action: 'cool {{cool_item}} with {{cool_rec}}')<br>local in: (cool_item: ItemName = {{ITEM_TO_COOL}}, cool_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_USE["PrimitiveAction: <br>(action: 'use {{use_item}}')<br>local in: (use_item: ItemName = {{DESKLAMP_TO_USE}})<br>out: (executed: Bool)"]:::PrimitiveAction
  

    %% ===================== LoopControl (use only if needed) =====================
    %% Guideline: Only foreach loops via LoopControl with edges 'body' and 'done'.
    %% Guideline: LoopControl nodes only operate on the loop variable (e.g., LOOP_FOR_LOCATIONS["LoopControl: <br>For receptacle_i in {{loop_receptacles}}<br>writes GLOBAL: (CURRENT_RECEPTACLE: ReceptacleName:=receptacle_i)<br>local in: (loop_receptacles: List_ReceptacleName = {{RECEPTACLE_CANDIDATES}})"]:::LoopControl, assign the global `CURRENT_RECEPTACLE` with local `receptacle_i` from the local `loop_receptacles` if global variable `RECEPTACLE_CANDIDATES` could be provided from the previous nodes).
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple LoopControl nodes with distinct IDs (e.g., LOOP_FOR_1, LOOP_FOR_2) to represent separate loop sites.
{LOOP_FOR_NODE}
    %% Guideline: Include a new LoopControl node if you must use a loop.


    %% ===================== Checks =====================
    %% Guideline: Checks branch on Yes/No only; inputs resolved from GLOBALS. 
    %% Guideline: Modify the check condition to improve workflow if needed.
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: Check nodes cannot contain any GLOBAL writes. 
    %% Guideline: When needed, duplicate Check nodes with unique IDs (e.g., C_IS_CLOSED_1, C_IS_CLOSED_2) to use the same predicate at different decision points.
{CHECK_NODE}
    %% Design new Check nodes with the domain predicates if needed
    %% Here are some examples. Note there just examples, not fixed inputs, adjust the check condition and parameters binding to fit the workflow.
    C_IS_CLOSED["Check: <br>is_closed({{rec_name_to_check}})<br>local in: (rec_name_to_check: ReceptacleName = {{CURRENT_RECEPTACLE}})"]:::Check
    C_HAS_TYPE["Check: <br>exists(Item, lambda x: contains({{rec_name_to_check}}, x) and is_item_of_type(x, {{item_type_to_check}}))<br>local in: (rec_name_to_check: ReceptacleName = {{CURRENT_RECEPTACLE}}, item_type_to_check: ItemType = {{TARGET_ITEM_TYPE}})"]:::Check

    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node below. 
{D_INIT_NODE}
    %% Guideline: Inside every node, operate only on variables declared in that node's `local in`; first bind any needed GLOBAL to a local there.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow; treat the shown globals as examples, not fixed inputs.
    %% Guideline: When needed, create multiple DataOp nodes by assigning unique IDs (e.g., D_SELECT_ONE_1, D_SELECT_ONE_2) so each instance can feed different downstream consumers.
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    %% This is an example: D_SELECT_ONE_ITEM_WITH_TYPE["DataOp: <br>writes GLOBAL: (TARGET_ITEM: ItemName = select_one(\'Item\', is_item_of_type(x, {{select_item_type}}))) <br>local in: (select_item_type: ItemType = {{TARGET_ITEM_TYPE}})"]:::DataOp
    %% This is an example: D_SELECT_ALL_RECEPTACLE_WITH_TYPE["DataOp: <br>writes GLOBAL: (SELECTED_RECEPTACLES: List_ReceptacleName = select_all(\'Receptacle\', is_receptacle_of_type(x, {{select_rec_type}}))) <br>local in: (select_rec_type: ReceptacleTypeName = {{TARGET_REC_TYPE}})"]:::DataOp
    %% Guideline: Modify the DataOp nodes to improve workflow if needed.
    %% Guideline: Revise the GLOBAL parameter assignment in the `local in` so each local reads from a global already produced in the workflow.
{DATA_OP_NODE}
    %% Design new DataOp nodes with the selection helper and domain predicates if needed
    
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

{CONTROL_FLOW_EDGE}

    
{HINTS}


# Output Format
- Output ONLY the Mermaid code block(s). No explanations or extra text. 
- Output `No Solution` whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow under the given constraints.
"""

# - Return a single JSON object with the following fields:
#   - `"major_revision"`: boolean indicating whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow; set to `true` only when the workflow cannot satisfy all constraints without revising the sub-goal intent or parameters.
#   - `"major_revision_reason"`: concise explanation describing what must change (e.g., adjust sub-goal intent, add/remove parameters) when `"major_revision"` is `true`; use an empty string when the flag is `false`.
#   - `"mermaid_code"`: the revised workflow as a single Mermaid string that compiles (empty string if `"major_revision"` is `true` and no valid workflow can be produced. Keep every %% comment instruction; never remove or rewrite these annotations.).
#   - `"change_summary"`: array of 3–5 bullet strings explaining key modifications and rationale.
#   - `"key_conditions"`: array listing the critical check predicates or lambda expressions referenced in the workflow.
# - Do not wrap the JSON in backticks or add extra commentary.



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
- `ItemType`: use the item category or concrete item label exactly as described in the trajectory (e.g., `apple`, `egg`).
- `ItemName`: use the precise item instance name from the trajectory (e.g., `apple 3`, `mug 1`).
- `ReceptacleType`: use the receptacle category label (e.g., `fridge`, `shelf`, `garbagecan`).
- `ReceptacleName`: use the surface receptacle name from the trajectory (e.g., `fridge 1`, `shelf 2`).
- `List_<T>` (e.g., `List_ItemName`, `List_ReceptacleType`): output an ordered list whose elements each satisfy the guidance for the element type `T`; preserve the order induced by the subgoal execution.

Task
-----
1. Restate the subgoal intent in your own words.
2. Scan the trajectory in order and identify the **minimal contiguous** span of steps that corresponds to executing this subgoal. Include `think` steps only if they contribute to deciding or executing the search. The span must satisfy all start conditions before it begins and all success conditions by its end.
3. For every step inside that span, note why it is relevant (e.g., “checks receptacle”, “opens closed receptacle”, “retrieves the item”).
4. Using that span, instantiate every parameter listed under `parameters` in the subgoal specification. Respect each parameter's declared type when formatting its value, following the type reference above.
5. Cite one or more steps that demonstrate each start condition held before the span and each success condition held at the end.
6. Summarize in natural language the key events that occurred **before** the selected span begins (i.e., steps with index before `start_index`) that are relevant for understanding the agent's situation. Avoid assumptions not grounded in the trajectory. The summary must be self-contained, and avoid citing supporting indexes. 
7. Explain, referencing the overall task and the pre-span context, why the agent initiates this subgoal at that specific moment (the intent behind starting the subgoal when it did). Ground the explanation in explicit evidence from the trajectory. The explanation must be self-contained, and avoid citing supporting indexes.

Output
-------
Return **only** valid JSON with this structure and double-quoted keys/strings:

{{
  "subgoal_name": string,
  "subtrajectory_span": {{"start_index": int, "end_index": int}},
  "subtrajectory_steps": [
    {{"index": int, "observation": string, "action": string, "why_relevant": string}}
  ],
  "parameter_bindings": {{"<parameter_name>": <value>, ...}},
  "start_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "success_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "pre_span_summary": string,
  "subgoal_initiation_intent": string,
  "notes": string
}}

Keep `why_relevant`, `explanation`, and `notes` concise; use "none" if no additional notes. 
`pre_span_summary` and `subgoal_initiation_intent` must stay self-contained, and avoid assumptions not grounded in the trajectory.
"""

TRAJ_SEGMENTATION_REVISION_PROMPT = """You are an expert analyst that maps sub-goal schemas onto embodied-agent trajectories. You can leverage previously verified matches to stay consistent with how the subgoal has been satisfied before, while still grounding all claims in the new trajectory.
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
- `ItemType`: use the item category or concrete item label exactly as described in the trajectory (e.g., `apple`, `egg`).
- `ItemName`: use the precise item instance name from the trajectory (e.g., `apple 3`, `mug 1`).
- `ReceptacleType`: use the receptacle category label (e.g., `fridge`, `shelf`, `garbagecan`).
- `ReceptacleName`: use the surface receptacle name from the trajectory (e.g., `fridge 1`, `shelf 2`).
- `List_<T>` (e.g., `List_ItemName`, `List_ReceptacleType`): output an ordered list whose elements each satisfy the guidance for the element type `T`; preserve the order induced by the subgoal execution.

Task
-----
1. Review the matched exemplars to extract representative intent wording, parameter semantics, and evidence patterns that should remain consistent when possible.
2. Restate the subgoal intent in your own words, adapting terminology so it aligns with both the exemplars and the current trajectory.
3. Scan the trajectory in order and identify the **minimal contiguous** span of steps that corresponds to executing this subgoal. Include `think` steps only if they contribute to deciding or executing the search. The span must satisfy all start conditions before it begins and all success conditions by its end.
4. For every step inside that span, note why it is relevant (e.g., “checks receptacle”, “opens closed receptacle”, “retrieves the item”).
5. Using that span, instantiate every parameter listed under `parameters` in the subgoal specification. Respect each parameter's declared type when formatting its value, following the type reference above. When multiple options are possible, prefer values consistent with the exemplars as long as they are supported by the current span.
6. Cite one or more steps that demonstrate each start condition held before the span and each success condition held at the end. Mirror the level of specificity used in the exemplars when it improves clarity, but keep the explanation grounded in the new trajectory.
7. Summarize in natural language the key events that occurred **before** the selected span begins (i.e., steps with index before `start_index`) that are relevant for understanding the agent's situation. Avoid assumptions not grounded in the trajectory. The summary must be self-contained, and avoid citing supporting indexes. 
8. Explain, referencing the overall task, the exemplars (if helpful), and the pre-span context, why the agent initiates this subgoal at that specific moment. Ground the explanation in explicit evidence from the trajectory. The explanation must be self-contained, and avoid citing supporting indexes.
9. Use the `notes` field to briefly mention whether any exemplar informed your decisions (reference them by short description or index if available); use "none" only when no exemplar insight was needed.

Output
-------
Return **only** valid JSON with this structure and double-quoted keys/strings:

{{
  "subgoal_name": string,
  "subtrajectory_span": {{"start_index": int, "end_index": int}},
  "subtrajectory_steps": [
    {{"index": int, "observation": string, "action": string, "why_relevant": string}}
  ],
  "parameter_bindings": {{"<parameter_name>": <value>, ...}},
  "start_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "success_condition_evidence": [
    {{"condition": string, "explanation": string}}
  ],
  "pre_span_summary": string,
  "subgoal_initiation_intent": string,
  "notes": string
}}

Keep `why_relevant`, `explanation`, and `notes` concise. If no exemplar insight is required, set `notes` to "none". `pre_span_summary` and `subgoal_initiation_intent` must stay self-contained, and avoid assumptions not grounded in the trajectory.
"""


SUBGOAL_REVISION_PROMPT = """
You are an AI planner for an embodied environment.
Revise the input parameters of sub-goals from the provided interaction trajectories. 

You are given:
1) A mermaid-based workflow skill for embodied tasks. It is specified to the given sub-goal.
2) An expert interaction summary and exact micro-trajectory
3) A workflow execution log showing a mismatch between the workflow's chosen action and the expert action

Your task: 
- Read the sub-goal and the interaction trajectories, revise the input parameters of the sub-goal if need.

## The Subgoal (revise it if need): 
{subgoal_spec}

## Matched expert trajectory segment

{matched_traj_segment}

## Unmatched expert trajectory segment

{unmatched_traj_segment}


# Detailed Instructions for Sub-Goal Generation:
- **Complexity**: Sub-goals should have mediocre complexity:
  - Must include at least two primitive actions; sub-goals implementable by a single action OR one condition check followed by one primitive action are NOT allowed.
  - Steps list length must be between 2 and 5 inclusive (reject 1-step designs).
- **Generality**: Sub-goals must be applicable to other similar tasks, not just the provided examples.
- **Arguments**: Use common variable types (e.g., strings, lists). Avoid complex inputs like another function.

- **Parameters and Types**: For each sub-goal, explicitly list the input parameters and their types.
  - Allowed parameter types (choose only from): ReceptacleName, ItemName, ReceptacleType, ItemType, List_T (e.g. List_ItemName, List_ReceptacleType).
  - Typed placeholder examples (primitive actions for illustration):
    - go to {{gotoRec: ReceptacleName}}
    - open {{openRec: ReceptacleName}}
    - take {{takeItem: ItemName}} from {{fromRec: ReceptacleName}}
    - cool {{coolItem: ItemName}} with {{recep: ReceptacleName}}
    - heat {{heatItem: ItemName}} with {{recep: ReceptacleName}}
    - clean {{cleanItem: ItemName}} with {{recep: ReceptacleName}}
    - move {{moveItem: ItemName}} to {{toRec: ReceptacleName}}
    - use {{desklamp: ItemName}}
  - Typed placeholder examples (overall goal for illustration):
    - 'find an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put an item of type {{itemType: ItemType}} on a receptacle of type {{RecType: ReceptacleType}}'
    - 'clean an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a clean item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a hot item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'heat an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a cool item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'cool an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'examine an item of type {{itemType: ItemType}} with the desklamp',
    - 'look at an item of type {{itemType: ItemType}} under the desklamp', 
    - 'put two items of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
  - In the JSON output, provide parameters as strings in the form "param_name: param_type".

- **Start Conditions (Preconditions)**: For the given sub-goal, specify clear, checkable preconditions under which the sub-goal can begin. If these are not satisfied, this sub-goal MUST NOT be selected.
  - Express conditions succinctly with parameter references (e.g., "the agent does not locate at {{gotoRec}}, and {{gotoRec}} is reachable", "the agent already locates at {{openRec}} and {{openRec}} is current closed", "the agent already holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open").
  - Prefer condition forms that can be grounded in observations or known state (e.g., reachable, is open/is closed, contains, holding, exists).
  - Keep preconditions minimal and necessary; avoid hidden assumptions. 

- **Success Conditions (Postconditions)**: For the given sub-goal, specify the observable state that must hold after successful completion. If these are not satisfied, the sub-goal is incomplete/failed.
  - Use the same expression style as start_conditions with parameter references (e.g., "the agent locates at {{gotoRec}}", "{{openRec}} is current open", "{{toRec}} contains {{moveItem}}", "the agent holds a {{moveItem}}").
  - Keep postconditions minimal, necessary, and directly verifiable from observations or world facts.

- **Naming**: Name the sub-goal to summarize a multi-step intent, not a single primitive action.
  - Be concise and descriptive; use lowercase with underscores.
  - Do NOT reuse primitive action names or their near variants: go_to, open, take, put, move, clean, heat, cool, use.
  - Prefer names like: locate_and_open_receptacle, retrieve_item_from_container, transfer_item_to_surface, clean_item_and_place, heat_item_then_place.

- **Steps Analysis**: For each sub-goal, describe the sequence of actions required to achieve it.
  - Use concise, descriptive parameter names aligned with the **Parameters and Types** of the current sub-goal.
  - Ground each step in actual primitive actions: go to, open, take, put, move, clean, heat, cool, use.
  - Express branches and loops using production rules in the form: "If <condition>, then <primitive action>". You may use "Else if" and a final default action when appropriate.
  - Represent loops compactly as: "For each <X> in <LIST>: <primitive action or rule>" (keep total steps ≤ 5).
  - Express condition in natural language with the parameters declared in "parameters".
  - Minimum steps policy: ensure the "steps" array contains 2–5 items and includes at least two non-redundant primitive actions across all rules; do not output single-step sub-goals.
  - Examples:
    - If the agent does not already locate at {{gotoRec}}, and {{gotoRec}} is reachable, then go to {{gotoRec}}
    - If the agent locates at {{openRec}} and {{openRec}} is current closed, then open {{openRec}}
    - If the agent holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open, then move {{moveItem}} to {{toRec}}


# Output Format:
```json {{
[
    {{
    "name": <sub_goal_1_name, concise multi-step summary; use '_' instead of ' '; MUST NOT equal or directly mirror a primitive action name such as 'go_to', 'open', 'take', 'put', 'move', 'clean', 'heat', 'cool', 'use'>,
    "parameters": [
        "param_name_1: param_type_1",
        "param_name_2: param_type_2",
        ...
    ],
    "start_conditions": [
        "<condition 1>",
        "<condition 2>",
        ...
    ],
    "success_conditions": [
        "<condition 1>",
        "<condition 2>",
        ...
    ],
    "description": "<sub-goal 1 description>"
    "steps": [
        "<step 1>",
        "<step 2>",
        ...
    ],
    "applicable_tasks": [
        "<task 1: copy verbatim>",
        "<task 2: copy verbatim>",
        ...
    ],
    }}, 
    {{
    "name": <sub_goal_2_name, concise multi-step summary; use '_' instead of ' '; MUST NOT equal or directly mirror a primitive action name>,
    "parameters": [
        "param_name_1: param_type_1",
        ...
    ],
    "start_conditions": [
        "<condition 1>",
        ...
    ],
    "success_conditions": [
        "<condition 1>",
        ...
    ],
    "description": "<sub-goal 2 description>"
    "steps": [
        "<step 1>",
        "<step 2>",
        ...
    ],
    "applicable_tasks": [
        "<task 1: copy verbatim>",
        "<task 2: copy verbatim>",
        ...
    ],
    }}, ...
]
}}
"""

SUB_GOAL_ABSTRACT_EVOLUTION_PROMPT = """
You are an AI planner for an embodied environment.

You are given:
1) The desired sub-goal to implement.
2) The expert interaction trajectory for the desired sub-goal.

Your task: 
- Revise the sub-goals from the provided interaction trajectories. 
- Strictly follow the previous sub-goal format.
- Do NOT change the parameters of the sub-goal.
- Revise the description, start_conditions, and steps to match the expert trajectory data.

# Data
Here are some interaction trajectories:
{DATA}

# Detailed Instructions for Sub-Goal Generation:
- **Complexity**: Sub-goals should have mediocre complexity:
  - Must include at least two primitive actions; sub-goals implementable by a single action OR one condition check followed by one primitive action are NOT allowed.
  - Steps list length must be between 2 and 5 inclusive (reject 1-step designs).
- **Generality**: Sub-goals must be applicable to other similar tasks, not just the provided examples.
- **Arguments**: Use common variable types (e.g., strings, lists). Avoid complex inputs like another function.

- **World State Analysis**: Use the provided world states within the trajectories to revise the `start_conditions`, `success_conditions`, and `steps` so they reflect the observed facts; this alignment is extremely important.

- **Parameters and Types**: For each sub-goal, explicitly list the input parameters and their types.
  - Allowed parameter types (choose only from): ReceptacleName, ItemName, ReceptacleType, ItemType, List_T (e.g. List_ItemName, List_ReceptacleType).
  - Typed placeholder examples (primitive actions for illustration):
    - go to {{gotoRec: ReceptacleName}}
    - open {{openRec: ReceptacleName}}
    - take {{takeItem: ItemName}} from {{fromRec: ReceptacleName}}
    - cool {{coolItem: ItemName}} with {{recep: ReceptacleName}}
    - heat {{heatItem: ItemName}} with {{recep: ReceptacleName}}
    - clean {{cleanItem: ItemName}} with {{recep: ReceptacleName}}
    - move {{moveItem: ItemName}} to {{toRec: ReceptacleName}}
    - use {{desklamp: ItemName}}
  - Typed placeholder examples (overall goal for illustration):
    - 'find an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put an item of type {{itemType: ItemType}} on a receptacle of type {{RecType: ReceptacleType}}'
    - 'clean an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a clean item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a hot item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'heat an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'put a cool item of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
    - 'cool an item of type {{itemType: ItemType}} and put it in a receptacle of type {{RecType: ReceptacleType}}'
    - 'examine an item of type {{itemType: ItemType}} with the desklamp',
    - 'look at an item of type {{itemType: ItemType}} under the desklamp', 
    - 'put two items of type {{itemType: ItemType}} in a receptacle of type {{RecType: ReceptacleType}}'
  - In the JSON output, provide parameters as strings in the form "param_name: param_type".
  - Note that only `ReceptacleName` is a valid location which would be located or reachable, `ReceptacleType` is a type of receptacle.
  - Note that `ItemName` is not a valid location, any items always is contained in the receptacles or is hold by agent.  
  
- **Start Conditions (Preconditions)**: For each sub-goal, specify clear, checkable preconditions under which the sub-goal can begin. If these are not satisfied, this sub-goal MUST NOT be selected.
  - Express conditions succinctly with parameter references (e.g., "the agent does not locate at {{gotoRec}}, and {{gotoRec}} is reachable", "the agent already locates at {{openRec}} and {{openRec}} is current closed", "the agent already holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open").
  - Prefer condition forms that can be grounded in observations or known state (e.g., reachable, is open/is closed, contains, holding, exists).
  - Keep preconditions minimal and necessary; avoid hidden assumptions. 

- **Success Conditions (Postconditions)**: For each sub-goal, specify the observable state that must hold after successful completion. If these are not satisfied, the sub-goal is incomplete/failed.
  - Use the same expression style as start_conditions with parameter references (e.g., "the agent locates at {{gotoRec}}", "{{openRec}} is current open", "{{toRec}} contains {{moveItem}}", "the agent holds a {{moveItem}}").
  - Keep postconditions minimal, necessary, and directly verifiable from observations or world facts.

- **Naming**: Name each sub-goal to summarize a multi-step intent, not a single primitive action.
  - Be concise and descriptive; use lowercase with underscores.
  - Do NOT reuse primitive action names or their near variants: go_to, open, take, put, move, clean, heat, cool, use.
  - Prefer names like: locate_and_open_receptacle, retrieve_item_from_container, transfer_item_to_surface, clean_item_and_place, heat_item_then_place.

- **Steps Analysis**: For each sub-goal, describe the sequence of actions required to achieve it.
  - Use concise, descriptive parameter names aligned with the **Parameters and Types** of the current sub-goal.
  - Ground each step in actual primitive actions: go to, open, take, put, move, clean, heat, cool, use.
  - Express branches and loops using production rules in the form: "If <condition>, then <primitive action>". You may use "Else if" and a final default action when appropriate.
  - Represent loops compactly as: "For each <X> in <LIST>: <primitive action or rule>" (keep total steps ≤ 5).
  - Express condition in natural language with the parameters declared in "parameters".
  - Minimum steps policy: ensure the "steps" array contains 2–5 items and includes at least two non-redundant primitive actions across all rules; do not output single-step sub-goals.
  - Examples:
    - If the agent does not already locate at {{gotoRec}}, and {{gotoRec}} is reachable, then go to {{gotoRec}}
    - If the agent locates at {{openRec}} and {{openRec}} is current closed, then open {{openRec}}
    - If the agent holds a {{moveItem}} and locates at {{toRec}} and the {{toRec}} is current open, then move {{moveItem}} to {{toRec}}

- **Applicable Tasks**: For each sub-goal, provide tasks where it applies.
  - Copy task names verbatim from the source data (no paraphrasing, no reformatting, preserve original casing and punctuation).

- **Major Revision**: 
  - If the current sub-goal definition, parameter contract, and the provided trajectories (including their world states) make it impossible to produce a compliant description, start_conditions, success_conditions, and steps, set a `major_revision` flag to true in the output JSON and explain why (e.g., the intent or parameters must be broadened/narrowed). Otherwise set the flag to false and leave the reason empty.


# Provided Sub-Goals:
{previous_sub_goals}

# Output Format:
```json {{
  "name": <keep the same as the provided sub-goal>,
  "parameters": <keep the same as the provided sub-goal>,
  "start_conditions": [
      "<condition 1>",
      "<condition 2>",
      ...
  ],
  "description": "<rewrite to align with trajectories>",
  "success_conditions": [
      "<condition 1>",
      "<condition 2>",
      ...
  ],
  "steps": [
      "<step 1>",
      "<step 2>",
      ...
  ],
  "major_revision": <true or false>,
  "major_revision_reason": "<brief justification if major_revision is true, otherwise empty string>"
}}
"""

# Prompt to revise a specific workflow to match expert trajectory for using a desklamp
SUB_GOAL_WORKFLOW_EVOLUTION_PROMPT = """
You are a workflow improver. 

You are given:
1) The desired sub-goal to implement.
2) A mermaid-based workflow skill for embodied tasks. It is specified to the given sub-goal.
3) A workflow execution log showing a match or mismatch between the workflow's chosen action and the expert action


Your task: 
- Modify the mermaid workflow to align with the expert interaction trajectories while preserving the sub-goal skill intent. 
- If the current sub-goal definition or parameter contract makes a compliant workflow impossible given the trajectories and world states, report this via the `major_revision` flag rather than producing an invalid workflow.

# Sub-goal to implement
{sub_goal}

# Mermaid workflow 
## Overall Guideline (must follow exactly):
- Use lowercase names for local variables and UPPERCASE names for globals (e.g., target_rec vs TARGET_RECEPTACLES).
- Local and global identifiers must stay distinct; do not reuse the same base name with different casing (e.g., avoid target vs TARGET).
- Use double braces `{{VAR}}` for all placeholders evaluated at run time.
- All effects on global state happen inside DataOp/LoopControl nodes via `writes GLOBAL: (...)`. Interface/Check/Action MUST NOT include `writes GLOBAL`.
- Branching is ONLY via Check nodes with labels `Yes`/`No`. LoopControl uses `body`/`done`.
- Every Check/Action with parameters must make them resolvable from GLOBALS or inline bindings in `local in: (...)`.
- Respect data-flow ordering: any GLOBAL referenced in a node's `local in` must already be assigned by an earlier node via `writes GLOBAL` (e.g., `CURRENT_RECEPTACLE` written inside a LoopControl or DataOp before `local in: (... = CURRENT_RECEPTACLE)`). Do not rely on implicit/undefined globals.
- Avoid unescaped quote characters within quoted strings; escape them to keep JSON and Python strings valid.
- Selection helpers `select_one` and `select_all` may be used only inside DataOp nodes; they are not allowed in Check/PrimitiveAction/LoopControl/Interface nodes.

Domain predicates (provided by runtime):
- locate({{receptacle_name}}); reachable({{receptacle_name}})
- contains({{receptacle_name}}, {{item_name}}); holding({{item_name}})
- is_open({{receptacle_name}}); is_closed({{receptacle_name}})
- is_cleaned({{item_name}}); is_heated({{item_name}}); is_cooled({{item_name}}); is_turned_on({{item_name}})
- type checks: is_item_of_type({{item_name}}, {{target_type}}); is_receptacle_of_type({{receptacle_name}}, {{target_type}})
- Quantifiers, e.g. `exists("Item", lambda x: contains({{receptacle_name}}, x) and is_item_of_type(x, {{target_type}}))`

Object Selection (provided by runtime):
- select_one(\'Item\' or \'Receptacle\', a FOL expression with domain predicates conditioned on `x`)
  * select the first Item/Receptacle that satisfies the expression, return None if no such object exists
  * e.g., select_one(\'Item\', is_item_of_type(x, {{target_type}})))
- select_all(\'Item\' or \'Receptacle\', a FOL expression with domain predicates conditioned on `x`)
  * select all Items/Receptacles that satisfies the expression, return an empty list if no such object exists
  * e.g., select_all(\'Item\', is_item_of_type(x, {{target_type}})))

Strict style scaffold (copy-paste exactly, then instantiate only the nodes you need):

%%{{init: {{'theme': 'default', 'themeVariables': {{'background': '#ffffff'}} }} }}%%
flowchart TD
    %% Guideline: Keep every %% comment instruction; never remove or rewrite these annotations.
    %% ===================== Class Definitions =====================
    %% Guideline: Strictly follow the class definitions.
{CLASS_DEFINITIONS_NODE}

    %% ===================== Type Alias Legend =====================
    %% Guideline: Strictly follow the type alias legend.
{TYPE_ALIAS_NODE}

    %% ===================== Spec =====================
    %% Guideline: Strictly follow the Spec.
{FLOW_SPEC_NODE}

    %% ===================== Interface & Spec =====================
    %% Guideline: START is entry; *_END are exits. No computations here. Line breaks use <br>.
    %% Guideline: Strictly follow the interface definition.
{INTERFACE_NODE}

    %% ===================== Primitive Actions =====================
    %% Guideline: Parameters of PrimitiveActions are bound from GLOBALS directly in 'local in'; no edge payloads.
    %% Guideline: Pick the PrimitiveActions you need from the list below. 
    %% Guideline: If needed, adjust the parameters assignment in the `local in` to match the designed workflow.

    A_GOTO["PrimitiveAction: <br>(action: 'go to {{goto_rec}}')<br>local in: (goto_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_OPEN["PrimitiveAction: <br>(action: 'open {{open_rec}}')<br>local in: (open_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_TAKE["PrimitiveAction: <br>(action: 'take {{take_item}} from {{from_rec}}')<br>local in: (take_item: ItemName = {{TARGET_ITEM}}, from_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_MOVE["PrimitiveAction: <br>(action: 'move {{move_item}} to {{to_rec}}')<br>local in: (move_item: ItemName = {{TARGET_ITEM}}, to_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_PUT["PrimitiveAction: <br>(action: 'put {{put_item}} in {{to_rec}}')<br>local in: (put_item: ItemName = {{TARGET_ITEM}}, to_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_CLEAN["PrimitiveAction: <br>(action: 'clean {{clean_item}} with {{clean_rec}}')<br>local in: (clean_item: ItemName = {{TARGET_ITEM}}, clean_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_HEAT["PrimitiveAction: <br>(action: 'heat {{heat_item}} with {{heat_rec}}')<br>local in: (heat_item: ItemName = {{TARGET_ITEM}}, heat_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_COOL["PrimitiveAction: <br>(action: 'cool {{cool_item}} with {{cool_rec}}')<br>local in: (cool_item: ItemName = {{TARGET_ITEM}}, cool_rec: ReceptacleName = {{CURRENT_RECEPTACLE}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_USE["PrimitiveAction: <br>(action: 'use {{use_item}}')<br>local in: (use_item: ItemName = {{TARGET_ITEM}})<br>out: (executed: Bool)"]:::PrimitiveAction
  
    %% ===================== LoopControl (use only if needed) =====================
    %% Guideline: Only foreach loops via LoopControl with edges 'body' and 'done'.
    %% Guideline: LoopControl nodes only operate on the loop variable (e.g., assign the global `CURRENT_RECEPTACLE` with local `receptacle_i` from the local `loop_receptacles` if global variable `RECEPTACLE_CANDIDATES` could be provided from the previous nodes).
    %% Guideline: Modify the loop variable assignment to improve workflow if needed.
{LOOP_FOR_NODE}
    %% Guideline: Include a new LoopControl node if you must use a loop.
    %% For Example:
    LOOP_FOR["LoopControl: <br>For receptacle_i in {{loop_receptacles}}<br>writes GLOBAL: (CURRENT_RECEPTACLE: ReceptacleName:=receptacle_i)<br>local in: (loop_receptacles: List_ReceptacleName = {{RECEPTACLE_CANDIDATES}})"]:::LoopControl

    %% ===================== Checks =====================
    %% Guideline: Checks branch on Yes/No only; inputs resolved from GLOBALS. 
    %% Guideline: Modify the check condition to improve workflow if needed.
{CHECK_NODE}
    %% Design new Check nodes with the domain predicates if needed
    %% For Example:
    C_IS_CLOSED["Check: <br>is_closed({{rec}})<br>local in: (rec: ReceptacleName = {{CURRENT_RECEPTACLE}})"]:::Check
    C_HAS_TYPE["Check: <br>exists(Item, lambda x: contains({{rec}}, x) and is_item_of_type(x, {{target_type}}))<br>local in: (rec: ReceptacleName = {{CURRENT_RECEPTACLE}}, target_type: ItemType = {{TARGET_ITEM_TYPE}})"]:::Check
    
    %% ===================== Data Operation =====================
    %% Guideline: All GLOBAL writes and data-binding happen in DataOp nodes. 
    %% Guideline: Strictly copy the provided D_INIT node. 
{D_INIT_NODE}
    %% Guideline: Selection helpers select_one/select_all are ONLY allowed inside DataOp nodes.
    
    %% Optional helpers (remove if not used): select_one/select_all → assign to GLOBAL before used by other nodes
    D_SELECT_ONE_ITEM_WITH_TYPE["DataOp: <br>writes GLOBAL: (TARGET_ITEM: ItemName = select_one(\'Item\', is_item_of_type(x, {{target_type}}))) <br>local in: (target_type: ItemType = {{TARGET_ITEM_TYPE}})"]:::DataOp
    D_SELECT_ALL_RECEPTACLE_WITH_TYPE["DataOp: <br>writes GLOBAL: (SELECTED_RECEPTACLES: List_ReceptacleName = select_all(\'Receptacle\', is_receptacle_of_type(x, {{target_type}}))) <br>local in: (target_type: ReceptacleTypeName = {{TARGET_REC_TYPE}})"]:::DataOp
    %% Guideline: Modify the DataOp nodes to improve workflow if needed.
{DATA_OP_NODE}
    %% Design new DataOp nodes with the selection helper and domain predicates if needed

    %% ===================== Node Class Assignments =====================
{CLASS_ASSIGNMENTS}

    %% ===================== Legend links =====================
    %% Guideline: Strictly follow the legend links.
{LEGEND_EDGE}

    %% ===================== Control-Flow Edges =====================
    %% Guideline: Keep edges control-only (no parameters); label branches with Yes/No/body/done.
    %% Guideline: Modify the control-flow edges to improve workflow if needed.
    %% Guideline: For outgoing edges, LoopControl nodes must have both `body` and `done`; Check nodes must have `Yes` and `No`; DataOp/PrimitiveAction nodes must emit exactly one edge.
{CONTROL_FLOW_EDGE}
    


## Matched expert trajectory segment

{matched_traj_segment}

## Unmatched expert trajectory segment

{unmatched_traj_segment}


{HINTS}

# Output Format
- Return a single JSON object with the following fields:
  - `"major_revision"`: boolean indicating whether the current sub-goal definition, parameter contract, or provided trajectories (including world states) prevent you from producing a compliant workflow; set to `true` only when the workflow cannot satisfy all constraints without revising the sub-goal intent or parameters.
  - `"major_revision_reason"`: concise explanation describing what must change (e.g., adjust sub-goal intent, add/remove parameters) when `"major_revision"` is `true`; use an empty string when the flag is `false`.
  - `"mermaid_code"`: the revised workflow as a single Mermaid string that compiles (empty string if `"major_revision"` is `true` and no valid workflow can be produced).
  - `"change_summary"`: array of 3–5 bullet strings explaining key modifications and rationale.
  - `"key_conditions"`: array listing the critical check predicates or lambda expressions referenced in the workflow.
- Do not wrap the JSON in backticks or add extra commentary.

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




TEXTUAL_GRADIENT_FROM_OFFLINE_FAILURE_PROMPT = """
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


# ===================== Online Skill Evolution Prompts =====================

EPISODE_RECOVERY_ANALYSIS_PROMPT = """You are analyzing an ALFWorld household agent's episode trajectory to identify valuable failure-to-recovery patterns that can improve the agent's skill system.

The agent has these skills (mermaid flowchart graphs):
{skill_descriptions}

When a skill graph reaches FAILURE_END, the planner recovers by issuing actions or calling other skills. We want to identify recovery patterns and classify HOW they should improve the system.

# Task Information
Task name: {task_name}
Task type: {task_type}

# Episode Success
{episode_success}

# Available Action Types
go to <receptacle>, open <receptacle>, close <receptacle>, take <item> from <receptacle>,
clean <item> with <receptacle>, cool <item> with <receptacle>, heat <item> with <receptacle>,
move <item> to <receptacle>, use <item>, examine <item/receptacle>

# Skill Execution Log (structured)
{skill_execution_log}

# Full Episode Trajectory
{trajectory}

# Instructions

Analyze the trajectory and identify valuable failure-then-recovery patterns.

## Pattern Types
**Search Expansion**: The skill searched a list of receptacles but did not find the target item. Recovery: the planner searched additional receptacles not in the original list.
**Precondition Recovery**: A skill failed because a precondition was not met (e.g., receptacle closed, item not held, wrong location). Recovery: the planner satisfied the precondition (e.g., opened the receptacle, picked up the item, navigated first).

## Evolution Type Classification (CRITICAL)
For each pattern, decide WHERE the fix belongs:

**`skill_graft`** — The recovery uses only operations that are within the failing skill's own responsibility.
- Example: `locate_and_retrieve_item` fails because it didn't check if a receptacle was closed before looking inside → recovery opens the receptacle. Opening containers IS part of "locate and retrieve", so this is an intra-skill improvement.
- Rule: The recovery actions are the SAME type of operations the skill already performs (e.g., navigation, opening, taking within a retrieve skill).

**`planner_routing`** — The recovery works by switching to a DIFFERENT skill or a fundamentally different strategy that is OUTSIDE the failing skill's responsibility.
- Example: `locate_and_retrieve_item` fails (can't find item) → planner recovers by realizing the item type was wrong and calling the skill again with different parameters, or switching to a different skill entirely.
- Rule: The failing skill has a clear, narrow responsibility. If the recovery requires a capability outside that scope, it's a planner routing issue, NOT a skill deficiency.

**`new_skill`** — The recovery pattern is reusable across multiple skills and represents a capability not covered by any existing skill.
- Rule: Use sparingly. Only if the pattern clearly doesn't fit as a graft or routing rule.

Only select patterns where:
1. The recovery was actually successful (the agent achieved what was missing)
2. The pattern is **structurally generalizable** — not a one-off fix for a specific item/receptacle
3. The failure node in the skill graph is identifiable from the diagnostic

# Output Format (strict JSON only):
{{
  "analysis": "brief reasoning about what happened in this episode",
  "patterns": [
    {{
      "pattern_type": "search_expansion" or "precondition_recovery",
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
- **General**: Apply to any item/receptacle, not just the specific ones in this episode
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

# Available Action Types
go to <receptacle>, open <receptacle>, close <receptacle>, take <item> from <receptacle>,
clean <item> with <receptacle>, cool <item> with <receptacle>, heat <item> with <receptacle>,
move <item> to <receptacle>, use <item>, examine <item/receptacle>

# Instructions

Clean the trajectory by:
1. **Remove** failed retry attempts (e.g., repeated "go to X" that all fail — keep only the last failed attempt if needed for context, but remove pure duplicates)
2. **Remove** irrelevant actions: "think: ..." or other hallucinated actions that don't contribute
3. **Remove** actions unrelated to resolving the specific failure
4. **Preserve** the essential causal chain: each action's output feeds the next action's input
5. **Preserve** completeness: the cleaned trajectory must still achieve the recovery from start to finish
6. **Verify** coherence: check that the sequence makes logical sense (you navigate before taking, open before looking inside)

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

# Available Action Types (reference for valid actions)
{action_types_sample}

# Instructions

The anti-patterns above are raw failure logs from repeated executions. Your job is to:

1. **Identify the root causes**: What general rules do these failures reveal? (e.g., "searchLocations must include all reachable receptacles of likely types, not just the first one found")
2. **Distill concise rules**: Replace N individual failure dumps with 1-3 general rules that prevent ALL of them
3. **Fix parameter_roles**: If any parameter role description is misleading, correct it
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
- `locate(receptacle: ReceptacleName)` → bool: agent is at the receptacle
- `reachable(receptacle: ReceptacleName)` → bool: agent can reach the receptacle
- `contains(receptacle: ReceptacleName, item: ItemName)` → bool: receptacle has the item
- `holding(item: ItemName)` → bool: agent is holding the item
- `is_open(receptacle: ReceptacleName)` → bool: receptacle is open
- `is_closed(receptacle: ReceptacleName)` → bool: receptacle is closed
- `is_cleaned(item: ItemName)` → bool: item has been cleaned
- `is_cooled(item: ItemName)` → bool: item has been cooled
- `is_heated(item: ItemName)` → bool: item has been heated
- `is_item_of_type(item: ItemName, type: ItemType)` → bool: item matches type
- `is_receptacle_of_type(receptacle: ReceptacleName, type: ReceptacleType)` → bool: receptacle matches type
- `can_goto(receptacle: ReceptacleName)` → bool
- `can_open(receptacle: ReceptacleName)` → bool
- `can_take(receptacle: ReceptacleName, item: ItemName)` → bool
- `can_put(receptacle: ReceptacleName, item: ItemName)` → bool
- `exists_item_in_list(list: List_ItemName, lambda: fn)` → bool
- `exists_receptacle_in_list(list: List_ReceptacleName, lambda: fn)` → bool

# Available Primitive Actions
- `go to {{receptacle}}` — navigate to a receptacle
- `open {{receptacle}}` — open a closed receptacle
- `take {{item}} from {{receptacle}}` — pick up an item
- `move {{item}} to {{receptacle}}` — place a held item (equivalent to put)
- `clean {{item}} with {{receptacle}}` — clean an item at a sink
- `cool {{item}} with {{receptacle}}` — cool an item in a fridge
- `heat {{item}} with {{receptacle}}` — heat an item in a microwave
- `use {{item}}` — use/turn on an item (e.g., desklamp)

# Instructions

## Critical: Generalization Principle
The trajectory above is ONE concrete example. Your graft must be a **general capability upgrade** to the skill, NOT a replay of this specific episode.

- NEVER hardcode specific item names, receptacle names, or IDs from the episode.
- ALL values must come from scope variables (`{{{{CURRENT_RECEPTACLE}}}}`, `{{{{TARGET_ITEM_TYPE}}}}`, etc.).
- The graft should work for ANY item/receptacle that could trigger the same failure mode.
- ONLY use predicate functions from the closed list above.

## Compositional Inlining
When the recovery trajectory called another skill, do NOT create an abstract "call skill" node. Instead, **inline that skill's graph structure** directly into your graft.

## Strict Mermaid Syntax (CRITICAL — violations cause runtime crashes)
Every variable reference MUST use DOUBLE braces `{{{{VAR}}}}`. Single braces `{{VAR}}` will crash the Python template engine.

## Variable Isolation (CRITICAL)
Use graft-scoped variable names with a `G_` prefix for any new variables. Do NOT overwrite parent variables.

## Edge Labels for PrimitiveAction
After a PrimitiveAction, insert a **Check node** to test whether the action succeeded.

## Sub-graph Requirements
1. Starts with `GRAFT_ENTRY`
2. Uses ONLY double-brace references
3. Uses `G_` prefix for new globals
4. Ends with `GRAFT_CONTINUE` (success) or `GRAFT_FAIL` (failure)
5. Uses only DataOp, Check, PrimitiveAction node types
6. After every PrimitiveAction, uses a Check node

For **search_expansion**: Search additional reachable receptacles not yet visited.
For **precondition_recovery**: Satisfy the unmet precondition then retry.

# Output Format (strict JSON only):
{{
  "reasoning": "brief explanation of the induced sub-graph logic",
  "graft_mermaid": "the mermaid node definitions and edges as a string",
  "graft_point_src": "source node ID where graft connects",
  "graft_point_dst": "original destination node being replaced",
  "continue_target": "node ID where GRAFT_CONTINUE should connect",
  "new_parameters": ["list of new parameter names if needed, or empty list"]
}}
"""

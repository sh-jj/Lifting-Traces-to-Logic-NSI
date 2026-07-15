from __future__ import annotations
import os
import json
import string
from ..base import BaseAgent
from ..utils import llm_response, get_price, get_token_counts
from json_repair import repair_json
from .prompts import OBSERVATION_TO_PREDICATES_PROMPT
from .prompts import SUBGOAL_PROMPT, SUB_GOAL_WORKFLOW_PROMPT
from .prompts import TRAJ_SEGMENTATION_PROMPT, TRAJ_SEGMENTATION_REVISION_PROMPT
from .prompts import MEMORY_CONSTRUCTION_INTENT_PROMPT, MEMORY_CONSTRUCTION_PREDICATE_ANALYSIS_PROMPT, MEMORY_CONSTRUCTION_INTENT_WITH_PREDICATE_PROMPT
from .prompts import SUB_GOAL_WORKFLOW_EVOLUTION_PROMPT, SUBGOAL_REVISION_PROMPT, SUB_GOAL_ABSTRACT_EVOLUTION_PROMPT
from .prompts import REACT_WITH_STATE_GROUNDING_PROMPT, SKILL_ACTION_ROUTER_PROMPT, EXECUTION_WITH_ROUTING_PROMPT
from .prompts import SKILL_PARAMETER_ANALYSIS_PROMPT, SKILL_PRECONDITION_SYNTH_PROMPT
from .prompts import TASK_LEVEL_SUMMARY_PROMPT, TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT

import traceback
import warnings
import random
from copy import deepcopy

import re
from .basic_actions import *
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
import ast

import numpy as np
from sklearn.cluster import KMeans

from .infoflow2graph import (
    parse_mermaid_with_info,
    validate_edge_payloads,
    validate_node_inputs,
    validate_node_inputs_with_prefix_node,
    validate_check_nodes_without_global_writes,
    validate_control_flow_outgoing_edges,
    validate_control_flow_node_definitions,
    validate_loop_entry_edges,
)
from .infoflow2graph import reharsal_with_info, reharsal_online_with_info, traverse_with_info
from .infoflow2graph import extract_start_inputs, inverse_all_items, _build_predicate_env, convert_all_items
from .infoflow2graph import Action_Unmatched_Exception
from .infoflow2graph import graft_subgraph_at_node, extract_graft_node_ids


from .symbolic_world import World, executor, DOMAIN
from .symbolic_world import get_pre_condition, canon, transform_item_name

ENV_TYPES = {'pick_and_place': 'put', 'pick_clean_then_place': 'clean', 'pick_heat_then_place': 'heat', 'pick_cool_then_place': 'cool', 'look_at_obj': 'examine', 'pick_two_obj': 'puttwo'}
ACTION_PATTERNS = [('goto', re.compile('^go to (?P<receptacle>.+)$'), lambda m: {'receptacle': canon(m['receptacle'])}), ('open', re.compile('^open (?P<receptacle>.+)$'), lambda m: {'receptacle': canon(m['receptacle'])}), ('close', re.compile('^close (?P<receptacle>.+)$'), lambda m: {'receptacle': canon(m['receptacle'])}), ('take', re.compile('^take (?P<item>.+) from (?P<src>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['src'])}), ('clean', re.compile('^clean (?P<item>.+) with (?P<src>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['src'])}), ('cool', re.compile('^cool (?P<item>.+) with (?P<src>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['src'])}), ('heat', re.compile('^heat (?P<item>.+) with (?P<src>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['src'])}), ('put', re.compile('^put (?P<item>.+) (?:in(?:/on)?|on) (?P<receptacle>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['receptacle'])}), ('move', re.compile('^move (?P<item>.+) to (?P<receptacle>.+)$'), lambda m: {'item': canon(m['item']), 'receptacle': canon(m['receptacle'])}), ('use', re.compile('^use (?P<item>.+)$'), lambda m: {'item': canon(m['item'])})]
PREDICATE_PATTERN = re.compile('^(?P<name>[^()]+)\\((?P<args>.*)\\)$')

reverse_list2str = lambda lst: '[' + ', '.join(lst[::-1]).replace('_', ' ') + ']'


LIST_TYPE_PATTERN = re.compile(r"^List[\s_\[]*(.+?)[\]\s]*$", re.IGNORECASE)
TYPE_BASE_ALIASES = {
    "agent": "agent",
    "bool": "bool",
    "boolean": "bool",
    "int": "int",
    "integer": "int",
    "item": "item",
    "itemname": "item",
    "itemtype": "item_type",
    "typename": "item_type",
    "receptacle": "receptacle",
    "receptaclename": "receptacle",
    "receptacletype": "receptacle_type",
    "objecttype": "object_type",
}
FUNCTION_SIGNATURES = {
    "exists_item_in_list": {"args": ["list[item]", ("lambda", "item")], "returns": "bool"},
    "forall_item_in_list": {"args": ["list[item]", ("lambda", "item")], "returns": "bool"},
    "exists_receptacle_in_list": {"args": ["list[receptacle]", ("lambda", "receptacle")], "returns": "bool"},
    "forall_receptacle_in_list": {"args": ["list[receptacle]", ("lambda", "receptacle")], "returns": "bool"},
    "holding": {"args": ["item"], "returns": "bool"},
    "locate": {"args": ["receptacle"], "returns": "bool"},
    "reachable": {"args": ["receptacle"], "returns": "bool"},
    "contains": {"args": ["receptacle", "item"], "returns": "bool"},
    "is_open": {"args": ["receptacle"], "returns": "bool"},
    "is_closed": {"args": ["receptacle"], "returns": "bool"},
    "is_cleaned": {"args": ["item"], "returns": "bool"},
    "is_cooled": {"args": ["item"], "returns": "bool"},
    "is_heated": {"args": ["item"], "returns": "bool"},
    "is_turned_on": {"args": ["item"], "returns": "bool"},
    "is_item_of_type": {"args": ["item", "item_type"], "returns": "bool"},
    "is_receptacle_of_type": {"args": ["receptacle", "receptacle_type"], "returns": "bool"},
    "can_goto": {"args": ["receptacle"], "returns": "bool"},
    "can_open": {"args": ["receptacle"], "returns": "bool"},
    "can_take": {"args": ["receptacle", "item"], "returns": "bool"},
    "can_put": {"args": ["receptacle", "item"], "returns": "bool"},
    "can_clean": {"args": ["receptacle", "item"], "returns": "bool"},
    "can_cool": {"args": ["receptacle", "item"], "returns": "bool"},
    "can_heat": {"args": ["receptacle", "item"], "returns": "bool"},
    "can_turn_on": {"args": ["item"], "returns": "bool"},
}


@dataclass(frozen=True)
class ParameterTypeInfo:
    canonical: str
    display: str
    source: Optional[str] = None
    origin: str = "param"


def _canonicalize_type_name(type_name: str) -> str:
    cleaned = (type_name or "").strip()
    if not cleaned:
        return "unknown"
    list_match = LIST_TYPE_PATTERN.match(cleaned)
    if list_match:
        inner = list_match.group(1).strip()
        inner_canon = _canonicalize_type_name(inner)
        if inner_canon == "unknown":
            return "unknown"
        return f"list[{inner_canon}]"
    normalized = re.sub(r"[\s_\-]", "", cleaned).lower()
    return TYPE_BASE_ALIASES.get(normalized, "unknown")


def _build_skill_parameter_type_map(skill_json: Dict) -> Dict[str, ParameterTypeInfo]:
    param_types: Dict[str, ParameterTypeInfo] = {}
    for declaration in skill_json.get("parameters", []):
        if ":" not in declaration:
            continue
        name, type_str = declaration.split(":", 1)
        canonical = _canonicalize_type_name(type_str)
        param_types[name.strip()] = ParameterTypeInfo(
            canonical=canonical,
            display=type_str.strip(),
            source=name.strip(),
            origin="param",
        )
    for name, type_str in skill_json.get("param", {}).items():
        canonical = _canonicalize_type_name(type_str)
        param_types[name.strip()] = ParameterTypeInfo(
            canonical=canonical,
            display=type_str.strip(),
            source=name.strip(),
            origin="param",
        )
    return param_types


def _prepare_clause_for_typecheck(
    clause: str, param_types: Dict[str, ParameterTypeInfo]
) -> Tuple[str, Dict[str, ParameterTypeInfo]]:
    formatter = string.Formatter()
    alias_env: Dict[str, ParameterTypeInfo] = {}
    expr_parts: List[str] = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(clause):
        expr_parts.append(literal_text)
        if field_name is None:
            continue
        safe_name = f"__skill_param_{field_name}"
        base_info = param_types.get(
            field_name,
            ParameterTypeInfo(
                canonical="unknown",
                display="unknown",
                source=field_name,
                origin="unknown",
            ),
        )
        alias_env[safe_name] = ParameterTypeInfo(
            canonical=base_info.canonical,
            display=base_info.display,
            source=field_name,
            origin=base_info.origin,
        )
        expr_parts.append(safe_name)
    return "".join(expr_parts), alias_env


def _types_compatible(actual: str, expected: str) -> bool:
    if expected == "any" or actual == "unknown":
        return True
    return actual == expected


def _describe_expected_type(type_label: str) -> str:
    if type_label.startswith("list["):
        return f"list of {type_label[5:-1]}"
    return type_label


class _ClauseTypeChecker(ast.NodeVisitor):
    def __init__(self, base_env: Dict[str, ParameterTypeInfo]):
        self.scopes: List[Dict[str, ParameterTypeInfo]] = [dict(base_env)]
        self.node_types: Dict[ast.AST, str] = {}
        self.node_sources: Dict[ast.AST, str] = {}
        self.errors: List[str] = []

    def check(self, expr: str) -> List[str]:
        try:
            parsed = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            self.errors.append(f"invalid expression '{expr}': {exc.msg}")
            return self.errors
        self.visit(parsed)
        return self.errors

    def visit_Expression(self, node: ast.Expression):
        self.visit(node.body)

    def visit_Name(self, node: ast.Name):
        info = self._lookup_type(node.id)
        if info is None:
            info = ParameterTypeInfo(
                canonical="unknown",
                display="unknown",
                source=node.id,
                origin="unknown",
            )
        self.node_types[node] = info.canonical
        label = info.source or node.id
        if info.origin == "lambda":
            prefix = label
        elif info.origin == "param":
            prefix = f"parameter '{label}'"
        else:
            prefix = label
        display = info.display or info.canonical
        if display and display != "unknown":
            self.node_sources[node] = f"{prefix} ({display})"
        else:
            self.node_sources[node] = prefix

    def visit_Constant(self, node: ast.Constant):
        value = node.value
        if isinstance(value, bool):
            self.node_types[node] = "bool"
        elif isinstance(value, int):
            self.node_types[node] = "int"
        elif isinstance(value, str):
            self.node_types[node] = "string"
        else:
            self.node_types[node] = "unknown"

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.visit(node.operand)
        self.node_types[node] = "bool"

    def visit_BoolOp(self, node: ast.BoolOp):
        for value in node.values:
            self.visit(value)
        self.node_types[node] = "bool"

    def visit_Compare(self, node: ast.Compare):
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)
        self.node_types[node] = "bool"

    def visit_Call(self, node: ast.Call):
        func_name = self._resolve_name(node.func)
        signature = FUNCTION_SIGNATURES.get(func_name)
        self.visit(node.func)
        if signature is None:
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)
            self.node_types[node] = "unknown"
            return
        expected_args = signature["args"]
        for idx, expected in enumerate(expected_args):
            if idx >= len(node.args):
                break
            arg = node.args[idx]
            if isinstance(expected, tuple) and expected[0] == "lambda":
                self._check_lambda_argument(arg, expected[1], func_name, idx + 1)
            else:
                self.visit(arg)
                arg_type = self.node_types.get(arg, "unknown")
                if not _types_compatible(arg_type, expected):
                    self._report_arg_type_error(func_name, idx + 1, expected, arg)
        for extra_arg in node.args[len(expected_args):]:
            self.visit(extra_arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self.node_types[node] = signature["returns"]

    def _check_lambda_argument(self, node: ast.AST, expected_elem_type: str, func_name: str, position: int):
        if not isinstance(node, ast.Lambda):
            self.visit(node)
            self._report_arg_type_error(func_name, position, f"lambda over {expected_elem_type}", node)
            return
        if not node.args.args:
            self.errors.append(f"{func_name} argument #{position} lambda must have at least one parameter.")
            return
        param_name = node.args.args[0].arg
        lambda_scope = {
            param_name: ParameterTypeInfo(
                canonical=expected_elem_type,
                display=expected_elem_type,
                source=f"lambda parameter '{param_name}'",
                origin="lambda",
            )
        }
        self.scopes.append(lambda_scope)
        self.visit(node.body)
        self.scopes.pop()
        self.node_types[node] = "lambda"

    def _resolve_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._resolve_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def _lookup_type(self, name: str) -> Optional[ParameterTypeInfo]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _report_arg_type_error(self, func_name: str, position: int, expected: str, node: ast.AST):
        source = self.node_sources.get(node)
        if source is None:
            source = getattr(ast, "unparse", lambda n: "<expression>")(node)
        readable_expected = _describe_expected_type(expected)
        self.errors.append(
            f"{func_name} argument #{position} expects {readable_expected}, but received {source}."
        )


def _type_check_clause(clause: str, alias_env: Dict[str, ParameterTypeInfo]) -> List[str]:
    checker = _ClauseTypeChecker(alias_env)
    return checker.check(clause)


def _check_precondition_clause_types(
    clause: str, param_types: Dict[str, ParameterTypeInfo]
) -> List[str]:
    prepared_expr, alias_env = _prepare_clause_for_typecheck(clause, param_types)

    # print(prepared_expr, alias_env)
    if not prepared_expr.strip():
        return []
    return _type_check_clause(prepared_expr, alias_env)

from .memory import (
    TrajectoryMemoryGraph,
    TrajectoryMemoryNode,
    SkillCallingMemoryGraph,
    SkillCallingMemoryNode,
)

from .memory import generate_embeddings

RAW_ACTION_TYPES = ["go to", "open", "close", "take", "clean", "cool", "heat", "put", "move", "use"]



def extract_nodes_from_mermaid_code(mermaid_code):
    # Extract key sections from the previous mermaid code while excluding commented lines.
    def _collapse_lines(lines: List[str]) -> str:
        return "\n".join(lines).strip("\n") if lines else ""
    
    section_header_re = re.compile(r"^%%\s*=+\s*(.+?)\s*=+\s*$")
    section_alias_map = {
        "class definitions": "class_definitions",
        "type alias legend": "type_alias_legend",
        "spec": "spec",
        "interface & spec": "interface_spec",
        "loopcontrol (use only if needed)": "loop_control",
        "checks": "checks",
        "primitive actions": "primitive_actions",
        "data operation": "data_operation",
        "node class assignments": "node_class_assignments",
        "legend links": "legend_links",
        "legend link": "legend_links",
        "control-flow edges": "control_flow_edges",
    }
    section_buffers = {
        "class_definitions": [],
        "type_alias_legend": [],
        "spec": [],
        "interface_spec": [],
        "loop_control": [],
        "checks": [],
        "primitive_actions": [],
        "node_class_assignments": [],
        "legend_links": [],
        "control_flow_edges": [],
    }
    data_operation_buffers = {"d_init": [], "others": []}
    current_section_key = None
    d_init_pattern = re.compile(r"^\{?D_INIT(?:_NODE)?\b")

    for raw_line in mermaid_code.splitlines():
        stripped = raw_line.strip()
        header_match = section_header_re.match(stripped)
        if header_match:
            header_label = re.sub(r"\s+", " ", header_match.group(1)).strip().lower()
            current_section_key = section_alias_map.get(header_label)
            continue
        if not stripped or stripped.startswith("%%"):
            continue
        if current_section_key == "data_operation":
            target_list = data_operation_buffers["d_init"] if d_init_pattern.match(stripped) else data_operation_buffers["others"]
            target_list.append(raw_line.rstrip())
        elif current_section_key in section_buffers:
            section_buffers[current_section_key].append(raw_line.rstrip())
    previous_mermaid_components = {
                "class_definitions": _collapse_lines(section_buffers["class_definitions"]),
                "type_alias_legend": _collapse_lines(section_buffers["type_alias_legend"]),
                "spec": _collapse_lines(section_buffers["spec"]),
                "interface_spec": _collapse_lines(section_buffers["interface_spec"]),
                "loop_control": _collapse_lines(section_buffers["loop_control"]),
                "checks": _collapse_lines(section_buffers["checks"]),
                "primitive_actions": _collapse_lines(section_buffers["primitive_actions"]),
                "data_operations": {
                    "d_init_node": _collapse_lines(data_operation_buffers["d_init"]),
                    "other_nodes": _collapse_lines(data_operation_buffers["others"]),
                },
                "node_class_assignments": _collapse_lines(section_buffers["node_class_assignments"]),
                "legend_links": _collapse_lines(section_buffers["legend_links"]),
                "control_flow_edges": _collapse_lines(section_buffers["control_flow_edges"]),
            }
    return previous_mermaid_components


def analyze_example(example, llm, temperature=0.0, verbose=True):
    
    previous_obs = ""
    previous_action = ""

    current_facts = []

    example_with_facts = ""

    for item in example.split("\n"):
        if verbose:
            print(item)

        if item.startswith("obs:"):
            previous_obs += item + "\n"
            
            
            # print(current_facts)
            prompt = OBSERVATION_TO_PREDICATES_PROMPT.format(facts=current_facts, 
                                                             new_action=previous_action, new_observation=item.split("obs: ")[1].strip())

            response = llm_response(prompt, llm, temperature=temperature)

            # print(response)
            # exit(0)
            response_json = json.loads(repair_json(response))
            # print(response_json)

            current_facts += response_json["add_facts"]
            for fact in response_json["remove_facts"]:
                if fact in current_facts:
                    current_facts.remove(fact)
                else:
                    raise ValueError(f"Fact {fact} to be removed is not in current facts.")
            
            example_with_facts += item.strip() + "\n"
            example_with_facts += f"facts: {current_facts}\n"
        
        elif item.startswith("> "):
            previous_action = item.split("> ")[1].strip()
            previous_obs += item + "\n"

            example_with_facts += previous_action.strip() + "\n"

            if previous_action.startswith("think"):
                continue

            # action check
            line = previous_action.strip() 
            expr_fn = None

            for name, pattern, builder in ACTION_PATTERNS:

                match = pattern.match(line)

                # print(name, match)

                if match:
                    action_param = {"type": name}
                    action_param.update(builder(match.groupdict()))

                    # print(action_param)
                    expr_fn, pre_condition_msg = get_pre_condition(action_param)

                    # print(expr_fn)
                    if expr_fn is None:
                        raise ValueError(f"Pre-condition for action {action_param} is not defined.")
                    
                    break
                else:
                    if verbose:
                        print(f"match {name} failed...")
            if expr_fn is None:
                raise ValueError(f"Action String [{previous_action}] is not supported.")
            
            current_world = World(predicates=current_facts)
            pre_condition_value = executor.execute(expr_fn(DOMAIN), current_world).value

            if verbose:
                print(action_param, "pre-condition checking:", pre_condition_value)
            if not pre_condition_value:
                raise ValueError(f"Pre-condition for action {action_param} is not satisfied.")
            


    return example_with_facts

def formulate_example_with_fact(example_with_fact, reduce_dummy=False):

    print("example with facts: ")
    print(example_with_fact)

    str_list = example_with_fact.splitlines()
    formatted_example = []
    
    current_step = 0
    for ii in range(len(str_list) // 3):

        assert str_list[ii * 3].startswith("obs:")
        assert str_list[ii * 3 + 1].startswith("facts:")


        info_ii = {
            "step": ii,
            "obs": str_list[ii * 3],
            "fact": ast.literal_eval(str_list[ii * 3 + 1].split("facts:")[1]),
            "action": str_list[ii * 3 + 2],
        }
        if not isinstance(info_ii["fact"], list):
            info_ii["fact"] = []
            
        formatted_example.append(info_ii)
    # last step:
    ii += 1
    info_ii = {
        "step": ii,
        "obs": str_list[ii * 3],
        "fact": ast.literal_eval(str_list[ii * 3 + 1].split("facts:")[1]),
        "action": None,
    }
    
        
    formatted_example.append(info_ii)
    # print(formatted_example)
    # exit(0)
    
    return formatted_example

    
    
class NesyMemorySkillAgent(BaseAgent):
    """
    Neural-Symbolic Skill Grounding Agent for learning and executing grounded skills.
    """

    def __init__(self, example_dir='', llm='', temperature=0.0, log_folder='',
                load_skill=False, load_call=False, verbose=True, all_ready=False, online_mode=False,
                induction_mode: str = "ilp"):
        """
        Initialize the NesySkillGroundingAgent.
        
        Args:
            example_dir: Directory for examples
            llm: Language model name
            temperature: Temperature for LLM
            load_skill: Whether to load existing skills
        """
        super().__init__()
        
        # Initialize prompt templates
        self.prompt_template = ""
        self.grounding_template = ""
        self.skill_template = ""
        
        self.example_dir = example_dir

        
        self.llm = llm
        self.temperature = temperature
        self.name = "nesy-skill-evoluation-agent"
        self.interaction_history = ""

        self.grounded_skill_list = []
        self.grounded_skill_set = {}
        self.valid_grounded_skills_cnt = 0
        
        # Grounding specific attributes
        self.skill_grounding_examples = []
        self.grounding_memory = {}
        self.skill_executor = {}


        self.react_examples = {}
        self.grounding_examples = {}

        num_example_per_task = 2
        data_traj_collection = ""

        self.skill_graph = {}
        self.failed_think_msg = {}
        self.task_level_guidelines = {}
        
        self.load_skill = load_skill
        self.load_call = load_call
        self.verbose = verbose
        self.all_ready = all_ready
        self.online_mode = online_mode
        # Offline induction can use a single-pass path or the structural ILP path.
        # The two modes write to separate workflow directories.
        if induction_mode not in ("single", "ilp"):
            raise ValueError(f"induction_mode must be 'single' or 'ilp', got {induction_mode!r}")
        self.induction_mode = induction_mode
        if induction_mode == "single":
            warnings.warn(
                "induction_mode='single' is a pre-paper baseline: it makes a single LLM "
                "call per sub-goal with no two-stage structural ERM and does NOT implement "
                "the NSI method. The paper uses induction_mode='ilp'; results will not "
                "reproduce under 'single'.",
                stacklevel=2,
            )


        self.log_folder = log_folder
        grounding_dir = os.path.join(self.log_folder, "grounding_examples")
        os.makedirs(grounding_dir, exist_ok=True)
        
        for _, task_type in ENV_TYPES.items():
            cur_example_dir = os.path.join(self.example_dir, task_type)
            cur_grounding_dir = os.path.join(self.log_folder, "grounding_examples", task_type)
            os.makedirs(cur_grounding_dir, exist_ok=True)

            example_files = [f"example_{ii}.txt" for ii in range(num_example_per_task)]

            self.react_examples[task_type] = []
            self.grounding_examples[task_type] = []
            
            for example_file in sorted(example_files):
                file_path = os.path.join(cur_example_dir, example_file)
                with open(file_path, "r") as file:
                    data_traj_i = file.read()
                    self.react_examples[task_type].append(data_traj_i)

                grounding_file = os.path.join(cur_grounding_dir, example_file.replace(".txt", "_grounding.txt"))
                if not os.path.exists(grounding_file):
                    with open(grounding_file, "w") as file:
                        example_with_fact = analyze_example(data_traj_i, self.llm, temperature=self.temperature, verbose=False)
                        file.write(example_with_fact)
                        # print(example_with_fact)
                        print("save example_with_fact to file: ", grounding_file)
                else:
                    with open(grounding_file, "r") as file:
                        example_with_fact = file.read()
                    print("load example_with_fact to file: ", grounding_file)
                
                # print(example_with_fact)
                self.grounding_examples[task_type].append(example_with_fact)
                
                task_instruction_i = data_traj_i.strip().splitlines()[0].split("Your task is to: ")[1].strip(".")

                data_traj_collection += f"Task name: {task_instruction_i}\n" + data_traj_i + "\n\n"

        self.trajectory_memory_graph = TrajectoryMemoryGraph()
        self.skill_calling_memory_graph = SkillCallingMemoryGraph()

        traj_memory_file_path = os.path.join(self.log_folder, "trajectory_memory_graph.json")

        if os.path.exists(traj_memory_file_path):
            with open(traj_memory_file_path, "r") as file:
                trajectory_memory_graph_dict = json.load(file)
                self.trajectory_memory_graph = TrajectoryMemoryGraph.from_dict(trajectory_memory_graph_dict)
            print("load the trajectory memory graph from json format: ", traj_memory_file_path)
        else:

            # Generate the trajectory memory graph
            for task_type in ENV_TYPES.values():
                for example_with_fact_i in self.grounding_examples[task_type]:
                    # print(example_with_fact_i)
                    # exit(0)
                    example_with_fact_list_i = formulate_example_with_fact(example_with_fact_i, reduce_dummy=True)
                    self.update_trajectory_memory(example_with_fact_list_i, success_flag = True)
                # break
            # # save the trajectory memory graph in json format, you need firstly convert the graph to a dictionary-format
            trajectory_memory_graph_dict = self.trajectory_memory_graph.to_dict()
            with open(traj_memory_file_path, "w") as file:
                json.dump(trajectory_memory_graph_dict, file, indent=4)
            print("save the trajectory memory graph to json format: ", traj_memory_file_path)
        
        for start_node in self.trajectory_memory_graph.nodes:
            if self.trajectory_memory_graph.nodes[start_node].task_instruction.endswith("."):
                self.trajectory_memory_graph.nodes[start_node].task_instruction = self.trajectory_memory_graph.nodes[start_node].task_instruction.strip(".")
        # self.trajectory_memory_graph.show()
        # exit(0)
        

        # self.intent_clustering(self.trajectory_memory_graph, n_cluster=10)
        sub_goals = self.decompose_sub_goals(data_traj_collection, llm=self.llm, temperature=self.temperature, load=True)

        self.sub_goals = sub_goals


        # self.node_invention(self.trajectory_memory_graph)
        matching_log = []

        self.skill_json_dict = {}

        # ILP mode requires the skill_calling_memory_graph BEFORE induction so that
        # `_best_of` / `_hardest_trajectory` / `_verify_improvement` can score
        # candidate programs against expert trajectories. In single mode the graph
        # is built AFTER induction (legacy order, untouched).
        self._calling_graph_already_built = False
        if self.induction_mode == "ilp" and not self.load_skill:
            self._build_calling_graph_for_subgoals(sub_goals)
            self._calling_graph_already_built = True

        for sub_goal in sub_goals:

            # if sub_goal["name"] != "transfer_item_to_receptacle":
            #     continue

            if not self.load_skill:
                if self.induction_mode == "ilp":
                    json_file = self.skill_ILP(sub_goal)
                else:
                    json_file = self.induce_sub_goal_skills(sub_goal)
            else:
                # In load_skill mode pick the directory that matches the active induction mode.
                load_dir = "ilp_workflow" if self.induction_mode == "ilp" else "induced_workflow"
                json_file = os.path.join(self.log_folder, load_dir, f"{sub_goal['name']}_valid.json")
            
            # exit(0)

            if json_file is None or not os.path.exists(json_file):
                print("Can not induce a valid skill for sub-goal: ", sub_goal["name"])
                continue
            
            with open(json_file, "r") as f:
                skill_json = json.load(f)
            print("load skill from json file: ", json_file)
            
            graph_i = parse_mermaid_with_info(skill_json['mermaid_code'])

            # validate the dataflow of graph 
            node_errors = validate_node_inputs(graph_i)
            prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
            check_write_errors = validate_check_nodes_without_global_writes(graph_i)
            control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
            control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
            loop_entry_errors = validate_loop_entry_edges(graph_i)
            
            self.extract_node_edge_errors(
                node_errors,
                prefix_errors,
                check_write_errors,
                control_flow_errors,
                control_flow_node_errors,
                loop_entry_errors,
            )
            
            if node_errors or prefix_errors or check_write_errors or control_flow_errors or control_flow_node_errors or loop_entry_errors:
                print("The loaded skill is invalid with dataflow errors")
                
                self.skill_json_dict[sub_goal["name"]] = None
                continue
            else:
                print("The loaded skill is valid without dataflow errors.")

                self.skill_json_dict[sub_goal["name"]] = skill_json
        
        

        if not self.load_call:
            if getattr(self, "_calling_graph_already_built", False):
                # Already built in ILP-mode pre-induction step; just persist + show.
                print("[induction_mode=ilp] skill_calling_memory_graph was built before induction; skipping rebuild.")
                self.skill_calling_memory_graph.show()
                skill_calling_graph_dict = self.skill_calling_memory_graph.to_dict()
                file_path = os.path.join(self.log_folder, "skill_calling_memory_graph.json")
                with open(file_path, "w") as file:
                    json.dump(skill_calling_graph_dict, file, indent=4)
                print("save the skill calling memory graph to json format: ", file_path)
            else:
                for sub_goal in sub_goals:

                    for problem_i in self.trajectory_memory_graph.trajectory_head_list:

                        print(self.trajectory_memory_graph.nodes[problem_i].task_instruction)
                        print(sub_goal["applicable_tasks"])

                        instruction_i = self.trajectory_memory_graph.nodes[problem_i].task_instruction.strip(".")
                        if not instruction_i in sub_goal["applicable_tasks"]:
                            continue
                        else:
                            calling_summary = self.generate_calling_summary(sub_goal, start_node=problem_i)

                            # self.skill_calling_memory_graph.add_node(calling_node)
                            self.skill_calling_memory_graph.add_node_from_calling_summary(calling_summary)



                self.skill_calling_memory_graph.show()
                skill_calling_graph_dict = self.skill_calling_memory_graph.to_dict()
                file_path = os.path.join(self.log_folder, "skill_calling_memory_graph.json")
                with open(file_path, "w") as file:
                    json.dump(skill_calling_graph_dict, file, indent=4)
                print("save the skill calling memory graph to json format: ", file_path)
        else:
            file_path = os.path.join(self.log_folder, "skill_calling_memory_graph.json")
            with open(file_path, "r") as file:
                saved_skill_calling_graph = json.load(file)
            self.skill_calling_memory_graph = SkillCallingMemoryGraph.from_dict(saved_skill_calling_graph)
            self.skill_calling_memory_graph.show()

        # nodes_by_subgoal = {}
        # for node_id, node in self.skill_calling_memory_graph.nodes.items():
        #     print(node_id, node.subgoal_name)
        #     if node.subgoal_name not in nodes_by_subgoal:
        #         nodes_by_subgoal[node.subgoal_name] = []
        #     nodes_by_subgoal[node.subgoal_name].append(node_id)
        # self.skill_calling_memory_graph.nodes_by_subgoal = nodes_by_subgoal
        # with open(file_path, "w") as file:
        #     skill_calling_graph_dict = self.skill_calling_memory_graph.to_dict()
        #     json.dump(skill_calling_graph_dict, file, indent=4)
        
        # --- Online Evolution State (always initialized) ---
        self.online_evolved_dir = os.path.join(self.log_folder, "online_evolved")
        self.evolution_ledger: Dict = {}
        self.routing_rules: List[Dict] = []
        self._skill_execution_log: List[Dict] = []
        self.traj_path: Optional[str] = None
        self._episode_task_name: str = ""
        self._episode_env_type: str = ""

        if self.all_ready:

            skill_json_file = os.path.join(self.log_folder, "skill_json_file_for_run.json")
            with open(skill_json_file, "r") as f:
                self.skill_json_dict = json.load(f)
            print("load skill json file from ", skill_json_file)

            mermaid_dir_name = "ilp_workflow" if self.induction_mode == "ilp" else "induced_workflow"
            mermaid_dir = os.path.join(self.log_folder, mermaid_dir_name)
            mermaid_file_list = os.listdir(mermaid_dir)
            for file_ii in mermaid_file_list:
                if file_ii.endswith("_valid.mmd"):
                    skill_name = file_ii.split("_valid.")[0]
                    skill_json = self.skill_json_dict[skill_name]
                    if skill_json is None:
                        continue

                    # print(self.skill_json_dict[skill_name]["mermaid_code"])

                    self.skill_json_dict[skill_name]["mermaid_code"] = open(os.path.join(mermaid_dir, file_ii), "r").read()

                    # print(self.skill_json_dict[skill_name]["mermaid_code"])
                    # if "multi" in skill_name:
                    #     exit(0)


            skill_memory_blocks_with_task_type_file = os.path.join(self.log_folder, "skill_memory_blocks_with_task_type.json")
            with open(skill_memory_blocks_with_task_type_file, "r") as f:
                self.skill_memory_blocks_with_task_type = json.load(f)
            print("load skill memory blocks with task type file from ", skill_memory_blocks_with_task_type_file)

        
            revised_calling_memory_file= os.path.join(self.log_folder, "revised_calling_memory.json")
            with open(revised_calling_memory_file, "r") as f:
                skill_calling_graph_dict = json.load(f)
                self.skill_calling_memory_graph = SkillCallingMemoryGraph.from_dict(skill_calling_graph_dict)
            
            print("load revised calling memory file from ", revised_calling_memory_file)
            
            guideline_file = os.path.join(self.log_folder, "task_level_guidelines.json")
            with open(guideline_file, "r") as f:
                self.task_level_guidelines = json.load(f)
            print("load task level guidelines from ", guideline_file)
                
            
            print("Ready for online evaluation!!!")

            if self.online_mode:
                self._load_online_evolved_skills()

            return



        self.skill_success_records = {}
        log_list = []
        for subgoal_name, calling_node_ids in self.skill_calling_memory_graph.nodes_by_subgoal.items():
            
            # if subgoal_name != "examine_item_with_desklamp":
            # if subgoal_name != "place_multiple_items_in_receptacle": 
            #     continue
            
            if (subgoal_name in self.skill_json_dict) and (self.skill_json_dict[subgoal_name] != None):
                pass
            else:
                print("skip the subgoal {} because the skill is not valid.".format(subgoal_name))
                continue

            success_matching, fail_matching = self.deductive_evaluation(subgoal_name, calling_node_ids, verbose=True)
            success_rate = len(success_matching) / len(calling_node_ids)
            print(f"subgoal {subgoal_name} success rate: {success_rate:.3f}, ({len(success_matching)}/{len(calling_node_ids)})")
            log_list.append(f"subgoal {subgoal_name} \nsuccess rate: {success_rate:.3f}, ({len(success_matching)}/{len(calling_node_ids)})")
            

            print("Success Logs: ")
            success_raw_traj_id_list = []
            success_calling_id_list = []
            for calling_node_id, info in success_matching:
                # print(calling_node_id, info)
                self.skill_calling_memory_graph.nodes[calling_node_id].show()
                success_raw_traj_id_list.append(self.skill_calling_memory_graph.nodes[calling_node_id].problem_node)
                success_calling_id_list.append(calling_node_id)
            
            
            
            print("Failed Logs: ")
            for calling_node_id, info in fail_matching:
                # print(calling_node_id, info)
                self.skill_calling_memory_graph.nodes[calling_node_id].show()

            if success_rate >= 0.5 and success_rate < 1.0:
                
                for calling_node_id, info in fail_matching:
                    problem_node_id = self.skill_calling_memory_graph.nodes[calling_node_id].problem_node
                    
                    if problem_node_id in success_raw_traj_id_list:
                        self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name].remove(calling_node_id)
                        continue                    
                    # calling_summary = self.generate_calling_summary(self.skill_json_dict[subgoal_name], start_node=problem_node_id)
                    # print(calling_summary)
                    
                    success_flag, revised_calling_node_id = self.abductive_recorrect_calling_summary(subgoal_name, problem_node_id, 
                                                                               related_calling_list=success_calling_id_list, 
                                                                               verbose=False)
                    
                    
                    self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name].remove(calling_node_id)

                    if success_flag:
                        success_calling_id_list.append(revised_calling_node_id)
                        
                
                print(f"success rate before abduction: {success_rate:.3f}({len(success_matching)}/{len(success_matching) + len(fail_matching)})")
                new_success_rate = len(success_calling_id_list) / len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])
                print(f"success rate after abduction: {new_success_rate:.3f}"
                      f"({len(success_calling_id_list)}/{len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])})")
                # exit(0)

                log_list.append("After abduction, success rate: {:.3f}({}/{})".format(new_success_rate, len(success_calling_id_list), len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])))
            
            # elif success_rate < 0.5:
            #     # log_list.append("The success rate is too low, skip the abduction.")

            #     print("Revise the skill via induction")

            #     for calling_node_id, info in fail_matching:
            #         problem_node_id = self.skill_calling_memory_graph.nodes[calling_node_id].problem_node
                    
            #         if problem_node_id in success_raw_traj_id_list:
            #             self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name].remove(calling_node_id)
            #             continue                    
            #         # calling_summary = self.generate_calling_summary(self.skill_json_dict[subgoal_name], start_node=problem_node_id)
            #         # print(calling_summary)
                    
            #         gradient_json = self.collect_gradient_from_failure(subgoal_name, calling_node_id, problem_node_id, trace_info=info, verbose=False)


                



            self.skill_success_records[subgoal_name] = success_calling_id_list


                
            # exit(0)

        for log_i in log_list:
            print(log_i)
        
        
        
        
        self.skill_memory_blocks_with_task_type = {}
        self.prepare_parameter_bindings_guidelines()


        self.summarize_task_level_guidelines()
        
        self.induce_pre_condition()

        skill_json_file = os.path.join(self.log_folder, "skill_json_file_for_run.json")
        with open(skill_json_file, "w") as f:
            json.dump(self.skill_json_dict, f, indent=4)
        print("save skill json file to ", skill_json_file)

        skill_memory_blocks_with_task_type_file = os.path.join(self.log_folder, "skill_memory_blocks_with_task_type.json")
        with open(skill_memory_blocks_with_task_type_file, "w") as f:
            json.dump(self.skill_memory_blocks_with_task_type, f, indent=4)
        print("save skill memory blocks with task type file to ", skill_memory_blocks_with_task_type_file)

        revised_calling_memory_file= os.path.join(self.log_folder, "revised_calling_memory.json")
        with open(revised_calling_memory_file, "w") as f:
            skill_calling_graph_dict = self.skill_calling_memory_graph.to_dict()
            json.dump(skill_calling_graph_dict, f, indent=4)
        print("save revised calling memory file to ", revised_calling_memory_file)
        
        return 
        exit(0)
    
        # self.evolve_skill_with_calling_graph()
        
        




        exit(0)
    
    def summarize_task_level_guidelines(self):
        """
         Goal:
        - For each `task_type`, summarize:
            * executable sub-goals in a sequential order; 
            * for each skill
                - in which *phase* of the episode, the skills are usually invoked (self-contained summary for each skill);
                - typical pre-skill context:
                    1. key world facts,
                    2. recent intents,
                    3. skill initiation rationale;
                - typical post-skill outcome (facts / follow-up intents).
            * summarize "pure raw-action" patterns for sub-trajectories where skills are rarely or never used. 
                - in which *phase* of the episode, the pure raw-action are usually invoked (self-contained summary for each raw-action);
                - typical pre-action outcome 
                    1. key world facts,
                    2. recent intents,
                    3. action initiation rationale;
                - typical post-action outcome (facts / follow-up intents).  
        """

        print("summarize task level guidelines...")
        

        self.task_level_guidelines = {}
        
        for _, task_type in ENV_TYPES.items():

            if task_type in self.skill_memory_blocks_with_task_type and len(self.skill_memory_blocks_with_task_type[task_type]) > 0:
                
                print(task_type)
                skill_info = ""
                for sub_goal_i, skill_memory_block in self.skill_memory_blocks_with_task_type[task_type].items():

                    skill_json = self.skill_json_dict[sub_goal_i]

                    skill_expression = "## " + sub_goal_i + "(" +  ", ".join(skill_json["parameters"]) + ")"

                    skill_description = "### Description: {}\n".format(skill_json["description"])
                    start_conditions = "### Start Conditions: \n"
                    for cond_i in skill_json["start_conditions"]:
                        start_conditions += " -  {}\n".format(cond_i)
                    success_conditions = "### Success Conditions: \n"
                    for cond_i in skill_json["success_conditions"]:
                        success_conditions += " -  {}\n".format(cond_i)
                    
                    skill_info += skill_expression + "\n" + skill_description + "\n" + start_conditions + success_conditions
                    

                    for index, call_node_id in enumerate(skill_memory_block):
                        
                        calling_node = self.skill_calling_memory_graph.nodes[call_node_id]
                        record_str_i = "### Example {}, overall task goal: {}".format(index+1, self.trajectory_memory_graph.nodes[calling_node.problem_node].task_instruction) + "\n"
                        record_str_i += " - Problem Initial Description: {}".format(self.trajectory_memory_graph.nodes[calling_node.problem_node].observation_current.replace('obs: ', '')) + "\n"
                        record_str_i += " - Pre-skill Interaction Summary: " + calling_node.pre_span_summary + "\n"
                        record_str_i += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
                        record_str_i += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"

                        skill_info += "\n" + record_str_i + "\n"
                
                raw_traj = ""
                for example_i, example in enumerate(self.grounding_examples[task_type]):
                    raw_traj += f"Example {example_i+1}:\n{example}\n\n"
                
                prompt = TASK_LEVEL_SUMMARY_PROMPT.format(
                    task_type=task_type,
                    skill_info=skill_info,
                    raw_traj=raw_traj,
                )

                response = llm_response(prompt, self.llm, self.temperature, max_tokens=4096)

                if self.verbose:
                    print("-"*20 + "prompt" + "-"*20)
                    print(prompt)
                    print("-"*20 + "response" + "-"*20)
                    print(response)

                try:
                    summary_json = json.loads(repair_json(response))
                except Exception:
                    summary_json = {"task_type": task_type, "raw_response": response}

                self.task_level_guidelines[task_type] = summary_json

        if len(self.task_level_guidelines) > 0:
            guideline_file = os.path.join(self.log_folder, "task_level_guidelines.json")
            with open(guideline_file, "w") as f:
                json.dump(self.task_level_guidelines, f, indent=4)
            print("save task level guidelines to ", guideline_file)
        # exit(0)
    
    def collect_gradient_from_failure(self, subgoal_name, calling_node_id, problem_node_id, trace_info, verbose=False):
        
        
        calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]
        calling_node.show()


        failure_interaction = ""
        iter_node_id = calling_node.trajectory_start_node
        while iter_node_id != None:
            current_node = self.trajectory_memory_graph.nodes[iter_node_id]
            # current_node.show()
            failure_interaction += f"obs: {current_node.observation_current}\nworld states: {current_node.facts_current}\naction: {current_node.action}\n"
            iter_node_id = current_node.trajectory_next_node_id

        skill_json = self.skill_json_dict[subgoal_name]
        skill_call_dict = {
            "subgoal": subgoal_name, 
            "description": skill_json["description"],
            "steps": skill_json["steps"],
            "start_conditions": skill_json["start_conditions"],
            "start_condition_evidence": calling_node.start_condition_evidence, 
            "success_conditions": skill_json["success_conditions"],
            "pre_span_summary": calling_node.pre_span_summary, 
            "subgoal_initiation_intent": calling_node.subgoal_initiation_intent,
            "parameters": skill_json["parameters"],
            "parameter_bindings": calling_node.parameter_bindings, 
        }
        
        prompt = TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT.format(
            skill_schema=skill_call_dict,
            mermaid_workflow=self.skill_json_dict[subgoal_name]["mermaid_code"],
            failure_interaction=failure_interaction,
            execution_trace=trace_info,
        )

        response = llm_response(prompt, self.llm, self.temperature, max_tokens=8192)

        if self.verbose:
            print("-"*20 + "prompt" + "-"*20)
            print(prompt)
            print("-"*20 + "response" + "-"*20)
            print(response)



        exit(0)
        pass

    def prepare_parameter_bindings_guidelines(self):
        for subgoal_name, calling_node_ids in self.skill_calling_memory_graph.nodes_by_subgoal.items():
            
            # if subgoal_name != "locate_and_retrieve_item":
            #     continue
            
            if (subgoal_name in self.skill_json_dict) and (self.skill_json_dict[subgoal_name] != None):
                print(self.skill_json_dict[subgoal_name]["applicable_tasks"])
        
                print("successful calling nodes list: ", self.skill_success_records[subgoal_name])


                for calling_node_id in self.skill_success_records[subgoal_name]:
                    
                    # self.skill_calling_memory_graph.nodes[calling_node_id].show()

                    task_instruction = self.trajectory_memory_graph.nodes[self.skill_calling_memory_graph.nodes[calling_node_id].problem_node].task_instruction
                    task_type = None

                    # print("task instruction: ", task_instruction)
                    if "clean" in task_instruction:
                        task_type = "clean"
                    elif "heat" in task_instruction or "hot" in task_instruction:
                        task_type = "heat"
                    elif "cool" in task_instruction or "cold" in task_instruction:
                        task_type = "cool"
                    elif "put two" in task_instruction:
                        task_type = "puttwo"
                    elif "examine" in task_instruction or "look at" in task_instruction:
                        task_type = "examine"
                    elif "put some" in task_instruction or "find some" in task_instruction:
                        task_type = "put"
                    else:
                        raise Exception(f"Unknown task type: {task_instruction}")
                

                    # print("task instruction: ", task_instruction, ", task type: ", task_type)

                    if task_type not in self.skill_memory_blocks_with_task_type:
                        self.skill_memory_blocks_with_task_type[task_type] = {}

                    if subgoal_name not in self.skill_memory_blocks_with_task_type[task_type]:
                        self.skill_memory_blocks_with_task_type[task_type][subgoal_name] = []
                    
                    self.skill_memory_blocks_with_task_type[task_type][subgoal_name].append(calling_node_id)

                query_json = {
                    "skill_name": subgoal_name,
                    "description": self.skill_json_dict[subgoal_name]["description"],
                    "parameters": self.skill_json_dict[subgoal_name]["parameters"], 
                    "start_conditions": self.skill_json_dict[subgoal_name]["start_conditions"],
                    "success_conditions": self.skill_json_dict[subgoal_name]["success_conditions"],
                }

                example_record = ""
                for index, calling_node_id in enumerate(self.skill_success_records[subgoal_name]):
                    calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]

                    example_record += "## Record {}".format(index+1) + "\n"
                    # example_record += " - Problem Initial Description: {}".format(self.trajectory_memory_graph.nodes[calling_node.problem_node].observation_current.replace('obs: ', '')) + "\n"
                    example_record += " - Pre-skill Interaction: \n"

                    iter_node_id = calling_node.problem_node
                    while iter_node_id:
                        example_record += self.trajectory_memory_graph.nodes[iter_node_id].observation_current + "\n"

                        if iter_node_id != calling_node.trajectory_start_node:
                            example_record += "action: " + self.trajectory_memory_graph.nodes[iter_node_id].action + "\n"
                        else:
                            break
                        iter_node_id = self.trajectory_memory_graph.nodes[iter_node_id].trajectory_next_node_id
                    
                    example_record += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
                    example_record += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"
                
                # print(query_json)
                # print(example_record)

                prompt = SKILL_PARAMETER_ANALYSIS_PROMPT.format(
                    skill_schema=json.dumps(query_json, indent=4), 
                    success_records=example_record
                )
                response = llm_response(prompt, self.llm, temperature=self.temperature)

                # print("-" * 50 + "prompt" + "-" * 50)
                # print(prompt)
                # print("-" * 50 + "response" + "-" * 50)
                # print(response)

                json_response = json.loads(repair_json(response))
                self.skill_json_dict[subgoal_name]["parameter_bindings_guidelines"] = json_response


    def induce_pre_condition(self):
        
        for subgoal_name in self.skill_json_dict:
            if subgoal_name not in self.skill_json_dict or self.skill_json_dict[subgoal_name] is None:
                continue
            skill_json = self.skill_json_dict[subgoal_name]
            info_json = {
                "name": skill_json["name"], 
                "description": skill_json["description"],
                "parameters": skill_json["parameters"], 
                "start_conditions": skill_json["start_conditions"],
                "success_conditions": skill_json["success_conditions"],
                "steps": skill_json["steps"],
            }
            param_type_env = _build_skill_parameter_type_map(skill_json)

            record_str = ""
            for index, calling_node_id in enumerate(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name]):
                calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]
                traj_node_head_id = calling_node.trajectory_start_node

                record_str += "## Record {}".format(index+1) + "\n"
                record_str += " - Pre-skill Interaction: \n"

                iter_node_id = calling_node.problem_node
                while iter_node_id:
                    record_str += self.trajectory_memory_graph.nodes[iter_node_id].observation_current + "\n"

                    if iter_node_id != traj_node_head_id:
                        record_str += "action: " + self.trajectory_memory_graph.nodes[iter_node_id].action + "\n"
                    else:
                        break
                    iter_node_id = self.trajectory_memory_graph.nodes[iter_node_id].trajectory_next_node_id
                
                record_str += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
                record_str += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"
                record_str += " - Current World Facts: " + str(self.trajectory_memory_graph.nodes[traj_node_head_id].facts_current) + "\n"

            print(info_json)
            print(record_str)

            prompt = SKILL_PRECONDITION_SYNTH_PROMPT.format(skill_schema=json.dumps(info_json, indent=4), success_records=record_str, hints="")
            response = llm_response(prompt, self.llm, temperature=self.temperature)

            print("-" * 50 + "prompt" + "-" * 50)
            print(prompt)
            print("-" * 50 + "response" + "-" * 50)
            print(response)

            json_response = json.loads(repair_json(response))
            # expr = json_response["pre_condition_expression"]
            
            pre_cond_list = []
            for expr_i in json_response["pre_condition_expression"]:
                expr = expr_i.replace("{{", "{").replace("}}", "}")
                # expr = expr.replace("{", "\"{").replace("}", "}\"")
                print("expr: ", expr)
                pre_cond_list.append(expr)
            
            formated_param_bindings = {}
            for param_name, param_value in calling_node.parameter_bindings.items():
                if isinstance(param_value, str):
                    _param_value = "\""+ transform_item_name(param_value) + "\""
                elif isinstance(param_value, list):
                    _param_value = ["\""+ transform_item_name(item) + "\"" for item in param_value]
                formated_param_bindings[param_name] = _param_value
            

            all_pass_flag = True
            passed_cond_list = []
            hints = ""
            for cond_idx, pred_cond_expr_i in enumerate(pre_cond_list):
                error_node_id = []

                # print("pred_cond_expr_i: ", pred_cond_expr_i)
                # print(param_type_env)
                type_errors = _check_precondition_clause_types(pred_cond_expr_i, param_type_env)

                # print("type_errors: ", type_errors)
            
                # exit(0)
                if type_errors:
                    hints += f"- Type issues in precondition {pred_cond_expr_i}:\n"
                    for err in type_errors:
                        hints += f"    - {err}\n"
                    hints += "Please revise this clause to satisfy the skill parameter types.\n"
                    continue
                for index, calling_node_id in enumerate(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name]):
                    calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]
                    traj_node_head_id = calling_node.trajectory_start_node
                    current_facts = self.trajectory_memory_graph.nodes[traj_node_head_id].facts_current

                    params = formated_param_bindings
                    converted_expr = pred_cond_expr_i.format(**params)
                    # print("converted_expr: ", converted_expr)

                    current_world = World(predicates=current_facts)

                    env = _build_predicate_env(executor, current_world, current_facts)
                    condition_value = bool(eval(converted_expr, env, env))

                    # print("condition_value", condition_value)
                    if not condition_value:
                        error_node_id.append(calling_node_id)
                        hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                        hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                        hints += f"    - The instantiated pre_condition, {converted_expr}, does not hold.\n"
                    # exit(0)
                if error_node_id:
                    hints += f"Please revise the precondition {pred_cond_expr_i} to fit all the records.\n"
                    all_pass_flag = False
                else:
                    hints += f"- Precondition {pred_cond_expr_i} holds for all records.\n"
                    passed_cond_list.append(pred_cond_expr_i)
            

            for retry in range(5):

                if all_pass_flag:
                    break
                
                prompt = SKILL_PRECONDITION_SYNTH_PROMPT.format(skill_schema=json.dumps(info_json, indent=4), success_records=record_str, hints=hints)
                response = llm_response(prompt, self.llm, temperature=self.temperature)
                json_response = json.loads(repair_json(response))
                # expr = json_response["pre_condition_expression"]
                
                pre_cond_list = []
                for expr_i in json_response["pre_condition_expression"]:
                    expr = expr_i.replace("{{", "{").replace("}}", "}")
                    # expr = expr.replace("{", "\"{").replace("}", "}\"")
                    print("expr: ", expr)
                    pre_cond_list.append(expr)
                    
                all_pass_flag = True
                passed_cond_list = []
                for cond_idx, pred_cond_expr_i in enumerate(pre_cond_list):
                    error_node_id = []

                    # print("pred_cond_expr_i: ", pred_cond_expr_i)
                    # print(param_type_env)
                    type_errors = _check_precondition_clause_types(pred_cond_expr_i, param_type_env)

                    print("type_errors: ", type_errors)

                    # exit(0)
                    if type_errors:
                        hints += f"- Type issues in precondition {pred_cond_expr_i}:\n"
                        for err in type_errors:
                            hints += f"    - {err}\n"
                        hints += "Please revise this clause to satisfy the skill parameter types.\n"
                        continue
                    for index, calling_node_id in enumerate(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name]):
                        calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]
                        traj_node_head_id = calling_node.trajectory_start_node
                        current_facts = self.trajectory_memory_graph.nodes[traj_node_head_id].facts_current


                        params = formated_param_bindings
                        converted_expr = pred_cond_expr_i.format(**params)
                        # print("converted_expr: ", converted_expr)

                        current_world = World(predicates=current_facts)

                        env = _build_predicate_env(executor, current_world, current_facts)
                        condition_value = bool(eval(converted_expr, env, env))

                        # print("condition_value", condition_value)
                        if not condition_value:
                            error_node_id.append(calling_node_id)
                            hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                            hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                            hints += f"    - The instantiated pre_condition, {converted_expr}, does not hold.\n"
                        # exit(0)
                    if error_node_id:
                        hints += f"Please revise the precondition {pred_cond_expr_i} to fit all the records.\n"
                        all_pass_flag = False
                    else:
                        hints += f"- Precondition {pred_cond_expr_i} holds for all records.\n"
                        passed_cond_list.append(pred_cond_expr_i)
            if all_pass_flag:
                self.skill_json_dict[subgoal_name]["pre_condition_expression"] = pre_cond_list
            else:
                self.skill_json_dict[subgoal_name]["pre_condition_expression"] = passed_cond_list


        print("induced pre_condition_expression: ")
        for subgoal_name in self.skill_json_dict:
            print(subgoal_name)
            skill_json = self.skill_json_dict.get(subgoal_name)
            if skill_json is None:
                print("  (skill is None — induction failed validation; skipping precondition print)")
                continue
            for cond_expr in skill_json.get("pre_condition_expression", []):
                print(cond_expr)

        
        return






    
    def abductive_recorrect_calling_summary(self, subgoal_name, problem_node_id, related_calling_list=[], verbose=True):
        
        # print(self.skill_json_dict[subgoal_name])

        skill_info = {
            "name": subgoal_name, 
            "parameters": self.skill_json_dict[subgoal_name]["parameters"],
            "start_conditions": self.skill_json_dict[subgoal_name]["start_conditions"],
            "success_conditions": self.skill_json_dict[subgoal_name]["success_conditions"],
            "description": self.skill_json_dict[subgoal_name]["description"],
            "steps": self.skill_json_dict[subgoal_name]["steps"],
        }
        if verbose:
            print(skill_info)

        matched_examples_str = ""

        for idx, call_node_id in enumerate(related_calling_list):

            
            matched_examples_str += f"### Example {idx+1} \n"

            node = self.skill_calling_memory_graph.nodes[call_node_id]
            node.show()

            sub_traj = node.expand_raw_trajectory(self.trajectory_memory_graph)
            # print(sub_traj)

            related_calling_json = {
                "overall_task_goal": self.trajectory_memory_graph.nodes[node.problem_node].task_instruction,
                "pre_span_summary": node.pre_span_summary,
                "subgoal_initiation_intent": node.subgoal_initiation_intent,
                "detailed_subtrajectory": sub_traj,

                "parameter_bindings": dict(node.parameter_bindings), 
                "start_condition_evidence": list(node.start_condition_evidence), 
                "success_condition_evidence": list(node.success_condition_evidence)
            }

            # print("-"* 50)
            # print(json.dumps(related_calling_json, indent=4))

            matched_examples_str += json.dumps(related_calling_json, indent=4) + "\n"

        traj_info = []
        index_c = 0
        current_node = problem_node_id
        node_idx = []
        
        while current_node is not None:
            node_i = self.trajectory_memory_graph.nodes[current_node]
            node_idx.append(current_node)

            # print(node_i.observation_current)
            # print(node_i.important_predicates_for_action)
            # print(node_i.action)
            traj_info.append({
                "index": index_c,
                "observation": node_i.observation_current.split("obs: ")[1].strip(), 
                "action": node_i.action.strip() if node_i.action is not None else "",
            })

            index_c += 1
            current_node = node_i.trajectory_next_node_id

        
        prompt = TRAJ_SEGMENTATION_REVISION_PROMPT.format(
            subgoal_spec=json.dumps(skill_info, indent=4),
            matched_examples=matched_examples_str, 
            task_instruction=self.trajectory_memory_graph.nodes[problem_node_id].task_instruction,
            traj_info=json.dumps(traj_info, indent=4)
        )
        if verbose:
            print("-"* 50 + "prompt" + "-"* 50)
            print(prompt)

        response = llm_response(prompt, self.llm, temperature=self.temperature)

        if verbose:
            print("-"* 50 + "response" + "-"* 50)
            print(response)

        json_response = json.loads(repair_json(response))

        subtrajectory_steps = [] 
        for step_i in json_response["subtrajectory_steps"]:
            subtrajectory_steps.append({
                "node_id": node_idx[step_i["index"]],
                "why_relevant": step_i["why_relevant"],
            })

        function_calling_summary = {
            "subgoal": subgoal_name, 
            "problem_node": problem_node_id, 
            "trajectory_start_node": node_idx[json_response["subtrajectory_span"]["start_index"]],
            "trajectory_end_node": node_idx[json_response["subtrajectory_span"]["end_index"]],
            "subtrajectory_mapping": subtrajectory_steps, 
            "parameter_bindings": json_response["parameter_bindings"], 
            "start_condition_evidence": json_response["start_condition_evidence"],
            "success_condition_evidence": json_response["success_condition_evidence"],
            "pre_span_summary": json_response["pre_span_summary"],
            "subgoal_initiation_intent": json_response["subgoal_initiation_intent"],
        }

        new_skill_calling_node_id = self.skill_calling_memory_graph.add_node_from_calling_summary(function_calling_summary)
        match_flag, logs = self.static_verify_with_calling(self.skill_json_dict[subgoal_name], 
                                                           self.skill_calling_memory_graph.nodes[new_skill_calling_node_id],
                                                           verbose=verbose)
        
        if verbose:
            print("Abduction success:", match_flag)

        if match_flag:
            return True, new_skill_calling_node_id
        else:
            # exit(0)
            return False, new_skill_calling_node_id
        # exit(0)


            





    def deductive_evaluation(self, subgoal_name, node_ids, verbose=True):
        
        success_matching, fail_matching = [], []
        success_cnt = 0

        for calling_node_id in node_ids:
            node = self.skill_calling_memory_graph.nodes[calling_node_id]

            match_flag, logs = self.static_verify_with_calling(self.skill_json_dict[subgoal_name], node, verbose=verbose)

            
            self.skill_calling_memory_graph.nodes[calling_node_id].set_success_flag(match_flag)

            success_cnt += int(match_flag)
            print("match_flag:", match_flag)
            for key, value in logs:
                print(f"{key}: {value}")
            if not match_flag:
                pass
                # exit(0)
                pass
            else:
                pass

            # info = convert_log2info(logs)
            info = ""
            for key, value in logs:
                if key == "START":
                    if len(node.parameter_bindings):
                        value = value + ", GLOBAL INPUTS: "
                    for param, param_b in node.parameter_bindings.items():
                        value = value + f"{param.upper()}_INPUT = {param_b}"
                        value += ", " if param != list(node.parameter_bindings.keys())[-1] else ""
                
                if '[node id' in value:
                    if key.startswith("A_"):
                        # remove the [node id xxx] in the value
                        value = value.split("[node id")[0].strip()
                    else:
                        # extract the node id from the value string
                        obs_node_id = int(value.split("[node id ")[1].split("]")[0].strip())
                        self.trajectory_memory_graph.nodes[obs_node_id].show()

                        # retrive the least observation:
                        while self.trajectory_memory_graph.nodes[obs_node_id].trajectory_prev_node_id is not None:
                            if self.trajectory_memory_graph.nodes[obs_node_id].observation_current != "obs: OK.":
                                break
                            obs_node_id = self.trajectory_memory_graph.nodes[obs_node_id].trajectory_prev_node_id
                        # self.trajectory_memory_graph.nodes[obs_node_id].show()
                        # replace the [node id xxx] in value with the actual observation
                        value = value.replace("[node id {}]".format(obs_node_id), self.trajectory_memory_graph.nodes[obs_node_id].observation_current)
                
                info += f"{key}: {value}\n"

            print(info)    
            if match_flag:
                success_matching.append((calling_node_id, info))
            else:
                fail_matching.append((calling_node_id, info))
            
        print(f"success rate: {success_cnt/len(node_ids)}, ({success_cnt}/{len(node_ids)})")
        return success_matching, fail_matching

    def extract_node_edge_errors(
        self,
        node_errors,
        prefix_errors,
        check_write_errors=None,
        control_flow_errors=None,
        control_flow_node_errors=None,
        loop_entry_errors=None,
    ):
        check_write_errors = check_write_errors or []
        control_flow_errors = control_flow_errors or []
        control_flow_node_errors = control_flow_node_errors or []
        loop_entry_errors = loop_entry_errors or []
        loop_entry_errors = loop_entry_errors or []
        if node_errors:
            print("Errors found in node payloads:")
            for error in node_errors:
                print(error)
        else:
            print("All node payloads are valid.")
        if prefix_errors:
            print("Validation errors [missing predecessor global writes]:")
            for e in prefix_errors:
                print("  -", e)
        else:
            print("Validation passed: all required globals are written before each node.")
        if check_write_errors:
            print("Validation errors [check nodes must be read-only]:")
            for e in check_write_errors:
                print("  -", e)
        else:
            print("Check nodes correctly avoid GLOBAL writes.")
        if control_flow_errors:
            print("Validation errors [control-flow edges must follow Loop/Check/DataOp/Action rules]:")
            for e in control_flow_errors:
                print("  -", e)
        else:
            print("Control-flow edges meet the branching/continuation requirements.")
        if control_flow_node_errors:
            print("Validation errors [Control-Flow Edges reference undefined nodes]:")
            for e in control_flow_node_errors:
                print("  -", e)
        else:
            print("All Control-Flow edge endpoints correspond to defined nodes.")
        if loop_entry_errors:
            print("Validation errors [LoopControl incoming edges must use Start_Loop/Continue_Loop]:")
            for e in loop_entry_errors:
                print("  -", e)
        else:
            print("LoopControl incoming edge labels meet Start/Continue requirements.")

        node_payloads_hints = ""
        if node_errors:
            node_payloads_hints = "There are some errors in the node payloads of current workflow:\n"
            for e in node_errors:
                node_payloads_hints += f"  - {e}\n"
            
            d_init_flag = any("Node D_INIT (dataop)" in e for e in node_errors)
            if d_init_flag:
                node_payloads_hints += "Strictly copy the provided D_INIT node, and do not modify it in any way.\n"

            node_payloads_hints += "Each payload key must be a local input of the node.\n"
            node_payloads_hints += "Each payload value must be a pre-defined global variable or an local input of the node.\n"
            node_payloads_hints += "Each writes GLOBAL expression must reference only values provided via the node's local inputs; bind globals to locals before writing.\n"
            node_payloads_hints += "Local input names must be lowercase and must not collide with globals by casing (e.g., avoid target vs TARGET).\n"
            node_payloads_hints += "Please fix the errors in the node payloads via adjusting the workflow.\n"

        prefix_errors_hints = ""
        if prefix_errors:
            prefix_errors_hints = "Some local inputs have validation issues:\n"
            for e in prefix_errors:
                prefix_errors_hints += f"  - {e}\n"

            missing_globals = any("expects globals" in e for e in prefix_errors)
            type_mismatches = any("mismatched" in e for e in prefix_errors)
            missing_type_info = any("cannot be verified" in e for e in prefix_errors)
            d_init_flag = any("Node D_INIT (dataop)" in e for e in prefix_errors)
            
            if d_init_flag:
                prefix_errors_hints += "Strictly copy the provided D_INIT node, and do not modify it in any way.\n"


            if missing_globals:
                prefix_errors_hints += "Ensure every referenced global is written earlier in the workflow or provided by START before the consuming node runs.\n"
                prefix_errors_hints += "Add a DataOp (or similar) node before the usage to write these globals, or adjust the loop/node bindings to use available globals.\n"
            if type_mismatches:
                prefix_errors_hints += "Align local input types with the globals they bind to; update either the local declaration or the global definition so their type annotations match.\n"
            if missing_type_info:
                prefix_errors_hints += "Provide explicit type annotations for the globals involved so the validator can confirm compatibility with the local inputs.\n"
            prefix_errors_hints += "Please revise the workflow so that each required global exists with the correct type before the node that consumes it.\n"

        check_write_hints = ""
        if check_write_errors:
            check_write_hints = "Check nodes must remain read-only and cannot perform `writes global` assignments:\n"
            for e in check_write_errors:
                check_write_hints += f"  - {e}\n"

            d_init_flag = any("Node D_INIT (dataop)" in e for e in check_write_errors)
            if d_init_flag:
                check_write_hints += "Strictly copy the provided D_INIT node, and do not modify it in any way.\n"

            
            check_write_hints += "Move any needed GLOBAL updates into a DataOp or PrimitiveAction node before or after the check so the predicate only reads data.\n"

        control_flow_hints = ""
        if control_flow_errors:
            control_flow_hints = "Control-flow edges violate the branching/continuation constraints:\n"
            for e in control_flow_errors:
                control_flow_hints += f"  - {e}\n"
            control_flow_hints += (
                "Ensure LoopControl nodes expose both 'body' and 'done' edges, Check nodes branch via 'Yes' and 'No', "
                "and each DataOp/PrimitiveAction has exactly one outgoing edge.\n"
            )

        control_flow_node_hints = ""
        if control_flow_node_errors:
            control_flow_node_hints = "Some Control-Flow edges mention nodes that do not have definitions above the edge list:\n"
            for e in control_flow_node_errors:
                control_flow_node_hints += f"  - {e}\n"
            control_flow_node_hints += (
                "Add the missing node blocks (PrimitiveAction/Check/DataOp/etc.) in their respective sections before the "
                "Control-Flow Edges, or remove the stray edges.\n"
            )

        loop_entry_hints = ""
        if loop_entry_errors:
            loop_entry_hints = "LoopControl incoming edges must carry Start_Loop/Continue_Loop and Start_Loop must appear at least once:\n"
            for e in loop_entry_errors:
                loop_entry_hints += f"  - {e}\n"
            loop_entry_hints += (
                "Label every edge entering a LoopControl node with Start_Loop (for external entry/reset) or Continue_Loop (for in-loop back edges); "
                "ensure at least one Start_Loop edge targets each LoopControl; do not use these labels on edges targeting non-loop nodes.\n"
            )

        return (
            node_payloads_hints,
            prefix_errors_hints,
            check_write_hints,
            control_flow_hints,
            control_flow_node_hints,
            loop_entry_hints,
        )
            
    def evolve_skill_with_calling_graph(self):

        matching_log_list = []
        for subgoal_name, node_ids in self.skill_calling_memory_graph.nodes_by_subgoal.items():
            
            # if subgoal_name != "examine_item_with_desklamp":
            #     continue
            
            # print(self.skill_json_dict)
            if self.skill_json_dict[subgoal_name] is None:
                print("skip the subgoal {} because the skill is not valid.".format(subgoal_name))
                continue
            print("*" * 30)
            print("SUB_GOAL: ", subgoal_name)
            print("*" * 30)

            success_cnt = 0

            success_matching = []
            fail_matching = []

            

            def summarize_traj(skill_calling_node):
                traj_description = ""
                traj_description += f"  - Problem description: {self.trajectory_memory_graph.nodes[skill_calling_node.problem_node].observation_current.replace('obs: ', '')}\n"
                traj_description += f"  - Interaction summary before the skill calling: {skill_calling_node.pre_span_summary}\n"
                traj_description += f"  - Skill intent: {skill_calling_node.subgoal_initiation_intent}\n"
                traj_description += f"  - Interaction from expert: \n"
                iter_node_index = skill_calling_node.trajectory_start_node
                while iter_node_index != None:
                    
                    iter_node = self.trajectory_memory_graph.nodes[iter_node_index]
                    # iter_node.show()
                    traj_description += iter_node.observation_current + "\n"
                    traj_description += f"world state: {iter_node.facts_current}\n"
                    if iter_node.action:
                        traj_description += iter_node.action + "\n"
                    elif iter_node.trajectory_next_node_id is None:
                        traj_description += "The overall task goal has been achieved.\n"

                    if iter_node_index == skill_calling_node.trajectory_end_node:
                        break
                    else:
                        iter_node_index = iter_node.trajectory_next_node_id
                return traj_description

            data_collection = ""
            sub_traj_count = 0
            success_hints = ""
            if len(success_matching) > 0:
                success_hints = f"Here are {len(success_matching)} successful examples:\n"
            else:
                success_hints = f"All examples are failed.\n"
            for node_id, info in success_matching:
                self.skill_calling_memory_graph.nodes[node_id].show()
                skill_calling_node = self.skill_calling_memory_graph.nodes[node_id]

                traj_description = summarize_traj(skill_calling_node)
                data_collection += f"### Record {sub_traj_count+1}\n{traj_description}"
                sub_traj_count += 1

            failed_hints = ""
            if len(fail_matching) > 0:
                failed_hints = f"Here are {len(fail_matching)} failed examples:\n"
            else:
                failed_hints = f"All examples are successful.\n"

            for idx, (node_id, info) in enumerate(fail_matching):
                self.skill_calling_memory_graph.nodes[node_id].show()
                skill_calling_node = self.skill_calling_memory_graph.nodes[node_id]

                traj_description = summarize_traj(skill_calling_node)

                data_collection += f"### Record {sub_traj_count+1}\n{traj_description}"
                sub_traj_count += 1

                failed_hints += f"### Record {idx+1}\n"
                failed_hints += f"- Trajectory description:\n{traj_description}\n"
                failed_hints += f"- Workflow Running Log:\n{info}\n"

                
                # print(traj_description)
                # print("-" * 50)
                # print(info)

            print(failed_hints)

            # print(self.skill_json_dict[subgoal_name])
            # update mermaid code
            sub_goal_spec = {
                "name": subgoal_name,
                "parameters": self.skill_json_dict[subgoal_name]["param"], 
                "start_conditions": self.skill_json_dict[subgoal_name]["start_conditions"],
                "success_conditions": self.skill_json_dict[subgoal_name]["success_conditions"],
                "description": self.skill_json_dict[subgoal_name]["description"],
                "steps": self.skill_json_dict[subgoal_name]["steps"],
            }

            prompt = SUB_GOAL_ABSTRACT_EVOLUTION_PROMPT.format(
                DATA=data_collection, 
                previous_sub_goals=json.dumps(sub_goal_spec, indent=4)
            )
            # print("-" * 50 + "prompt" + "-"* 50)
            # print(prompt)
            response = llm_response(prompt, self.llm, self.temperature)

            # print(response)
            json_response = json.loads(repair_json(response))

            if json_response["major_revision"]:
                raise Exception(f"Major revision is required.\n{json.dumps(sub_goal_spec, indent=4)}")
            
            # exit(0)
            print("revise the sub-goal abstract: ")
            print(json.dumps(sub_goal_spec, indent=4))
            print("--->")
            print(json.dumps(json_response, indent=4))

            mermaid_file_dir = os.path.join(self.log_folder, "evolved_workflow")
            if not os.path.exists(mermaid_file_dir):
                os.makedirs(mermaid_file_dir)


            sub_goal_spec["start_conditions"] = json_response["start_conditions"]
            sub_goal_spec["success_conditions"] = json_response["success_conditions"]
            sub_goal_spec["description"] = json_response["description"]
            sub_goal_spec["steps"] = json_response["steps"]


            previous_mermaid_code = self.skill_json_dict[subgoal_name]["mermaid_code"]
            mermaid_file_path = os.path.join(mermaid_file_dir, f"{subgoal_name}_v0.mmd")
            with open(mermaid_file_path, "w") as f:
                f.write(previous_mermaid_code)

            # for key, value in previous_mermaid_components.items():
            #     print(f"{key}:\n{value}")

            json_response = self.generate_mermaid_code_with_calling_info(sub_goal_spec, 
                                                                          success_hints,
                                                                          failed_hints,
                                                                          previous_mermaid_code)
            
            version_count = 1
            
            current_mermaid_code = json_response["mermaid_code"]
            mermaid_file_path = os.path.join(mermaid_file_dir, f"{subgoal_name}_v{version_count}.mmd")
            with open(mermaid_file_path, "w") as f:
                f.write(current_mermaid_code)
            print(f"Mermaid code saved to {mermaid_file_path}")
            

            current_graph = parse_mermaid_with_info(current_mermaid_code)

            valid_file = None

            for error_try in range(10):
                
                node_errors = validate_node_inputs(current_graph)
                prefix_errors = validate_node_inputs_with_prefix_node(current_graph)
                check_write_errors = validate_check_nodes_without_global_writes(current_graph)
                control_flow_errors = validate_control_flow_outgoing_edges(current_graph)
                control_flow_node_errors = validate_control_flow_node_definitions(current_graph)
                loop_entry_errors = validate_loop_entry_edges(current_graph)
                (
                    node_payloads_hints,
                    prefix_errors_hints,
                    check_write_hints,
                    control_flow_hints,
                    control_flow_node_hints,
                    loop_entry_hints,
                ) = self.extract_node_edge_errors(
                    node_errors,
                    prefix_errors,
                    check_write_errors,
                    control_flow_errors,
                    control_flow_node_errors,
                    loop_entry_errors,
                )


                if (
                    len(node_errors)
                    + len(prefix_errors)
                    + len(check_write_errors)
                    + len(control_flow_errors)
                    + len(control_flow_node_errors)
                    + len(loop_entry_errors)
                    == 0
                ):
                    valid_file = mermaid_file_path
                    break
                
                json_response = self.generate_mermaid_code_with_calling_info(sub_goal_spec, 
                                                                          success_hints,
                                                                          failed_hints,
                                                                          previous_mermaid_code,
                                                                          syntax_hints=node_payloads_hints + prefix_errors_hints + check_write_hints + control_flow_hints + control_flow_node_hints
                                                                          )
                
                
                
                current_mermaid_code = json_response["mermaid_code"]

                version_count += 1
                mermaid_file_path = os.path.join(mermaid_file_dir, f"{subgoal_name}_v{version_count}.mmd")
                with open(mermaid_file_path, "w") as f:
                    f.write(current_mermaid_code)
                print(f"Mermaid code saved to {mermaid_file_path}")
            
                current_graph = parse_mermaid_with_info(current_mermaid_code)

            if valid_file is None:
                raise Exception(f"Failed to generate valid mermaid code for {subgoal_name} after {error_try} tries.")
            else:

                mermaid_file_path = os.path.join(mermaid_file_dir, f"{subgoal_name}_valid.mmd")
                with open(mermaid_file_path, "w") as f:
                    f.write(current_mermaid_code)
                print(f"Valid mermaid code saved to {mermaid_file_path}")
            exit(0)



        
        for match_log_i in matching_log_list:
            print(match_log_i)
        exit(0)
    
    def generate_mermaid_code_with_calling_info(self, sub_goal_spec, success_hints, failed_hints, previous_mermaid_code, syntax_hints=""):
        previous_mermaid_components = extract_nodes_from_mermaid_code(previous_mermaid_code)
        prompt = SUB_GOAL_WORKFLOW_EVOLUTION_PROMPT.format(
            sub_goal=json.dumps(sub_goal_spec, indent=4),
            matched_traj_segment=success_hints,
            unmatched_traj_segment=failed_hints, 
            CLASS_DEFINITIONS_NODE=previous_mermaid_components["class_definitions"],
            TYPE_ALIAS_NODE=previous_mermaid_components["type_alias_legend"], 
            FLOW_SPEC_NODE=previous_mermaid_components["spec"], 
            INTERFACE_NODE=previous_mermaid_components["interface_spec"],
            LOOP_FOR_NODE=previous_mermaid_components["loop_control"],
            CHECK_NODE=previous_mermaid_components["checks"],
            D_INIT_NODE= previous_mermaid_components["data_operations"]["d_init_node"],
            DATA_OP_NODE= previous_mermaid_components["data_operations"]["other_nodes"],
            CLASS_ASSIGNMENTS=previous_mermaid_components["node_class_assignments"], 
            LEGEND_EDGE=previous_mermaid_components["legend_links"], 
            CONTROL_FLOW_EDGE=previous_mermaid_components["control_flow_edges"],
            HINTS=syntax_hints,
        )
        print("-" * 50 + "prompt" + "-"* 50)
        print(prompt)

        response = llm_response(prompt, self.llm, self.temperature)
        json_response = json.loads(repair_json(response))

        if json_response["major_revision"]:
            raise Exception(f"Major revision is required.\n{json.dumps(sub_goal_spec, indent=4)}")
        
        print(json_response["mermaid_code"])
        print(json_response["change_summary"])

        
        
        return json_response
    
    def static_verify_with_calling(self, skill_json, function_calling_node, verbose=True):
        
        
        skill_graph = parse_mermaid_with_info(skill_json["mermaid_code"])

        if verbose:
            function_calling_node.show()
            
            iterative_node_index = function_calling_node.trajectory_start_node

            # print(iterative_node_index)
            while iterative_node_index is not None:
                node = self.trajectory_memory_graph.nodes[iterative_node_index]
                node.show()
                iterative_node_index = node.trajectory_next_node_id



        
        
        params_verify = {}

        current_traj_node_id = function_calling_node.trajectory_start_node
        params_verify["current_traj_node"] = self.trajectory_memory_graph.nodes[current_traj_node_id]

        for key, value in function_calling_node.parameter_bindings.items():
            if value == "":
                params_verify[f"{key.upper()}_INPUT"] = None
            else:
                params_verify[f"{key.upper()}_INPUT"] = value
        
        
        params_verify["world_builder"] = World
        params_verify["domain"] = DOMAIN
        params_verify["executor"] = executor

        if verbose:
            print("CONTEXT INPUT: ")
            print(params_verify)

        worlflow_trace, skill_success = self.reharsal_with_memory(skill_graph, ctx=params_verify, verbose=verbose)
        
        if not skill_success:
            print("Skill matching failed.")
            return False, worlflow_trace

        else:
            print("Skill matching successful!")
        
        # exit(0)
        
        
        return True, worlflow_trace

    def reharsal_with_memory(self, skill_workflow, ctx, verbose=True) -> None:
        """
        Reharsal the skill workflow with the given context.
        """
        _name_token_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

        def _build_render_context(local_args: Dict[str, object]) -> Dict[str, object]:
            merged: Dict[str, object] = {}

            def _add_source(src: Dict[str, object]) -> None:
                for key, value in src.items():
                    if not isinstance(key, str):
                        continue
                    if not _name_token_re.match(key):
                        continue
                    # for alias in _candidate_variants(key):
                    for alias in key:
                        merged[alias] = value
            _add_source(dict(skill_workflow.global_vars))
            _add_source({k: v for k, v in ctx.items() if isinstance(k, str)})
            _add_source(local_args)
            return merged

        def decide(expr: str, ctx: dict, local_args: dict) -> bool:
            # Normalize
            raw_expr = expr.strip()

            # Helper: evaluate quantifier/boolean expressions directly with Python env.
            def _eval_bool_with_env(expr_src: str) -> Tuple[bool, str]:
                expr_fmt = expr_src.replace("{{", "{").replace("}}", "}")
                expr_fmt = expr_fmt.replace("{", "\"{").replace("}", "}\"")
                # render_env = _build_render_context(local_args)
                rendered = expr_fmt.format(**local_args)
                converted_expr = convert_all_items(rendered)

                execu = ctx["executor"]
                facts = ctx["current_traj_node"].facts_current
                # print(facts)
                
                try:
                    current_world = ctx["world_builder"](predicates=facts)
                    env = _build_predicate_env(execu, current_world, facts)
                    condition_value = bool(eval(converted_expr, env, env))
                except Exception as e:
                    print(f"Error evaluating expression: {converted_expr}")
                    print(f"Exception: {e}")

                
                if verbose:
                    print(f"node id {ctx['current_traj_node'].id}, {ctx['current_traj_node'].observation_current}")
                    print(f"node id {ctx['current_traj_node'].id}, decide: {converted_expr}, value: {condition_value}")

                log_str = converted_expr.replace("lambda d: d.f_", "")
                return condition_value, f"CHECK {log_str}, the observation: [node id {ctx['current_traj_node'].id}], decision value: {condition_value}. "
            
            # print("here is a check node: ", raw_expr)

            if re.search(r"\b(exists|forall)\s*\(", raw_expr, flags=re.IGNORECASE):
                return _eval_bool_with_env(raw_expr)

            expr_q = raw_expr.replace("{{", "{").replace("}}", "}")
            expr_q = expr_q.replace("{", "\"{").replace("}", "}\"")
            expr_lambda = "lambda d: d.f_" + expr_q
            # render_env = _build_render_context(local_args)

            if verbose:
                print("raw_expr: ", raw_expr)
                print(expr_lambda)
            converted_expr = convert_all_items(expr_lambda.format(**local_args))
            eval_locals = dict(ctx)
            eval_locals.update(local_args)
            eval_locals.setdefault("ctx", ctx)

            if verbose:
                print("converted_expr: ", converted_expr)
            expr_fn = eval(converted_expr, globals(), eval_locals)

            facts = ctx["current_traj_node"].facts_current
            
            # print(facts)
            try:
                current_world = ctx["world_builder"](predicates=facts)
                condition_value = ctx["executor"].execute(expr_fn(ctx["domain"]), current_world).value
            except Exception as e:
                print(f"Error evaluating expression: {converted_expr}")
                print(f"Exception: {e}")
                
            # print(condition_value)
            
            if verbose:
                print(f"node id {ctx['current_traj_node'].id}, {ctx['current_traj_node'].observation_current}")
                print(f"node id {ctx['current_traj_node'].id}, decide: {converted_expr}, value: {condition_value}")

            log_str = converted_expr.replace("lambda d: d.f_", "")
            return condition_value, f"CHECK {log_str}, the observation: [node id {ctx['current_traj_node'].id}], decision value: {condition_value}."


        def act(act_name: str, args: str, ctx: dict, local_args: dict) -> None:
            
            # print("action node: ", act_name, args, ctx.keys())
            # print(ctx["current_traj_node"].action)
            while ctx["current_traj_node"].action.startswith("think"):
                
                if verbose:
                    print(f"node {ctx['current_traj_node'].id}, action: {ctx['current_traj_node'].action}")
                
                next_node_id = ctx["current_traj_node"].trajectory_next_node_id
                
                if next_node_id is None:
                    raise RuntimeError("No more steps in history")
                else:
                    ctx["current_traj_node"] = self.trajectory_memory_graph.nodes[next_node_id]
            
            try: 
                template = (args or "").strip()
                if not template and act_name:
                    template = act_name
                # render_env = _build_render_context(local_args)

                # print("local_args: ", local_args)
                
                expr_fmt = template.replace("{{", "{").replace("}}", "}")
                action_str = expr_fmt.format(**local_args)

                action_str = inverse_all_items(action_str)
                # print("action_str: ", action_str)
            except Exception as e:
                print(f"Error formatting action: {template}")
                print(f"Exception: {e}")
                raise e
            # print(template)
            # print(local_args)

            
            if ctx["current_traj_node"].action == action_str:
                # action matches, move to next step
                if verbose:
                    print(f"node id {ctx['current_traj_node'].id}, action matched: {action_str}")
                next_node_id = ctx["current_traj_node"].trajectory_next_node_id
                ctx["current_traj_node"] = self.trajectory_memory_graph.nodes[next_node_id]
                
                return f"Action matched: {action_str}. [node id {ctx['current_traj_node'].id}]"
            else:
                # print("action: ", action)
                # print("args: ", args)
                # print("local_args: ", local_args)
                # print(action.format(**local_args))
                
                # raise RuntimeError(f"Action `{action_str}` does not match history `{ctx['current_traj_node'].action}`")
                raise Action_Unmatched_Exception(f"Action match failed: workflow chose the `{action_str}`, and does not match expert history `{ctx['current_traj_node'].action}`. [node id {ctx['current_traj_node'].id}]")
            
            # No-op (you could log/update ctx)
            return f"ACTION {act_name}({action_str})"
        
        try:
            trace = traverse_with_info(skill_workflow, decide, act, ctx, verbose=verbose)

        except Exception as e:
            print(f"Skill execution failed. Error: {e}")
            # print the traceback of error e
            print(traceback.format_exc())
            return False, None

        if verbose:
            print("Trace of Workflow: ")
            for nid, info in trace:
                print(f"{nid}: {info}")
            
        
        if trace[-1][0] == "SUCCESS_END":
            return trace, True
        return trace, False


        



    def generate_calling_summary(self, sub_goal, start_node):
        
        # print(self.trajectory_memory_graph.nodes[start_node].task_instruction)
        # print(sub_goal)


        traj_info = []
        index_c = 0
        current_node = start_node
        node_idx = []
        
        while current_node is not None:
            node_i = self.trajectory_memory_graph.nodes[current_node]
            node_idx.append(current_node)

            # print(node_i.observation_current)
            # print(node_i.important_predicates_for_action)
            # print(node_i.action)
            traj_info.append({
                "index": index_c,
                "observation": node_i.observation_current.split("obs: ")[1].strip(), 
                "action": node_i.action.strip() if node_i.action is not None else "",
            })

            index_c += 1
            current_node = node_i.trajectory_next_node_id
        
        # traj_info.append({
        #         "index": index_c,
        #         "observation": "the overall task , {}, has been achieved.".format(self.trajectory_memory_graph.nodes[start_node].task_instruction), 
        #         "action": "",
        #     })
        # node_idx.append(None)
        
        # print(traj_info)

        
        prompt = TRAJ_SEGMENTATION_PROMPT.format(
            task_instruction=self.trajectory_memory_graph.nodes[start_node].task_instruction,
            subgoal_spec=json.dumps(sub_goal, indent=4),
            traj_info=json.dumps(traj_info, indent=4),
        )

        response = llm_response(prompt, self.llm, temperature=self.temperature)

        # print("="*20 + " prompt " + "="*20)
        # print(prompt)
        # print("="*20 + " response " + "="*20)
        # print(response)


        json_response = json.loads(repair_json(response))
        print(json.dumps(json_response, indent=4))

        subtrajectory_steps = [] 
        for step_i in json_response["subtrajectory_steps"]:
            subtrajectory_steps.append({
                "node_id": node_idx[step_i["index"]],
                "why_relevant": step_i["why_relevant"],
            })

        function_calling_summary = {
            "subgoal": sub_goal["name"], 
            "problem_node": start_node, 
            "trajectory_start_node": node_idx[json_response["subtrajectory_span"]["start_index"]],
            "trajectory_end_node": node_idx[json_response["subtrajectory_span"]["end_index"]],
            "subtrajectory_mapping": subtrajectory_steps, 
            "parameter_bindings": json_response["parameter_bindings"], 
            "start_condition_evidence": json_response["start_condition_evidence"],
            "success_condition_evidence": json_response["success_condition_evidence"],
            "pre_span_summary": json_response["pre_span_summary"],
            "subgoal_initiation_intent": json_response["subgoal_initiation_intent"],
        }
        return function_calling_summary

    def _ilp_induce_with_validation(self, sub_goal, data_traj, initial_hints="", previous_mermaid_code="", max_attempts=11, label=""):
        """Wrap `induce_skill` with the same validate-and-fix loop `induce_sub_goal_skills`
        uses. Each LLM call is followed by 6 dataflow validators; if any fail, errors
        are formatted via `extract_node_edge_errors` and fed back as `hints` to the
        next `induce_skill` call. Returns (induced_file, mermaid_code) on first all-clean
        candidate, or (induced_file, mermaid_code) of the last attempt if max_attempts
        exhausted without convergence.

        Without this validation loop, induced mermaids could pass `static_verify_with_calling`
        (which only checks empirical replay) yet fail load-time dataflow validators
        (Start_Loop edge labels, undefined globals, single-brace placeholders, etc.).
        """
        try:
            induced_file, mermaid_code = self.induce_skill(
                sub_goal, data_traj, self.llm,
                temperature=self.temperature,
                previous_mermaid_code=previous_mermaid_code,
                hints=initial_hints,
            )
        except Exception as e:
            print(f"[ilp][{label}] induce_skill crashed on first attempt: {e}")
            return None, None
        # Always normalize single-brace placeholders before validation (cheap, idempotent).
        mermaid_code = self._fix_single_brace_placeholders(mermaid_code)

        last_errors_summary = ""
        for attempt in range(max_attempts):
            try:
                graph_i = parse_mermaid_with_info(mermaid_code)
            except Exception as e:
                # Treat parse failure as a single "node error" so the LLM gets retry feedback.
                node_errors = [f"Parse failure: {e}"]
                prefix_errors, check_write_errors = [], []
                control_flow_errors, control_flow_node_errors, loop_entry_errors = [], [], []
            else:
                node_errors = validate_node_inputs(graph_i)
                prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
                check_write_errors = validate_check_nodes_without_global_writes(graph_i)
                control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
                control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
                loop_entry_errors = validate_loop_entry_edges(graph_i)

            total_errors = (
                len(node_errors) + len(prefix_errors) + len(check_write_errors)
                + len(control_flow_errors) + len(control_flow_node_errors) + len(loop_entry_errors)
            )
            if total_errors == 0:
                if attempt > 0:
                    print(f"[ilp][{label}] passed validators after {attempt} retry/retries.")
                return induced_file, mermaid_code

            # Format errors for the next prompt.
            (
                node_payloads_hints, prefix_errors_hints, check_write_hints,
                control_flow_hints, control_flow_node_hints, loop_entry_hints,
            ) = self.extract_node_edge_errors(
                node_errors, prefix_errors, check_write_errors,
                control_flow_errors, control_flow_node_errors, loop_entry_errors,
            )
            combined_hints = (
                initial_hints
                + "\n\n[Validator feedback from previous attempt — fix these in the next mermaid]\n"
                + node_payloads_hints + prefix_errors_hints + check_write_hints
                + control_flow_hints + control_flow_node_hints + loop_entry_hints
            )
            last_errors_summary = f"{total_errors} errors (node={len(node_errors)} prefix={len(prefix_errors)} check_write={len(check_write_errors)} ctrl={len(control_flow_errors)} ctrl_node={len(control_flow_node_errors)} loop={len(loop_entry_errors)})"
            print(f"[ilp][{label}] attempt {attempt+1} validators failed ({last_errors_summary}); retrying.")

            try:
                induced_file, mermaid_code = self.induce_skill(
                    sub_goal, data_traj, self.llm,
                    temperature=self.temperature,
                    previous_mermaid_code=mermaid_code,
                    hints=combined_hints,
                )
            except Exception as e:
                print(f"[ilp][{label}] induce_skill crashed on attempt {attempt+1}: {e}")
                # Return the last good (or last attempted) mermaid; caller decides.
                return induced_file, mermaid_code
            mermaid_code = self._fix_single_brace_placeholders(mermaid_code)

        print(f"[ilp][{label}] exhausted {max_attempts} attempts; final state has errors ({last_errors_summary}). Returning last candidate.")
        return induced_file, mermaid_code

    @staticmethod
    def _fix_single_brace_placeholders(text):
        """Normalize single-brace `{FOO}` placeholders to the double-brace `{{FOO}}` form
        the constructor's dataflow validators (validate_node_inputs etc.) require.

        This normalization keeps induced outputs compatible with the load-time validators.
        """
        if not text:
            return text
        _re = re.compile(r"(?<!\{)\{(?!\{)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})")
        return _re.sub(lambda m: "{{" + m.group(1).strip() + "}}", text)

    # ============================================================
    # Two-Stage Structural ERM (paper §5) — opt-in via induction_mode="ilp"
    # Mirrors textcraft_runs/.../nsms.py:skill_ILP. ALFWorld-specific
    # adaptations: (1) the LLM induction primitive is `induce_skill`
    # (single LLM call producing one mermaid), (2) verification uses
    # `static_verify_with_calling`, (3) outputs go to `ilp_workflow/`.
    #
    # The implementation synthesizes local candidates, validates them, and then
    # performs cross-trajectory generalization.
    # ============================================================

    def _build_calling_graph_for_subgoals(self, sub_goals):
        """Populate skill_calling_memory_graph from expert trajectories using sub_goal dicts.

        ILP mode runs this BEFORE skill induction so candidate programs can be
        scored against expert calls. Prefers a cached skill_calling_memory_graph.json
        if present (cheap, deterministic); otherwise calls generate_calling_summary
        per (sub_goal, expert_trajectory) pair and persists the result.
        """
        cache_path = os.path.join(self.log_folder, "skill_calling_memory_graph.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as fh:
                    cached = json.load(fh)
                self.skill_calling_memory_graph = SkillCallingMemoryGraph.from_dict(cached)
                print(f"[ilp] loaded cached skill_calling_memory_graph from {cache_path}")
                self.skill_calling_memory_graph.show()
                return
            except Exception as e:
                print(f"[ilp] cached calling graph at {cache_path} could not be loaded ({e}); rebuilding.")

        # Reset and rebuild — safer than appending into a possibly-half-built graph.
        self.skill_calling_memory_graph = SkillCallingMemoryGraph()
        for sub_goal in sub_goals:
            for problem_i in self.trajectory_memory_graph.trajectory_head_list:
                instruction_i = self.trajectory_memory_graph.nodes[problem_i].task_instruction.strip(".")
                if instruction_i not in sub_goal["applicable_tasks"]:
                    continue
                calling_summary = self.generate_calling_summary(sub_goal, start_node=problem_i)
                self.skill_calling_memory_graph.add_node_from_calling_summary(calling_summary)
        self.skill_calling_memory_graph.show()
        os.makedirs(self.log_folder, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(self.skill_calling_memory_graph.to_dict(), fh, indent=4)
        print(f"[ilp] saved pre-induction skill_calling_memory_graph -> {cache_path}")

    def skill_ILP(self, sub_goal, max_global_iters: int = 5):
        """Two-stage Structural ERM induction for a single sub_goal.

        Returns the path to a JSON file `{name}_valid.json` under `ilp_workflow/`,
        or None if no candidate produced a parseable program.
        """
        ilp_dir = os.path.join(self.log_folder, "ilp_workflow")
        os.makedirs(ilp_dir, exist_ok=True)
        per_traj_dir = os.path.join(ilp_dir, "trace2code")
        gen_dir = os.path.join(ilp_dir, "generalizer")
        os.makedirs(per_traj_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        sub_goal_name = sub_goal["name"]
        out_json_file = os.path.join(ilp_dir, f"{sub_goal_name}_valid.json")
        out_mmd_file = os.path.join(ilp_dir, f"{sub_goal_name}_valid.mmd")

        # If artifact already exists, treat as cached (mirrors TextCraft behavior).
        if os.path.exists(out_json_file):
            print(f"[ilp] cached ILP skill found, skipping induction: {out_json_file}")
            return out_json_file

        nodes_by_subgoal = getattr(self.skill_calling_memory_graph, "nodes_by_subgoal", {}) or {}
        calling_id_list = list(nodes_by_subgoal.get(sub_goal_name, []))
        if not calling_id_list:
            print(f"[ilp] no expert calling nodes for sub_goal {sub_goal_name!r}; skipping.")
            return None

        print("=" * 60)
        print(f"[ilp] skill_ILP for sub_goal: {sub_goal_name}  ({len(calling_id_list)} expert calls)")
        print("=" * 60)

        # ---- Stage 0: Warm-start candidate from all trajectories (Path A) ----
        warmstart_dir = os.path.join(ilp_dir, "warmstart")
        os.makedirs(warmstart_dir, exist_ok=True)
        sigma_local_pool: List[Dict] = []
        sigma_warmstart = self._ilp_warmstart_all_traces(sub_goal, warmstart_dir)
        if sigma_warmstart is not None:
            sigma_local_pool.append(sigma_warmstart)
            print(f"[ilp] Stage-0 warmstart added to candidate pool")

        # ---- Stage 1: Intra-Trajectory — one local program per expert call ----
        for traj_calling_id in calling_id_list:
            sigma_tau = self._ilp_trace2code(sub_goal, traj_calling_id, per_traj_dir)
            if sigma_tau is None:
                continue
            sigma_local_pool.append(sigma_tau)

        if not sigma_local_pool:
            print(f"[ilp] no local program could be synthesized for {sub_goal_name}.")
            return None

        # ---- Score each local program over ALL expert calls ----
        for sigma_tau in sigma_local_pool:
            self._ilp_score_program(sub_goal, sigma_tau, calling_id_list)

        # Pick the local program with highest cross-trajectory coverage as the seed global.
        best_idx = max(range(len(sigma_local_pool)),
                       key=lambda i: sigma_local_pool[i]["coverage_across_traj"])
        sigma_glb = deepcopy(sigma_local_pool[best_idx])
        for ss in sigma_local_pool:
            print(f"[ilp]   local from call {ss['source_traj']}: coverage={ss['coverage_across_traj']:.3f} succ_calls={ss['succ_call_idx']}")
        print(f"[ilp] seed global: coverage={sigma_glb['coverage_across_traj']:.3f} from call {sigma_glb['source_traj']}")

        # ---- Stage 2: Inter-Trajectory — hardest-tau driven generalization ----
        remaining = list(calling_id_list)
        for iter_idx in range(max_global_iters):
            if sigma_glb["coverage_across_traj"] >= 1.0 - 1e-9:
                print(f"[ilp] stage-2 converged: full coverage at iter={iter_idx}.")
                break

            tau_hard = None
            retrieve_sigma = None
            cand = next((c for c in remaining if c not in sigma_glb["succ_call_idx"]), None)
            if cand is None:
                # All remaining trajectories already covered by sigma_glb -> done.
                print(f"[ilp] stage-2: no uncovered trajectory remains at iter={iter_idx}.")
                break

            # Among local programs that DO cover cand, pick the one with the highest
            # global coverage (per paper §5.2: "donor with most leverage").
            donor = None
            donor_cov = -1.0
            for ss in sigma_local_pool:
                if cand in ss["succ_call_idx"] and ss["coverage_across_traj"] > donor_cov:
                    donor = ss
                    donor_cov = ss["coverage_across_traj"]
            tau_hard = cand
            retrieve_sigma = donor  # may be None — generalizer handles donor=None
            if donor is None:
                print(f"[ilp] stage-2: tau_hard={cand} has no donor; will expand sigma_glb directly from this trajectory.")

            sigma_prop = self._ilp_apply_generalizer(sub_goal, sigma_glb, tau_hard, retrieve_sigma, gen_dir, attempt_idx=iter_idx)
            if sigma_prop is None or sigma_prop.get("program") is None:
                # Generalizer failed to produce a parseable candidate; drop tau_hard and continue.
                if tau_hard in remaining:
                    remaining.remove(tau_hard)
                continue

            self._ilp_score_program(sub_goal, sigma_prop, calling_id_list)
            # Accept non-decreasing coverage so structural improvements with equal
            # empirical coverage remain eligible.
            if sigma_prop["coverage_across_traj"] >= sigma_glb["coverage_across_traj"]:
                prev = sigma_glb["coverage_across_traj"]
                sigma_glb = sigma_prop
                print(f"[ilp] iter={iter_idx} accepted candidate: coverage {prev:.3f} -> {sigma_glb['coverage_across_traj']:.3f}")
            else:
                print(f"[ilp] iter={iter_idx} rejected candidate: coverage {sigma_prop['coverage_across_traj']:.3f} < {sigma_glb['coverage_across_traj']:.3f}")
                if tau_hard in remaining:
                    remaining.remove(tau_hard)

        # ---- Persist final global program ----
        skill_json_out = deepcopy(sub_goal)
        skill_json_out["mermaid_code"] = sigma_glb["program"]
        skill_json_out["ilp_meta"] = {
            "final_coverage": sigma_glb["coverage_across_traj"],
            "source_traj": sigma_glb["source_traj"],
            "succ_call_idx": sigma_glb["succ_call_idx"],
            "n_expert_calls": len(calling_id_list),
            "n_local_pool": len(sigma_local_pool),
        }
        with open(out_json_file, "w") as f:
            json.dump(skill_json_out, f, indent=4)
        with open(out_mmd_file, "w") as f:
            f.write(sigma_glb["program"])
        print(f"[ilp] saved final ILP skill: {out_json_file}")
        print(f"[ilp]   final coverage = {sigma_glb['coverage_across_traj']:.3f}")
        return out_json_file

    def _ilp_trace2code(self, sub_goal, traj_calling_id, out_dir):
        """Stage-1: synthesize a local program for one expert calling node.

        Reuses ALFWorld's existing `induce_skill` (one LLM call -> one mermaid).
        The data fed to the LLM is restricted to this single calling-node's trace.
        """
        try:
            calling_node = self.skill_calling_memory_graph.nodes[traj_calling_id]
        except KeyError:
            print(f"[ilp][trace2code] calling id {traj_calling_id} missing; skip")
            return None

        # Build a single-trajectory data string in the same format induce_sub_goal_skills uses.
        traj_str = ""
        node_id = calling_node.trajectory_start_node
        seen_actions = set()
        while node_id is not None:
            tn = self.trajectory_memory_graph.nodes[node_id]
            traj_str += tn.observation_current.strip() + "\n"
            if tn.trajectory_next_node_id is not None:
                traj_str += f"important_predicates_for_action: {getattr(tn, 'important_predicates_for_action', '')}\n"
                if tn.action is not None:
                    traj_str += f"action: {tn.action.strip()}\n"
                    seen_actions.add(tn.action.strip().split()[0] if tn.action.strip() else "")
                if getattr(tn, "intent_for_action", None):
                    traj_str += f"intent_for_action: {tn.intent_for_action.strip()}\n"
            else:
                traj_str += "The Overall Task Success!\n"
            if node_id == calling_node.trajectory_end_node:
                break
            node_id = tn.trajectory_next_node_id

        per_traj_log = os.path.join(out_dir, f"{sub_goal['name']}_call{traj_calling_id}.txt")
        with open(per_traj_log, "w") as f:
            f.write(traj_str)

        # Use induce_skill but redirect default output via a temporary override of the
        # standard induced_workflow path: we copy the produced files into our per-traj dir.
        induced_file, mermaid_code = self._ilp_induce_with_validation(
            sub_goal, traj_str,
            initial_hints="",
            previous_mermaid_code="",
            label=f"trace2code:call{traj_calling_id}",
        )
        if mermaid_code is None:
            print(f"[ilp][trace2code] induce_skill (with validation) failed for call {traj_calling_id}")
            return None

        # Copy artifacts into ilp_workflow/trace2code/ for traceability.
        try:
            local_mmd = os.path.join(out_dir, f"{sub_goal['name']}_call{traj_calling_id}.mmd")
            with open(local_mmd, "w") as f:
                f.write(mermaid_code)
        except Exception as e:
            print(f"[ilp][trace2code] could not copy mermaid for call {traj_calling_id}: {e}")

        return {
            "program": mermaid_code,
            "program_file": induced_file,
            "source_traj": traj_calling_id,
            "coverage_across_traj": 0.0,   # filled by _ilp_score_program
            "succ_call_idx": [],
        }

    def _ilp_warmstart_all_traces(self, sub_goal, out_dir):
        """Stage-0 (Path A): warm-start global candidate using all expert trajectories
        at once. Mirrors single-mode `induce_sub_goal_skills` data formatting so the
        produced mermaid carries cross-trajectory inductive bias.
        """
        nodes_by_subgoal = getattr(self.skill_calling_memory_graph, "nodes_by_subgoal", {}) or {}
        calling_id_list = list(nodes_by_subgoal.get(sub_goal["name"], []))
        if not calling_id_list:
            return None

        sub_goal_data_collection = ""
        for problem in self.trajectory_memory_graph.trajectory_head_list:
            node_id = problem
            if self.trajectory_memory_graph.nodes[node_id].task_instruction.strip(".") not in sub_goal["applicable_tasks"]:
                continue
            traj_i = ""
            while node_id is not None:
                tn = self.trajectory_memory_graph.nodes[node_id]
                traj_i += tn.observation_current.strip() + "\n"
                if tn.trajectory_next_node_id is not None:
                    traj_i += f"important_predicates_for_action: {getattr(tn, 'important_predicates_for_action', '')}\n"
                    if tn.action is not None:
                        traj_i += f"action: {tn.action.strip()}\n"
                    if getattr(tn, "intent_for_action", None):
                        traj_i += f"intent_for_action: {tn.intent_for_action.strip()}\n"
                else:
                    traj_i += "The Overall Task Success!\n"
                node_id = tn.trajectory_next_node_id
            sub_goal_data_collection += traj_i + "\n\n"

        if not sub_goal_data_collection.strip():
            print(f"[ilp][warmstart] no applicable trajectories found for {sub_goal['name']}")
            return None

        log_path = os.path.join(out_dir, f"{sub_goal['name']}_alltraces.txt")
        with open(log_path, "w") as f:
            f.write(sub_goal_data_collection)

        induced_file, mermaid_code = self._ilp_induce_with_validation(
            sub_goal, sub_goal_data_collection,
            initial_hints="",
            previous_mermaid_code="",
            label="warmstart:all_traces",
        )
        if mermaid_code is None:
            print(f"[ilp][warmstart] induce failed for {sub_goal['name']}")
            return None

        try:
            local_mmd = os.path.join(out_dir, f"{sub_goal['name']}_warmstart.mmd")
            with open(local_mmd, "w") as f:
                f.write(mermaid_code)
        except Exception as e:
            print(f"[ilp][warmstart] could not persist warmstart mermaid: {e}")

        return {
            "program": mermaid_code,
            "program_file": induced_file,
            "source_traj": "warmstart",   # string sentinel so logs distinguish from int call ids
            "coverage_across_traj": 0.0,
            "succ_call_idx": [],
        }

    def _ilp_score_program(self, sub_goal, sigma, calling_id_list):
        """Run static_verify_with_calling for `sigma` over every expert calling node;
        write results back into sigma['coverage_across_traj'] and sigma['succ_call_idx'].
        """
        succ_calls: List[int] = []
        total = 0
        for cid in calling_id_list:
            try:
                calling_node = self.skill_calling_memory_graph.nodes[cid]
            except KeyError:
                continue
            skill_check = {
                "name": sub_goal["name"],
                "description": sub_goal.get("description", ""),
                "parameters": sub_goal.get("parameters", []),
                "start_conditions": sub_goal.get("start_conditions", []),
                "success_conditions": sub_goal.get("success_conditions", []),
                "steps": sub_goal.get("steps", []),
                "mermaid_code": sigma["program"],
            }
            try:
                ok, _ = self.static_verify_with_calling(skill_check, calling_node, verbose=False)
            except Exception as e:
                print(f"[ilp][score] static_verify crashed for call {cid}: {e}")
                ok = False
            total += 1
            if ok:
                succ_calls.append(cid)
        sigma["succ_call_idx"] = succ_calls
        sigma["coverage_across_traj"] = (len(succ_calls) / total) if total > 0 else 0.0

    def _ilp_apply_generalizer(self, sub_goal, sigma_glb, tau_hard, sigma_donor, out_dir, attempt_idx):
        """Stage-2 candidate synthesis: ask the LLM to merge the global program with
        a donor program that covers tau_hard, OR to expand the global directly from
        tau_hard when no donor exists.

        We feed both programs and the hard-trajectory record into
        ALFWorld's existing `induce_skill` via the `previous_mermaid_code` and `hints`
        slots (no new prompt yet). The English hints sketch the four structural
        operators (Branching / Crossover / Lifting / LoopFold) so the LLM has the
        same guidance as TextCraft's generalizer prompt provides.
        """
        try:
            calling_node_hard = self.skill_calling_memory_graph.nodes[tau_hard]
        except KeyError:
            return None

        donor_available = sigma_donor is not None

        # Build a "data_traj" packet: A (current global) + optional B (donor) + hard trajectory.
        success_record = "## Applicable Mermaid Workflow A (current global, covers {} calls)\n".format(
            len(sigma_glb.get("succ_call_idx", []))
        )
        success_record += sigma_glb["program"] + "\n\n"

        if donor_available:
            donor_record = "## Applicable Mermaid Workflow B (donor, covers {} calls including the hard one)\n".format(
                len(sigma_donor.get("succ_call_idx", []))
            )
            donor_record += sigma_donor["program"] + "\n\n"
        else:
            donor_record = (
                "## No Donor Available\n"
                "No existing local program covers the Hard Trajectory below. "
                "You must expand Workflow A directly so it covers the new trajectory "
                "without breaking the trajectories A already handles.\n\n"
            )

        # Hard trajectory expert trace
        hard_record = "## Hard Trajectory (call_id={}, currently NOT covered by Workflow A)\n".format(tau_hard)
        hard_record += f"  - Problem description: {self.trajectory_memory_graph.nodes[calling_node_hard.problem_node].observation_current.replace('obs: ', '')}\n"
        hard_record += f"  - Pre-skill summary: {calling_node_hard.pre_span_summary}\n"
        hard_record += f"  - Skill intent: {calling_node_hard.subgoal_initiation_intent}\n"
        hard_record += f"  - Parameter bindings: {json.dumps(calling_node_hard.parameter_bindings, indent=2)}\n"
        hard_record += "  - Expert interaction:\n"
        node_id = calling_node_hard.trajectory_start_node
        while node_id is not None:
            tn = self.trajectory_memory_graph.nodes[node_id]
            hard_record += tn.observation_current + "\n"
            hard_record += f"world state: {tn.facts_current}\n"
            if tn.action is not None:
                hard_record += tn.action + "\n"
            elif tn.trajectory_next_node_id is None:
                hard_record += "Task succeeded.\n"
            if node_id == calling_node_hard.trajectory_end_node:
                break
            node_id = tn.trajectory_next_node_id

        data_traj = success_record + donor_record + hard_record

        if donor_available:
            hints = (
                "[ILP Stage-2 Generalization Guidance — donor available]\n"
                "- Align the two candidate workflows above (A=current global, B=donor).\n"
                "- For divergent control paths, invent a CheckOp node with a discriminative predicate (Branching + Predicate Invention).\n"
                "- Reuse useful subgraphs from the donor by rebinding inputs to the current execution context (Crossover).\n"
                "- Lift hardcoded entities to type-based parameters when they vary across the trajectories (Lifting).\n"
                "- If you see repeated sub-structures, fold them into a LoopControl with a termination invariant (LoopFold).\n"
                "- The new program MUST keep covering all expert calls A already handles, and additionally cover the Hard Trajectory.\n"
            )
        else:
            hints = (
                "[ILP Stage-2 Generalization Guidance — donor MISSING]\n"
                "- Workflow A is the current global program; no donor program covers the Hard Trajectory.\n"
                "- Expand Workflow A by introducing new branches/loops/checks to also handle the Hard Trajectory.\n"
                "- For divergent control paths, invent a CheckOp node with a discriminative predicate (Branching + Predicate Invention).\n"
                "- Lift hardcoded entities to type-based parameters when they vary across the trajectories (Lifting).\n"
                "- If you see repeated sub-structures, fold them into a LoopControl with a termination invariant (LoopFold).\n"
                "- The new program MUST keep covering all expert calls A already handles AND additionally cover the Hard Trajectory.\n"
                "- Do NOT regress on the calls A already handles.\n"
            )

        induced_file, mermaid_code = self._ilp_induce_with_validation(
            sub_goal, data_traj,
            initial_hints=hints,
            previous_mermaid_code=sigma_glb["program"],
            label=f"generalizer:iter{attempt_idx}",
        )
        if mermaid_code is None:
            return None

        # Persist attempt for traceability.
        try:
            attempt_mmd = os.path.join(out_dir, f"{sub_goal['name']}_iter{attempt_idx}_call{tau_hard}.mmd")
            with open(attempt_mmd, "w") as f:
                f.write(mermaid_code or "")
        except Exception:
            pass

        if not mermaid_code or mermaid_code == "No Solution":
            return None

        return {
            "program": mermaid_code,
            "program_file": induced_file,
            "source_traj": tau_hard,
            "parent_global": sigma_glb.get("source_traj"),
            "donor": sigma_donor.get("source_traj") if donor_available else None,
            "coverage_across_traj": 0.0,
            "succ_call_idx": [],
        }

    def induce_sub_goal_skills(self, sub_goal):
        induced_skills = []
        induced_files_list = []
        json_file_list = []
        skill_names = []
    

        print(json.dumps(sub_goal, indent=4))

        sub_goal_data_collection = ""
        for problem in self.trajectory_memory_graph.trajectory_head_list:
            
            node_id = problem

            print(self.trajectory_memory_graph.nodes[node_id].task_instruction)
            print(sub_goal["applicable_tasks"])
            if self.trajectory_memory_graph.nodes[node_id].task_instruction.strip(".") not in sub_goal["applicable_tasks"]:
                continue
            
            traj_i = ""
            while node_id is not None:
                
                traj_i += self.trajectory_memory_graph.nodes[node_id].observation_current.strip() + "\n"
                
                if self.trajectory_memory_graph.nodes[node_id].trajectory_next_node_id is not None:  
                    traj_i += f"important_predicates_for_action: {self.trajectory_memory_graph.nodes[node_id].important_predicates_for_action}" + "\n"
                    traj_i += f"action: {self.trajectory_memory_graph.nodes[node_id].action.strip()}" + "\n"
                    traj_i += f"intent_for_action: {self.trajectory_memory_graph.nodes[node_id].intent_for_action.strip()}" + "\n"
                else:
                    traj_i += "The Overall Task Success!" + "\n"

                node_id = self.trajectory_memory_graph.nodes[node_id].trajectory_next_node_id
            
            sub_goal_data_collection += traj_i + "\n\n"
        
        # print(sub_goal_data_collection)

        raw_name = sub_goal["name"]
        sub_goal["name"] = f"{raw_name}_v0"

        induced_skill_file, mermaid_code_i = self.induce_skill(sub_goal, sub_goal_data_collection, self.llm, temperature=self.temperature)
        induced_files_list.append(induced_skill_file)
        

        # induced_skill_file = os.path.join(self.log_folder, "induced_workflow", 'find_and_take_item.mmd')
        # mermaid_code_i = open(induced_skill_file, "r").read()
        # print("load from ", induced_skill_file)
        
        print(sub_goal)
        print(mermaid_code_i)
        
        current_result_file = induced_skill_file

        graph_i = parse_mermaid_with_info(mermaid_code_i)

        result_file = None

        for error_try in range(11):
            
            node_errors = validate_node_inputs(graph_i)
            prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
            check_write_errors = validate_check_nodes_without_global_writes(graph_i)
            control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
            control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
            loop_entry_errors = validate_loop_entry_edges(graph_i)

            generation_file = current_result_file.replace(".mmd", f"_generation.json")
            with open(generation_file, "r") as f:
                generation_log = json.load(f)
            generation_log["node_errors"] = node_errors
            generation_log["prefix_errors"] = prefix_errors
            generation_log["check_write_errors"] = check_write_errors
            generation_log["control_flow_errors"] = control_flow_errors
            generation_log["control_flow_node_errors"] = control_flow_node_errors
            generation_log["loop_entry_errors"] = loop_entry_errors
            with open(generation_file, "w") as f:
                json.dump(generation_log, f, indent=4)
            print("update error for the mermaid generation logs >>>", generation_file)
            
            if error_try >= 10:
                break
            
            
            # exit(0)
            if (
                len(node_errors)
                + len(prefix_errors)
                + len(check_write_errors)
                + len(control_flow_errors)
                + len(control_flow_node_errors)
                + len(loop_entry_errors)
                == 0
            ):
                result_file = current_result_file
                break
            else:
                (
                    node_payloads_hints,
                    prefix_errors_hints,
                    check_write_hints,
                    control_flow_hints,
                    control_flow_node_hints,
                    loop_entry_hints,
                ) = self.extract_node_edge_errors(
                    node_errors,
                    prefix_errors,
                    check_write_errors,
                    control_flow_errors,
                    control_flow_node_errors,
                    loop_entry_errors,
                )
            
            
            sub_goal["name"] = f"{raw_name}_v{error_try + 1}"
            current_result_file, mermaid_code_i = self.induce_skill(
                sub_goal,
                sub_goal_data_collection,
                self.llm,
                temperature=self.temperature,
                previous_mermaid_code=mermaid_code_i,
                hints=node_payloads_hints + prefix_errors_hints + check_write_hints + control_flow_hints + control_flow_node_hints + loop_entry_hints,
            )
            
            graph_i = parse_mermaid_with_info(mermaid_code_i)

            print("save to ", current_result_file)
        
        
        if result_file is None:
            # fail after 3 times
            return None
        else:
            # # Derive param keys and simple types directly from START node
            sub_goal["param"] = extract_start_inputs(mermaid_code_i)

            sub_goal["mermaid_code"] = mermaid_code_i
            json_file = os.path.join(self.log_folder, "induced_workflow", f"{sub_goal['name']}.json")
            with open(json_file, "w") as file:
                json.dump(sub_goal, file, indent=4)
            print("save json >>>", json_file)

            # replace the "_v?" with "_valid"
            valid_mmd_name = os.path.join(self.log_folder, "induced_workflow", f"{sub_goal['name'].split('_v')[0]}_valid.mmd")
            with open(valid_mmd_name, "w") as file:
                file.write(mermaid_code_i)
            print("save mermaid >>>", valid_mmd_name)

            valid_name = os.path.join(self.log_folder, "induced_workflow", f"{sub_goal['name'].split('_v')[0]}_valid.json")
            sub_goal["name"] = raw_name
            with open(valid_name, "w") as file:
                json.dump(sub_goal, file, indent=4)
            print("save json >>>", valid_name)

            return valid_name
            
            


        exit(0)
        return skill_names
    def node_invention(self, graph: TrajectoryMemoryGraph):
        """
        Invent new nodes in the trajectory memory graph.
        
        Args:
            graph: Trajectory memory graph
        """

        action_node_collection = {}
        for action_type_i in RAW_ACTION_TYPES:

            node_list = []
            for node_id, node in graph.nodes.items():
                if node.action_type == action_type_i:
                    node_list.append(node)
            
            for node in node_list:
                print(f"node_id: {node.id}, action: {node.action}, intent: {node.intent_for_action}")

            action_node_collection[action_type_i] = node_list
        
        for problem_i in graph.trajectory_head_list:
            node_id = problem_i

            while node_id is not None:

                action_type = graph.nodes[node_id].action_type

                node_id = graph.nodes[node_id].next_node_id


        exit(0)
        
    def intent_clustering(self, graph, n_cluster=10):
        """
        Cluster the intent of the trajectory memory graph.
        
        Args:
            n_cluster: Number of clusters
        """

        embeddings, node_ids = [], []
        for node in graph.nodes.values():
            if node.intent_embedding:
                embeddings.append(node.intent_embedding)
                node_ids.append(node.id)

        embeddings = np.array(embeddings, dtype=np.float32)

        labels = KMeans(n_clusters=n_cluster, random_state=0).fit_predict(embeddings)
        clusters = {}
        for nid, label in zip(node_ids, labels):
            clusters.setdefault(int(label), []).append(int(nid))
        
        # There are m non-empty clusters
        m = len([cluster_id for cluster_id, nids in clusters.items() if nids])
        print(f"Number of non-empty clusters: {m}")
        
        for cluster_id, nids in clusters.items():
            print(f"Cluster {cluster_id} ({len(nids)} nodes): ")

            for nid in nids:
                print(f"    - {nid}: action = {graph.nodes[nid].action}, intent = {graph.nodes[nid].intent_for_action}, anonymized_intent = {graph.nodes[nid].intent_anonymized}")

        print(f"Number of non-empty clusters: {m}")
        # exit(0)


    def update_trajectory_memory(self, example_with_fact_list, success_flag=True):
        """
        Update the trajectory memory with a new example.
        
        Args:
            example_with_fact_list: Formatted example with facts
            success_flag: Whether the example was successful
        """
        
        
        if success_flag:
        
            query_instruction = example_with_fact_list[0]["obs"].split("Your task is to: ")[1]
            print(query_instruction)
            
            trajectory_wo_facts = ""
            pre_node_id = None
            for i in range(len(example_with_fact_list)):

                obs_i = example_with_fact_list[i]["obs"]
                fact_i = example_with_fact_list[i]["fact"]
                action_i = example_with_fact_list[i]["action"]

                remove_fact_i = []
                add_fact_i = []

                if i < len(example_with_fact_list) - 1:
                    obs_next = example_with_fact_list[i + 1]["obs"]
                    fact_next = example_with_fact_list[i + 1]["fact"]
                
                    for predicate_i in fact_i:
                        if predicate_i not in fact_next:
                            remove_fact_i.append(predicate_i)
                    for predicate_i in fact_next:
                        if predicate_i not in fact_i:
                            add_fact_i.append(predicate_i)

                # print(obs_i, fact_i, action_i, remove_fact_i, add_fact_i)
                
                # print(example_with_fact_list[i])

                trajectory_wo_facts += obs_i.strip() + "\n"

                node_kwargs = {
                    "task_instruction": query_instruction, 
                    "observation_current": obs_i, 
                    "facts_current": fact_i, 
                    "action": action_i, 
                    "action_remove_fact": remove_fact_i, 
                    "action_add_fact": add_fact_i, 
                }
                node_i = TrajectoryMemoryNode(
                    **node_kwargs,
                    verbose=False)
                
                
                if node_i.action_type in RAW_ACTION_TYPES:
                    prompt = MEMORY_CONSTRUCTION_PREDICATE_ANALYSIS_PROMPT.format(
                        interaction_history=trajectory_wo_facts,
                        facts_current=fact_i,
                        new_action=action_i)
                    important_predicates = llm_response(prompt, model=self.llm, temperature=self.temperature)
                    # print(important_predicates)
                    # important_predicates is a string of python list of string
                    important_predicates = eval(important_predicates)
                    for predicate_i in important_predicates:
                        if predicate_i not in fact_i:
                            important_predicates.remove(predicate_i)
                    node_i.important_predicates_for_action = important_predicates

                    prompt = MEMORY_CONSTRUCTION_INTENT_WITH_PREDICATE_PROMPT.format(
                    interaction_history=trajectory_wo_facts,
                    important_predicates_for_action=important_predicates,
                    new_action=action_i)
                    intent = llm_response(prompt, model=self.llm, temperature=self.temperature)
                    node_i.set_intent(intent)

                elif node_i.action_type == "think":
                    node_i.important_predicates_for_action = None
                    
                    prompt = MEMORY_CONSTRUCTION_INTENT_PROMPT.format(
                    interaction_history=trajectory_wo_facts,
                    new_action=action_i)
                    intent = llm_response(prompt, model=self.llm, temperature=self.temperature)
                    node_i.set_intent(intent)
                elif node_i.action is None:
                    intent = "The overall task goal has been achieved."
                    node_i.set_intent(intent)
                    
                # node_i.intent_embedding = generate_embeddings(text = node_i.intent_anonymized)
                node_i.intent_embedding = []
                node_id = self.trajectory_memory_graph.add_node(node_i)

                if i > 0:
                    self.trajectory_memory_graph.add_edge(pre_node_id, node_id, edge_type='trajectory')
                else:
                    self.trajectory_memory_graph.trajectory_head_list.append(node_id)

                pre_node_id = node_id

                if action_i:
                    trajectory_wo_facts += action_i.strip() + "\n"
                
            
        else:
            raise Exception("Failed examples not supported yet.")
        

    def decompose_sub_goals(self, data_traj_collection, llm, temperature=0.0, load=False):
        
        file_path = os.path.join(self.log_folder, "sub_goals.json")
        if load:
        
            try:
                
                with open(file_path, "r") as f:
                    sub_goals = json.loads(f.read())
                print(f"sub_goals loaded from {file_path}")
                return sub_goals
            except FileNotFoundError:
                print("sub_goals.json not found")
        
        prompt = SUBGOAL_PROMPT.format(examples=data_traj_collection)

        response = llm_response(prompt, llm, temperature=temperature)

        sub_goals = json.loads(repair_json(response))
        
        for sub_goal_i in sub_goals:
            print(json.dumps(sub_goal_i, indent=4))

        with open(file_path, "w") as f:
            f.write(json.dumps(sub_goals, indent=4))
        print(f"sub_goals saved to {file_path}")

        return sub_goals



    def induce_skill(self, sub_goal_i, data_traj, llm, temperature=0.0, previous_mermaid_code="", hints=""):
        
        induced_workflow_dir = os.path.join(self.log_folder, "induced_workflow")
        os.makedirs(induced_workflow_dir, exist_ok=True)

        def _sanitize_input_name(name: str) -> str:
            safe = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip()).upper()
            if not safe:
                safe = "INPUT"
            if not safe.endswith("_INPUT"):
                safe = f"{safe}_INPUT"
            return safe

        def _format_interface_block(parameters: List[str]) -> Tuple[str, str, str]:
            parsed_params: List[Tuple[str, str]] = []
            for param in parameters or []:
                if not isinstance(param, str):
                    continue
                if ':' not in param:
                    continue
                name_part, type_part = param.split(':', 1)
                name = name_part.strip()
                param_type = type_part.strip()
                if not name or not param_type:
                    continue
                parsed_params.append((name, param_type))

            if not parsed_params:
                start_inputs = "(none)"
                flow_inputs = ""
            else:
                interface_lines = [f"{_sanitize_input_name(name)}: {ptype}" for name, ptype in parsed_params]
                start_inputs = "<br>".join(interface_lines)
                flow_inputs = ", ".join(interface_lines)

            start_block = (
                "START([\"Interface: <br> Inputs: <br>"
                f"{start_inputs}\"]):::Interface"
            )
            if flow_inputs:
                flow_spec_block = (
                    "FLOW_SPEC{{\"Spec: <br>Flow Spec<br>inputs: {{"
                    f"{flow_inputs}"
                    "}}<br>outputs: {{SUCCESS_FLAG: Bool}}\"}}:::Spec"
                )
            else:
                flow_spec_block = (
                    "FLOW_SPEC{{\"Spec: <br>Flow Spec<br>inputs: {{}}<br>outputs: {{SUCCESS_FLAG: Bool}}\"}}:::Spec"
                )

            global_lines: List[str] = []
            local_in_parts: List[str] = []
            seen_globals: Set[str] = set()
            seen_locals: Set[str] = set()

            def _derive_local_name(global_name: str) -> str:
                base = re.sub(r"__+", "_", global_name.strip("_").lower())
                if not base:
                    base = "input"
                candidate = f"{base}_init"
                counter = 1
                while candidate in seen_locals:
                    counter += 1
                    candidate = f"{base}_init{counter}"
                seen_locals.add(candidate)
                return candidate

            for name, ptype in parsed_params:
                input_name = _sanitize_input_name(name)
                global_name = input_name[:-6] if input_name.endswith("_INPUT") else input_name
                local_name = _derive_local_name(global_name)

                if global_name not in seen_globals:
                    global_lines.append(f"{global_name}: {ptype}:={{{{{local_name}}}}}")
                    seen_globals.add(global_name)

                local_in_parts.append(f"{local_name}: {ptype} = {{{{{input_name}}}}}")

            default_globals = [
                ("SUCCESS_FLAG", "Bool", "False"),
            ]
            for g_name, g_type, g_value in default_globals:
                if g_name not in seen_globals:
                    global_lines.append(f"{g_name}: {g_type}:= {g_value}")
                    seen_globals.add(g_name)

            global_assignments = "<br>".join(global_lines) if global_lines else "SUCCESS_FLAG: Bool:=False"
            local_in_str = ", ".join(local_in_parts)

            d_init_block = (
                "D_INIT[\"DataOp: <br>writes GLOBAL: ("
                f"{global_assignments}"
                ")<br>local in: ("
                f"{local_in_str}"
                ")\"]:::DataOp"
            )

            return start_block, flow_spec_block, d_init_block

        start_block, flow_spec_block, d_init_block = _format_interface_block(sub_goal_i.get("parameters", []))
        
        if previous_mermaid_code == "":
            previous_mermaid_components = {
                "primitive_actions": "", 
                "loop_control" : """
    LOOP_FOR_LOCATIONS["LoopControl: <br>For receptacle_i in {{loop_receptacles}}<br>writes GLOBAL: (CURRENT_RECEPTACLE: ReceptacleName = receptacle_i)<br>local in: (loop_receptacles: List_ReceptacleName = {{RECEPTACLE_CANDIDATES}})"]:::LoopControl
""",
                "checks": "", 
                "data_operations": {
                    "other_nodes": "", 
                }, 
                "node_class_assignments" : """
    class START,SUCCESS_END,FAILURE_END Interface
    class LOOP_FOR_LOCATIONS LoopControl
    class A_GOTO,A_OPEN PrimitiveAction
    class C_IS_CLOSED,C_HAS_TYPE Check
""", 
                "control_flow_edges" : """
    %% Here is an example of iteration over `RECEPTACLE_CANDIDATES` to find a item with given `TARGET_TYPE`.
    %% Enter workflow and initialize globals
    START --> D_INIT
    %% Begin enumerating receptacle candidates to find a matching type
    D_INIT --> |Start_Loop| LOOP_FOR_LOCATIONS

    %% Loop body visits each candidate; done when exhausted
    LOOP_FOR_LOCATIONS --> |body| A_GOTO
    LOOP_FOR_LOCATIONS --> |done| FAILURE_END

    %% After going to candidate, check if closed
    A_GOTO --> C_IS_CLOSED
    %% If closed, open it
    C_IS_CLOSED -->|Yes| A_OPEN
    %% If already open, skip to check object type
    C_IS_CLOSED -->|No| C_HAS_TYPE
    %% After opening, proceed to check object type
    A_OPEN --> C_HAS_TYPE
    %% Found desired type, succeed
    C_HAS_TYPE -->|Yes| SUCCESS_END
    %% Not matching type: continue current loop iteration
    C_HAS_TYPE -->|No, Continue_Loop| LOOP_FOR_LOCATIONS""",   
            }
        else:
            previous_mermaid_components = extract_nodes_from_mermaid_code(previous_mermaid_code)

        prompt = SUB_GOAL_WORKFLOW_PROMPT.format(
            sub_goal=json.dumps(sub_goal_i, indent=4),
            examples=data_traj,
            FLOW_SPEC_NODE="    " + flow_spec_block,
            START_INTERFACE_NODE="    " + start_block,
            ACTION_NODE=previous_mermaid_components["primitive_actions"], 
            LOOP_FOR_NODE=previous_mermaid_components["loop_control"],
            CHECK_NODE=previous_mermaid_components["checks"],
            D_INIT_NODE= "    " + d_init_block,
            DATA_OP_NODE= previous_mermaid_components["data_operations"]["other_nodes"],
            CLASS_ASSIGNMENTS=previous_mermaid_components["node_class_assignments"], 
            CONTROL_FLOW_EDGE=previous_mermaid_components["control_flow_edges"],
            HINTS=hints
        )
        response = llm_response(prompt, llm, temperature=temperature)
        
        
        mermaid_code = response.strip()
        if mermaid_code.startswith("```mermaid"):
            mermaid_code = mermaid_code[len("```mermaid"):]
        if mermaid_code.endswith("```"):
            mermaid_code = mermaid_code[:-3]
        mermaid_code = mermaid_code.strip()
        
        print(mermaid_code)

        print("-"* 50 + "prompt" + "-"*50)
        print(prompt)
        
        print("-"* 50 + "response" + "-"*50)
        print(mermaid_code)
        print(sub_goal_i)
        

        induced_file = os.path.join(induced_workflow_dir, f"{sub_goal_i['name']}.mmd")
        induced_record_file = os.path.join(induced_workflow_dir, f"{sub_goal_i['name']}_generation.json")

        
        
        with open(induced_file, "w") as f:
            f.write(mermaid_code)
        print(f"induced workflow saved to {induced_file}")
        generation_json = {
            "prompt": prompt,
            "mermaid_file": induced_file, 
        }
        with open(induced_record_file, "w") as f:
            json.dump(generation_json, f, indent=4)
        print(f"induced record saved to {induced_record_file}")
        
        if mermaid_code == 'No Solution':
            print("No Solution")
            raise Exception(f"No Solution.\n LOG FILE: {induced_record_file}")
        # exit(0)

        return induced_file, mermaid_code



    def reset(self, env_type, env_id=None):
        """
        Reset the agent for a new environment and load relevant examples.

        Args:
            env_type: Type of environment (e.g., 'pick_cool_then_place', 'pick_clean_then_place', etc.)
            env_id: Optional episode identifier string
        """
        self.current_env_type = env_type
        self.reset_token_counts()
        self._skill_execution_log = []

        target_trace = ""
        
        # Build example trace from react examples

        # for example_i in self.grounding_examples[env_type]:
        #     target_trace += example_i + "\n\n"
        
        for example_i in self.react_examples[env_type]:
            target_trace += example_i + "\n\n"
            
        self.example_trace = target_trace
        self.interaction_history = ""

        self.current_facts = []
        
        self.previous_action = ""
        self.agent_log = ""
        self.preceived_flag = False

        skill_info = ""

        # for sub_goal_i, skill_i, mermaid_i, params_i in self.induce_skills:
        #     skill_info += f"{skill_i['expression']} # {skill_i['description']}, example: {skill_i['example']}\n"
        
        print(self.current_env_type)

        for sub_goal_i, calling_node_list in self.skill_memory_blocks_with_task_type[self.current_env_type].items():
            
            print(sub_goal_i, calling_node_list)

            if sub_goal_i in ["examine_item_with_desklamp", "place_multiple_items_in_receptacle"]:
                continue
            
            skill_json = self.skill_json_dict[sub_goal_i]

            skill_expression = sub_goal_i
            # print(skill_json)

            skill_expression = "## " + sub_goal_i + "(" +  ", ".join(skill_json["parameters"]) + ")"
            # print(skill_expression)

            skill_description = "### Description: {}\n".format(skill_json["description"])
            start_conditions = "### Start Conditions: \n"
            for cond_i in skill_json["start_conditions"]:
                start_conditions += " -  {}\n".format(cond_i)
            success_conditions = "### Success Conditions: \n"
            for cond_i in skill_json["success_conditions"]:
                success_conditions += " -  {}\n".format(cond_i)
            
            skill_info += skill_expression + "\n" + skill_description + "\n" + start_conditions + success_conditions
            

            for index, call_node_id in enumerate(calling_node_list):
                
                calling_node = self.skill_calling_memory_graph.nodes[call_node_id]
                record_str_i = "### Example {}, overall task goal: {}".format(index+1, self.trajectory_memory_graph.nodes[calling_node.problem_node].task_instruction) + "\n"
                record_str_i += " - Problem Initial Description: {}".format(self.trajectory_memory_graph.nodes[calling_node.problem_node].observation_current.replace('obs: ', '')) + "\n"
                record_str_i += " - Pre-skill Interaction Summary: " + calling_node.pre_span_summary + "\n"
                record_str_i += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
                record_str_i += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"

                skill_info += "\n" + record_str_i + "\n"

            skill_info += "### **Guidelines for your Parameter Bindings**: " + json.dumps(skill_json["parameter_bindings_guidelines"], indent=4) + "\n"
                
            # print(skill_info)
        

        # print(skill_info)
        # exit(0)
        self.skill_info_for_current_task = skill_info

        task_guideline = self.transform_task_guideline_to_text(self.task_level_guidelines[self.current_env_type])
        task_guideline_json = json.dumps(self.task_level_guidelines[self.current_env_type], indent=4)

        # print(task_guideline)
        # exit(0)
        
        self.current_task_guideline = task_guideline     
        self.current_task_guideline_json = task_guideline_json
               

        


        self.trajectory_memory_node_id = None
        self.current_query_instruction = None
    
    def transform_task_guideline_to_text(self, guideline_json):
        def format_list(title, items):
            lines = [title]
            for idx, val in enumerate(items, 1):
                lines.append(f"  {idx}. {val}")
            return "\n".join(lines) + "\n"

        def describe_scenarios(scenarios, include_records=False):
            lines = []
            for idx, sc in enumerate(scenarios, 1):
                phase = sc.get("phase", "")
                when_to_invoke = sc.get("when_to_invoke", [])
                pre_ctx = sc.get("typical_pre_context", {})
                post_ctx = sc.get("typical_post_outcome", {})
                record_indices = sc.get("record_indices", []) if include_records else []

                lines.append(f"  - Scenario {idx}: aligns to sub-task '{phase}' from the ordered list.")
                if when_to_invoke:
                    lines.append("    When this applies (triggers/cues):")
                    for trig in when_to_invoke:
                        lines.append(f"      - {trig}")
                lines.append("    Before starting (typical context):")
                for entry in pre_ctx.get("key_world_facts", []):
                    lines.append(f"      - world fact: {entry}")
                for entry in pre_ctx.get("recent_intents", []):
                    lines.append(f"      - recent intent: {entry}")
                init = pre_ctx.get("initiation_rationale")
                if init:
                    lines.append(f"      - initiation rationale: {init}")

                lines.append("    Expected after completion:")
                for entry in post_ctx.get("expected_state_change", []):
                    lines.append(f"      - state change: {entry}")
                for entry in post_ctx.get("follow_up_intents", []):
                    lines.append(f"      - follow-up intent: {entry}")

                if include_records and record_indices:
                    lines.append(f"    Evidence: supported by success record indices {record_indices}")
            return "\n".join(lines)

        # Build a readable task guideline
        task_guideline_parts = []
        task_guideline_parts.append("# Task Guideline")
        task_guideline_parts.append(format_list("## Sub-Task Decomposition (ordered)", guideline_json.get("ordered_subtasks", [])))

        task_guideline_parts.append("## Skill Section Guideline")
        for skill_entry in guideline_json.get("skill_guidelines", []):
            task_guideline_parts.append(f"- Skill: {skill_entry.get('skill', '')}")
            task_guideline_parts.append("  Scenarios: phase must match an ordered sub-task; triggers tell when to call; pre/post describe context and outcomes; record indices cite supporting examples.")
            scenarios = skill_entry.get("applicable_scenarios") or skill_entry.get("scenarios", [])
            task_guideline_parts.append(describe_scenarios(scenarios, include_records=True))

        task_guideline_parts.append("## Raw Action Guidelines")
        for action_entry in guideline_json.get("raw_action_guidelines", []):
            task_guideline_parts.append(f"- Raw Action: {action_entry.get('action', '')}")
            task_guideline_parts.append("  Scenarios: phase must match an ordered sub-task; triggers tell when to use the raw action; pre/post describe context and outcomes.")
            scenarios = action_entry.get("applicable_scenarios") or action_entry.get("scenarios", [])
            task_guideline_parts.append(describe_scenarios(scenarios, include_records=False))

        task_guideline = "\n".join(task_guideline_parts) + "\n"
        return task_guideline

    def _generate_grounding_prompt(self):
        """
        Generate prompt for skill grounding.
        """
        pass

    def _generate_execution_prompt(self):
        """
        Generate prompt for skill execution.
        """
        pass

    def perceive(self, obs, current_facts):

        obs_str = f"obs: {obs.strip()}\n"
        self.interaction_history += obs_str.strip() + "\n"

        prompt = OBSERVATION_TO_PREDICATES_PROMPT.format(facts=current_facts, 
                                                        new_action=self.previous_action, new_observation=obs)

        response = llm_response(prompt, self.llm, temperature=self.temperature)

        # print(response)
        response_json = json.loads(repair_json(response))
            # print(response_json)
        

        self.agent_log += "="*20 + "state prompt" + "="*20 + "\n{content}\n".format(content=prompt.split("\nFollow exactly this schema for each new observation.\n")[1])
        self.agent_log += "="*20 + "state response" + "="*20 + "\n{content}\n".format(content=response)

        current_facts += response_json["add_facts"]
        for fact in response_json["remove_facts"]:
            if fact in current_facts:
                current_facts.remove(fact)
            else:
                pass
                # raise ValueError(f"Fact {fact} to be removed is not in current facts.")
        # self.interaction_history += f"facts: {self.current_facts}\n"

        if self.current_query_instruction == None:
            self.current_query_instruction = obs_str.split("Your task is to: ")[1]
        
        node_kwargs = {
            "task_instruction": self.current_query_instruction, 
            "observation_current": obs_str, 
            "facts_current": current_facts, 
            "action": None, 
            "action_remove_fact": None, 
            "action_add_fact": None, 
        }
        new_node = TrajectoryMemoryNode(
            **node_kwargs,
            verbose=False)
        
        node_id = self.trajectory_memory_graph.add_node(new_node)
        if self.trajectory_memory_node_id == None:
            self.trajectory_memory_node_id = node_id
            self.current_trajectory_memory_problem_node_id = node_id
        else:
            self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action_remove_fact = response_json["remove_facts"]
            self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action_add_fact = response_json["add_facts"]

            self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].trajectory_next_node_id = node_id
            self.trajectory_memory_graph.nodes[node_id].trajectory_pre_node_id = self.trajectory_memory_node_id
            self.trajectory_memory_node_id = node_id

        return current_facts

    def act(self, obs):
        """
        Generate action based on observation.

        # self.skill_memory_blocks_with_task_type[task_type][subgoal_name].append(calling_node_id)
        
        """
        
        if self.preceived_flag:
            self.preceived_flag = False
            pass
        else:
            self.current_facts = self.perceive(obs, self.current_facts)
        

        precondition_error_msg, error_action = None, None
        cumulative_error_message = "At the current step, the following actions are not valid. \n"
        
        skill_info = self.skill_info_for_current_task
        task_guideline = self.current_task_guideline
        task_guideline_json = getattr(self, "current_task_guideline_json", json.dumps(self.task_level_guidelines.get(self.current_env_type, {}), indent=4))

            
        
        # exit(0)

        # Detect a "stuck on same skill" pattern. If the planner has called
        # the same skill 3+ times in a row (regardless of params) since the last raw
        # action, inject an anti-stuck warning into both router & execution prompts.
        # This is reflective-planning's "switch strategy" prompt-level cue, not skill mutation.
        if not hasattr(self, "_recent_skill_call_history"):
            self._recent_skill_call_history = []  # list of skill names in chronological order
        recent = self._recent_skill_call_history[-3:]
        if len(recent) >= 3 and len(set(recent)) == 1:
            stuck_skill = recent[0]
            repeat_warning = (
                f"WARNING: You have called the skill `{stuck_skill}` {len(recent)} times in a row "
                f"and it has not made progress on the overall task (no new key facts changed). "
                f"DO NOT pick `{stuck_skill}` again this step. Choose a different skill, OR a raw "
                f"action (such as 'go to <new receptacle>'), OR explicitly broaden parameters "
                f"(e.g., add more receptacles to searchLocations). Recovering from this loop is "
                f"more important than honoring the prior recommendation."
            )
        else:
            repeat_warning = "(no repeat-call issue detected)"

        for try_i in range(3):


            if try_i > 0:
                cumulative_error_message += f"action: {error_action}\n"
                cumulative_error_message += f"reason: {precondition_error_msg}\n"
                error_message = cumulative_error_message
            else:
                error_message = ""

            router_prompt = SKILL_ACTION_ROUTER_PROMPT.format(
                task_message=self.interaction_history,
                facts_current=json.dumps(self.current_facts, indent=2),
                task_guideline=task_guideline,
                routing_rules=self._get_routing_rules_text() if self.online_mode else "none",
                repeat_warning=repeat_warning,
            )

            router_response = llm_response(router_prompt, self.llm, temperature=self.temperature, stop_strs=[])

            if self.verbose:
                print("="*20 + "prompt" + "="*20)
                # print(prompt)
                try:
                    print(router_prompt.split("\n\n\n# Here is the task.")[1])
                except Exception:
                    print(router_prompt)
                print("="*20 + "response" + "="*20)
                print(router_response)

            try:
                router_json = json.loads(repair_json(router_response))
            except Exception:
                router_json = {
                    "active_subtask": "",
                    "subtask_reason": "",
                    "candidates": [],
                    "recommendation": {"type": "none", "name": "", "why": "router parsing failed"},
                    "missing_info": []
                }

            routing_recommendation = json.dumps(router_json, indent=4)

            prompt = EXECUTION_WITH_ROUTING_PROMPT.format(
                examples=self.example_trace,
                task_message=self.interaction_history,
                action_check_error = error_message,
                skill_info = skill_info,
                task_guideline = task_guideline,
                routing_recommendation = routing_recommendation,
                facts_current = json.dumps(self.current_facts, indent=2),
                repeat_warning=repeat_warning,
                )

            # print(prompt)
            # exit(0)
            response = llm_response(prompt, self.llm, temperature=self.temperature, stop_strs=[])
            
            if self.verbose:
                print("="*20 + "prompt" + "="*20)
                # print(prompt)
                try:
                    print(prompt.split("\n\n\n# Here is the task.")[1])
                except Exception:
                    print(prompt)
                print("="*20 + "response" + "="*20)
                print(response)

            # exit(0)

            self.agent_log += "="*20 + "router prompt" + "="*20 + "\n{content}\n".format(content=router_prompt)
            self.agent_log += "="*20 + "router response" + "="*20 + "\n{content}\n".format(content=router_response)

            try:
                prompt_content_for_log = prompt.split("\n\n\n# Here is the task.")[1]
            except Exception:
                prompt_content_for_log = prompt

            self.agent_log += "="*20 + "action prompt" + "="*20 + "\n{content}\n".format(content=prompt_content_for_log)
            self.agent_log += "="*20 + "action response" + "="*20 + "\n{content}\n".format(content=response)


            response_json = json.loads(repair_json(response))


            if response_json["action"]:

                self.previous_action = response_json["action"]

                self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action = response_json["action"].strip()

                # A raw action breaks any skill repeat streak.
                self._recent_skill_call_history.append("__RAW__")
                self._recent_skill_call_history = self._recent_skill_call_history[-10:]

                return response_json["action"], "textual action"
            
            # self.previous_action = response_json
            # return response_json, "skill action"

            # if action.startswith("think"):
            #     break
            # precondition_satisfied, precondition_error_msg, action_param = check_action_precondition(action, self.current_facts, self.induce_skills)

            precondition_expression_list = self.skill_json_dict[response_json["skill"]]["pre_condition_expression"]
            
            formated_param_bindings = {}
            for param_name, param_value in response_json["parameter_bindings"].items():
                if isinstance(param_value, str):
                    _param_value = "\""+ transform_item_name(param_value) + "\""
                elif isinstance(param_value, list):
                    _param_value = ["\""+ transform_item_name(item) + "\"" for item in param_value]
                formated_param_bindings[param_name] = _param_value
            
                    
            precondition_satisfied = True
            precondition_error_msg = ""
            for precondition_expression in precondition_expression_list:
                params = formated_param_bindings
                converted_expr = precondition_expression.format(**params)
                
                print("convert expr: ", precondition_expression, " --> ", converted_expr)

                current_world = World(predicates=self.current_facts)

                env = _build_predicate_env(executor, current_world, self.current_facts)
                condition_value = bool(eval(converted_expr, env, env))

                if not condition_value:
                    precondition_satisfied = False
                    precondition_error_msg += f"Precondition {converted_expr} is not satisfied.\n"
            
            if precondition_satisfied:
                self.previous_action = response_json
                # Record skill choice in history for repeat detection.
                self._recent_skill_call_history.append(response_json.get("skill") or "__UNKNOWN__")
                self._recent_skill_call_history = self._recent_skill_call_history[-10:]
                return response_json, "skill action"
            else:
                error_message, error_action = precondition_error_msg, response_json

        self.previous_action = response_json
        # Record skill choice even on a precondition-failed final fallback.
        self._recent_skill_call_history.append(response_json.get("skill") or "__UNKNOWN__")
        self._recent_skill_call_history = self._recent_skill_call_history[-10:]
        return response_json, "skill action"
        # self.previous_action = action
        # return action, "textual action"

    def update_skill_grounding(self, skill_name, execution_history, success_flag):
        """
        Update skill grounding based on execution feedback.
        """
        pass

    def execute_grounded_skill(self, skill_name, parameters):
        """
        Execute a grounded skill with given parameters.
        """
        pass

    def execute_skill_online(self, function_calling_json, env, process_ob):

        skill_str = str(function_calling_json["skill"]) + "(" + ", ".join([f"{s_i[0]}={s_i[1]}" for s_i in function_calling_json["parameter_bindings"].items()]) + ")"

        pre_skill_facts = list(self.current_facts)  # snapshot for online evolution logging

        def _log_skill_exec(success, diagnostic="", trace_nodes=None, interaction_list=None):
            """Record skill execution outcome for online evolution."""
            self._skill_execution_log.append({
                "skill_name": function_calling_json["skill"],
                "parameter_bindings": function_calling_json.get("parameter_bindings", {}),
                "success": success,
                "diagnostic": diagnostic,
                "trace_nodes": trace_nodes or [],
                "interaction_history": list(interaction_list or []),
                "pre_facts": pre_skill_facts,
                "post_facts": list(self.current_facts),
            })

        self.update_history(">> "+skill_str) 

        skill_workflow = parse_mermaid_with_info(self.skill_json_dict[function_calling_json["skill"]]["mermaid_code"])
        
        if self.verbose:
            print(function_calling_json)
        
        params_verify = {}

        params_verify["online_env"] = env

        for key, value in function_calling_json["parameter_bindings"].items():
            if value == "":
                params_verify[f"{key.upper()}_INPUT"] = None
            else:
                params_verify[f"{key.upper()}_INPUT"] = value
        
        
        params_verify["world_builder"] = World
        params_verify["domain"] = DOMAIN
        params_verify["executor"] = executor
        params_verify["process_ob"] = process_ob
        params_verify["online_env"] = env
        params_verify["current_facts"] = self.current_facts
        params_verify["skill_interaction"] = []
        params_verify["task_success_flag"] = False
        params_verify["_skill_name"] = function_calling_json["skill"]


        if self.verbose:
            print("CONTEXT INPUT: ")
            print(params_verify)

        calling_start_node_id = self.trajectory_memory_node_id

        interaction_info, skill_success, env_done_flag, task_success_flag = self.execute_skill_workflow(skill_workflow, ctx=params_verify, verbose=self.verbose)

        interaction_history_list, execution_trace_list = interaction_info
        calling_end_node_id = self.trajectory_memory_node_id

        # --- Online Evolution: Log skill execution + Graft lifecycle tracking ---
        trace = execution_trace_list
        skill_done = skill_success

        if skill_done:
            _log_skill_exec(True, diagnostic="", trace_nodes=[n for n, _ in trace] if trace else [],
                            interaction_list=interaction_history_list)
        else:
            diag_msg = ""
            if trace:
                diag_msg = f"Skill reached {trace[-1][0] if trace else 'unknown'}"
            _log_skill_exec(False, diagnostic=diag_msg,
                            trace_nodes=[n for n, _ in trace] if trace else [],
                            interaction_list=interaction_history_list)

        if self.online_mode and self.evolution_ledger and trace:
            skill_name = function_calling_json["skill"]
            visited_nodes = {n for n, _ in trace}
            for graft_id, graft_info in list(self.evolution_ledger.items()):
                if graft_info.get("skill_name") != skill_name:
                    continue
                if graft_info.get("status") != "tentative":
                    continue
                graft_nodes = set(graft_info.get("graft_node_ids", []))
                if not graft_nodes.intersection(visited_nodes):
                    continue
                # Graft was exercised
                if skill_done:
                    graft_info["success_count"] = graft_info.get("success_count", 0) + 1
                    if graft_info["success_count"] >= 1:
                        graft_info["status"] = "solidified"
                        print(f"[ONLINE] Graft {graft_id} solidified after {graft_info['success_count']} successes")
                else:
                    graft_info["fail_count"] = graft_info.get("fail_count", 0) + 1
                    if graft_info["fail_count"] >= 1:
                        print(f"[ONLINE] Discarding graft {graft_id} after {graft_info['fail_count']} failures")
                        self._discard_graft(graft_id)
                    else:
                        self._save_online_skills()

        iter_node_id = self.current_trajectory_memory_problem_node_id

        function_calling_summary = {
        "subgoal": function_calling_json["skill"], 
        "problem_node": self.current_trajectory_memory_problem_node_id, 
        "trajectory_start_node": calling_start_node_id,
        "trajectory_end_node": calling_end_node_id,
        "subtrajectory_mapping": [], 
        "parameter_bindings": function_calling_json["parameter_bindings"], 
        "start_condition_evidence": function_calling_json["start_condition_evidence"],
        "success_condition_evidence": [],
        "pre_span_summary": function_calling_json["pre_span_summary"],
        "subgoal_initiation_intent": function_calling_json["intent"],
        }
        
        if self.verbose:
            print(function_calling_summary)
        
        new_node_id = self.skill_calling_memory_graph.add_node_from_calling_summary(function_calling_summary)
        self.skill_calling_memory_graph.nodes[new_node_id].set_success_flag(skill_success)
        
        if not skill_success:
            self.update_history(">> "+skill_str + "did not achieve its sub-goal, " + function_calling_json["skill"]) 
            
            if self.verbose:
                print("\n".join(interaction_history_list))
            
            # execution_trace_list
            execution_trace_info = ""
            if self.verbose and execution_trace_list:
                for iii, (node_id, node_info) in enumerate(execution_trace_list):
                    execution_trace_info += f"step {iii}: {node_id}, {node_info}\n"


            failure_interaction = ""
            iter_node_id = calling_start_node_id
            while iter_node_id != None:
                current_node = self.trajectory_memory_graph.nodes[iter_node_id]
                # current_node.show()
                failure_interaction += f"obs: {current_node.observation_current}\nworld states: {current_node.facts_current}\naction: {current_node.action}\n"
                iter_node_id = current_node.trajectory_next_node_id

            skill_json = self.skill_json_dict[function_calling_json["skill"]]
            skill_call_dict = {
                "subgoal": function_calling_json["skill"], 
                "description": skill_json["description"],
                "steps": skill_json["steps"],
                "start_conditions": skill_json["start_conditions"],
                "start_condition_evidence": function_calling_json["start_condition_evidence"],
                "success_conditions": skill_json["success_conditions"],
                "pre_span_summary": function_calling_json["pre_span_summary"],
                "subgoal_initiation_intent": function_calling_json["intent"],
                "parameters": skill_json["parameters"],
                "parameter_bindings": function_calling_json["parameter_bindings"], 
            }
            
            prompt = TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT.format(
                skill_schema=skill_call_dict,
                mermaid_workflow=self.skill_json_dict[function_calling_json["skill"]]["mermaid_code"],
                failure_interaction=failure_interaction,
                execution_trace=execution_trace_info,
            )

            response = llm_response(prompt, self.llm, self.temperature, max_tokens=8192)

            if self.verbose:
                print("-"*20 + "prompt" + "-"*20)
                print(prompt)
                print("-"*20 + "response" + "-"*20)
                print(response)

            # exit(0)

        else:
            self.update_history(">> "+skill_str + " have achieved its sub-goal, " + function_calling_json["skill"]) 
        

        return interaction_history_list[-1], "\n".join(interaction_history_list), env_done_flag, task_success_flag
    
    def execute_skill_workflow(self, skill_workflow, ctx, verbose=True):
        """
        Execute skill workflow.
        """

        _name_token_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

        def _build_render_context(local_args: Dict[str, object]) -> Dict[str, object]:
            merged: Dict[str, object] = {}

            def _add_source(src: Dict[str, object]) -> None:
                for key, value in src.items():
                    if not isinstance(key, str):
                        continue
                    if not _name_token_re.match(key):
                        continue
                    # for alias in _candidate_variants(key):
                    for alias in key:
                        merged[alias] = value
            _add_source(dict(skill_workflow.global_vars))
            _add_source({k: v for k, v in ctx.items() if isinstance(k, str)})
            _add_source(local_args)
            return merged

        def decide(expr: str, ctx: dict, local_args: dict) -> bool:
            # Normalize
            raw_expr = expr.strip()

            # Helper: evaluate quantifier/boolean expressions directly with Python env.
            def _eval_bool_with_env(expr_src: str) -> Tuple[bool, str]:
                expr_fmt = expr_src.replace("{{", "{").replace("}}", "}")
                expr_fmt = expr_fmt.replace("{", "\"{").replace("}", "}\"")
                # render_env = _build_render_context(local_args)
                rendered = expr_fmt.format(**local_args)
                converted_expr = convert_all_items(rendered)

                execu = ctx["executor"]
                facts = ctx["current_facts"]
                # print(facts)
                
                try:
                    current_world = ctx["world_builder"](predicates=facts)
                    env = _build_predicate_env(execu, current_world, facts)
                    condition_value = bool(eval(converted_expr, env, env))
                except Exception as e:
                    print(f"Error evaluating expression: {converted_expr}")
                    print(f"Exception: {e}")

                
                if verbose:
                    if ctx["skill_interaction"]:
                        print(f"last observation: {ctx['skill_interaction'][-1]}")
                    print(f"current facts: {ctx['current_facts']}")

                log_str = converted_expr.replace("lambda d: d.f_", "")
                
                last_observation = self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].observation_current

                return condition_value, f"CHECK {log_str}, the last observation: {last_observation}, decision value: {condition_value}."
            # print("here is a check node: ", raw_expr)

            if re.search(r"\b(exists|forall)\s*\(", raw_expr, flags=re.IGNORECASE):
                return _eval_bool_with_env(raw_expr)

            expr_q = raw_expr.replace("{{", "{").replace("}}", "}")
            expr_q = expr_q.replace("{", "\"{").replace("}", "}\"")
            expr_lambda = "lambda d: d.f_" + expr_q
            # render_env = _build_render_context(local_args)

            if verbose:
                print("raw_expr: ", raw_expr)
                print(expr_lambda)
            converted_expr = convert_all_items(expr_lambda.format(**local_args))
            eval_locals = dict(ctx)
            eval_locals.update(local_args)
            eval_locals.setdefault("ctx", ctx)

            if verbose:
                print("converted_expr: ", converted_expr)
            expr_fn = eval(converted_expr, globals(), eval_locals)

            facts = ctx["current_facts"]
            
            # print(facts)
            try:
                current_world = ctx["world_builder"](predicates=facts)
                condition_value = ctx["executor"].execute(expr_fn(ctx["domain"]), current_world).value
            except Exception as e:
                print(f"Error evaluating expression: {converted_expr}")
                print(f"Exception: {e}")
                raise e
            # print(condition_value)
            
            if verbose:
                if ctx["skill_interaction"]:
                    print(f"last observation: {ctx['skill_interaction'][-1]}")
                print(f"current facts: {ctx['current_facts']}")
            log_str = converted_expr.replace("lambda d: d.f_", "")

            last_observation = self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].observation_current
            return condition_value, f"CHECK {log_str}, the last observation: {last_observation}, decision value: {condition_value}."


        def act(act_name: str, args: str, ctx: dict, local_args: dict) -> None:
            
            # print("action node: ", act_name, args, ctx.keys())

            template = (args or "").strip()
            if not template and act_name:
                template = act_name

            expr_fmt = template.replace("{{", "{").replace("}}", "}")
            action_str = expr_fmt.format(**local_args)

            action_str = inverse_all_items(action_str)
            
            if self.verbose:
                print("action_str: ", action_str)

            observation, reward, done, info = ctx["online_env"].step([action_str.strip()])
            observation, is_nothing_happens = ctx["process_ob"](observation[0], track_nothing_happens=True)

            self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action = action_str.strip()
            self.update_history("> "+action_str.strip())

            if self.verbose:
                print("observation: ", observation)

            ctx["skill_interaction"].append(action_str.strip())
            ctx["skill_interaction"].append(observation.strip())

            self.preceived_flag = True
            if not is_nothing_happens:
                ctx["current_facts"] = self.perceive(observation, ctx["current_facts"])
            
            if done[0] or info["won"][0]:
                if info["won"][0]:
                    ctx["task_success_flag"] = True
                raise Exception("Environment terminated.")
            
            return action_str.strip()
        
        try:
            trace = traverse_with_info(skill_workflow, decide, act, ctx, verbose=verbose)

        except Exception as e:

            if str(e) == "Environment terminated.":
                return (ctx["skill_interaction"], None), True, True, ctx["task_success_flag"]

            print(f"Skill execution failed. Error: {e}")
            print(traceback.format_exc())

            # Check if the error is caused by a tentative graft (KeyError on graft variable)
            if isinstance(e, KeyError) and self.online_mode and self.evolution_ledger:
                skill_name_ctx = ctx.get("_skill_name", "")
                if skill_name_ctx and skill_name_ctx in self.skill_json_dict and self.skill_json_dict[skill_name_ctx]:
                    for graft_id, graft_info in list(self.evolution_ledger.items()):
                        if (graft_info.get("skill_name") == skill_name_ctx
                                and graft_info.get("status") == "tentative"):
                            print(f"[ONLINE] Graft {graft_id} caused KeyError '{e}', discarding and retrying")
                            self._discard_graft(graft_id)
                            # Retry with restored original mermaid
                            try:
                                restored_workflow = parse_mermaid_with_info(
                                    self.skill_json_dict[skill_name_ctx]["mermaid_code"])
                                trace = traverse_with_info(restored_workflow, decide, act, ctx, verbose=verbose)
                                if trace[-1][0] == "SUCCESS_END":
                                    return (ctx["skill_interaction"], trace), True, False, False
                                return (ctx["skill_interaction"], trace), False, False, False
                            except Exception as retry_e:
                                if str(retry_e) == "Environment terminated.":
                                    return (ctx["skill_interaction"], None), True, True, ctx["task_success_flag"]
                                print(f"Retry also failed: {retry_e}")
                            break

            return (ctx["skill_interaction"], None), False, False, False

        if verbose:
            print("Trace of Workflow: ")
            for nid, info in trace:
                print(f"{nid}: {info}")

        if trace[-1][0] == "SUCCESS_END":
            return (ctx["skill_interaction"], trace), True, False, False
        return (ctx["skill_interaction"], trace), False, False, False

    # ===================== Online Skill Evolution =====================

    def _append_agent_log(self, text: str) -> None:
        """Append text to the agent log."""
        self.agent_log += text

    def set_episode_info(self, traj_path: str = "", task_name: str = "", env_type: str = "") -> None:
        """Set per-episode metadata for online evolution."""
        self.traj_path = traj_path
        self._episode_task_name = task_name
        self._episode_env_type = env_type

    def _update_routing_rules(self, pattern: Dict, episode_idx: int) -> None:
        """Extract a planner routing rule from a planner_routing pattern."""
        from .prompts import ROUTING_RULE_EXTRACTION_PROMPT

        failing_skill = pattern.get("failing_skill", "")
        skill_json = self.skill_json_dict.get(failing_skill, {})

        skill_desc_parts = []
        for sname, sj in self.skill_json_dict.items():
            if sj is None:
                continue
            params = ", ".join(sj.get("parameters", []))
            desc = sj.get("description", "")
            skill_desc_parts.append(f"- {sname}({params}): {desc}")

        prompt = ROUTING_RULE_EXTRACTION_PROMPT.format(
            failing_skill=failing_skill,
            failing_skill_description=skill_json.get("description", "") if skill_json else "",
            failure_diagnostic=pattern.get("failure_diagnostic", ""),
            recovery_segment=pattern.get("recovery_segment", ""),
            recovery_skill_used=pattern.get("recovery_skill_used", "none"),
            skill_descriptions="\n".join(skill_desc_parts),
        )

        response = llm_response(prompt, self.llm, temperature=0.0)
        self._append_agent_log("=" * 20 + " routing rule extraction " + "=" * 20 + f"\n{prompt}\n{response}\n")

        rule = json.loads(repair_json(response))
        rule["source_episode"] = episode_idx
        rule["failing_skill"] = failing_skill
        rule["pattern_type"] = pattern.get("pattern_type", "")

        for existing in self.routing_rules:
            if (existing.get("failing_skill") == failing_skill
                    and existing.get("pattern_type") == rule["pattern_type"]):
                existing.setdefault("episodes_seen", []).append(episode_idx)
                print(f"[ONLINE] Routing rule already exists for {failing_skill}/{rule['pattern_type']}, updated episodes_seen")
                self._save_online_skills()
                return

        self.routing_rules.append(rule)
        self._save_online_skills()
        print(f"[ONLINE] New routing rule: {rule.get('rule_summary', '')}")

    def _log_new_skill_opportunity(self, pattern: Dict, episode_idx: int) -> None:
        """Log a new_skill pattern for future consideration."""
        log_path = os.path.join(self.online_evolved_dir, "new_skill_opportunities.json")
        existing = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                existing = json.load(f)
        existing.append({
            "episode_idx": episode_idx,
            "pattern": pattern,
        })
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[ONLINE] Logged new_skill opportunity from episode {episode_idx}")

    def _refine_parameter_guidelines(self, episode_idx: int) -> None:
        """Refine skill parameter guidelines by distilling accumulated anti_patterns."""
        from .prompts import PARAMETER_GUIDELINE_REFINEMENT_PROMPT

        ANTI_PATTERN_THRESHOLD = 5

        for skill_name, skill_json in self.skill_json_dict.items():
            if skill_json is None:
                continue
            guidelines = skill_json.get("parameter_bindings_guidelines", {})
            anti_patterns = guidelines.get("anti_patterns", [])

            raw_failures = [a for a in anti_patterns if a.startswith("FAILED with bindings")]
            if len(raw_failures) < ANTI_PATTERN_THRESHOLD:
                continue

            print(f"[HONING] Refining {skill_name} guidelines ({len(raw_failures)} raw failures)")

            action_types = ("go to, open, close, take ... from, clean ... with, "
                           "cool ... with, heat ... with, move ... to, use, examine")

            prompt = PARAMETER_GUIDELINE_REFINEMENT_PROMPT.format(
                skill_name=skill_name,
                parameters=", ".join(skill_json.get("parameters", [])),
                description=skill_json.get("description", ""),
                parameter_roles=json.dumps(guidelines.get("parameter_roles", {}), indent=2),
                anti_patterns="\n".join(f"- {a}" for a in anti_patterns),
                action_types_sample=action_types,
            )

            response = llm_response(prompt, self.llm, temperature=0.0)
            self._append_agent_log("=" * 20 + f" guideline refinement {skill_name} " + "=" * 20 + f"\n{response}\n")

            result = json.loads(repair_json(response))

            if result.get("refined_parameter_roles"):
                guidelines["parameter_roles"] = result["refined_parameter_roles"]
            if result.get("refined_anti_patterns"):
                guidelines["anti_patterns"] = result["refined_anti_patterns"]
            if result.get("refined_checklist"):
                guidelines["checklist_before_call"] = result["refined_checklist"]

            skill_json["parameter_bindings_guidelines"] = guidelines
            self._save_online_skills()
            print(f"[HONING] Refined {skill_name}: {len(raw_failures)} raw failures -> {len(result.get('refined_anti_patterns', []))} rules")

    def _get_routing_rules_text(self) -> str:
        """Format routing rules for injection into the planner prompt."""
        if not self.routing_rules:
            return "none"
        parts = []
        for rule in self.routing_rules:
            summary = rule.get("rule_summary", "")
            if not summary:
                summary = f"{rule.get('condition', '')} -> {rule.get('action', '')}"
            parts.append(f"- {summary}")
        return "\n".join(parts)

    def _load_online_evolved_skills(self) -> None:
        """Load online-evolved skill graphs if available, else keep offline base."""
        os.makedirs(self.online_evolved_dir, exist_ok=True)
        os.makedirs(os.path.join(self.online_evolved_dir, "grafts"), exist_ok=True)
        os.makedirs(os.path.join(self.online_evolved_dir, "episode_analysis"), exist_ok=True)

        online_json = os.path.join(self.online_evolved_dir, "skill_json_online.json")
        ledger_path = os.path.join(self.online_evolved_dir, "evolution_ledger.json")

        if os.path.exists(online_json):
            with open(online_json, "r") as f:
                self.skill_json_dict = json.load(f)
            print(f"[ONLINE] Loaded evolved skills from {online_json}")
        else:
            print("[ONLINE] No evolved skills found, using offline base")

        if os.path.exists(ledger_path):
            with open(ledger_path, "r") as f:
                self.evolution_ledger = json.load(f)
            print(f"[ONLINE] Loaded evolution ledger ({len(self.evolution_ledger)} grafts)")

        routing_path = os.path.join(self.online_evolved_dir, "routing_rules.json")
        if os.path.exists(routing_path):
            with open(routing_path, "r") as f:
                self.routing_rules = json.load(f)
            print(f"[ONLINE] Loaded {len(self.routing_rules)} routing rule(s)")

    def _save_online_skills(self) -> None:
        """Persist current skill_json_dict, evolution_ledger, and routing_rules."""
        os.makedirs(self.online_evolved_dir, exist_ok=True)
        online_json = os.path.join(self.online_evolved_dir, "skill_json_online.json")
        with open(online_json, "w") as f:
            json.dump(self.skill_json_dict, f, indent=2, ensure_ascii=False)

        ledger_path = os.path.join(self.online_evolved_dir, "evolution_ledger.json")
        with open(ledger_path, "w") as f:
            json.dump(self.evolution_ledger, f, indent=2, ensure_ascii=False)

        routing_path = os.path.join(self.online_evolved_dir, "routing_rules.json")
        with open(routing_path, "w") as f:
            json.dump(self.routing_rules, f, indent=2, ensure_ascii=False)

    def _discard_graft(self, graft_id: str) -> None:
        """Remove a tentative graft: restore original mermaid and remove from ledger."""
        graft_info = self.evolution_ledger.get(graft_id)
        if not graft_info:
            return
        skill_name = graft_info["skill_name"]
        original_mermaid = graft_info.get("original_mermaid")
        if original_mermaid and skill_name in self.skill_json_dict:
            self.skill_json_dict[skill_name]["mermaid_code"] = original_mermaid
            print(f"[ONLINE] Discarded graft {graft_id} for {skill_name}")
        del self.evolution_ledger[graft_id]
        self._save_online_skills()

    def _next_graft_prefix(self, skill_name: str) -> str:
        """Return the next unused graft prefix (G0_, G1_, ...) for a skill."""
        existing = [g for g in self.evolution_ledger.values() if g.get("skill_name") == skill_name]
        return f"G{len(existing)}_"

    def reveiew_episode(self, success_flag: bool = False, **kwargs) -> None:
        """Post-episode online evolution analysis.

        Called after each episode. When online_mode is enabled, analyzes the
        episode trajectory to find valuable failure->recovery patterns, induces
        sub-graph fragments, verifies them deductively, and grafts them onto
        failure points as tentative extensions.
        """
        if not self.online_mode:
            return

        episode_idx = kwargs.get("episode_idx", -1)
        task_name = kwargs.get("task_name", self._episode_task_name)
        env_type = kwargs.get("env_type", self._episode_env_type)

        # Phase 0: Refine parameter guidelines if anti_patterns accumulated
        try:
            self._refine_parameter_guidelines(episode_idx)
        except Exception as e:
            print(f"[ONLINE] Guideline refinement failed: {e}")
            traceback.print_exc()

        if not success_flag:
            print(f"[ONLINE] Episode {episode_idx}: task failed, skipping evolution")
            return

        # Phase 1: Skip if no skill failures occurred
        failed_calls = [e for e in self._skill_execution_log if not e.get("success")]
        if not failed_calls:
            print(f"[ONLINE] Episode {episode_idx}: no skill failures, skipping analysis")
            return

        # Read full trajectory
        traj_text = ""
        if hasattr(self, "traj_path") and self.traj_path and os.path.exists(self.traj_path):
            with open(self.traj_path, "r") as f:
                traj_text = f.read()

        if not traj_text:
            return

        print(f"[ONLINE] Episode {episode_idx} ({task_name}, type={env_type}): "
              f"{len(failed_calls)} skill failures, analyzing...")

        # Phase 2: LLM recovery analysis
        try:
            valuable_patterns = self._analyze_episode_for_recovery(
                traj_text, failed_calls, success_flag, task_name, env_type,
            )
        except Exception as e:
            print(f"[ONLINE] Recovery analysis failed: {e}")
            traceback.print_exc()
            return

        if not valuable_patterns:
            print(f"[ONLINE] No valuable recovery patterns found")
            return

        print(f"[ONLINE] Found {len(valuable_patterns)} valuable pattern(s)")

        # Phase 3: Route each pattern by evolution_type
        for pattern in valuable_patterns:
            evo_type = pattern.get("evolution_type", "skill_graft")
            try:
                if evo_type == "skill_graft":
                    print(f"[ONLINE] Pattern -> skill_graft: {pattern.get('failing_skill')}/{pattern.get('pattern_type')}")
                    self._process_recovery_pattern(pattern, episode_idx)
                elif evo_type == "planner_routing":
                    print(f"[ONLINE] Pattern -> planner_routing: {pattern.get('failing_skill')} -> {pattern.get('recovery_skill_used', '?')}")
                    self._update_routing_rules(pattern, episode_idx)
                elif evo_type == "new_skill":
                    print(f"[ONLINE] Pattern -> new_skill opportunity logged")
                    self._log_new_skill_opportunity(pattern, episode_idx)
                else:
                    print(f"[ONLINE] Unknown evolution_type '{evo_type}', defaulting to skill_graft")
                    self._process_recovery_pattern(pattern, episode_idx)
            except Exception as e:
                print(f"[ONLINE] Failed to process {evo_type} pattern: {e}")
                traceback.print_exc()
                continue

        # Save episode analysis for debugging
        analysis_dir = os.path.join(self.online_evolved_dir, "episode_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        analysis_path = os.path.join(analysis_dir, f"ep_{episode_idx}_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump({
                "episode_idx": episode_idx,
                "task_name": task_name,
                "env_type": env_type,
                "success_flag": success_flag,
                "num_failures": len(failed_calls),
                "valuable_patterns": valuable_patterns,
            }, f, indent=2, ensure_ascii=False)

    def _analyze_episode_for_recovery(
        self, traj_text: str, failed_calls: List[Dict],
        success_flag: bool, task_name: str, env_type: str,
    ) -> List[Dict]:
        """Phase 2: LLM analyzes trajectory to find valuable failure->recovery patterns."""
        from .prompts import EPISODE_RECOVERY_ANALYSIS_PROMPT

        skill_exec_summary = []
        for call in self._skill_execution_log:
            skill_exec_summary.append({
                "skill_name": call.get("skill_name", ""),
                "parameter_bindings": call.get("parameter_bindings", {}),
                "success": call.get("success", False),
                "diagnostic": call.get("diagnostic", ""),
            })

        skill_desc_parts = []
        for sname, sj in self.skill_json_dict.items():
            if sj is None:
                continue
            params = ", ".join(sj.get("parameters", []))
            desc = sj.get("description", "")
            skill_desc_parts.append(f"- {sname}({params}): {desc}")
        skill_descriptions = "\n".join(skill_desc_parts) if skill_desc_parts else "none"

        prompt = EPISODE_RECOVERY_ANALYSIS_PROMPT.format(
            trajectory=traj_text,
            skill_execution_log=json.dumps(skill_exec_summary, indent=2),
            task_name=task_name,
            task_type=env_type,
            episode_success=success_flag,
            skill_descriptions=skill_descriptions,
        )

        response = llm_response(prompt, self.llm, temperature=0.0)
        self._append_agent_log("=" * 20 + " recovery analysis prompt " + "=" * 20 + f"\n{prompt}\n")
        self._append_agent_log("=" * 20 + " recovery analysis response " + "=" * 20 + f"\n{response}\n")

        result = json.loads(repair_json(response))
        patterns = result.get("patterns", [])
        return [p for p in patterns if p.get("value_assessment") == "high"]

    def _process_recovery_pattern(self, pattern: Dict, episode_idx: int) -> None:
        """Phase 3-4: Clean trajectory, induce sub-graph, verify, graft."""
        from .prompts import TRAJECTORY_CLEANING_PROMPT, GRAFT_INDUCTION_PROMPT

        skill_name = pattern.get("failing_skill", "")
        if skill_name not in self.skill_json_dict or self.skill_json_dict[skill_name] is None:
            print(f"[ONLINE] Unknown skill {skill_name}, skipping")
            return

        pattern_type = pattern.get("pattern_type", "unknown").replace(" ", "_")

        # Dedup: skip if an active graft already exists for this skill + pattern_type
        for graft_info in self.evolution_ledger.values():
            existing_pt = graft_info.get("pattern_type", "").replace(" ", "_")
            if (graft_info.get("skill_name") == skill_name
                    and existing_pt == pattern_type
                    and graft_info.get("status") in ("tentative", "solidified")):
                seen = graft_info.setdefault("episodes_seen", [])
                if episode_idx not in seen:
                    seen.append(episode_idx)
                    self._save_online_skills()
                print(f"[ONLINE] Skip: {skill_name} already has active {pattern_type} graft")
                return

        original_mermaid = self.skill_json_dict[skill_name]["mermaid_code"]
        failure_diagnostic = pattern.get("failure_diagnostic", "")
        recovery_segment = pattern.get("recovery_segment", "")

        # Step A: Clean trajectory
        clean_prompt = TRAJECTORY_CLEANING_PROMPT.format(
            skill_name=skill_name,
            failure_node="FAILURE_END",
            diagnostic=failure_diagnostic,
            recovery_segment=recovery_segment,
        )
        clean_response = llm_response(clean_prompt, self.llm, temperature=0.0)
        cleaned_trajectory = json.loads(repair_json(clean_response)).get("cleaned_trajectory", "")

        if not cleaned_trajectory:
            print(f"[ONLINE] Empty cleaned trajectory, skipping")
            return

        # Step B: Induce sub-graph
        import re as _re
        referenced_skills = set()
        combined_text = recovery_segment + "\n" + cleaned_trajectory
        for ref_name in self.skill_json_dict:
            if self.skill_json_dict[ref_name] is None:
                continue
            if ref_name != skill_name and ref_name in combined_text:
                referenced_skills.add(ref_name)
        # Detect action patterns that hint at skill usage
        if _re.search(r'\bgo to\b', combined_text, _re.IGNORECASE):
            if "locate_and_retrieve_item" in self.skill_json_dict:
                referenced_skills.add("locate_and_retrieve_item")
        if _re.search(r'\bmove\b.*\bto\b', combined_text, _re.IGNORECASE):
            if "transfer_item_to_receptacle" in self.skill_json_dict:
                referenced_skills.add("transfer_item_to_receptacle")

        ref_graphs_text = ""
        if referenced_skills:
            for ref_name in sorted(referenced_skills):
                ref_sj = self.skill_json_dict[ref_name]
                if ref_sj is None:
                    continue
                ref_graphs_text += f"## Skill: {ref_name}\n"
                ref_graphs_text += f"Parameters: {ref_sj.get('parameters', [])}\n"
                ref_graphs_text += f"```mermaid\n{ref_sj['mermaid_code']}\n```\n\n"
        else:
            ref_graphs_text = "(none -- recovery used only primitive actions)"

        # Determine scope variables based on skill
        scope_vars_map = {
            "locate_and_retrieve_item": "ITEMTYPE, SEARCHLOCATIONS, CURRENT_RECEPTACLE, TARGET_ITEM_TYPE, ITEM_TO_TAKE",
            "transfer_item_to_receptacle": "ITEMNAME, TOREC",
            "clean_item_and_place": "CLEANITEM, SINK, TOREC",
            "heat_item_then_place": "HEATITEM, MICROWAVE, TOREC",
            "cool_item_then_place": "COOLITEM, FRIDGE, TOREC",
        }
        scope_vars = scope_vars_map.get(skill_name, "TARGET_ITEM, TARGET_RECEPTACLE")

        graft_prefix = self._next_graft_prefix(skill_name)

        # Resolve actual graft point
        default_src = "LOOP_FOR_LOCATIONS" if skill_name == "locate_and_retrieve_item" else "FAILURE_END"
        default_dst = "FAILURE_END"

        graft_mermaid = None
        induced = None
        combined = None
        for attempt in range(3):
            induce_prompt = GRAFT_INDUCTION_PROMPT.format(
                original_mermaid=original_mermaid,
                failure_node="FAILURE_END",
                failure_diagnostic=failure_diagnostic,
                cleaned_trajectory=cleaned_trajectory,
                pattern_type=pattern_type,
                scope_variables=scope_vars,
                referenced_skill_graphs=ref_graphs_text,
            )
            induce_response = llm_response(induce_prompt, self.llm, temperature=0.0)
            induced = json.loads(repair_json(induce_response))
            graft_mermaid = induced.get("graft_mermaid", "")

            graft_src = induced.get("graft_point_src", default_src)
            graft_dst = induced.get("graft_point_dst", default_dst)

            if graft_mermaid:
                try:
                    combined = graft_subgraph_at_node(
                        base_mermaid=original_mermaid,
                        graft_mermaid=graft_mermaid,
                        remove_edge_src=graft_src,
                        remove_edge_dst=graft_dst,
                        continue_target=induced.get("continue_target", "SUCCESS_END"),
                        graft_prefix=graft_prefix,
                    )
                    parse_mermaid_with_info(combined)
                    break
                except Exception as e:
                    print(f"[ONLINE] Graft validation failed (attempt {attempt+1}): {e}")
                    graft_mermaid = None
                    combined = None

        if not graft_mermaid or not combined:
            print(f"[ONLINE] Could not produce valid graft after 3 attempts")
            return

        # Step C: Deductive verify
        try:
            combined_graph = parse_mermaid_with_info(combined)
            all_dsts = {e.dst for e in combined_graph.edges}
            continue_tgt = induced.get("continue_target", "SUCCESS_END")
            fail_tgt = induced.get("graft_point_dst", "FAILURE_END")
            has_continue = continue_tgt in all_dsts
            has_fail = fail_tgt in all_dsts

            if not has_continue:
                print(f"[ONLINE] Deductive verify failed: no path to {continue_tgt}")
                return
            if not has_fail:
                print(f"[ONLINE] Deductive verify warning: no fallback path to {fail_tgt}")

            print(f"[ONLINE] Deductive verify passed for {skill_name} ({pattern_type})")
        except Exception as e:
            print(f"[ONLINE] Deductive verify failed: {e}")
            return

        # Step D: Graft + persist
        graft_id = f"{skill_name}__{pattern_type}"
        graft_node_ids = list(extract_graft_node_ids(combined, graft_prefix))

        self.skill_json_dict[skill_name]["mermaid_code"] = combined
        self.evolution_ledger[graft_id] = {
            "graft_id": graft_id,
            "status": "tentative",
            "skill_name": skill_name,
            "graft_point_src": induced.get("graft_point_src", default_src),
            "graft_point_dst": induced.get("graft_point_dst", default_dst),
            "pattern_type": pattern_type,
            "episodes_seen": [episode_idx],
            "graft_node_ids": graft_node_ids,
            "original_mermaid": original_mermaid,
            "success_count": 0,
            "fail_count": 0,
        }

        grafts_dir = os.path.join(self.online_evolved_dir, "grafts")
        os.makedirs(grafts_dir, exist_ok=True)
        graft_path = os.path.join(grafts_dir, f"{graft_id}.mmd")
        with open(graft_path, "w") as f:
            f.write(graft_mermaid)

        self._save_online_skills()
        print(f"[ONLINE] Grafted {graft_id} (tentative) -> {skill_name}")

    # ===================== End Online Skill Evolution =====================

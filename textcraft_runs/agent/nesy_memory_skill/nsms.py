import json
import os
import re
import string
from typing import Dict, List, Optional, Set, Tuple
import ast
import traceback
from copy import deepcopy
import random
from dataclasses import dataclass

from ..base import BaseAgent
from ..utils import llm_response, get_price, get_token_counts

from json_repair import repair_json

from .prompts import OBSERVATION_TO_PREDICATES_PROMPT
from .prompts import MEMORY_CONSTRUCTION_PREDICATE_ANALYSIS_PROMPT, MEMORY_CONSTRUCTION_INTENT_WITH_PREDICATE_PROMPT, MEMORY_CONSTRUCTION_INTENT_PROMPT
from .prompts import SUBGOAL_PROMPT, SUB_GOAL_WORKFLOW_PROMPT, TRAJ_SEGMENTATION_PROMPT
from .prompts import TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT, TASK_LEVEL_SUMMARY_PROMPT
from .prompts import SUBGOAL_IDENTIFICATION_PROMPT
from .prompts import SKILL_PARAMETER_ANALYSIS_PROMPT, SKILL_PRECONDITION_SYNTH_PROMPT
from .prompts import SKILL_ACTION_ROUTER_PROMPT, EXECUTION_WITH_ROUTING_PROMPT
from .prompts import SUB_GOAL_WORKFLOW_TRACE2CODE_PROMPT, SUB_GOAL_WORKFLOW_TRACE2CODE_PROMPT_V2
from .prompts import SUB_GOAL_WORKFLOW_GENERALIZER_PROMPT
from .prompts import TRAJ_SEGMENTATION_REVISION_PROMPT
from .prompts import SUBGOAL_JUDGEMENT_PROMPT, SUBGOAL_EXECUTION_RETHINK_PROMPT
from .prompts import REGRESSION_TOPOLOGY_PROMPT, PLANNING_PROMPT



from .infoflow2graph import (
    parse_mermaid_with_info,
    validate_node_inputs,
    validate_node_inputs_with_prefix_node,
    validate_check_nodes_without_global_writes,
    validate_control_flow_outgoing_edges,
    validate_control_flow_node_definitions,
    validate_loop_entry_edges,
    validate_check_expr_syntax,
    extract_start_inputs,
    traverse_with_info,
    inverse_all_items,
    _build_predicate_env,
    convert_all_items,
    Action_Unmatched_Exception,
)

from .memory import TrajectoryMemoryNode4TextCraft as TrajectoryMemoryNode
from .memory import TrajectoryMemoryGraph4TextCraft as TrajectoryMemoryGraph
from .memory import SkillCallingMemoryGraph4TextCraft as SkillCallingMemoryGraph
from .memory import SkillCallingMemoryNode4TextCraft as SkillCallingMemoryNode


from .symbolic_world import World, executor, parser, DOMAIN
from .symbolic_world import get_pre_condition, canon, transform_item_name
from .symbolic_world import Variable, T_ITEM, T_INT
            
MERMAID_TYPE_MAPPING = {
    "ItemName": T_ITEM,
    "Count": T_INT,
}

LIST_TYPE_PATTERN = re.compile(r"^List[\s_\[]*(.+?)[\]\s]*$", re.IGNORECASE)
TYPE_BASE_ALIASES = {
    "agent": "agent",
    "bool": "bool",
    "boolean": "bool",
    "count": "int",
    "int": "int",
    "integer": "int",
    "item": "item",
    "itemid": "item",
    "item_id": "item",
    "itemname": "item",
    "string": "string",
}
FUNCTION_SIGNATURES = {
    "goal": {"args": ["item", "int"], "returns": "bool"},
    "inventory": {"args": ["item", "int"], "returns": "bool"},
    "unavailable": {"args": ["item"], "returns": "bool"},
    "current_count_of_item": {"args": ["item"], "returns": "int"},
    "count": {"args": ["item"], "returns": "int"},
    "is_holding": {"args": ["item"], "returns": "bool"},
    "has_enough": {"args": ["item", "int"], "returns": "bool"},
    "needs_item": {"args": ["item", "int"], "returns": "bool"},
    "can_get": {"args": ["item", "int"], "returns": "bool"},
    "can_craft": {"args": ["item", "int"], "returns": "bool"},
    "can_inventory": {"args": [], "returns": "bool"},
    "numerical_greater_than": {"args": ["int", "int"], "returns": "bool"},
    "numerical_greater_equal": {"args": ["int", "int"], "returns": "bool"},
    "numerical_less_than": {"args": ["int", "int"], "returns": "bool"},
    "numerical_less_equal": {"args": ["int", "int"], "returns": "bool"},
    "numerical_equal": {"args": ["int", "int"], "returns": "bool"},
    "greater_than": {"args": ["int", "int"], "returns": "bool"},
    "exists_item_in_list": {"args": ["list[item]", ("lambda", "item")], "returns": "bool"},
    "forall_item_in_list": {"args": ["list[item]", ("lambda", "item")], "returns": "bool"},
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

    if not prepared_expr.strip():
        return []
    return _type_check_clause(prepared_expr, alias_env)

def _fix_single_brace_placeholders(text: str) -> str:
    """Convert single-brace placeholders to double-brace form."""

    # Regex to find single-braced placeholders like {foo} that aren't already doubled.
    _SINGLE_BRACED_PLACEHOLDER = re.compile(r"(?<!\{)\{(?!\{)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})")

    if not text:
        return text

    def _repl(match: re.Match) -> str:
        return "{{" + match.group(1).strip() + "}}"

    return _SINGLE_BRACED_PLACEHOLDER.sub(_repl, text)


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
    
    print(example)

    previous_obs = ""
    previous_action = ""

    current_facts = []

    example_with_facts = ""
    
    crafting_commands = example.split("obs")[0]
    example = example.split(crafting_commands)[1]

    print("crafting_commands: ", crafting_commands)
    print(example)
    for item in example.split("\n"):
        if verbose:
            print(item)

        
        if item.startswith("obs:"):
            previous_obs += item.replace("obs:", "observation:") + "\n"
            
            
            if not previous_action.startswith("think"):
                # print(current_facts)
                prompt = OBSERVATION_TO_PREDICATES_PROMPT.format(facts=current_facts, 
                                                             new_action=previous_action, new_observation=item.split("obs: ")[1].strip())

                response = llm_response(prompt, llm, temperature=temperature)


                prompt_context = prompt.split("Follow exactly this schema for each new observation.")[1]
                print("-" * 50 + "prompt" + "-"*50)
                print(prompt_context)
                print("-"* 50 + "response" + "-"*50)
                print(response)
                # print(response)
                # exit(0)
                response_json = json.loads(repair_json(response))
                # print(response_json)
            else:
                response_json = {
                    "add_facts": [],
                    "remove_facts": [],
                }

            # current_facts += response_json["add_facts"]
            for fact in response_json["remove_facts"]:
                if fact in current_facts:
                    current_facts.remove(fact)
                else:
                    raise ValueError(f"Fact {fact} to be removed is not in current facts.")
            
            for fact in response_json["add_facts"]:
                if fact not in current_facts:
                    current_facts.append(fact)
            
            example_with_facts += item.strip() + "\n"
            example_with_facts += f"facts: {current_facts}\n"

            print("-"*50 + "updated current_facts: " + "-"*50)
            print(current_facts)
        else: 
            # previous_action = item.split("> ")[1].strip()
            previous_action = item.strip()
            previous_obs += item + "\n"

            example_with_facts += "action: " + previous_action.strip() + "\n"

            if previous_action.startswith("think"):
                continue
        

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


ENV_TYPES = [
    "direct_fetch", 
    "craft_depth_1", 
    "craft_depth_2",
    "craft_depth_3",
    "craft_depth_4",
]

class NesyMemorySkillAgent(BaseAgent):
    """
    NSI agent for TextCraft.

    The module path retains a historical implementation identifier for backward
    compatibility.
    """

    def _append_agent_log(self, content: str) -> None:
        self.agent_log += content
        if self.agent_log_path:
            with open(self.agent_log_path, "a", encoding="utf-8") as handle:
                handle.write(content)

    def __init__(
        self,
        example_path: Optional[str] = None,
        example_dir: str = "",
        log_folder: str = "",
        llm: str = "",
        temperature: float = 0.0,
        load_skill: bool = False,
        load_call: bool = False,
        all_ready: bool = False,
        online_mode: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        # Initialize prompt templates
        self.prompt_template = ""
        self.grounding_template = ""
        self.skill_template = ""
        

        self.example_dir = example_dir
        self.llm = llm
        self.temperature = temperature
        self.verbose = verbose

        self.name = "nsms-agent"
        self.interaction_history = ""
        self.agent_log = ""
        self.agent_log_path = None
        self.item_name_domain: List[str] = []


        self.react_examples: Dict = self._load_examples(example_path)
        self.grounding_examples: Dict = {}

        self.grounded_skill_list: List = []
        self.grounded_skill_set: Dict = {}
        self.valid_grounded_skills_cnt = 0

        self.skill_grounding_examples: List = []
        self.grounding_memory: Dict = {}
        self.skill_executor: Dict = {}

        self.skill_graph: Dict = {}
        self.failed_think_msg: Dict = {}
        self.task_level_guidelines: Dict = {}
        self.subgoal_failure_memory: List[Dict] = []
        self.current_regression_topology: Optional[Dict] = None

        self.load_skill = load_skill
        self.load_call = load_call
        self.all_ready = all_ready
        self.online_mode = online_mode

        # Prepare log/grounding directories so follow-up steps can drop files here.
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        
        self.grounding_example_dir = os.path.join(self.log_folder, "grounding_examples")
        os.makedirs(self.grounding_example_dir, exist_ok=True)

        # Runtime state is needed by both from-scratch and all-ready evaluation.
        # Keep persistent online state across episodes; reset episode-local traces
        # in reset().
        self.online_evolved_dir = os.path.join(self.log_folder, "online_evolved")
        self.evolution_ledger: Dict = {}
        self.routing_rules: List[Dict] = []
        self._consecutive_subgoal_failures: Dict[str, int] = {}
        self._skill_execution_log: List[Dict] = []


        example_file_list = sorted(
            file_i for file_i in os.listdir(self.example_dir)
            if file_i.endswith(".txt")
        )
        if not example_file_list:
            raise ValueError(f"No TextCraft demonstration .txt files found in {self.example_dir}")

        self.react_examples = []
        self.grounding_examples = []
        data_traj_collection = ""        
        crafting_commands_list = []
        for file_i in example_file_list:
            filename_match = re.fullmatch(
                r"example_(?P<index>\d+)_(?P<goal>.+)_depth_(?P<depth>\d+)\.txt",
                file_i,
            )
            if filename_match is None:
                raise ValueError(f"Unexpected TextCraft demonstration filename: {file_i}")
            depth_i = int(filename_match.group("depth"))
            goal_i = filename_match.group("goal")

            with open(os.path.join(self.example_dir, file_i), "r") as file:
                data_traj_i = file.read()
                self.react_examples.append((data_traj_i, goal_i, depth_i))

            grounding_file_i = os.path.join(self.grounding_example_dir, file_i.replace(".txt", "_grounding.txt"))
            if not os.path.exists(grounding_file_i):
                with open(grounding_file_i, "w") as file:
                    example_with_fact = analyze_example(data_traj_i, self.llm, temperature=self.temperature, verbose=False)
                    file.write(example_with_fact)
                    # print(example_with_fact)
                    print("save example_with_fact to file: ", grounding_file_i)
            else:
                with open(grounding_file_i, "r") as file:
                    example_with_fact = file.read()
                print("load example_with_fact to file: ", grounding_file_i)
            
            print(example_with_fact)
            self.grounding_examples.append((example_with_fact, goal_i, depth_i))
            
            crafting_commands_i = data_traj_i.split("obs")[0]
            data_traj_i = data_traj_i.split(crafting_commands_i)[1]
            task_instruction_i = data_traj_i.strip().splitlines()[0].split("Your task is to: ")[1].strip(".")
            task_instruction_i += "\nAll crafting actions must use the provided crafting commands.\n" + crafting_commands_i + "\n"
            data_traj_collection += f"Task name: {task_instruction_i}\n" + data_traj_i + "\n\n"

            crafting_commands_list.append(crafting_commands_i)
        
        # print(data_traj_collection)
        # exit(0)

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
            for traj_idx, (example_with_fact_i, goal_i, depth_i) in enumerate(self.grounding_examples):
                # print(example_with_fact_i)
                # exit(0)
                crafting_commands_i = crafting_commands_list[traj_idx]
                example_with_fact_list_i = formulate_example_with_fact(example_with_fact_i, reduce_dummy=True)
                self.update_trajectory_memory(example_with_fact_list_i, crafting_commands_i,success_flag = True)
                # break
            # # save the trajectory memory graph in json format, you need firstly convert the graph to a dictionary-format
            trajectory_memory_graph_dict = self.trajectory_memory_graph.to_dict()
            with open(traj_memory_file_path, "w") as file:
                json.dump(trajectory_memory_graph_dict, file, indent=4)
            print("save the trajectory memory graph to json format: ", traj_memory_file_path)
            self.trajectory_memory_graph.show()
        
        sub_goals = self.decompose_sub_goals(data_traj_collection, llm=self.llm, temperature=self.temperature, load=True)
        # sub_goals = self.decompose_sub_goals_v2(llm=self.llm, temperature=self.temperature, load=False)
        self.sub_goals = sub_goals
        
        # exit(0)



        # self.node_invention(self.trajectory_memory_graph)
        matching_log = []

        self.skill_json_dict = {}
        if not self.load_call:
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

            # exit(0)

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

        
            

        # exit(0)

        for sub_goal in sub_goals:
            
            # if sub_goal["name"] != "check_and_fetch_item":
            #     continue

            print(json.dumps(sub_goal, indent=4))
            # exit(0)

            if not self.load_skill:  
                # json_file = self.induce_sub_goal_skills(sub_goal)    
                json_file = self.skill_ILP(sub_goal)
            else:
                json_file = os.path.join(self.log_folder, "ilp_workflow", f"{sub_goal['name']}_valid.json")
                # if not os.path.exists(json_file):
                #     json_file = self.induce_sub_goal_skills(sub_goal)  
            # continue
            # exit(0)

            if json_file is None or not os.path.exists(json_file):
                print("Can not induce a valid skill for sub-goal: ", sub_goal["name"])
                continue
            
            with open(json_file, "r") as f:
                skill_json = json.load(f)
            print("load skill from json file: ", json_file)


            mmd_file = json_file.replace(".json", ".mmd")
            with open(mmd_file, "r") as f:
                mmd_code = f.read()
            skill_json['mermaid_code'] = mmd_code
            
            graph_i = parse_mermaid_with_info(skill_json['mermaid_code'])

            # validate the dataflow of graph 
            node_errors = validate_node_inputs(graph_i)
            prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
            check_write_errors = validate_check_nodes_without_global_writes(graph_i)
            check_expr_errors = validate_check_expr_syntax(graph_i)
            control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
            control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
            loop_entry_errors = validate_loop_entry_edges(graph_i)
            
            self.extract_node_edge_errors(
                node_errors,
                prefix_errors,
                check_write_errors,
                check_expr_errors,
                control_flow_errors,
                control_flow_node_errors,
                loop_entry_errors,
            )
            
            if node_errors or prefix_errors or check_write_errors or check_expr_errors or control_flow_errors or control_flow_node_errors or loop_entry_errors:
                print("The loaded skill is invalid with dataflow errors")
                
                self.skill_json_dict[sub_goal["name"]] = None
                continue
            else:
                print("The loaded skill is valid without dataflow errors.")

                self.skill_json_dict[sub_goal["name"]] = skill_json
        
        # print(self.skill_json_dict["check_and_fetch_item"]["mermaid_code"])
        # exit(0)
        
        if self.all_ready:
            
            skill_json_file = os.path.join(self.log_folder, "skill_json_file_for_run.json")
            with open(skill_json_file, "r") as f:
                self.skill_json_dict = json.load(f)
            print("load skill json file from ", skill_json_file)

            
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

        
        # exit(0)
        self.skill_success_records = {}
        log_list = []
        for subgoal_name, calling_node_ids in self.skill_calling_memory_graph.nodes_by_subgoal.items():
            
            # if subgoal_name != "check_and_fetch_item":
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
                                                                               verbose=self.verbose)
                    
                    
                    self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name].remove(calling_node_id)

                    if success_flag:
                        success_calling_id_list.append(revised_calling_node_id)
                        
                
                print(f"success rate before abduction: {success_rate:.3f}({len(success_matching)}/{len(success_matching) + len(fail_matching)})")
                new_success_rate = len(success_calling_id_list) / len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])
                print(f"success rate after abduction: {new_success_rate:.3f}"
                      f"({len(success_calling_id_list)}/{len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])})")
                # exit(0)

                log_list.append("After abduction, success rate: {:.3f}({}/{})".format(new_success_rate, len(success_calling_id_list), len(self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name])))
            
            elif success_rate < 0.5:
                    # log_list.append("The success rate is too low, skip the abduction.")

                print("Revise the skill via induction")

                failed_information = []
                for calling_node_id, info in fail_matching:
                    problem_node_id = self.skill_calling_memory_graph.nodes[calling_node_id].problem_node
                    
                    if problem_node_id in success_raw_traj_id_list:
                        self.skill_calling_memory_graph.nodes_by_subgoal[subgoal_name].remove(calling_node_id)
                        continue                    
                    # calling_summary = self.generate_calling_summary(self.skill_json_dict[subgoal_name], start_node=problem_node_id)
                    # print(calling_summary)
                    
                    gradient_json = self.collect_gradient_from_failure(subgoal_name, calling_node_id, problem_node_id, trace_info=info, verbose=False)
                    failed_information.append(gradient_json)

                for failed_info in failed_information:
                    print(failed_info)

            self.skill_success_records[subgoal_name] = success_calling_id_list

        for log_i in log_list:
            print(log_i)
        
        # exit(0)

        
        
        self.skill_memory_blocks_with_task_type = {}
        self.prepare_parameter_bindings_guidelines()


        self.summarize_task_level_guidelines()
        self.induce_pre_condition()
        
        # self.induce_pre_condition()
        print(self.skill_json_dict["check_and_fetch_item"]["mermaid_code"])


        skill_json_file = os.path.join(self.log_folder, "skill_json_file_for_run.json")
        with open(skill_json_file, "w") as f:
            json.dump(self.skill_json_dict, f, indent=4)
        print("save skill json file to ", skill_json_file)

        # exit(0)

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




        self.skill_dir = os.path.join(self.log_folder, "induced_workflow")
        os.makedirs(self.skill_dir, exist_ok=True)

        # Episode-scoped fields.
        self.commands = ""
        self.task = ""
        self.env_type = "textcraft"
    
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

                print("pred_cond_expr_i: ", pred_cond_expr_i)
                print(param_type_env)
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

                    try:
                        condition_value = bool(eval(converted_expr, env, env))

                        # print("condition_value", condition_value)
                        if not condition_value:
                            error_node_id.append(calling_node_id)
                            hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                            hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                            hints += f"    - The instantiated pre_condition, {converted_expr}, does not hold.\n"
                    except Exception as e:
                        print("e: ", e)
                        error_node_id.append(calling_node_id)
                        hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                        hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                        hints += f"    - The instantiated pre_condition, {converted_expr} raised the following error: {e}.\n"

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
                        try:
                            condition_value = bool(eval(converted_expr, env, env))

                            # print("condition_value", condition_value)
                            if not condition_value:
                                error_node_id.append(calling_node_id)
                                hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                                hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                                hints += f"    - The instantiated pre_condition, {converted_expr}, does not hold.\n"
                        except Exception as e:
                            print("e: ", e)
                            error_node_id.append(calling_node_id)
                            hints += f"- Precondition {pred_cond_expr_i} does not hold for records {index+1}.\n"
                            hints += f"    - World Facts in record {index+1}: {current_facts}\n"
                            hints += f"    - The instantiated pre_condition, {converted_expr} raised the following error: {e}.\n"
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
            for cond_expr in self.skill_json_dict[subgoal_name]["pre_condition_expression"]:
                print(cond_expr)

        
        return

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
        
        # ENV_TYPES = {"craft": "craft"}
        for task_type in ENV_TYPES:

            if task_type in self.skill_memory_blocks_with_task_type and len(self.skill_memory_blocks_with_task_type[task_type]) > 0:
                
                # print(task_type)
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
                
                for example_i, (example_with_fact_i, goal_i, depth_i) in enumerate(self.grounding_examples):
                    if depth_i == 0 and task_type == "direct_fetch":
                        raw_traj += f"Example {example_i+1}:\n{example_with_fact_i}\n\n"
                    elif depth_i == 1 and task_type == "craft_depth_1":
                        raw_traj += f"Example {example_i+1}:\n{example_with_fact_i}\n\n"
                
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
                
                # exit(0)

                self.task_level_guidelines[task_type] = summary_json

        if len(self.task_level_guidelines) > 0:
            guideline_file = os.path.join(self.log_folder, "task_level_guidelines.json")
            with open(guideline_file, "w") as f:
                json.dump(self.task_level_guidelines, f, indent=4)
            print("save task level guidelines to ", guideline_file)

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
                    print(task_instruction)

                    if "fetch" in task_instruction:
                        task_type = "direct_fetch"
                    else:
                        task_type = "craft_depth_1"
                    # task_type = "craft"

                    # print("task instruction: ", task_instruction, ", task type: ", task_type)

                    if task_type not in self.skill_memory_blocks_with_task_type:
                        self.skill_memory_blocks_with_task_type[task_type] = {}

                    if subgoal_name not in self.skill_memory_blocks_with_task_type[task_type]:
                        self.skill_memory_blocks_with_task_type[task_type][subgoal_name] = []
                    
                    self.skill_memory_blocks_with_task_type[task_type][subgoal_name].append(calling_node_id)
                # exit(0)
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


    def collect_gradient_from_failure(self, subgoal_name, calling_node_id, problem_node_id, trace_info, skill_json=None, mermaid_code=None, verbose=False):
        
        
        calling_node = self.skill_calling_memory_graph.nodes[calling_node_id]
        calling_node.show()


        failure_interaction = ""
        iter_node_id = calling_node.trajectory_start_node
        while iter_node_id != None:
            current_node = self.trajectory_memory_graph.nodes[iter_node_id]
            # current_node.show()
            failure_interaction += f"obs: {current_node.observation_current}\nworld states: {current_node.facts_current}\naction: {current_node.action}\n"
            iter_node_id = current_node.trajectory_next_node_id

        if skill_json is None:
            skill_json = self.skill_json_dict[subgoal_name]
        if mermaid_code is None:
            mermaid_code = self.skill_json_dict[subgoal_name]["mermaid_code"]

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
            mermaid_workflow=mermaid_code,
            failure_interaction=failure_interaction,
            execution_trace=trace_info,
        )

        response = llm_response(prompt, self.llm, self.temperature, max_tokens=8192)

        if self.verbose:
            print("-"*20 + "prompt" + "-"*20)
            print(prompt)
            print("-"*20 + "response" + "-"*20)
            print(response)



        # exit(0)
        pass

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
        
        crafting_commands = self.trajectory_memory_graph.nodes[problem_node_id].crafting_commands
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
            CRAFT_COMMANDS=crafting_commands,
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
            "related_crafting_commands":json_response["related_crafting_commands"],
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

        print(f"evaluate the subgoal {subgoal_name} with {len(node_ids)} calling nodes.")
        
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
        params_verify["parser"] = parser
        params_verify["crafting_commands"] = self.trajectory_memory_graph.nodes[current_traj_node_id].crafting_commands



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

        def decide(expr: str, ctx: dict, local_args: dict) -> Tuple[bool, str]:
            
            # Normalize
            raw_expr = expr.strip()

            # Helper: evaluate quantifier/boolean expressions directly with Python env.
            def _eval_bool_with_env(expr_src: str) -> Tuple[bool, str]:
                expr_fmt = expr_src.replace("{{", "{").replace("}}", "}")
                expr_fmt = expr_fmt.replace("{", "\"{").replace("}", "}\"")
                # render_env = _build_render_context(local_args)
                # rendered = expr_fmt.format(**local_args)
                # converted_expr = convert_all_items(rendered)


                converted_args = {}
                for key, value in local_args.items():
        
                    print("key, value: ", key, value)
                    if isinstance(value, str):
                        converted_args[key] = convert_all_items(value)
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        converted_args[key] = [convert_all_items(item) for item in value]
                    else:
                        converted_args[key] = value

                converted_expr = expr_fmt.format(**converted_args)

                print("offline decide: ", converted_expr)

                # parser = ctx["parser"]
                # print(local_args)
                # exit(0)

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
            # expr_q = expr_q.replace("{", "\"{").replace("}", "}\"")
            # expr_lambda = "lambda d: d.f_" + expr_q
            # render_env = _build_render_context(local_args)
            

            if verbose:
                print("raw_expr: ", raw_expr)
                # print(expr_lambda)
            converted_args = {}
            for key, value in local_args.items():

                print("key, value: ", key, value)
                if isinstance(value, str):
                    converted_args[key] = convert_all_items(value)
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    converted_args[key] = [convert_all_items(item) for item in value]
                else:
                    converted_args[key] = value
                
            if verbose:
                print("expr_q: ", expr_q)
                print("local_args: ", local_args)
                print("converted_args: ", converted_args)
            converted_expr = expr_q.format(**converted_args)

            local_type_map_src = ctx.get("_current_local_types", {}) or {}
            local_arg_types = {name: local_type_map_src.get(name) for name in local_args}
            
            eval_locals = dict(ctx)
            eval_locals.update(local_args)
            eval_locals.setdefault("ctx", ctx)

            if verbose:
                print("converted_expr: ", converted_expr)
                print("local arg types: ", local_arg_types)

            from .symbolic_world import Variable, T_ITEM, T_INT
            
            # print("offline decide: ", converted_expr)
            expr_fol = converted_expr
            parser = ctx["parser"]
            # print(local_args)
            # print(expr_fol)

            variable_types = [Variable(local_args[name], MERMAID_TYPE_MAPPING.get(local_arg_types.get(name), T_ITEM)) for name in local_args]
            # print(variable_types)
            fol_expression = parser.parse_expression(
                expr_fol,
                variable_types
            )
            if verbose:
                print('\nParsed expression:')
                print(repr(fol_expression))
            facts = ctx["current_traj_node"].facts_current
            current_world = ctx["world_builder"](predicates=facts)
            try:
                condition_value = executor.execute(fol_expression, current_world).value
            except Exception as e:
                print(f"Error evaluating expression: {fol_expression}")
                print(f"Exception: {e}")
            # print(condition_value)
            return condition_value, f"CHECK {fol_expression}, the observation: [node id {ctx['current_traj_node'].id}], decision value: {condition_value}."


            
            # exit(0)
            
            # expr_fn = eval(converted_expr, globals(), eval_locals)

            # facts = ctx["current_traj_node"].facts_current
            
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
                if verbose:
                    print(args, local_args)
                craft_cmds = ctx.get("crafting_commands", "")

                def _match_craft_command(item_name, item_count, ingred_counts, ingred_names, craft_commands):
                    """Try to find the exact craft command string from the provided crafting_commands block."""
                    lines = [
                        ln.strip()
                        for ln in craft_commands.splitlines()
                        if ln.strip().lower().startswith("craft ")
                    ]

                    def _norm_item(txt):
                        txt = str(txt).replace("_", " ")
                        return txt.strip().lower().rstrip(".").rstrip("s")

                    def _norm_parts(counts, names):

                        print("type of the counts and names", type(counts), type(names))
                        if type(counts) == str:
                            counts = [int(c.strip("'")) for c in counts.strip("[]").split(",")]
                        if type(names) == str:
                            names = [n.strip("'") for n in names.strip("[]").split(",")]
                        parts = []
                        for c, n in zip(counts, names):
                            n_norm = _norm_item(str(n))
                            parts.append(f"{c} {n_norm}")

                            print(f"{c} {n}")
                            print(f"{c} {n_norm}")
                        return parts

                    target_item = _norm_item(item_name)
                    target_count = str(item_count)
                    target_ing = set(_norm_parts(ingred_counts or [], ingred_names or []))

                    print(ingred_counts, ingred_names)
                    for ln in lines:
                        if " using " not in ln:
                            continue
                        left = ln.split(" using ", 1)[0].strip()  # e.g., craft 4 oak planks
                        right = ln.split(" using ", 1)[1].strip()

                        parts = left.split()
                        if len(parts) < 3:
                            continue
                        count_part = parts[1]
                        item_part = " ".join(parts[2:])
                        if count_part != target_count:
                            continue
                        if _norm_item(item_part) != target_item:
                            continue

                        print("target item: ", target_item)
                        print("target_count: ", target_count)


                        # parse ingredients (comma separated)
                        ing_lines = [seg.strip() for seg in right.split(",") if seg.strip()]
                        cand_ing = set()
                        for seg in ing_lines:
                            seg_parts = seg.split()
                            if len(seg_parts) < 2:
                                continue
                            cnt = seg_parts[0]
                            nm = " ".join(seg_parts[1:])
                            cand_ing.add(f"{cnt} {_norm_item(nm)}")
                        
                        print("target_ing: ", target_ing)
                        print("cand_ing: ", cand_ing)
                        if target_ing and cand_ing != target_ing:
                            continue
                        return ln
                    return None

                if args.startswith("get "):
                    if local_args["action_item_count"] > 1:
                        action_str = action_str + "s"
                elif args.startswith("craft "):
                    matched_cmd = _match_craft_command(
                        local_args.get("action_item_name", ""),
                        local_args.get("action_item_count", ""),
                        local_args.get("action_ingredients_count_list", []),
                        local_args.get("action_ingredients_name_list", []),
                        craft_cmds,
                    )
                    if matched_cmd:
                        action_str = matched_cmd
                    else:
                        # When strict
                        # (count+item+ingredients) match fails but a recipe
                        # with the same output item exists, use that recipe
                        # line as action_str. TextCraft env semantics: the
                        # recipe's output count is intrinsic to the recipe;
                        # the workflow's chosen count is irrelevant.
                        def _norm(t):
                            return str(t).replace("_", " ").strip().lower().rstrip(".").rstrip("s")
                        req_name = _norm(local_args.get("action_item_name", ""))
                        near_miss = ""
                        for _ln in craft_cmds.splitlines():
                            _ln_s = _ln.strip()
                            if not _ln_s.lower().startswith("craft ") or " using " not in _ln_s:
                                continue
                            _recipe_item = " ".join(_ln_s.split()[2:]).split(" using ")[0]
                            if _norm(_recipe_item) == req_name:
                                near_miss = _ln_s
                                break
                        if near_miss:
                            action_str = near_miss
                        else:
                            # (B) Structured diagnostic — no recipe with this
                            # output name exists at all. List up to 5 recipes
                            # whose output token is a substring/superstring of
                            # the requested item, so the LLM proposer can spot
                            # likely typos / wrong abstraction levels.
                            req_tokens = set(req_name.split())
                            similar: List[str] = []
                            for _ln in craft_cmds.splitlines():
                                _ln_s = _ln.strip()
                                if not _ln_s.lower().startswith("craft ") or " using " not in _ln_s:
                                    continue
                                _recipe_item_2 = _norm(" ".join(_ln_s.split()[2:]).split(" using ")[0])
                                _recipe_tokens = set(_recipe_item_2.split())
                                if req_tokens & _recipe_tokens:
                                    similar.append(_ln_s)
                                    if len(similar) >= 5:
                                        break
                            similar_block = (
                                "\n  Similar recipes (share at least one token with requested item):\n    - "
                                + "\n    - ".join(similar)
                            ) if similar else (
                                "\n  No recipe in `Crafting commands` has an output name token-overlapping with the requested item; "
                                "this item may need to be obtained via `get` instead of `craft`, or you may have routed to the wrong PrimitiveAction."
                            )
                            raise ValueError(
                                f"Craft action mismatch: workflow attempted "
                                f"`craft {local_args['action_item_count']} {local_args['action_item_name']} using "
                                f"{local_args['action_ingredients_count_list']} {local_args['action_ingredients_name_list']}`, "
                                f"but no recipe in `Crafting commands` has output name `{local_args['action_item_name']}`."
                                f"{similar_block}\n  Fix: rebind A_CRAFT's TARGET_ITEM/INGREDIENTS to one of the similar recipes "
                                "above, or replace the craft step with a `get` PrimitiveAction if the item is directly fetchable."
                            )

            except Exception as e:
                print(f"Error formatting action: {template}")
                print(f"Exception: {e}")
                action_str = str(e)
            # print(template)
            # print(local_args)

            if verbose:
                print(ctx["current_traj_node"].action.strip("."))
                print(action_str)

            if ctx["current_traj_node"].action.strip(".") == action_str:
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
        
        # exit(0)
        
        if trace[-1][0] == "SUCCESS_END":
            return trace, True
        return trace, False
    
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
        params_verify["parser"] = parser
        params_verify["crafting_commands"] = self.commands
        params_verify["current_facts"] = self.current_facts
        params_verify["skill_interaction"] = []
        params_verify["task_success_flag"] = False
        params_verify["_skill_name"] = function_calling_json["skill"]

        if callable(process_ob):
            def _process_ob(obs):
                try:
                    return process_ob(obs, track_nothing_happens=True)
                except TypeError:
                    return process_ob(obs)
            params_verify["process_ob"] = _process_ob
        else:
            params_verify["process_ob"] = lambda obs, **kwargs: obs

        if self.verbose:
            print("CONTEXT INPUT: ")
            print(params_verify)

        calling_start_node_id = self.trajectory_memory_node_id

        def decide(expr: str, ctx: dict, local_args: dict) -> Tuple[bool, str]:
            raw_expr = expr.strip()

            def _eval_bool_with_env(expr_src: str) -> Tuple[bool, str]:
                expr_fmt = expr_src.replace("{{", "{").replace("}}", "}")
                expr_fmt = expr_fmt.replace("{", "\"{").replace("}", "}\"")

                converted_args = {}
                for key, value in local_args.items():
                    if isinstance(value, str):
                        converted_args[key] = convert_all_items(value)
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        converted_args[key] = [convert_all_items(item) for item in value]
                    else:
                        converted_args[key] = value

                converted_expr = expr_fmt.format(**converted_args)

                execu = ctx["executor"]
                facts = ctx["current_facts"]
                
                try:
                    current_world = ctx["world_builder"](predicates=facts)
                    env_eval = _build_predicate_env(execu, current_world, facts)
                    condition_value = bool(eval(converted_expr, env_eval, env_eval))
                except Exception as e:
                    print(f"Error evaluating expression: {converted_expr}")
                    print(f"Exception: {e}")
                    condition_value = False

                if self.verbose:
                    if ctx["skill_interaction"]:
                        print(f"last observation: {ctx['skill_interaction'][-1]}")
                    print(f"current facts: {ctx['current_facts']}")

                log_str = converted_expr.replace("lambda d: d.f_", "")
                last_observation = self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].observation_current
                return condition_value, f"CHECK {log_str}, the last observation: {last_observation}, decision value: {condition_value}."

            if re.search(r"\b(exists|forall)\s*\(", raw_expr, flags=re.IGNORECASE):
                return _eval_bool_with_env(raw_expr)
            
            expr_q = raw_expr.replace("{{", "{").replace("}}", "}")

            if self.verbose:
                print("raw_expr: ", raw_expr)

            converted_args = {}
            for key, value in local_args.items():
                if isinstance(value, str):
                    converted_args[key] = convert_all_items(value)
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    converted_args[key] = [convert_all_items(item) for item in value]
                else:
                    converted_args[key] = value

            if self.verbose:
                print("expr_q: ", expr_q)
                print("local_args: ", local_args)
                print("converted_args: ", converted_args)

            converted_expr = expr_q.format(**converted_args)

            local_type_map_src = ctx.get("_current_local_types", {}) or {}
            local_arg_types = {name: local_type_map_src.get(name) for name in local_args}

            if self.verbose:
                print("converted_expr: ", converted_expr)
                print("local arg types: ", local_arg_types)

            variable_types = [
                Variable(local_args[name], MERMAID_TYPE_MAPPING.get(local_arg_types.get(name), T_ITEM))
                for name in local_args
            ]

            parser_local = ctx["parser"]
            fol_expression = parser_local.parse_expression(
                converted_expr,
                variable_types
            )
            if self.verbose:
                print('\nParsed expression:')
                print(repr(fol_expression))

            facts = ctx["current_facts"]
            current_world = ctx["world_builder"](predicates=facts)
            try:
                condition_value = executor.execute(fol_expression, current_world).value
            except Exception as e:
                print(f"Error evaluating expression: {fol_expression}")
                print(f"Exception: {e}")
                condition_value = False
            return condition_value, f"CHECK {fol_expression}, the last observation: [node id {self.trajectory_memory_node_id}], decision value: {condition_value}."

        def act(act_name: str, args: str, ctx: dict, local_args: dict) -> None:
            template = (args or "").strip()
            if not template and act_name:
                template = act_name

            try:
                expr_fmt = template.replace("{{", "{").replace("}}", "}")
                action_str = expr_fmt.format(**local_args)
                action_str = inverse_all_items(action_str)

                if self.verbose:
                    print(args, local_args)

                craft_cmds = ctx.get("crafting_commands", "")

                def _match_craft_command(item_name, item_count, ingred_counts, ingred_names, craft_commands):
                    lines = [
                        ln.strip()
                        for ln in craft_commands.splitlines()
                        if ln.strip().lower().startswith("craft ")
                    ]

                    def _norm_item(txt):
                        txt = str(txt).replace("_", " ")
                        return txt.strip().lower().rstrip(".").rstrip("s")

                    def _norm_parts(counts, names):
                        if type(counts) == str:
                            counts = [int(c.strip("'")) for c in counts.strip("[]").split(",") if c.strip()]
                        if type(names) == str:
                            names = [n.strip("'") for n in names.strip("[]").split(",") if n.strip()]
                        parts = []
                        for c, n in zip(counts, names):
                            n_norm = _norm_item(str(n))
                            parts.append(f"{c} {n_norm}")
                        return parts

                    target_item = _norm_item(item_name)
                    target_count = str(item_count)
                    target_ing = set(_norm_parts(ingred_counts or [], ingred_names or []))

                    for ln in lines:
                        if " using " not in ln:
                            continue
                        left = ln.split(" using ", 1)[0].strip()
                        right = ln.split(" using ", 1)[1].strip()

                        parts = left.split()
                        if len(parts) < 3:
                            continue
                        count_part = parts[1]
                        item_part = " ".join(parts[2:])
                        if count_part != target_count:
                            continue
                        if _norm_item(item_part) != target_item:
                            continue

                        ing_lines = [seg.strip() for seg in right.split(",") if seg.strip()]
                        cand_ing = set()
                        for seg in ing_lines:
                            seg_parts = seg.split()
                            if len(seg_parts) < 2:
                                continue
                            cnt = seg_parts[0]
                            nm = " ".join(seg_parts[1:])
                            cand_ing.add(f"{cnt} {_norm_item(nm)}")
                        if target_ing and cand_ing != target_ing:
                            continue
                        return ln
                    return None

                if args.startswith("get "):
                    if local_args.get("action_item_count", 1) > 1:
                        # action_str = action_str + "s"
                        action_str = action_str

                elif args.startswith("craft "):
                    matched_cmd = _match_craft_command(
                        local_args.get("action_item_name", ""),
                        local_args.get("action_item_count", ""),
                        local_args.get("action_ingredients_count_list", []),
                        local_args.get("action_ingredients_name_list", []),
                        craft_cmds,
                    )
                    if matched_cmd:
                        action_str = matched_cmd
                    else:
                        # Fallback: match by item name only, use recipe's own count
                        def _norm(t):
                            return str(t).replace("_", " ").strip().lower().rstrip(".").rstrip("s")
                        req_name = _norm(local_args.get("action_item_name", ""))
                        fallback_cmd = ""
                        for _ln in craft_cmds.splitlines():
                            _ln_s = _ln.strip()
                            if not _ln_s.lower().startswith("craft ") or " using " not in _ln_s:
                                continue
                            _recipe_item = " ".join(_ln_s.split()[2:]).split(" using ")[0]
                            if _norm(_recipe_item) == req_name:
                                fallback_cmd = _ln_s
                                break
                        if fallback_cmd:
                            action_str = fallback_cmd
                        else:
                            raise ValueError(
                                "Craft command not found for "
                                f"{local_args.get('action_item_name', '')} "
                                f"{local_args.get('action_item_count', '')} using "
                                f"{local_args.get('action_ingredients_count_list', [])} "
                                f"{local_args.get('action_ingredients_name_list', [])}."
                            )
            except Exception as e:
                print(f"Error formatting action: {template}")
                print(f"Exception: {e}")
                action_str = f"think: action formatting error: {e}"

            if self.verbose:
                print("action_str: ", action_str)

            observation, reward, terminated, truncated, info = ctx["online_env"].step(action_str.strip())
            process_result = ctx["process_ob"](observation)
            if isinstance(process_result, tuple):
                observation = process_result[0]
                is_nothing_happens = bool(process_result[1]) if len(process_result) > 1 else False
            else:
                observation = process_result
                is_nothing_happens = False

            if self.trajectory_memory_node_id is not None:
                self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action = action_str.strip()
            self.update_history("> "+action_str.strip())

            ctx["skill_interaction"].append(action_str.strip())
            ctx["skill_interaction"].append(str(observation).strip())

            self.previous_action = action_str.strip()
            self.preceived_flag = True
            if not is_nothing_happens:
                ctx["current_facts"] = self.perceive(str(observation), ctx["current_facts"])

            if reward > 0:
                ctx["task_success_flag"] = True

            if terminated or truncated:
                raise Exception("Environment terminated.")

            # Detect action failure directly from observation
            obs_str = str(observation).strip()
            if obs_str.startswith("Could not"):
                # Build diagnostic from action context + observation
                skill_name = ctx.get("_skill_name", "skill")
                diag_parts = [f"{skill_name} action failed: {action_str.strip()} -> {obs_str}."]
                target = ctx.get("TARGET_ITEM", "")
                need = ctx.get("NEED_COUNT", "")
                if target:
                    diag_parts.append(f"Target: {target}, need_count: {need}.")
                ctx["_diagnostic"] = " ".join(diag_parts)

                # Check if we're inside a graft node — if so, don't abort;
                # let the graph's Check nodes handle the failure via branching.
                import re as _re_act
                current_node = ctx.get("_current_node_id", "")
                is_graft_node = bool(_re_act.match(r'^G\d+_', current_node))

                # Persist get failures so planner won't retry fetch via skill
                if action_str.strip().startswith("get "):
                    # Look up if a recipe exists for this item
                    failed_item = action_str.strip().split("get ", 1)[-1].strip().rstrip("s")
                    recipe_hint = ""
                    craft_cmds = ctx.get("crafting_commands", "")
                    for ln in craft_cmds.splitlines():
                        ln_lower = ln.strip().lower()
                        if ln_lower.startswith("craft ") and failed_item.lower().replace("_", " ") in ln_lower.split(" using ")[0]:
                            recipe_hint = f" Recipe exists: '{ln.strip()}'. Craft it instead of fetching."
                            break
                    if not is_graft_node:
                        self._record_subgoal_failure(
                            action_str.strip(),
                            f"{action_str.strip()}: item unavailable in environment.{recipe_hint}",
                            ""
                        )

                if is_graft_node:
                    # Inside a graft: continue traversal so Check nodes can branch
                    print(f"[GRAFT] Action failed at {current_node}, continuing graph traversal: {obs_str}")
                    return action_str.strip()

                raise RuntimeError(f"Action failed: {obs_str}")

            return action_str.strip()
        
        interaction_history_before_skill_execution = self.interaction_history
        try:
            trace = traverse_with_info(skill_workflow, decide, act, params_verify, verbose=self.verbose)
        except Exception as e:
            if str(e) == "Environment terminated.":
                interaction_history_list = params_verify["skill_interaction"]
                last_obs = interaction_history_list[-1] if interaction_history_list else ""
                _log_skill_exec(params_verify["task_success_flag"], interaction_list=interaction_history_list)
                return last_obs, "\n".join(interaction_history_list), True, params_verify["task_success_flag"]
            
            print(f"Skill execution failed. Error: {e}")
            interaction_history_list = params_verify["skill_interaction"]
            # Action failure with diagnostic — fall through to the diagnostic path below
            # by setting skill_done=False and letting the else branch handle it
            trace = []  # empty trace → skill_done = False

        interaction_history_list = params_verify["skill_interaction"]
        skill_done = bool(trace) and trace[-1][0] == "SUCCESS_END"

        # --- Online Evolution: Tentative graft lifecycle tracking ---
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
                    # depth_composition grafts may fail on recipe mismatches initially;
                    # allow more attempts before discarding
                    discard_threshold = 3 if "depth" in graft_info.get("pattern_type", "") else 1
                    if graft_info["fail_count"] >= discard_threshold:
                        print(f"[ONLINE] Discarding graft {graft_id} after {graft_info['fail_count']} failures")
                        self._discard_graft(graft_id)
                    else:
                        print(f"[ONLINE] Graft {graft_id} failed ({graft_info['fail_count']}/{discard_threshold}), keeping")
                        self._save_online_skills()

            # --- Auto-create depth_composition graft when graft get fails ---
            # When the graph reaches FAILURE_END through graft nodes and the
            # failed ingredient has a recipe, create a deeper graft to craft it.
            if not skill_done and trace[-1][0] == "FAILURE_END":
                import re as _re_graft
                # Find the last graft GET node that was visited
                last_graft_get = None
                for node_id, _ in trace:
                    if _re_graft.match(r'^G\d+_GRAFT_GET$', node_id):
                        last_graft_get = node_id
                if last_graft_get:
                    diagnostic = params_verify.get("_diagnostic", "")
                    failed_item = ""
                    if "Could not find" in str(diagnostic):
                        # Extract the item that couldn't be gotten
                        m = _re_graft.search(r'get \d+ (\S+)', str(diagnostic))
                        if m:
                            failed_item = m.group(1).strip()
                    if not failed_item:
                        # Try from ctx G_TARGET_ITEM
                        failed_item = str(params_verify.get("G_TARGET_ITEM", "")).strip()
                    if failed_item:
                        # Check if this item has a recipe
                        craft_cmds = params_verify.get("crafting_commands", "")
                        recipe_line = ""
                        for ln in craft_cmds.splitlines():
                            ln_s = ln.strip().lower()
                            if ln_s.startswith("craft ") and " using " in ln_s:
                                recipe_item = " ".join(ln_s.split()[2:]).split(" using ")[0]
                                if recipe_item.replace("_", " ") == failed_item.replace("_", " "):
                                    recipe_line = ln.strip()
                                    break
                        if recipe_line:
                            depth_pt = "depth_composition"
                            self._auto_depth_graft(
                                skill_name, failed_item, recipe_line,
                                last_graft_get, depth_pt,
                            )
                            print(f"[ONLINE] Auto depth_composition: {failed_item} has recipe '{recipe_line}'")

        if self.verbose:
            print("skill_done flag: ", skill_done)

        if skill_done:
            # Graph reached SUCCESS_END — post-action checks in the workflow already verified success.
            # No LLM JUDGEMENT needed: trust the symbolic verification.
            if True:
                skill_success = True
                self.update_history(">> "+skill_str + "achieved its sub-goal, " + function_calling_json["skill"]) 
            
                calling_end_node_id = self.trajectory_memory_node_id

                related_crafting_commands = function_calling_json.get("related_crafting_commands")
                if related_crafting_commands is None:
                    if self.commands:
                        related_crafting_commands = [
                            line.strip() for line in self.commands.splitlines()
                            if line.strip().lower().startswith("craft ")
                        ]
                    else:
                        related_crafting_commands = []

                function_calling_summary = {
                    "subgoal": function_calling_json["skill"], 
                    "problem_node": self.current_trajectory_memory_problem_node_id, 
                    "trajectory_start_node": calling_start_node_id,
                    "trajectory_end_node": calling_end_node_id,
                    "subtrajectory_mapping": [], 
                    "parameter_bindings": function_calling_json["parameter_bindings"], 
                    "related_crafting_commands": related_crafting_commands,
                    "start_condition_evidence": function_calling_json["start_condition_evidence"],
                    "success_condition_evidence": [],
                    "pre_span_summary": function_calling_json["pre_span_summary"],
                    "subgoal_initiation_intent": function_calling_json["intent"],
                }
                if self.current_trajectory_memory_problem_node_id is not None:
                    new_node_id = self.skill_calling_memory_graph.add_node_from_calling_summary(function_calling_summary)
                    self.skill_calling_memory_graph.nodes[new_node_id].set_success_flag(skill_success)
                self._record_parameter_binding_feedback(
                    function_calling_json["skill"],
                    success=True,
                    parameter_bindings=function_calling_json.get("parameter_bindings", {}),
                    start_condition_evidence=function_calling_json.get("start_condition_evidence", ""),
                )
        
            else: # LEGACY: was used when JUDGEMENT_PROMPT could reject SUCCESS_END
                # With post-action verification in workflows, SUCCESS_END is trustworthy.
                # Keeping as dead code for reference.
                skill_success = False

                if self.verbose:
                    print("\n".join(interaction_history_list))
                
                updated_start_conditions = json_response["updated_start_conditions"]
                updated_success_conditions = json_response["updated_success_conditions"]
                
                # note = json_response["note"]
                # if not "note" in self.skill_json_dict[function_calling_json["skill"]]:
                #     self.skill_json_dict[function_calling_json["skill"]]["note"] = [note]
                # else:
                #     self.skill_json_dict[function_calling_json["skill"]]["note"].append(note)
                
                for cond_i in updated_start_conditions:
                    if not cond_i in self.skill_json_dict[function_calling_json["skill"]]["start_conditions"]:
                        self.skill_json_dict[function_calling_json["skill"]]["start_conditions"].append(cond_i)
                for cond_i in updated_success_conditions:
                    if not cond_i in self.skill_json_dict[function_calling_json["skill"]]["success_conditions"]:
                        self.skill_json_dict[function_calling_json["skill"]]["success_conditions"].append(cond_i)
                
                self._record_parameter_binding_feedback(
                    function_calling_json["skill"],
                    success=False,
                    parameter_bindings=function_calling_json.get("parameter_bindings", {}),
                    start_condition_evidence=function_calling_json.get("start_condition_evidence", ""),
                    failure_analysis=json_response.get("failure_analysis", ""),
                    following_suggestions=json_response.get("following_suggestions", ""),
                )
                self._update_parameter_bindings_guidelines(
                    function_calling_json["skill"],
                    json_response.get("updated_parameter_bindings_guidelines", {}),
                )
                
                if self.verbose:
                    print("update the specification of skill, ", function_calling_json["skill"])
                    print("failure analysis: ", json_response["failure_analysis"])
                    print("following suggestions: ", json_response["following_suggestions"])
                    print("updated start conditions: ", updated_start_conditions)
                    print("updated success conditions: ", updated_success_conditions)
                
                self.update_history(">> "+skill_str + " did not achieve its sub-goal. ")

                think_msg = " Failure_analysis: " + json_response["failure_analysis"] + " Following suggestions: " + json_response["following_suggestions"] 
                self.update_history("> think: " + think_msg)
                self.update_history("obs: OK.")

                self._record_subgoal_failure(
                    function_calling_json["skill"],
                    json_response["failure_analysis"],
                    json_response["following_suggestions"],
                )

                # --- Online skill honing: learn from craft failures in JUDGEMENT path ---
                for hist_item in interaction_history_list:
                    if "Craft command not found" in str(hist_item):
                        bindings = function_calling_json.get("parameter_bindings", {})
                        anti = (
                            f"FAILED with bindings {json.dumps(bindings)}: craft command not found. "
                            f"All parameter counts must exactly match a single recipe from the crafting commands. "
                            f"Do not scale or combine recipe counts."
                        )
                        self._update_parameter_bindings_guidelines(function_calling_json["skill"], {
                            "anti_patterns": [anti],
                        })
                        if self.verbose:
                            print(f"[SKILL HONING] Updated {function_calling_json['skill']} guidelines with anti-pattern")
                        break

                interaction_history_list.append("think: "+ think_msg)
                interaction_history_list.append("OK.")



                # exit(0)
                    

        
        else:   # execution failed

            skill_success = False

            if self.verbose:
                print("\n".join(interaction_history_list))

            # ---- Read diagnostic from Failure Node template (auto-resolved by graph engine) ----
            graph_failure_reason = params_verify.get("_diagnostic")

            if graph_failure_reason:
                think_msg = f" Failure_analysis: {graph_failure_reason}"
                self.update_history(">> " + skill_str + " did not achieve its sub-goal. ")
                self.update_history("> think: " + think_msg)
                self.update_history("obs: OK.")

                # Record with specific (skill, target_item) key so planner can distinguish
                target_item = function_calling_json.get("parameter_bindings", {}).get("target_item", "")
                failure_key = f"{function_calling_json['skill']}({target_item})" if target_item else function_calling_json["skill"]
                self._record_subgoal_failure(
                    failure_key,
                    str(graph_failure_reason),
                    "",
                )
                self._record_parameter_binding_feedback(
                    function_calling_json["skill"],
                    success=False,
                    parameter_bindings=function_calling_json.get("parameter_bindings", {}),
                    start_condition_evidence=function_calling_json.get("start_condition_evidence", ""),
                    failure_analysis=str(graph_failure_reason),
                    following_suggestions="",
                )

                # --- Online skill honing: learn parameter constraints from failure ---
                # Check interaction history for "Craft command not found" (batch mismatch)
                skill_name = function_calling_json["skill"]
                for hist_item in interaction_history_list:
                    if "Craft command not found" in str(hist_item):
                        bindings = function_calling_json.get("parameter_bindings", {})
                        anti = (
                            f"FAILED with bindings {json.dumps(bindings)}: craft command not found. "
                            f"All parameter counts and ingredient names must exactly match a single recipe. "
                            f"Do not scale counts. If the recipe uses a generic ingredient (e.g. 'planks'), "
                            f"do NOT call this skill with a specific variant (e.g. 'oak planks'). "
                            f"Instead use a raw craft action: 'craft N item using M specific_variant'."
                        )
                        self._update_parameter_bindings_guidelines(skill_name, {
                            "anti_patterns": [anti],
                        })
                        if self.verbose:
                            print(f"[SKILL HONING] Updated {skill_name} guidelines with anti-pattern")
                        break

                interaction_history_list.append("think: " + think_msg)
                interaction_history_list.append("OK.")

                if self.verbose:
                    print(f"[GRAPH DIAGNOSTIC] {graph_failure_reason}")

                last_obs = interaction_history_list[-1] if interaction_history_list else ""
                _log_skill_exec(False, diagnostic=str(graph_failure_reason),
                                trace_nodes=[n for n, _ in trace] if trace else [],
                                interaction_list=interaction_history_list)
                return last_obs, "\n".join(interaction_history_list), False, params_verify["task_success_flag"]
            # ---- End structured diagnostic path ----

            execution_trace_info = ""
            if self.verbose and trace:
                for iii, (node_id, node_info) in enumerate(trace):
                    execution_trace_info += f"step {iii}: {node_id}, {node_info}\n"

            skill_execution_record = ""
            for iii, item in enumerate(interaction_history_list):
                if iii % 2 == 0:
                    skill_execution_record += "> " + item.strip() + "\n"
                else:
                    skill_execution_record += "obs: " + item.strip() + "\n"
            
            # print("skill_execution_record")
            # print(skill_execution_record)
            # print("-"*20)
            
            skill_execution_record = ""
            iter_node_id = calling_start_node_id
            while iter_node_id is not None:
                current_node = self.trajectory_memory_graph.nodes[iter_node_id]
                skill_execution_record += (
                    f"{current_node.observation_current.strip()}\n"
                    f"world states: {current_node.facts_current}\n"
                )
                if current_node.action is not None:
                    skill_execution_record += (
                        f"action: {current_node.action.strip()}\n"
                    )
                iter_node_id = current_node.trajectory_next_node_id
            # print(skill_execution_record)
            
            skill_json = self.skill_json_dict[function_calling_json["skill"]]
            skill_json_call = deepcopy(skill_json)
            skill_json_call.pop("mermaid_code")
            skill_json_call.pop("parameter_binding_feedback", None)
            skill_json_call.pop("applicable_tasks")

            skill_json_call["start_condition_evidence"] = function_calling_json["start_condition_evidence"]
            skill_json_call["parameter_bindings"] = function_calling_json["parameter_bindings"]

            # print(json.dumps(skill_json_call, ensure_ascii=False, indent=4))

            prompt = SUBGOAL_EXECUTION_RETHINK_PROMPT.format(
                subgoal=json.dumps(skill_json_call, ensure_ascii=False, indent=4),
                skill_calling=skill_str,
                mermaid_workflow=skill_json["mermaid_code"],
                interaction_history_before_skill_execution=interaction_history_before_skill_execution,
                skill_execution_record=skill_execution_record,
                execution_trace=execution_trace_info,
            )

            response = llm_response(prompt, self.llm, self.temperature)
            if self.verbose:
                print("-"*20 + "subgoal judgement prompt: ", "-"*20)
                print(prompt)
                print("-"*20 + "subgoal judgement response: ", "-"*20)
                print(response)
            
            

            self._append_agent_log("="*20 + "subgoal judgement prompt" + "="*20 + "\n{content}\n".format(content=prompt))
            self._append_agent_log("="*20 + "subgoal judgement response" + "="*20 + "\n{content}\n".format(content=response))
            json_response = json.loads(repair_json(response))

            # print()
            # exit(0)

            updated_start_conditions = json_response["updated_start_conditions"]
            updated_success_conditions = json_response["updated_success_conditions"]
            
            # note = json_response["note"]
            # if not "note" in self.skill_json_dict[function_calling_json["skill"]]:
            #     self.skill_json_dict[function_calling_json["skill"]]["note"] = [note]
            # else:
            #     self.skill_json_dict[function_calling_json["skill"]]["note"].append(note)
            
            for cond_i in updated_start_conditions:
                if not cond_i in self.skill_json_dict[function_calling_json["skill"]]["start_conditions"]:
                    self.skill_json_dict[function_calling_json["skill"]]["start_conditions"].append(cond_i)
            for cond_i in updated_success_conditions:
                if not cond_i in self.skill_json_dict[function_calling_json["skill"]]["success_conditions"]:
                    self.skill_json_dict[function_calling_json["skill"]]["success_conditions"].append(cond_i)
            self._record_parameter_binding_feedback(
                function_calling_json["skill"],
                success=False,
                parameter_bindings=function_calling_json.get("parameter_bindings", {}),
                start_condition_evidence=function_calling_json.get("start_condition_evidence", ""),
                failure_analysis=json_response.get("failure_analysis", ""),
                following_suggestions=json_response.get("following_suggestions", ""),
            )
            self._update_parameter_bindings_guidelines(
                function_calling_json["skill"],
                json_response.get("updated_parameter_bindings_guidelines", {}),
            )
            
            if self.verbose:
                print("update the specification of skill, ", function_calling_json["skill"])
                print("failure analysis: ", json_response["failure_analysis"])
                print("following suggestions: ", json_response["following_suggestions"])
                print("updated start conditions: ", updated_start_conditions)
                print("updated success conditions: ", updated_success_conditions)
            
            self.update_history(">> "+skill_str + " did not achieve its sub-goal. ")

            think_msg = " Failure_analysis: " + json_response["failure_analysis"] + " Following suggestions: " + json_response["following_suggestions"] 
            self.update_history("> think: " + think_msg)
            self.update_history("obs: OK.")


            interaction_history_list.append("think: "+ think_msg)
            interaction_history_list.append("OK.")
            

            # pass

            # failure_interaction = ""
            # iter_node_id = calling_start_node_id
            # while iter_node_id is not None:
            #     current_node = self.trajectory_memory_graph.nodes[iter_node_id]
            #     failure_interaction += (
            #         f"obs: {current_node.observation_current}\n"
            #         f"world states: {current_node.facts_current}\n"
            #         f"action: {current_node.action}\n"
            #     )
            #     iter_node_id = current_node.trajectory_next_node_id

            # skill_json = self.skill_json_dict[function_calling_json["skill"]]
            # skill_call_dict = {
            #     "subgoal": function_calling_json["skill"], 
            #     "description": skill_json["description"],
            #     "steps": skill_json["steps"],
            #     "start_conditions": skill_json["start_conditions"],
            #     "start_condition_evidence": function_calling_json["start_condition_evidence"],
            #     "success_conditions": skill_json["success_conditions"],
            #     "pre_span_summary": function_calling_json["pre_span_summary"],
            #     "subgoal_initiation_intent": function_calling_json["intent"],
            #     "parameters": skill_json["parameters"],
            #     "parameter_bindings": function_calling_json["parameter_bindings"], 
            # }
            
            # prompt = TEXTUAL_GRADIENT_FROM_FAILURE_PROMPT.format(
            #     skill_schema=skill_call_dict,
            #     mermaid_workflow=self.skill_json_dict[function_calling_json["skill"]]["mermaid_code"],
            #     failure_interaction=failure_interaction,
            #     execution_trace=execution_trace_info,
            # )

            # print(prompt)
            # exit(0)

            # response = llm_response(prompt, self.llm, self.temperature, max_tokens=8192)

            # if self.verbose:
            #     print("-"*20 + "prompt" + "-"*20)
            #     print(prompt)
            #     print("-"*20 + "response" + "-"*20)
            #     print(response)
        
        # exit(0)

        last_obs = interaction_history_list[-1] if interaction_history_list else ""
        _log_skill_exec(skill_done, diagnostic=params_verify.get("_diagnostic", ""),
                        trace_nodes=[n for n, _ in trace] if trace else [],
                        interaction_list=interaction_history_list)
        return last_obs, "\n".join(interaction_history_list), False, params_verify["task_success_flag"]

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
                "world_facts": node_i.facts_current, 
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

        crafting_commands = self.trajectory_memory_graph.nodes[start_node].crafting_commands.strip()
        
        prompt = TRAJ_SEGMENTATION_PROMPT.format(
            task_instruction=self.trajectory_memory_graph.nodes[start_node].task_instruction,
            subgoal_spec=json.dumps(sub_goal, indent=4),
            traj_info=json.dumps(traj_info, indent=4),
            CRAFT_COMMANDS= crafting_commands,
        )

        response = llm_response(prompt, self.llm, temperature=self.temperature)

        if self.verbose:
            print("="*20 + " prompt " + "="*20)
            print(prompt)
            print("="*20 + " response " + "="*20)
            print(response)
        
        


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
            "related_crafting_commands": json_response["related_crafting_commands"],
        }
        return function_calling_summary
    
    def extract_node_edge_errors(
        self,
        node_errors,
        prefix_errors,
        check_write_errors=None,
        check_expr_errors=None,
        control_flow_errors=None,
        control_flow_node_errors=None,
        loop_entry_errors=None,
    ):
        check_write_errors = check_write_errors or []
        check_expr_errors = check_expr_errors or []
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
        if check_expr_errors:
            print("Validation errors [check nodes must use only allowed operators/predicates]:")
            for e in check_expr_errors:
                print("  -", e)
        else:
            print("Check node expressions comply with allowed predicates/operators.")
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

        check_expr_hints = ""
        if check_expr_errors:
            check_expr_hints = "Check node expressions must only use allowed predicates, locals, and the basic boolean/comparison/arithmetic operators:\n"
            for e in check_expr_errors:
                check_expr_hints += f"  - {e}\n"
            check_expr_hints += (
                "Avoid Python helpers (e.g., all/any/zip/comprehensions), keyword arguments, attribute/subscript access, "
                "or predicates outside the allowed set; rewrite using simple predicate calls and and/or/not with ==, !=, <, >, <=, >= and +, -, *, /.\n"
            )

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
            check_expr_hints,
            control_flow_hints,
            control_flow_node_hints,
            loop_entry_hints,
        )
    
    def skill_ILP(
        self,
        sub_goal,
        synthesizer=None,
        max_local_iters: int = 10,
        max_global_iters: int = 5,
        p_merge: float = 0.5,
    ):
        """
        Two-Stage Structural ERM for Skill Induction (pseudocode scaffold).

        Args:
            expert_trajectories: iterable of expert trajectories (τ in D).
            target_sub_goal: target sub-goal g'.
            synthesizer: LLM-based synthesizer providing Consolidate/Refine.
            max_local_iters: cap for per-trajectory refinement loop.
            max_global_iters: cap for global consolidation loop.
            p_merge: probability threshold for choosing structure merge vs. param/abstraction refinement.

        Returns:
            sigma_glb: global generalized skill program.
            Sigma_local: list of locally overfit expert programs.
        """
        print("Skill Inductive Logical Program Synthesis for Sub-Goal: ", sub_goal["name"])

        synthesizer = synthesizer or getattr(self, "llm", None)
        Sigma_local: List = []
        trajectory_to_local = {}

        def _trace2code(traj_calling_id, verbose=False):
            """
            Stage-1 local synthesis: bootstrap a local program sigma_tau from a
            single expert trajectory. Builds the trajectory context (initial
            state, action sequence, available primitive actions) and induces the
            draft workflow with the LLM via induce_skill_from_single_trajectory.
            """
            calling_node = self.skill_calling_memory_graph.nodes[traj_calling_id]
            calling_node.show()


            sub_traj_info = ""
            iter_node_id = calling_node.trajectory_start_node

            action_sequence = []
            available_actions = set()
            while iter_node_id != None:
                current_node = self.trajectory_memory_graph.nodes[iter_node_id]
                # current_node.show()
                sub_traj_info += f"{current_node.observation_current}\nworld states: {current_node.facts_current}\naction: {current_node.action}\n"
                
                if current_node.action is not None:
                    if "get" in current_node.action:
                        available_actions.add("get")
                    elif "craft" in current_node.action:
                        available_actions.add("craft")
                    elif "inventory" in current_node.action:
                        available_actions.add("inventory")

                    

                    if not (current_node.action.startswith("think")): 
                        action_sequence.append(current_node.action)

                if iter_node_id == calling_node.trajectory_end_node:
                    break
                iter_node_id = current_node.trajectory_next_node_id
            
            if traj_calling_id == 3: 
                print(sub_traj_info)
            sub_traj_info += f"\n**Please stictly follow the action sequence to arange the PrimitiveAction Nodes**"
            sub_traj_info += f"\naction sequence: {'->'.join(action_sequence)}\n"

            trajectory_summary = self.trajectory_memory_graph.nodes[calling_node.trajectory_start_node].crafting_commands
            trajectory_summary += "\n" + self.trajectory_memory_graph.nodes[calling_node.trajectory_start_node].task_instruction
            trajectory_summary += "\nSummary of the interaction before the sub-goal initiation: " + calling_node.pre_span_summary + "\n"

            available_actions_str = ""
            if "inventory" in available_actions:
                available_actions_str += """    A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "get" in available_actions:
                available_actions_str += """    A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_name: ItemName = {{CURRENT_INGREDIENT}}, action_item_count: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "craft" in available_actions:
                available_actions_str += """    A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"

            skill_json = sub_goal
            skill_call_dict = {
                "subgoal": sub_goal["name"], 
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
            if verbose:
                print(skill_call_dict)
            os.makedirs(os.path.join(self.log_folder, "induced_workflow"), exist_ok=True)
            induced_workflow_dir = os.path.join(self.log_folder, "induced_workflow", str(sub_goal["name"]) + "_" + str(traj_calling_id))
            os.makedirs(induced_workflow_dir, exist_ok=True)

            valid_file = os.path.join(induced_workflow_dir, f"{sub_goal['name']}_valid.mmd")
            # valid_file = os.path.join(induced_workflow_dir, "xxx")
            # print(valid_file)
            if os.path.exists(valid_file):
                induced_file = valid_file
                mermaid_code = open(valid_file, "r").read()
                if verbose:
                    print("load valid mermaid code from file: ", valid_file)

            else:
                induced_file, mermaid_code = self.induce_skill_from_single_trajectory(
                skill_call_dict=skill_call_dict,
                trajectory_summary=trajectory_summary,
                available_actions_str=available_actions_str,
                sub_traj_info=sub_traj_info,
                llm=self.llm,
                temperature=self.temperature,
                previous_mermaid_code="",
                hints="",
                induced_workflow_dir=induced_workflow_dir,
                max_attempts=10, 
                verbose=verbose,
                )
            skill_json_for_check = {
                "name": skill_call_dict.get("subgoal", "single_trajectory_skill"),
                "description": skill_call_dict.get("description", ""),
                "parameters": skill_call_dict.get("parameters", []),
                "start_conditions": skill_call_dict.get("start_conditions", []),
                "success_conditions": skill_call_dict.get("success_conditions", []),
                "steps": skill_call_dict.get("steps", []),
                "mermaid_code": mermaid_code,
            }
            match_flag, logs = self.static_verify_with_calling(skill_json_for_check, calling_node, verbose=False)
            
            print("single-trajectory verification match_flag:", match_flag)
            if verbose:
                for key, value in logs:
                    print(f"{key}: {value}")
            if not match_flag:
                try:
                    # gradient_json = self.collect_gradient_from_failure(sub_goal["name"], traj_calling_id, calling_node.problem_node, trace_info=logs, 
                    #                                                    mermaid_code=mermaid_code, skill_json=skill_json_for_check, verbose=False)
                    
                    # print(gradient_json)

                    return {
                        "program": mermaid_code,
                        "program_file": induced_file,
                        "source_traj": traj_calling_id,
                        "coverage": 0.0,
                    }

                except Exception as e:
                    print(f"collect_gradient_from_failure failed: {e}")
            
            
            return {
                "program": mermaid_code,
                "program_file": induced_file,
                "source_traj": traj_calling_id,
                "coverage": 1.0,
            }

        def _find_error_state(traj, sigma_tau):
            """
            Consistency Check: locate an uncovered state s_err in S_g'^τ.
            Replays sigma_tau against the demo trajectory; returns the first
            divergence as a dict {skill_node_id, predicted_action, expert_action,
            traj_node_id} or None when fully consistent.
            """
            calling_node = self.skill_calling_memory_graph.nodes[traj]
            skill_json_for_check = {
                "name": sub_goal["name"],
                "description": sub_goal["description"],
                "parameters": sub_goal["parameters"],
                "start_conditions": sub_goal["start_conditions"],
                "success_conditions": sub_goal["success_conditions"],
                "steps": sub_goal["steps"],
                "mermaid_code": sigma_tau["program"],
            }
            try:
                match_flag, logs = self.static_verify_with_calling(
                    skill_json_for_check, calling_node, verbose=False
                )
            except Exception as e:
                print(f"[Stage-1] _find_error_state verify crashed: {e}")
                return {
                    "skill_node_id": "<verifier_crash>",
                    "predicted_action": "<unknown>",
                    "expert_action": "<unknown>",
                    "traj_node_id": "<unknown>",
                    "raw": str(e),
                }
            if match_flag:
                return None
            if not logs:
                return {
                    "skill_node_id": "<no_trace>",
                    "predicted_action": "<unknown>",
                    "expert_action": "<unknown>",
                    "traj_node_id": "<unknown>",
                    "raw": "verifier returned no trace",
                }
            last_node_id, last_info = logs[-1]
            m = re.search(
                r"workflow chose the `(.+?)`,\s*and does not match expert history `(.+?)`\.\s*\[node id (\d+)\]",
                str(last_info),
            )
            if m:
                return {
                    "skill_node_id": last_node_id,
                    "predicted_action": m.group(1),
                    "expert_action": m.group(2),
                    "traj_node_id": m.group(3),
                    "raw": str(last_info),
                }
            return {
                "skill_node_id": last_node_id,
                "predicted_action": "<unknown>",
                "expert_action": "<unknown>",
                "traj_node_id": "<unknown>",
                "raw": str(last_info),
            }

        def _error_to_code(error_state):
            """
            Convert uncovered state + expert continuation into a refinement hint
            string suitable for induce_skill_from_single_trajectory's `hints` slot.
            The hint exposes the four structural operators (Branching / Crossover
            / Lifting / LoopFold) so the LLM can pick the structurally appropriate
            consolidation.
            """
            hint = (
                "## Stage-1 Intra-Trajectory Refinement Hint\n"
                f"At skill node `{error_state.get('skill_node_id', '<?>')}` "
                f"(trajectory state node id `{error_state.get('traj_node_id', '<?>')}`),\n"
                f"the current workflow chose action `{error_state.get('predicted_action', '<?>')}` "
                f"but the expert took `{error_state.get('expert_action', '<?>')}`.\n"
                f"Verifier message: {error_state.get('raw', '')}\n\n"
                "Apply ONE of the four Structural Operators to consolidate:\n"
                "- **Branching**: insert a CheckOp before this PrimitiveAction with a discriminative\n"
                "  domain predicate (e.g., inventory(x, n), unavailable(x), goal(x, n)) that\n"
                "  distinguishes the contexts in which the expert chooses one action vs the other.\n"
                "- **Crossover**: if the expert action already exists elsewhere in the workflow,\n"
                "  route control to that subgraph by rebinding inputs.\n"
                "- **Lifting**: if the divergence is on a hardcoded constant (item name, count),\n"
                "  lift it to a skill-level variable derived from the trajectory's parameter bindings.\n"
                "- **LoopFold**: if the divergence is one of N similar steps, fold the repeated\n"
                "  structure into a LoopControl over the relevant list/range.\n\n"
                "Goal: extend the workflow so this trajectory step matches the expert AND\n"
                "previously matched steps remain consistent. Output the FULL revised Mermaid\n"
                "workflow (not a diff)."
            )
            return {"program": None, "hint_text": hint, "source": error_state}

        def _consolidate(base_prog, refinement_prog, traj_calling_id, iter_idx):
            """
            Invoke LM.Consolidate: re-induce a single-trajectory workflow that uses
            base_prog as starting context and refinement_prog['hint_text'] as the
            structural-consolidation directive. Returns a sigma dict with the new
            program string, or None when the LLM call fails.
            """
            if synthesizer is None or refinement_prog is None or refinement_prog.get("hint_text") is None:
                return None

            calling_node = self.skill_calling_memory_graph.nodes[traj_calling_id]

            sub_traj_info = ""
            iter_node_id = calling_node.trajectory_start_node
            action_sequence: List[str] = []
            available_actions: Set[str] = set()
            while iter_node_id is not None:
                current_node = self.trajectory_memory_graph.nodes[iter_node_id]
                sub_traj_info += (
                    f"{current_node.observation_current}\n"
                    f"world states: {current_node.facts_current}\n"
                    f"action: {current_node.action}\n"
                )
                if current_node.action is not None:
                    if "get" in current_node.action:
                        available_actions.add("get")
                    elif "craft" in current_node.action:
                        available_actions.add("craft")
                    elif "inventory" in current_node.action:
                        available_actions.add("inventory")
                    if not current_node.action.startswith("think"):
                        action_sequence.append(current_node.action)
                if iter_node_id == calling_node.trajectory_end_node:
                    break
                iter_node_id = current_node.trajectory_next_node_id

            sub_traj_info += "\n**Please stictly follow the action sequence to arange the PrimitiveAction Nodes**"
            sub_traj_info += f"\naction sequence: {'->'.join(action_sequence)}\n"

            trajectory_summary = self.trajectory_memory_graph.nodes[calling_node.trajectory_start_node].crafting_commands
            trajectory_summary += "\n" + self.trajectory_memory_graph.nodes[calling_node.trajectory_start_node].task_instruction
            trajectory_summary += "\nSummary of the interaction before the sub-goal initiation: " + calling_node.pre_span_summary + "\n"

            available_actions_str = ""
            if "inventory" in available_actions:
                available_actions_str += """    A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "get" in available_actions:
                available_actions_str += """    A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_name: ItemName = {{CURRENT_INGREDIENT}}, action_item_count: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "craft" in available_actions:
                available_actions_str += """    A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"

            skill_call_dict = {
                "subgoal": sub_goal["name"],
                "description": sub_goal["description"],
                "steps": sub_goal["steps"],
                "start_conditions": sub_goal["start_conditions"],
                "start_condition_evidence": calling_node.start_condition_evidence,
                "success_conditions": sub_goal["success_conditions"],
                "pre_span_summary": calling_node.pre_span_summary,
                "subgoal_initiation_intent": calling_node.subgoal_initiation_intent,
                "parameters": sub_goal["parameters"],
                "parameter_bindings": calling_node.parameter_bindings,
            }

            iter_dir = os.path.join(
                self.log_folder, "induced_workflow",
                f"{sub_goal['name']}_{traj_calling_id}_iter_{iter_idx}",
            )
            os.makedirs(iter_dir, exist_ok=True)
            iter_valid_file = os.path.join(iter_dir, f"{sub_goal['name']}_valid.mmd")

            if os.path.exists(iter_valid_file):
                mermaid_code = open(iter_valid_file, "r").read()
                induced_file = iter_valid_file
                print(f"[Stage-1] _consolidate cache hit at iter {iter_idx}: {iter_valid_file}")
            else:
                try:
                    induced_file, mermaid_code = self.induce_skill_from_single_trajectory(
                        skill_call_dict=skill_call_dict,
                        trajectory_summary=trajectory_summary,
                        available_actions_str=available_actions_str,
                        sub_traj_info=sub_traj_info,
                        llm=self.llm,
                        temperature=self.temperature,
                        previous_mermaid_code=base_prog["program"] or "",
                        hints=refinement_prog["hint_text"],
                        induced_workflow_dir=iter_dir,
                        max_attempts=6,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"[Stage-1] _consolidate induce crashed at iter {iter_idx}: {e}")
                    return None

                if mermaid_code is None or not str(mermaid_code).strip():
                    return None

                with open(iter_valid_file, "w") as f:
                    f.write(mermaid_code)

            return {
                "program": mermaid_code,
                "program_file": induced_file,
                "source_traj": traj_calling_id,
                "coverage": None,
                "iter_idx": iter_idx,
                "hint": refinement_prog["hint_text"],
            }

        def _coverage_expanded(old_prog, new_prog, traj):
            """
            Stage-1 empirical guardrail: True iff `new_prog` covers strictly more
            of the single trajectory `traj` than `old_prog`. Sets
            new_prog["coverage"] in place so the caller can read it.
            For TextCraft single-calling Stage-1 the per-traj coverage is binary
            (match / no match), so the strict-`>` check reduces to
            `new_match and not old_match`.
            """
            if new_prog is None or new_prog.get("program") is None:
                return False
            calling_node = self.skill_calling_memory_graph.nodes[traj]
            skill_json_for_check = {
                "name": sub_goal["name"],
                "description": sub_goal["description"],
                "parameters": sub_goal["parameters"],
                "start_conditions": sub_goal["start_conditions"],
                "success_conditions": sub_goal["success_conditions"],
                "steps": sub_goal["steps"],
                "mermaid_code": new_prog["program"],
            }
            try:
                match_flag, _ = self.static_verify_with_calling(
                    skill_json_for_check, calling_node, verbose=False
                )
            except Exception as e:
                print(f"[Stage-1] _coverage_expanded verify crashed: {e}")
                new_prog["coverage"] = 0.0
                return False
            new_cov = 1.0 if match_flag else 0.0
            new_prog["coverage"] = new_cov
            old_cov = float(old_prog.get("coverage", 0.0)) if old_prog else 0.0
            return new_cov > old_cov

        def _best_of(calling_list, local_pool):
            """
            Select initial global candidate with highest coverage (BestOf).
            """
            if not local_pool:
                return None
            
            best_idx = 0
            for idx, sigma_tau in enumerate(local_pool):
                
                traj_cnt, traj_succ = 0, 0
                sigma_tau["succ_call_idx"] = []

                for _, calling_node_idx in enumerate(calling_list):
                    calling_node = self.skill_calling_memory_graph.nodes[calling_node_idx]
                    mermaid_code = sigma_tau["program"]
                    skill_json_for_check = {
                        "name": sub_goal["name"],
                        "description": sub_goal["description"],
                        "parameters": sub_goal["parameters"],
                        "start_conditions": sub_goal["start_conditions"],
                        "success_conditions": sub_goal["success_conditions"],
                        "steps": sub_goal["steps"],
                        "mermaid_code": mermaid_code,
                    }
                    match_flag, logs = self.static_verify_with_calling(skill_json_for_check, calling_node, verbose=True)
                    
                    traj_cnt += 1
                    traj_succ += int(match_flag)

                    if not match_flag:
                        
                        for key, value in logs:
                            print(f"{key}: {value}")
                        # exit(0)
                    else:
                        sigma_tau["succ_call_idx"].append(calling_node_idx)
                        
                sigma_tau["coverage_across_traj"] = traj_succ / traj_cnt
                if sigma_tau["coverage_across_traj"] > local_pool[best_idx]["coverage_across_traj"]:
                    best_idx = idx
                print("local tau from the skill-calling node: ", sigma_tau["source_traj"], "success traj: ", sigma_tau["succ_call_idx"])
                
            for sigma_tau in local_pool:
                print("local expert from the skill-calling node: ", calling_node_idx, "success ratio: ", sigma_tau["coverage_across_traj"])
            

            return local_pool[best_idx], local_pool

        def _hardest_trajectory(trajs, sigma_glb):
            """
            Identify τ_hard = argmin_τ Consistency(sigma_glb, τ).
            """
            if not trajs or sigma_glb is None:
                return None
            
            print(sigma_glb.keys())
            # print(sigma_glb)
            for calling_idx in trajs:

                # print("calling_idx: ", calling_idx, "succ_call_idx: ", sigma_glb["succ_call_idx"])
                if calling_idx in sigma_glb["succ_call_idx"]:
                    continue
                else:
                    return calling_idx
            return None

        def _should_merge():
            """Flip a biased coin to decide Consolidate vs. Refine branch."""
            return random.random() < p_merge

        def _apply_generalizer(sigma_glb, traj, retrive_sigma=None):
            """
            Apply abstraction/generalization operator (variable lifting, loop folding, etc.).
            """
            if synthesizer is None:
                return sigma_glb
            # Stage-2 generalization: assemble the inter-trajectory consolidation
            # context (sigma_glb as the success workflow, plus a donor/failed
            # workflow and their matched trajectory records), then induce a
            # lifted workflow with the LLM generalizer.

            action_sequence = []
            available_actions = set()

            success_record = ""

            success_record += "## Applicable Mermaid-style Workflow 1\n"
            success_record += sigma_glb["program"] + "\n"
            cnt = 0
            for calling_idx in sigma_glb["succ_call_idx"]:

                calling_node = self.skill_calling_memory_graph.nodes[calling_idx]
                traj_node_idx = calling_node.trajectory_start_node
                traj_node = self.trajectory_memory_graph.nodes[traj_node_idx]
                
                cnt += 1
                traj_info_i = f"### Mathced Data Record {cnt}, {traj_node.task_instruction}\n"
                traj_info_i += f"{traj_node.crafting_commands}\n"

                traj_info_i += " - Problem Initial Description: {}".format(self.trajectory_memory_graph.nodes[calling_node.problem_node].observation_current.replace('obs: ', '')) + "\n"
                traj_info_i += " - Pre-skill Interaction Summary: " + calling_node.pre_span_summary + "\n"
                traj_info_i += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
                traj_info_i += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"

                while traj_node_idx is not None:
                    traj_node = self.trajectory_memory_graph.nodes[traj_node_idx]

                    traj_info_i += f"{traj_node.observation_current}\nworld states: {traj_node.facts_current}\naction: {traj_node.action}\n"
                    
                    if traj_node.action is not None:
                        if "get" in traj_node.action:
                            available_actions.add("get")
                        elif "craft" in traj_node.action:
                            available_actions.add("craft")
                        elif "inventory" in traj_node.action:
                            available_actions.add("inventory")

                    
                    if traj_node_idx == calling_node.trajectory_end_node:
                        break
                    traj_node_idx = traj_node.trajectory_next_node_id
                
                success_record += traj_info_i + "\n"
            

            
            failed_recorad = ""
            failed_recorad += "## Applicable Mermaid-style Workflow 2\n"
            
            if retrive_sigma is not None:
                failed_recorad += retrive_sigma["program"] + "\n"
            else:
                failed_recorad += "No applicable Mermaid-style Workflow for this trajectory.\n"

            cnt = 0
            calling_idx = traj

            calling_node = self.skill_calling_memory_graph.nodes[calling_idx]
            traj_node_idx = calling_node.trajectory_start_node
            traj_node = self.trajectory_memory_graph.nodes[traj_node_idx]
            
            cnt += 1
            traj_info_i = f"### Mathced Data Record {cnt}, {traj_node.task_instruction}\n"
            traj_info_i += f"{traj_node.crafting_commands}\n"

            traj_info_i += " - Problem Initial Description: {}".format(self.trajectory_memory_graph.nodes[calling_node.problem_node].observation_current.replace('obs: ', '')) + "\n"
            traj_info_i += " - Pre-skill Interaction Summary: " + calling_node.pre_span_summary + "\n"
            traj_info_i += " - Skill-Call Rationale: " + calling_node.subgoal_initiation_intent + "\n"
            traj_info_i += " - Parameter Bindings: " + json.dumps(calling_node.parameter_bindings, indent=4) + "\n"

            while traj_node_idx is not None:
                traj_node = self.trajectory_memory_graph.nodes[traj_node_idx]

                traj_info_i += f"{traj_node.observation_current}\nworld states: {traj_node.facts_current}\naction: {traj_node.action}\n"
                
                if traj_node.action is not None:
                    if "get" in traj_node.action:
                        available_actions.add("get")
                    elif "craft" in traj_node.action:
                        available_actions.add("craft")
                    elif "inventory" in traj_node.action:
                        available_actions.add("inventory")

                
                if traj_node_idx == calling_node.trajectory_end_node:
                    break
                traj_node_idx = traj_node.trajectory_next_node_id
            
            failed_recorad += traj_info_i + "\n"
            

            # if self.verbose:
            #     print("-" * 50 + "Success Record" + "-" * 50)
            #     print(success_record)
            #     print("-" * 50 + "Failed Record" + "-" * 50)
            #     print(failed_recorad)

            workflow_records = success_record + "\n" + failed_recorad
            
            available_actions_str = ""
            if "inventory" in available_actions:
                available_actions_str += """    A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "get" in available_actions:
                available_actions_str += """    A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_name: ItemName = {{CURRENT_INGREDIENT}}, action_item_count: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"
            if "craft" in available_actions:
                available_actions_str += """    A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction""" + "\n"


            os.makedirs(os.path.join(self.log_folder, "induced_workflow"), exist_ok=True)
            generalized_workflow_dir = os.path.join(self.log_folder, "induced_workflow", str(sub_goal["name"]) + "_" + "generalizer")
            os.makedirs(generalized_workflow_dir, exist_ok=True)

            valid_file = os.path.join(generalized_workflow_dir, f"{sub_goal['name']}_valid.mmd")
            # valid_file = os.path.join(generalized_workflow_dir, "xxx")
            # print(valid_file)

            mermaid_code = None
            if os.path.exists(valid_file):
                generalized_file = valid_file
                mermaid_code = open(valid_file, "r").read()
                if self.verbose:
                    print("load valid mermaid code from file: ", valid_file)

            else:
                valid_file, evolved_mermaid_code = self.evolve_skill(sub_goal_json=sub_goal, 
                                  workflow_records=workflow_records, 
                                  available_actions_str=available_actions_str, 
                                  previous_mermaid_code=sigma_glb["program"],
                                  generalized_workflow_dir=generalized_workflow_dir)
                if valid_file is not None:
                    generalized_file = valid_file
                    mermaid_code = evolved_mermaid_code

            
            # print(sigma_glb["program"])

            # print("-" * 50)

            # print(mermaid_code)
            # exit(0)


            return {"program": mermaid_code, "parent": sigma_glb, "hint_traj": traj}

        def _verify_improvement(current_prog, prog_baseline, calling_list):
            """
            Empirical verification step: ensure candidate keeps prior coverage and fixes τ_hard.
            """
            traj_cnt, traj_succ = 0, 0
            current_prog["succ_call_idx"] = []
            

            for _, calling_node_idx in enumerate(calling_list):
                calling_node = self.skill_calling_memory_graph.nodes[calling_node_idx]
                mermaid_code = current_prog["program"]
                skill_json_for_check = {
                    "name": sub_goal["name"],
                    "description": sub_goal["description"],
                    "parameters": sub_goal["parameters"],
                    "start_conditions": sub_goal["start_conditions"],
                    "success_conditions": sub_goal["success_conditions"],
                    "steps": sub_goal["steps"],
                    "mermaid_code": mermaid_code,
                }
                match_flag, logs = self.static_verify_with_calling(skill_json_for_check, calling_node, verbose=True)
                
                traj_cnt += 1
                traj_succ += int(match_flag)

                if not match_flag:
                    
                    for key, value in logs:
                        print(f"{key}: {value}")
                    # exit(0)
                else:
                    current_prog["succ_call_idx"].append(calling_node_idx)
                    
            current_prog["coverage_across_traj"] = traj_succ / traj_cnt

            print("current_prog coverage_across_traj: ", current_prog["coverage_across_traj"])
            print("prog_baseline coverage_across_traj: ", prog_baseline["coverage_across_traj"])


            if current_prog["coverage_across_traj"] >= prog_baseline["coverage_across_traj"]:
                return True, current_prog["succ_call_idx"]
            else:
                return False, current_prog["succ_call_idx"]
                
        

        expert_trajectories_calling_list = []

        # print skill-calling memory for current sub-goal
        for calling_id in self.skill_calling_memory_graph.nodes_by_subgoal[sub_goal["name"]]:
            self.skill_calling_memory_graph.nodes[calling_id].show()
            iter_node_idx = self.skill_calling_memory_graph.nodes[calling_id].trajectory_start_node
            
            while iter_node_idx is not None:
                iter_node = self.trajectory_memory_graph.nodes[iter_node_idx]
                iter_node.show()
                print(iter_node.facts_current)

                if iter_node_idx == self.skill_calling_memory_graph.nodes[calling_id].trajectory_end_node:
                    break
                iter_node_idx = iter_node.trajectory_next_node_id

            
            
            expert_trajectories_calling_list.append(calling_id)
        # exit(0)

        # exit(0)
        
        local_skill_json_list = []

        # Stage 1: Intra-Trajectory Logic Consolidation (local experts)
        for idx, tau in enumerate(expert_trajectories_calling_list):

            sigma_tau = _trace2code(tau)
            print("program_file:\n", sigma_tau["program_file"])
            print("source_traj:\n", sigma_tau["source_traj"])
            print("coverage (initial):\n", sigma_tau["coverage"])

            # Stage-1 intra-trajectory refinement loop (paper §5.1):
            # iterate consistency-check + structural-consolidation until the
            # local skill replays this single demo, capped by max_local_iters
            # with break-on-no-progress (2 consecutive non-improving iters).
            no_progress = 0
            for k in range(max_local_iters):
                if sigma_tau.get("coverage", 0.0) == 1.0:
                    break
                s_err = _find_error_state(tau, sigma_tau)
                if s_err is None:
                    break
                sigma_err = _error_to_code(s_err)
                sigma_new = _consolidate(sigma_tau, sigma_err, traj_calling_id=tau, iter_idx=k + 1)
                if sigma_new is None:
                    no_progress += 1
                    print(f"[Stage-1] tau={tau} iter {k+1}: induce returned None (no_progress={no_progress})")
                    if no_progress >= 2:
                        print(f"[Stage-1] tau={tau}: break-on-no-progress at iter {k+1}")
                        break
                    continue
                if _coverage_expanded(sigma_tau, sigma_new, tau):
                    print(
                        f"[Stage-1] tau={tau} iter {k+1}: refinement accepted "
                        f"(cov {sigma_tau.get('coverage', 0.0):.2f} -> {sigma_new.get('coverage', 0.0):.2f})"
                    )
                    sigma_tau = sigma_new
                    no_progress = 0
                else:
                    no_progress += 1
                    print(
                        f"[Stage-1] tau={tau} iter {k+1}: refinement rejected "
                        f"(cov {sigma_tau.get('coverage', 0.0):.2f} vs cand {sigma_new.get('coverage', 0.0):.2f}, no_progress={no_progress})"
                    )
                    if no_progress >= 2:
                        print(f"[Stage-1] tau={tau}: break-on-no-progress at iter {k+1}")
                        break

            print("coverage (final):\n", sigma_tau.get("coverage", 0.0))
            if sigma_tau.get("coverage", 0.0) == 0.0:
                continue
            Sigma_local.append(sigma_tau)
            trajectory_to_local[idx] = sigma_tau

        # Stage 2: Inter-Trajectory Structural Consolidation (global)
        sigma_glb, Sigma_local = _best_of(expert_trajectories_calling_list, Sigma_local)
        
        for ss in Sigma_local:
            print("sigma_tau ratio: ", ss["coverage_across_traj"], ", from calling id: ", ss["source_traj"], "success record idx: ", ss["succ_call_idx"])
        print("sigma_glb ratio: ", sigma_glb["coverage_across_traj"], ", from calling id: ", sigma_glb["source_traj"], "success record idx: ", sigma_glb["succ_call_idx"])

        if sigma_glb is None:
            return None, Sigma_local

        traj_collection = deepcopy(expert_trajectories_calling_list)

        for iter_idx in range(max_global_iters):

            current_glb_success_ratio = sigma_glb["coverage_across_traj"]

            tau_hard = _hardest_trajectory(traj_collection, sigma_glb)
            if tau_hard is None:
                # (D) All trajs already covered by sigma_glb (e.g. after the
                # craft-action fallback in `_match_craft_command` made every
                # local sigma cross-cover). Per paper §5.2, Stage-2's job is
                # still to consolidate multiple locals into one structurally
                # general skill — pick a secondary local as donor and force
                # one structural-pass at iter 0 so Lifting/LoopFold can fire.
                if iter_idx == 0 and len(Sigma_local) >= 2:
                    secondaries = [s for s in Sigma_local if s is not sigma_glb]
                    if secondaries:
                        donor = max(secondaries, key=lambda s: s.get("coverage_across_traj", 0.0))
                        donor_traj = donor.get("source_traj")
                        if donor_traj is not None:
                            print(f"[Stage-2] forcing structural pass at iter 0: donor source_traj={donor_traj}")
                            tau_hard = donor_traj
                            retrive_sigma = donor
                            sigma_prop = _apply_generalizer(sigma_glb, tau_hard, retrive_sigma)
                            if sigma_prop["program"] is not None:
                                improve_flag, succ_call_idx = _verify_improvement(
                                    sigma_prop, sigma_glb, expert_trajectories_calling_list
                                )
                                if improve_flag:
                                    sigma_glb = sigma_prop
                                    sigma_glb["succ_call_idx"] = succ_call_idx
                                    print(
                                        f"[Stage-2] structural pass accepted "
                                        f"(cov {current_glb_success_ratio:.2f} -> {sigma_glb['coverage_across_traj']:.2f})"
                                    )
                                else:
                                    print("[Stage-2] structural pass rejected by _verify_improvement")
                # Whether the structural pass fired or not, no more hard trajs
                # remain — exit the main loop.
                break

            maxx_succ_ratio_from_tau_hard = 0.
            retrive_sigma = None
            for ss in Sigma_local:
                if tau_hard in ss["succ_call_idx"]:
                    if ss["coverage_across_traj"] > maxx_succ_ratio_from_tau_hard:
                        maxx_succ_ratio_from_tau_hard = ss["coverage_across_traj"]
                        retrive_sigma = ss

            # When no donor covers tau_hard, still invoke the generalizer
            # with (sigma_glb, tau_hard, donor=None) — the LLM is asked to
            # expand sigma_glb directly from the hard trajectory.
            sigma_prop = _apply_generalizer(sigma_glb, tau_hard, retrive_sigma)

            if sigma_prop["program"] is None:
                # Drop tau_hard from the candidate pool and try the next iter.
                if tau_hard in traj_collection:
                    traj_collection.remove(tau_hard)
                continue

            

            improve_flag, succ_call_idx = _verify_improvement(sigma_prop, sigma_glb, expert_trajectories_calling_list)
            if improve_flag:
                sigma_glb = sigma_prop
                sigma_glb["succ_call_idx"] = succ_call_idx
                print(f"success ratio: {current_glb_success_ratio:.2f} -> {sigma_glb['coverage_across_traj']:.2f}")
            else:
                sigma_prop["succ_call_idx"] = succ_call_idx

                

            if sigma_glb["coverage_across_traj"] >= 1.0:
                break

            if iter_idx >= max_global_iters:
                break
        
        ilp_dir = os.path.join(self.log_folder, "ilp_workflow")
        if not os.path.exists(ilp_dir):
            os.makedirs(ilp_dir)
        ilp_file = os.path.join(ilp_dir, f"{sub_goal['name']}_valid.json")
        skill_json_copy = deepcopy(sub_goal)
        skill_json_copy["mermaid_code"] = sigma_glb["program"]
        with open(ilp_file, "w") as f:
            json.dump(skill_json_copy, f, indent=4)
        code_file = os.path.join(ilp_dir, f"{sub_goal['name']}_valid.mmd")
        with open(code_file, "w") as f:
            f.write(sigma_glb["program"])
        
        
        return ilp_file
    
    def induce_sub_goal_skills_evolution(self, sub_goal):

        print(json.dumps(sub_goal, indent=4))

        

        candidates_pool = []
        for instance_i in sub_goal["instances"]:

            sub_goal_info = deepcopy(sub_goal)

            sub_goal_info.pop("instances")
            sub_goal_info["concrete_outcome"] = instance_i["concrete_outcome"]

            print(instance_i)

            trajectory_id = instance_i["trajectory_id"]
            sub_goal_terminal_step = instance_i["step_id"]

            node_id = self.trajectory_memory_graph.trajectory_head_list[trajectory_id - 1]
            current_node = self.trajectory_memory_graph.nodes[node_id]
            task_instruction_for_instance = current_node.observation_current.replace('obs: ', '')
            

            traj_info = []
            index_c = 0
            node_idx = []
            
            while node_id is not None:
                current_node = self.trajectory_memory_graph.nodes[node_id]
                node_idx.append(node_id)
                index_c += 1
                current_node.show()
                traj_info.append({
                    "index": index_c,
                    "observation": current_node.observation_current.split("obs: ")[1].strip(), 
                    "action": current_node.action.strip() if current_node.action is not None else "",
                })

                # if index_c == sub_goal_terminal_step:
                #     break
                node_id = current_node.trajectory_next_node_id
            
            prompt = TRAJ_SEGMENTATION_PROMPT.format(
                task_instruction=task_instruction_for_instance,
                subgoal_spec=json.dumps(sub_goal_info, indent=4),
                traj_info=json.dumps(traj_info, indent=4),
            )
            response = llm_response(prompt, self.llm, self.temperature)
            print(response)

            continue
            exit(0)



            

            

            different_suffixes = []
            num_suffixes = 0

            node_id = self.trajectory_memory_graph.trajectory_head_list[trajectory_id - 1]
            current_node = self.trajectory_memory_graph.nodes[node_id]


            for pre_idx, _ in enumerate(range(sub_goal_terminal_step)):

                if node_id is None:
                    break
                
                current_node = self.trajectory_memory_graph.nodes[node_id]
                # current_node.show()
                current_suffix = "### Example {example_id}, overall task goal:" + current_node.task_instruction + "\n"
                current_suffix += " - Problem Initial Description: {}".format(task_instruction_for_instance) + "\n"
                current_suffix += " - Pre-skill Interaction Facts: " + str(current_node.facts_current) + "\n"
                different_suffixes.append(current_suffix)
                num_suffixes += 1

                node_data = self.trajectory_memory_graph.nodes[node_id]
                update_text = node_data.observation_current.strip() + "\n"
                if node_data.trajectory_next_node_id is not None:
                    update_text += f"current_predicates: {node_data.facts_current}\n"
                    update_text += f"action: {node_data.action.strip()}\n"
                    update_text += f"intent_for_action: {node_data.intent_for_action.strip()}\n"
                else:
                    update_text += "The Overall Task Success!\n"

                for suffix_idx in range(len(different_suffixes)):
                    different_suffixes[suffix_idx] += update_text
                
                node_id = current_node.trajectory_next_node_id
            
            print("-" * 50 + f"{num_suffixes} suffixes" + "-" * 50)

            for suffix_i in range(num_suffixes):
                print(different_suffixes[suffix_i].format(example_id=suffix_i + 1))
                print("-" * 50)
            
        # exit(0)
        


        


            

        


        # exit(0)
        
    def _sanitize_input_name(self, name: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip()).upper()
        if not safe:
            safe = "INPUT"
        if not safe.endswith("_INPUT"):
            safe = f"{safe}_INPUT"
        return safe

    def _format_interface_block(self, parameters: List[str]) -> Tuple[str, str, str]:
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
            interface_lines = [f"{self._sanitize_input_name(name)}: {ptype}" for name, ptype in parsed_params]
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
            input_name = self._sanitize_input_name(name)
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
                    traj_i += f"current_predicates: {self.trajectory_memory_graph.nodes[node_id].facts_current}" + "\n"
                    traj_i += f"action: {self.trajectory_memory_graph.nodes[node_id].action.strip()}" + "\n"
                    traj_i += f"intent_for_action: {self.trajectory_memory_graph.nodes[node_id].intent_for_action.strip()}" + "\n"
                else:
                    traj_i += "The Overall Task Success!" + "\n"

                node_id = self.trajectory_memory_graph.nodes[node_id].trajectory_next_node_id
            
            sub_goal_data_collection += traj_i + "\n\n"
        
        raw_name = sub_goal["name"]
        sub_goal["name"] = f"{raw_name}_v0"

        # print(sub_goal_data_collection)
        # exit(0)

        induced_skill_file, mermaid_code_i = self.induce_skill(sub_goal, sub_goal_data_collection, self.llm, temperature=self.temperature, 
                                                               hints="The current workflow is an example. Please refer to it when designing a new workflow to fit the needs of the sub-goals.")
        induced_files_list.append(induced_skill_file)
        
        print(sub_goal)
        print(mermaid_code_i)
        
        current_result_file = induced_skill_file

        graph_i = parse_mermaid_with_info(mermaid_code_i)


        result_file = None

        for error_try in range(11):
            
            node_errors = validate_node_inputs(graph_i)
            prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
            check_write_errors = validate_check_nodes_without_global_writes(graph_i)
            check_expr_errors = validate_check_expr_syntax(graph_i)
            control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
            control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
            loop_entry_errors = validate_loop_entry_edges(graph_i)
            # action_shape_errors = validate_action_shape(graph_i)

            generation_file = current_result_file.replace(".mmd", f"_generation.json")
            with open(generation_file, "r") as f:
                generation_log = json.load(f)
            generation_log["node_errors"] = node_errors
            generation_log["prefix_errors"] = prefix_errors
            generation_log["check_write_errors"] = check_write_errors
            generation_log["check_expr_errors"] = check_expr_errors
            generation_log["control_flow_errors"] = control_flow_errors
            generation_log["control_flow_node_errors"] = control_flow_node_errors
            generation_log["loop_entry_errors"] = loop_entry_errors
            with open(generation_file, "w") as f:
                json.dump(generation_log, f, indent=4)
            print("update error for the mermaid generation logs >>>", generation_file)
            
            # exit(0)
            if error_try >= 10:
                break
            
            
            if (
                len(node_errors)
                + len(prefix_errors)
                + len(check_write_errors)
                + len(check_expr_errors)
                + len(control_flow_errors)
                + len(control_flow_node_errors)
                + len(loop_entry_errors)
                # + len(action_shape_errors)
                == 0
            ):
                result_file = current_result_file
                break
            else:
                (
                    node_payloads_hints,
                    prefix_errors_hints,
                    check_write_hints,
                    check_expr_hints,
                    control_flow_hints,
                    control_flow_node_hints,
                    loop_entry_hints,
                ) = self.extract_node_edge_errors(
                    node_errors,
                    prefix_errors,
                    check_write_errors,
                    check_expr_errors,
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
                hints=node_payloads_hints + prefix_errors_hints + check_write_hints + check_expr_hints + control_flow_hints + control_flow_node_hints + loop_entry_hints,
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
    
    def evolve_skill(self, sub_goal_json, workflow_records, 
            previous_mermaid_code="",
            hints="",
            available_actions_str="""
        A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction
        A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_count: ItemName = {{CURRENT_INGREDIENT}}, action_item_name: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction
        A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS_LIST}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction
        """, 

        generalized_workflow_dir=None,
        max_attempts: int = 10,
        verbose: bool = None):

        if verbose is None:
            verbose = self.verbose

        skill_name = sub_goal_json["name"]
        skill_name_safe = re.sub(r"[^0-9A-Za-z_]+", "_", skill_name)
        _action_node_re = re.compile(r'([A-Za-z0-9_]+)\["PrimitiveAction:\s*<br>(.*?)"\]\s*:::PrimitiveAction', re.DOTALL)


        if generalized_workflow_dir is None:
            generalized_workflow_dir = os.path.join(self.log_folder, "generalized_workflow", skill_name_safe)
        os.makedirs(generalized_workflow_dir, exist_ok=True)

        

        def _canonical_action_body(body: str) -> str:
            def _anon_local_in(match: re.Match) -> str:
                local_in = re.sub(r"\{\{[^}]*\}\}", "{{}}", match.group(2))
                return f"{match.group(1)}{local_in}{match.group(3)}"

            body = re.sub(r"(local in:\s*\()(.*?)(\))", _anon_local_in, body, flags=re.DOTALL)
            body = re.sub(r"\s+", "", body)
            return body

        def _extract_action_shapes(src: str) -> List[Tuple[str, str]]:
            return [(nid, _canonical_action_body(body)) for nid, body in _action_node_re.findall(src or "")]

        allowed_action_shapes = set(shape for _, shape in _extract_action_shapes(available_actions_str))
        start_block, flow_spec_block, d_init_block = self._format_interface_block(sub_goal_json.get("parameters", []))
        

        # print(available_actions_str)
        # print(allowed_action_shapes)

        # exit(0)

        current_mermaid_code = previous_mermaid_code
        current_hints = hints
        result_file = None
        valid_mermaid_code = None

        for attempt in range(max_attempts):

            if current_mermaid_code == "":
                raise ValueError("Empty seed mermaid code provided.")
            else:
                previous_mermaid_components = extract_nodes_from_mermaid_code(current_mermaid_code)

            prompt = SUB_GOAL_WORKFLOW_GENERALIZER_PROMPT.format(
                sub_goal=json.dumps(sub_goal_json, indent=4),
                workflow_records=json.dumps(workflow_records, indent=4),
                FLOW_SPEC_NODE="    " + flow_spec_block,
                START_INTERFACE_NODE="    " + start_block,
                ACTION_NODE=previous_mermaid_components["primitive_actions"], 
                AVAILABLE_ACTIONS=available_actions_str,
                LOOP_FOR_NODE=previous_mermaid_components["loop_control"],
                CHECK_NODE=previous_mermaid_components["checks"],
                D_INIT_NODE= "    " + d_init_block,
                DATA_OP_NODE= previous_mermaid_components["data_operations"]["other_nodes"],
                CLASS_ASSIGNMENTS=previous_mermaid_components["node_class_assignments"], 
                CONTROL_FLOW_EDGE=previous_mermaid_components["control_flow_edges"],
                HINTS=current_hints,
                )
    
            response = llm_response(prompt, model=self.llm, temperature=self.temperature)
            
            
            mermaid_code = response.strip()
            if mermaid_code.startswith("```mermaid"):
                mermaid_code = mermaid_code[len("```mermaid"):]
            if mermaid_code.endswith("```"):
                mermaid_code = mermaid_code[:-3]
            mermaid_code = _fix_single_brace_placeholders(mermaid_code.strip())
            
            if verbose:
                print(mermaid_code)

                print("-"* 50 + "prompt" + "-"*50)
                print(prompt)
                
                print("-"* 50 + "response" + "-"*50)
                print(mermaid_code)
                print(sub_goal_json)

                # exit(0)

            
            induced_file = os.path.join(generalized_workflow_dir, f"{skill_name_safe}_v{attempt}.mmd")
            induced_record_file = os.path.join(generalized_workflow_dir, f"{skill_name_safe}_v{attempt}_generation.json")

            with open(induced_file, "w") as f:
                f.write(mermaid_code)
            print(f"induced workflow saved to {induced_file}")

            try:
                graph_i = parse_mermaid_with_info(mermaid_code)
                node_errors = validate_node_inputs(graph_i)
                prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
                check_write_errors = validate_check_nodes_without_global_writes(graph_i)
                check_expr_errors = validate_check_expr_syntax(graph_i)
                control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
                control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
                loop_entry_errors = validate_loop_entry_edges(graph_i)
            except Exception as e:
                node_errors = [f"Parsing failure: {e}"]
                prefix_errors = []
                check_write_errors = []
                check_expr_errors = []
                control_flow_errors = []
                control_flow_node_errors = []
                loop_entry_errors = []
            action_shape_errors = []
            for node_id, canon_body in _extract_action_shapes(mermaid_code):
                if canon_body not in allowed_action_shapes:
                    action_shape_errors.append(
                        f"Action node {node_id} does not match any available action template (after anonymizing bindings)."
                    )

            generation_json = {
                "prompt": prompt,
                "mermaid_file": induced_file, 
                "skill_call_dict": sub_goal_json,
                "sub_traj_info": workflow_records,
                "node_errors": node_errors,
                "prefix_errors": prefix_errors,
                "check_write_errors": check_write_errors,
                "check_expr_errors": check_expr_errors,
                "control_flow_errors": control_flow_errors,
                "control_flow_node_errors": control_flow_node_errors,
                "loop_entry_errors": loop_entry_errors,
                "action_shape_errors": action_shape_errors,
            }
            with open(induced_record_file, "w") as f:
                json.dump(generation_json, f, indent=4)
            print(f"induced record saved to {induced_record_file}")
            
            if mermaid_code == 'No Solution':
                print("No Solution")
                raise Exception(f"No Solution.\n LOG FILE: {induced_record_file}")

            if (
                len(node_errors)
                + len(prefix_errors)
                + len(check_write_errors)
                + len(check_expr_errors)
                + len(control_flow_errors)
                + len(control_flow_node_errors)
                + len(loop_entry_errors)
                + len(action_shape_errors)
                == 0
            ):
                result_file = induced_file
                valid_mermaid_code = mermaid_code
                break
            else:
                (
                    node_payloads_hints,
                    prefix_errors_hints,
                    check_write_hints,
                    check_expr_hints,
                    control_flow_hints,
                    control_flow_node_hints,
                    loop_entry_hints,
                ) = self.extract_node_edge_errors(
                    node_errors,
                    prefix_errors,
                    check_write_errors,
                    check_expr_errors,
                    control_flow_errors,
                    control_flow_node_errors,
                    loop_entry_errors,
                )
                action_shape_hint = ""
                if action_shape_errors:
                    action_shape_hint = "PrimitiveAction nodes must align with available action templates (ignoring binding globals). Allowed canonical forms:\\n"
                    for shape in allowed_action_shapes:
                        action_shape_hint += f"  - {shape}\\n"
                    # prune invalid action nodes from the previous code before the next iteration
                    invalid_node_ids = [err.split(" ")[2] for err in action_shape_errors if err.startswith("Action node")]
                    for nid in invalid_node_ids:
                        pattern = rf".*{nid}\\[.*\\]\\s*:::PrimitiveAction.*\\n?"
                        current_mermaid_code = re.sub(pattern, "", current_mermaid_code)
                current_hints = (
                    node_payloads_hints
                    + prefix_errors_hints
                    + check_write_hints
                    + check_expr_hints
                    + control_flow_hints
                    + control_flow_node_hints
                    + loop_entry_hints
                    + action_shape_hint
                )
                current_mermaid_code = mermaid_code

                # if attempt >= 3:
                #     print(current_hints)
                #     exit(0)
        
        if valid_mermaid_code is None:
            return None, None

        final_mmd = os.path.join(generalized_workflow_dir, f"{skill_name_safe}_valid.mmd")
        with open(final_mmd, "w") as f:
            f.write(valid_mermaid_code)
        print(f"save mermaid >>> {final_mmd}")

        return result_file, valid_mermaid_code
    

    def induce_skill_from_single_trajectory(
        self,
        skill_call_dict,
        sub_traj_info,
        llm,
        trajectory_summary,
        temperature=0.0,
        previous_mermaid_code="",
        hints="",
        available_actions_str="""
    A_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction
    A_GET["PrimitiveAction: <br>(action: 'get {{action_item_count}} {{action_item_name}}')<br>local in: (action_item_count: ItemName = {{CURRENT_INGREDIENT}}, action_item_name: Count = {{COUNT_TO_GET}})<br>out: (executed: Bool)"]:::PrimitiveAction
    A_CRAFT["PrimitiveAction: <br>(action: 'craft {{action_item_count}} {{action_item_name}} using {{action_ingredients_count_list}} {{action_ingredients_name_list}}')<br>local in: (action_item_name: ItemName = {{TARGET_ITEM}}, action_item_count: Count = {{COUNT_TO_CRAFT}}, action_ingredients_name_list: List_ItemName = {{INGREDIENTS_LIST}}, action_ingredients_count_list: List_Count = {{INGREDIENTS_COUNT}})<br>out: (executed: Bool)"]:::PrimitiveAction
""", 

        induced_workflow_dir=None,
        max_attempts: int = 6,
        verbose: bool = False,
    ):
        
        if induced_workflow_dir is None:
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

        start_block, flow_spec_block, d_init_block = _format_interface_block(skill_call_dict.get("parameters", []))
        skill_name = skill_call_dict.get("subgoal", "single_trajectory_skill")
        skill_name_safe = re.sub(r"[^0-9A-Za-z_]+", "_", skill_name)

        _action_node_re = re.compile(r'([A-Za-z0-9_]+)\["PrimitiveAction:\s*<br>(.*?)"\]\s*:::PrimitiveAction', re.DOTALL)

        def _canonical_action_body(body: str) -> str:
            def _anon_local_in(match: re.Match) -> str:
                local_in = re.sub(r"\{\{[^}]*\}\}", "{{}}", match.group(2))
                return f"{match.group(1)}{local_in}{match.group(3)}"

            body = re.sub(r"(local in:\s*\()(.*?)(\))", _anon_local_in, body, flags=re.DOTALL)
            body = re.sub(r"\s+", "", body)
            return body

        def _extract_action_shapes(src: str) -> List[Tuple[str, str]]:
            return [(nid, _canonical_action_body(body)) for nid, body in _action_node_re.findall(src or "")]

        allowed_action_shapes = set(shape for _, shape in _extract_action_shapes(available_actions_str))

        current_mermaid_code = previous_mermaid_code
        current_hints = hints
        result_file = None
        valid_mermaid_code = None

        for attempt in range(max_attempts):

            if current_mermaid_code == "":
                previous_mermaid_components = {
                    "primitive_actions": "", 
                    "loop_control" : """
    LOOP_FOR_INGREDIENTS["LoopControl: <br>For idx, ingredient_i in enumerate({{loop_items}})<br>writes GLOBAL: (CURRENT_INGREDIENT: ItemName:=ingredient_i, LOOP_INDEX: Count:=idx)<br>local in: (loop_items: List_ItemName = {{TARGET_ITEMS}})"]:::LoopControl
""",
                    "checks": "", 
                    "data_operations": {
                        "other_nodes": "", 
                    }, 
                    "node_class_assignments" : """
    class START,SUCCESS_END,FAILURE_END Interface
    class LOOP_FOR_INGREDIENTS LoopControl
    class A_GET PrimitiveAction
    class C_INGREDIENT_READY Check
    class D_INIT,D_SELECT_TARGET_COUNT,D_FETCH_AVAILABLE,D_COMPUTE_MISSING DataOp
""", 
                    "control_flow_edges" : """
    %% Here is an example of iterating over `TARGET_ITEMS` and gathering the corresponding number of `TARGET_COUNTS`.
    %% Enter workflow and initialize globals
    START --> D_INIT
    D_INIT --> |Start_Loop| LOOP_FOR_INGREDIENTS
    LOOP_FOR_INGREDIENTS --> |done| SUCCESS_END
    LOOP_FOR_INGREDIENTS --> |body| D_SELECT_TARGET_COUNT
    D_SELECT_TARGET_COUNT --> D_FETCH_AVAILABLE
    D_FETCH_AVAILABLE --> C_INGREDIENT_READY
    C_INGREDIENT_READY -->|Yes, Continue_Loop| LOOP_FOR_INGREDIENTS
    C_INGREDIENT_READY -->|No| D_COMPUTE_MISSING
    D_COMPUTE_MISSING --> A_GET
    A_GET --> |Continue_Loop| LOOP_FOR_INGREDIENTS
""",   
                }
            else:
                previous_mermaid_components = extract_nodes_from_mermaid_code(current_mermaid_code)

            prompt = SUB_GOAL_WORKFLOW_TRACE2CODE_PROMPT_V2.format(
                skill_call_dict=json.dumps(skill_call_dict, indent=4),
                trajectory_summary=trajectory_summary,
                sub_traj_info=sub_traj_info,
                AVAILABLE_ACTIONS=available_actions_str,
                FLOW_SPEC_NODE="    " + flow_spec_block,
                START_INTERFACE_NODE="    " + start_block,
                ACTION_NODE=previous_mermaid_components["primitive_actions"], 
                LOOP_FOR_NODE=previous_mermaid_components["loop_control"],
                CHECK_NODE=previous_mermaid_components["checks"],
                D_INIT_NODE= "    " + d_init_block,
                DATA_OP_NODE= previous_mermaid_components["data_operations"]["other_nodes"],
                CLASS_ASSIGNMENTS=previous_mermaid_components["node_class_assignments"], 
                CONTROL_FLOW_EDGE=previous_mermaid_components["control_flow_edges"],
                HINTS=current_hints
            )
            response = llm_response(prompt, llm, temperature=temperature)
            
            
            mermaid_code = response.strip()
            if mermaid_code.startswith("```mermaid"):
                mermaid_code = mermaid_code[len("```mermaid"):]
            if mermaid_code.endswith("```"):
                mermaid_code = mermaid_code[:-3]
            mermaid_code = _fix_single_brace_placeholders(mermaid_code.strip())
            
            print(mermaid_code)

            print("-"* 50 + "prompt" + "-"*50)
            print(prompt)
            
            print("-"* 50 + "response" + "-"*50)
            print(mermaid_code)
            print(skill_call_dict)
            
            induced_file = os.path.join(induced_workflow_dir, f"{skill_name_safe}_v{attempt}.mmd")
            induced_record_file = os.path.join(induced_workflow_dir, f"{skill_name_safe}_v{attempt}_generation.json")

            with open(induced_file, "w") as f:
                f.write(mermaid_code)
            print(f"induced workflow saved to {induced_file}")


            try:
                graph_i = parse_mermaid_with_info(mermaid_code)
                node_errors = validate_node_inputs(graph_i)
                prefix_errors = validate_node_inputs_with_prefix_node(graph_i)
                check_write_errors = validate_check_nodes_without_global_writes(graph_i)
                check_expr_errors = validate_check_expr_syntax(graph_i)
                control_flow_errors = validate_control_flow_outgoing_edges(graph_i)
                control_flow_node_errors = validate_control_flow_node_definitions(graph_i)
                loop_entry_errors = validate_loop_entry_edges(graph_i)
            except Exception as e:
                node_errors = [f"Parsing failure: {e}"]
                prefix_errors = []
                check_write_errors = []
                check_expr_errors = []
                control_flow_errors = []
                control_flow_node_errors = []
                loop_entry_errors = []
            action_shape_errors = []
            for node_id, canon_body in _extract_action_shapes(mermaid_code):
                if canon_body not in allowed_action_shapes:
                    action_shape_errors.append(
                        f"Action node {node_id} does not match any available action template (after anonymizing bindings)."
                    )

            generation_json = {
                "prompt": prompt,
                "mermaid_file": induced_file, 
                "skill_call_dict": skill_call_dict,
                "sub_traj_info": sub_traj_info,
                "node_errors": node_errors,
                "prefix_errors": prefix_errors,
                "check_write_errors": check_write_errors,
                "check_expr_errors": check_expr_errors,
                "control_flow_errors": control_flow_errors,
                "control_flow_node_errors": control_flow_node_errors,
                "loop_entry_errors": loop_entry_errors,
                "action_shape_errors": action_shape_errors,
            }
            with open(induced_record_file, "w") as f:
                json.dump(generation_json, f, indent=4)
            print(f"induced record saved to {induced_record_file}")
            
            if mermaid_code == 'No Solution':
                print("No Solution")
                raise Exception(f"No Solution.\n LOG FILE: {induced_record_file}")

            if (
                len(node_errors)
                + len(prefix_errors)
                + len(check_write_errors)
                + len(check_expr_errors)
                + len(control_flow_errors)
                + len(control_flow_node_errors)
                + len(loop_entry_errors)
                + len(action_shape_errors)
                == 0
            ):
                result_file = induced_file
                valid_mermaid_code = mermaid_code
                break
            else:
                (
                    node_payloads_hints,
                    prefix_errors_hints,
                    check_write_hints,
                    check_expr_hints,
                    control_flow_hints,
                    control_flow_node_hints,
                    loop_entry_hints,
                ) = self.extract_node_edge_errors(
                    node_errors,
                    prefix_errors,
                    check_write_errors,
                    check_expr_errors,
                    control_flow_errors,
                    control_flow_node_errors,
                    loop_entry_errors,
                )
                action_shape_hint = ""
                if action_shape_errors:
                    action_shape_hint = "PrimitiveAction nodes must align with available action templates (ignoring binding globals). Allowed canonical forms:\\n"
                    for shape in allowed_action_shapes:
                        action_shape_hint += f"  - {shape}\\n"
                    # prune invalid action nodes from the previous code before the next iteration
                    invalid_node_ids = [err.split(" ")[2] for err in action_shape_errors if err.startswith("Action node")]
                    for nid in invalid_node_ids:
                        pattern = rf".*{nid}\\[.*\\]\\s*:::PrimitiveAction.*\\n?"
                        current_mermaid_code = re.sub(pattern, "", current_mermaid_code)
                current_hints = (
                    node_payloads_hints
                    + prefix_errors_hints
                    + check_write_hints
                    + check_expr_hints
                    + control_flow_hints
                    + control_flow_node_hints
                    + loop_entry_hints
                    + action_shape_hint
                )
                current_mermaid_code = mermaid_code

                
                print(current_hints)
                if attempt >= 3:
                    break
                # exit(0)
        
        if valid_mermaid_code is None:
            return None, None

        final_mmd = os.path.join(induced_workflow_dir, f"{skill_name_safe}_valid.mmd")
        with open(final_mmd, "w") as f:
            f.write(valid_mermaid_code)
        print(f"save mermaid >>> {final_mmd}")

        return result_file, valid_mermaid_code
    
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
    LOOP_FOR_INGREDIENTS["LoopControl: <br>For idx, ingredient_i in enumerate({{loop_items}})<br>writes GLOBAL: (CURRENT_INGREDIENT: ItemName:=ingredient_i, LOOP_INDEX: Count:=idx)<br>local in: (loop_items: List_ItemName = {{TARGET_ITEMS}})"]:::LoopControl
""",
                "checks": "", 
                "data_operations": {
                    "other_nodes": "", 
                }, 
                "node_class_assignments" : """
    class START,SUCCESS_END,FAILURE_END Interface
    class LOOP_FOR_INGREDIENTS LoopControl
    class A_GET PrimitiveAction
    class C_INGREDIENT_READY Check
    class D_INIT,D_SELECT_TARGET_COUNT,D_FETCH_AVAILABLE,D_COMPUTE_MISSING DataOp
""", 
                "control_flow_edges" : """
    %% Here is an example of iterating over `TARGET_ITEMS` and gathering the corresponding number of `TARGET_COUNTS`.
    %% Enter workflow and initialize globals
    START --> D_INIT
    D_INIT --> |Start_Loop| LOOP_FOR_INGREDIENTS
    LOOP_FOR_INGREDIENTS --> |done| SUCCESS_END
    LOOP_FOR_INGREDIENTS --> |body| D_SELECT_TARGET_COUNT
    D_SELECT_TARGET_COUNT --> D_FETCH_AVAILABLE
    D_FETCH_AVAILABLE --> C_INGREDIENT_READY
    C_INGREDIENT_READY -->|Yes, Continue_Loop| LOOP_FOR_INGREDIENTS
    C_INGREDIENT_READY -->|No| D_COMPUTE_MISSING
    D_COMPUTE_MISSING --> A_GET
    A_GET --> |Continue_Loop| LOOP_FOR_INGREDIENTS
""",   
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
        mermaid_code = _fix_single_brace_placeholders(mermaid_code.strip())
        
        print(mermaid_code)

        print("-"* 50 + "prompt" + "-"*50)
        print(prompt)
        
        print("-"* 50 + "response" + "-"*50)
        print(mermaid_code)
        print(sub_goal_i)
        
        # fix single brace placeholders to double brace placeholders

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
    
    def decompose_sub_goals(self, data_traj_collection, llm, temperature=0.0, load=False):
        
        # print(data_traj_collection)
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

        if self.verbose:
            print("-" * 50 + "prompt" + "-" * 50)
            print(prompt)
        
        for sub_goal_i in sub_goals:
            print(json.dumps(sub_goal_i, indent=4))
        
        with open(file_path, "w") as f:
            f.write(json.dumps(sub_goals, indent=4))
        print(f"sub_goals saved to {file_path}")

        return sub_goals

    def decompose_sub_goals_v2(self, llm, temperature=0.0, load=False):
        
        file_path = os.path.join(self.log_folder, "sub_goals_v2.json")
        if load:
            try:
                with open(file_path, "r") as f:
                    payload = json.loads(f.read())
                print(f"sub_goals_v2 loaded from {file_path}")
                if isinstance(payload, dict) and "sub_goal_library" in payload:
                    self.trajectory_decompositions = payload.get("trajectory_decompositions", [])
                    self.sub_goals_v2_validation = payload.get("validation", {})
                    return payload["sub_goal_library"]
                return payload
            except FileNotFoundError:
                print("sub_goals_v2.json not found")
        
        data_traj_collection = ""
        traj_step_counts = {}
        
        

        for problem_idx, problem_i in enumerate(self.trajectory_memory_graph.trajectory_head_list):
            print(self.trajectory_memory_graph.nodes[problem_i].task_instruction)

            data_traj_collection += f"### Trajectory {problem_idx + 1}: {self.trajectory_memory_graph.nodes[problem_i].task_instruction}\n"
            node_id = problem_i
            step_count = 0
            while node_id != None:
                current_node = self.trajectory_memory_graph.nodes[node_id]
                step_count += 1
                data_traj_collection += f"Step {step_count}: \nState: {current_node.observation_current}\nFact: {current_node.facts_current}\n"
                action_str = f"Action: {current_node.action}, Diff:"
                for fact_i in current_node.action_remove_fact:
                    action_str += f" Remove {fact_i}, "
                for fact_i in current_node.action_add_fact:
                    action_str += f" Add {fact_i}, "
                if (len(current_node.action_remove_fact) == 0 and len(current_node.action_add_fact) == 0):
                    action_str += " None, "
                data_traj_collection += action_str.strip(", ") + "\n"
                node_id = self.trajectory_memory_graph.nodes[node_id].trajectory_next_node_id
            
            data_traj_collection += "\n\n"
            traj_step_counts[problem_idx + 1] = step_count
        print(data_traj_collection)

        self.trajectory_step_counts = traj_step_counts

        def _validate_decompositions(payload_obj):
            issues = []
            if not isinstance(payload_obj, dict):
                return False, ["Top-level JSON must be an object (dict), not a list."]
            if "sub_goal_library" not in payload_obj:
                issues.append("Missing required key: sub_goal_library.")
            if "trajectory_decompositions" not in payload_obj:
                issues.append("Missing required key: trajectory_decompositions.")
                return False, issues

            decomps = payload_obj.get("trajectory_decompositions")
            if not isinstance(decomps, list):
                issues.append("trajectory_decompositions must be a list.")
                return False, issues

            decomp_by_id = {}
            for d in decomps:
                if isinstance(d, dict) and "trajectory_id" in d:
                    decomp_by_id[d["trajectory_id"]] = d

            for traj_id, step_count in traj_step_counts.items():
                d = decomp_by_id.get(traj_id)
                if d is None:
                    issues.append(f"Missing decomposition for trajectory_id={traj_id}.")
                    continue

                overall_step_count = d.get("overall_step_count")
                if isinstance(overall_step_count, int) and overall_step_count != step_count:
                    issues.append(
                        f"trajectory_id={traj_id}: overall_step_count={overall_step_count} != actual_step_count={step_count}."
                    )

                segments = d.get("segments")
                if not isinstance(segments, list) or len(segments) == 0:
                    issues.append(f"trajectory_id={traj_id}: segments must be a non-empty list.")
                    continue

                first = segments[0]
                if not isinstance(first, dict) or first.get("start_step_id") != 1:
                    issues.append(f"trajectory_id={traj_id}: first segment start_step_id must be 1.")

                prev_end = None
                for idx, seg in enumerate(segments):
                    if not isinstance(seg, dict):
                        issues.append(f"trajectory_id={traj_id}: segment[{idx}] must be an object.")
                        continue
                    start = seg.get("start_step_id")
                    end = seg.get("end_step_id")
                    if not isinstance(start, int) or not isinstance(end, int):
                        issues.append(f"trajectory_id={traj_id}: segment[{idx}] start/end must be integers.")
                        continue
                    if start < 1 or end < 1 or start > step_count or end > step_count:
                        issues.append(
                            f"trajectory_id={traj_id}: segment[{idx}] start/end out of range (1..{step_count})."
                        )
                    if end < start:
                        issues.append(f"trajectory_id={traj_id}: segment[{idx}] end_step_id < start_step_id.")
                    if prev_end is not None and start != prev_end + 1:
                        issues.append(
                            f"trajectory_id={traj_id}: segment[{idx}] start_step_id={start} "
                            f"must equal previous end_step_id+1={prev_end + 1}."
                        )
                    prev_end = end

                last = segments[-1]
                if isinstance(last, dict) and last.get("end_step_id") != step_count:
                    issues.append(
                        f"trajectory_id={traj_id}: last segment end_step_id must be {step_count}."
                    )

            return len(issues) == 0, issues

        step_count_hint = "\n".join([f"- Trajectory {tid}: overall_step_count = {cnt}" for tid, cnt in traj_step_counts.items()])
        base_prompt = SUBGOAL_IDENTIFICATION_PROMPT.format(examples=data_traj_collection)
        base_prompt += "\n\n# Trajectory Step Counts (Ground Truth)\n" + step_count_hint + "\n"

        payload = None
        response = None
        last_issues = None
        for attempt in range(2):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = (
                    "You are correcting an invalid previous JSON output.\n"
                    "Return ONLY a valid JSON object following the schema, and fix the segmentation issues.\n\n"
                    f"Trajectory step counts:\n{step_count_hint}\n\n"
                    f"Segmentation issues to fix:\n- " + "\n- ".join(last_issues or []) + "\n\n"
                    "Previous JSON (for reference):\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False)
                )

            response = llm_response(prompt, llm, temperature=temperature)

            payload = json.loads(repair_json(response))
            ok, issues = _validate_decompositions(payload)
            if ok:
                break
            last_issues = issues

        # Attach validation result computed locally (do not rely solely on model self-reporting).
        ok, issues = _validate_decompositions(payload)
        if not isinstance(payload, dict):
            payload = {
                "sub_goal_library": payload,
                "trajectory_decompositions": [],
                "validation": {
                    "coverage_ok": False,
                    "notes": "Invalid output format: " + "; ".join(issues[:10]),
                },
            }
        else:
            payload.setdefault("validation", {})
            payload["validation"]["coverage_ok"] = bool(ok)
            if not ok:
                payload["validation"]["notes"] = "Invalid segmentation: " + "; ".join(issues[:10])
            payload["validation"].setdefault("notes", "")

        if isinstance(payload, dict) and "sub_goal_library" in payload:
            sub_goals = payload["sub_goal_library"]
            self.trajectory_decompositions = payload.get("trajectory_decompositions", [])
            self.sub_goals_v2_validation = payload.get("validation", {})
        else:
            sub_goals = payload
        
        for sub_goal_i in sub_goals:
            print(json.dumps(sub_goal_i, indent=4))
        if isinstance(payload, dict) and payload.get("trajectory_decompositions") is not None:
            print("-" * 20 + " trajectory_decompositions " + "-" * 20)
            print(json.dumps(payload.get("trajectory_decompositions", []), indent=2))
            print("-" * 20 + " validation " + "-" * 20)
            print(json.dumps(payload.get("validation", {}), indent=2))

        with open(file_path, "w") as f:
            f.write(json.dumps(payload, indent=4))
        print(f"sub_goals_v2 saved to {file_path}")

        # exit(0)

        return sub_goals



    def update_trajectory_memory(self, example_with_fact_list, crafting_commands, success_flag=True):
        """
        Update the trajectory memory with a new example.
        
        Args:
            example_with_fact_list: Formatted example with facts
            success_flag: Whether the example was successful
        """
        
        
        
        if success_flag:
        
            query_instruction = example_with_fact_list[0]["obs"].split("Your task is to: ")[1].strip(".")
            print(query_instruction)
            print(crafting_commands)
            
            trajectory_wo_facts = ""
            pre_node_id = None
            for i in range(len(example_with_fact_list)):

                obs_i = example_with_fact_list[i]["obs"]
                fact_i = example_with_fact_list[i]["fact"]
                action_i = example_with_fact_list[i]["action"]

                if action_i and action_i.startswith("action: "):
                    action_i = action_i.split("action: ")[1].strip()


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
                    "crafting_commands": crafting_commands,
                    "observation_current": obs_i, 
                    "facts_current": fact_i, 
                    "action": action_i, 
                    "action_remove_fact": remove_fact_i, 
                    "action_add_fact": add_fact_i, 
                }
                node_i = TrajectoryMemoryNode(
                    **node_kwargs,
                    verbose=False)
                
                
                if node_i.action_type in ["get", "inventory", "craft"]:
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
                else:
                    raise Exception("Action cannot be matched in the env: ", node_i.action, node_i.action_type)
                    
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

    def _load_examples(self, example_path: Optional[str]) -> Dict:
        """Load few-shot examples if a path is provided; otherwise return an empty dict."""
        if not example_path:
            return {}
        if not os.path.exists(example_path):
            if self.verbose:
                print(f"[NSMS] Example path not found, skipping load: {example_path}")
            return {}
        try:
            with open(example_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            if self.verbose:
                print(f"[NSMS] Failed to load examples from {example_path}: {exc}")
            return {}

    def reset(self, commands: Optional[str] = None, task: Optional[str] = None, env_type: Optional[str] = None, 
              craft_depth: int = None, agent_log_path: Optional[str] = None, traj_path: Optional[str] = None):
        """Reset episode-specific buffers; mirrors the React agent signature."""
        self.reset_token_counts()
        
        self.commands = commands.strip() if commands else ""
        self.task = task.strip() if task else ""
        self.env_type = env_type or "textcraft"
        self.current_env_type = f"craft_depth_{craft_depth}"
        self.current_facts = []
        self.previous_action = ""
        self.agent_log = ""
        self.agent_log_path = agent_log_path
        self.traj_path = traj_path
        self.preceived_flag = False
        self.current_query_instruction = None
        self.trajectory_memory_node_id = None
        self.current_trajectory_memory_problem_node_id = None
        self.skill_info_for_current_task = ""
        self.current_task_guideline_json = json.dumps(self.task_level_guidelines.get(self.current_env_type, {}), indent=4)
        self.current_task_guideline = self.current_task_guideline_json
        self.example_trace = "\n\n".join(example[0] for example in self.react_examples) if self.react_examples else ""
        self.subgoal_failure_memory = []
        self.current_regression_topology = None
        self._skill_execution_log = []
        self._last_action_failure = None

        self.interaction_history = self.commands + "\n"
        
        self.re_plan_count_per_query = 0


        skill_info = ""

        # for sub_goal_i, skill_i, mermaid_i, params_i in self.induce_skills:
        #     skill_info += f"{skill_i['expression']} # {skill_i['description']}, example: {skill_i['example']}\n"
        
        if self.verbose:
            print(self.current_env_type)
        self.current_env_type

        print(self.skill_memory_blocks_with_task_type)
        # exit(0)


        for task_type in self.skill_memory_blocks_with_task_type.keys():
            for sub_goal_i, calling_node_list in self.skill_memory_blocks_with_task_type[task_type].items():
                
                print(sub_goal_i, calling_node_list)
                
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
                skill_info += self._format_parameter_binding_feedback(skill_json)
                skill_info += self._format_parameter_binding_feedback(skill_json)
                    
            # print(skill_info)
        

        # print(skill_info)
        # exit(0)
        self.skill_info_for_current_task = skill_info


        if self.current_env_type in self.task_level_guidelines:
            task_guideline = self.transform_task_guideline_to_text(self.task_level_guidelines[self.current_env_type])
            task_guideline_json = json.dumps(self.task_level_guidelines[self.current_env_type], indent=4)
        else:
            task_guideline = task_guideline_json = None
        self.current_task_guideline = task_guideline     
        self.current_task_guideline_json = task_guideline_json

        # print(self.current_task_guideline)
        # print(self.current_task_guideline_json)

        self.current_regression_json = None
        # exit(0)

    def transform_task_guideline_to_text(self, guideline_json, task_type_name):
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
        task_guideline_parts.append("# Guideline on Different Tasks")
        task_guideline_parts.append(task_type_name)

        task_guideline_parts.append("## Sub-Task Decomposition")
        subgoal_topology = guideline_json.get("subgoal_topology")
        task_guideline_parts.append("abstract sub-goal sequences: " + " -> ".join(subgoal_topology["subgoals"]))
        for sub_goal_i, conditions_i in subgoal_topology["subgoal_success_conditions"].items():
            task_guideline_parts.append(f"   - success_condiitions for <{sub_goal_i}>: ")
            task_guideline_parts.append(f"   {conditions_i}")

        task_guideline_parts.append("## Skill Section Guideline")
        for skill_entry in guideline_json.get("skill_guidelines", []):
            task_guideline_parts.append(f"- Skill: {skill_entry.get('skill', '')}")
            task_guideline_parts.append("  Scenarios: phase must match an ordered sub-task; triggers tell when to call; pre/post describe context and outcomes; record indices cite supporting examples.")
            scenarios = skill_entry.get("applicable_scenarios") or skill_entry.get("scenarios", [])
            task_guideline_parts.append(describe_scenarios(scenarios, include_records=False))

        task_guideline_parts.append("## Raw Action Guidelines")
        for action_entry in guideline_json.get("raw_action_guidelines", []):
            task_guideline_parts.append(f"- Raw Action: {action_entry.get('action', '')}")
            task_guideline_parts.append("  Scenarios: phase must match an ordered sub-task; triggers tell when to use the raw action; pre/post describe context and outcomes.")
            scenarios = action_entry.get("applicable_scenarios") or action_entry.get("scenarios", [])
            task_guideline_parts.append(describe_scenarios(scenarios, include_records=False))

        task_guideline = "\n".join(task_guideline_parts) + "\n"
        return task_guideline

    def _format_subgoal_traces_for_regression(self) -> str:

        trace = []
        for task_type in ENV_TYPES:

            if task_type not in self.task_level_guidelines:
                continue

            guideline = self.task_level_guidelines[task_type]
            subgoal_topology = guideline.get("subgoal_topology")


            if guideline is None or subgoal_topology is None:
                continue
            trace.append({"task_type": task_type, "subgoal_topology": subgoal_topology})

        return json.dumps(trace, indent=2)

    def _record_subgoal_failure(self, subgoal_name: str, failure_analysis: str, following_suggestions: str) -> None:
        entry = None
        for item in self.subgoal_failure_memory:
            if item.get("subgoal") == subgoal_name:
                entry = item
                break
        if entry is None:
            entry = {"subgoal": subgoal_name, "count": 0, "last_failure": "", "last_suggestion": ""}
            self.subgoal_failure_memory.append(entry)
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_failure"] = failure_analysis
        entry["last_suggestion"] = following_suggestions

    def _record_parameter_binding_feedback(
        self,
        subgoal_name: str,
        success: bool,
        parameter_bindings: Dict,
        start_condition_evidence: str = "",
        success_conditions_analysis: str = "",
        failure_analysis: str = "",
        following_suggestions: str = "",
    ) -> None:
        skill_json = self.skill_json_dict.get(subgoal_name)
        if not skill_json:
            return
        feedback = {
            "success": success,
            "parameter_bindings": parameter_bindings or {},
        }
        if start_condition_evidence:
            feedback["start_condition_evidence"] = start_condition_evidence
        if success:
            if success_conditions_analysis:
                feedback["success_conditions_analysis"] = success_conditions_analysis
        else:
            if failure_analysis:
                feedback["failure_analysis"] = failure_analysis
            if following_suggestions:
                feedback["following_suggestions"] = following_suggestions
        feedback_list = skill_json.setdefault("parameter_binding_feedback", [])
        feedback_list.append(feedback)
        max_keep = 5
        if len(feedback_list) > max_keep:
            del feedback_list[:-max_keep]

    def _update_parameter_bindings_guidelines(self, subgoal_name: str, updated_guidelines: Dict) -> None:
        if not updated_guidelines:
            return
        skill_json = self.skill_json_dict.get(subgoal_name)
        if not skill_json:
            return
        current = skill_json.get("parameter_bindings_guidelines")
        if not isinstance(current, dict):
            skill_json["parameter_bindings_guidelines"] = updated_guidelines
            return
        if not isinstance(updated_guidelines, dict):
            return
        merged = deepcopy(current)
        for key, value in updated_guidelines.items():
            if key in ["selection_guidelines", "anti_patterns", "checklist_before_call"]:
                if isinstance(value, list):
                    existing = merged.get(key, [])
                    if not isinstance(existing, list):
                        existing = []
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                    merged[key] = existing
            elif key == "parameter_roles":
                if isinstance(value, dict):
                    existing = merged.get(key, {})
                    if not isinstance(existing, dict):
                        existing = {}
                    for role_key, role_val in value.items():
                        existing[role_key] = role_val
                    merged[key] = existing
            elif key == "fail_reason":
                if value not in [None, "", "null"]:
                    merged[key] = value
            else:
                merged[key] = value
        skill_json["parameter_bindings_guidelines"] = merged

    def _format_parameter_binding_feedback(self, skill_json: Dict, max_items: int = 3) -> str:
        feedback_list = skill_json.get("parameter_binding_feedback") or []
        if not feedback_list:
            return ""
        lines = ["### Recent Parameter Binding Outcomes:"]
        for idx, item in enumerate(feedback_list[-max_items:], 1):
            outcome = "success" if item.get("success") else "failure"
            lines.append(f" - Record {idx} ({outcome})")
            if "parameter_bindings" in item:
                lines.append(" - Parameter Bindings: " + json.dumps(item.get("parameter_bindings", {}), indent=4))
            if item.get("failure_analysis"):
                lines.append(f" - Failure Analysis: {item.get('failure_analysis')}")
            if item.get("following_suggestions"):
                lines.append(f" - Following Suggestions: {item.get('following_suggestions')}")
            if item.get("success_conditions_analysis"):
                lines.append(f" - Success Evidence: {item.get('success_conditions_analysis')}")
        return "\n".join(lines) + "\n"

    def _should_replan_from_router(self, router_json: Dict) -> bool:
        if not router_json:
            return True

        active_subtask = router_json.get("active_subtask", None)
        if active_subtask == "completed":
            return True
        

        recommendation = router_json.get("recommendation", {}) or {}
        if recommendation.get("type") == "none":
            return True
        if not router_json.get("active_subtask"):
            return True
        if router_json.get("missing_info"):
            return True
        return False

    def _get_recipe_output_items(self) -> Set[str]:
        outputs: Set[str] = set()
        for line in (self.commands or "").splitlines():
            line = line.strip()
            if not line.lower().startswith("craft "):
                continue
            match = re.match(r"^craft\s+\d+\s+(?P<item>.+?)\s+using\s+", line, flags=re.IGNORECASE)
            if not match:
                continue
            outputs.add(transform_item_name(match.group("item")))
        return outputs

    def _parse_predicate(self, predicate: str) -> Tuple[Optional[str], List[str]]:
        match = re.match(r"^(?P<name>[^()]+)\((?P<args>.*)\)$", predicate.strip())
        if not match:
            return None, []
        name = match.group("name").strip()
        args_blob = match.group("args").strip()
        if not args_blob:
            return name, []
        args = [arg.strip() for arg in args_blob.split(",") if arg.strip()]
        return name, args

    def _parse_int(self, value: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(str(value)))
            except Exception:
                return 0

    def _clean_predicate_arg(self, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", "\""):
            cleaned = cleaned[1:-1]
        return cleaned

    def _extract_inventory_item_from_subgoal(self, subgoal: Optional[str]) -> Optional[str]:
        if not subgoal:
            return None
        name, args = self._parse_predicate(subgoal)
        if name != "inventory" or not args:
            return None
        item = self._clean_predicate_arg(args[0])
        if not item:
            return None
        return transform_item_name(item)

    def _extract_craft_output_item(self, solution_name: Optional[str]) -> Optional[str]:
        if not solution_name:
            return None
        match = re.match(r"^craft\s+(?:\d+\s+)?(?P<item>.+?)\s+using\s+", solution_name, flags=re.IGNORECASE)
        if not match:
            return None
        return transform_item_name(match.group("item"))

    def _normalize_craft_command(self, command: str) -> str:
        return " ".join(command.strip().lower().split())

    def _get_craft_command_set(self) -> Set[str]:
        commands: Set[str] = set()
        for line in (self.commands or "").splitlines():
            line = line.strip()
            if not line.lower().startswith("craft "):
                continue
            commands.add(self._normalize_craft_command(line))
        return commands

    def _filter_decomposition_solutions_by_subgoal(
        self, updated_decomposition: Optional[List[Dict]]
    ) -> Optional[List[Dict]]:
        if not updated_decomposition:
            return updated_decomposition
        craft_commands = self._get_craft_command_set()
        for entry in updated_decomposition:
            target_item = self._extract_inventory_item_from_subgoal(entry.get("subgoal"))
            if not target_item:
                continue
            filtered_solutions = []
            for solution in entry.get("solutions", []) or []:
                solution_name = (solution.get("name") or "").strip()
                if solution_name.lower().startswith("craft "):
                    # Allow tag substitution (e.g. "oak planks" for "planks")
                    pass
                craft_item = self._extract_craft_output_item(solution_name)
                if craft_item and craft_item != target_item:
                    continue
                filtered_solutions.append(solution)
            entry["solutions"] = filtered_solutions
        return updated_decomposition

    def _condition_satisfied(self, condition: str, world: World, recipe_outputs: Set[str]) -> Optional[bool]:
        condition = (condition or "").strip()
        if not condition:
            return True
        negated = False
        if condition.startswith("not "):
            negated = True
            condition = condition[4:].strip()
        name, args = self._parse_predicate(condition)
        if not name:
            return None
        args = [self._clean_predicate_arg(arg) for arg in args]
        result: Optional[bool] = None

        
        if name == "inventory" and len(args) >= 2:
            item = transform_item_name(args[0])
            required = self._parse_int(args[1])
            result = world.inventory.get(item, 0) >= required
        elif name == "unavailable" and len(args) >= 1:
            item = transform_item_name(args[0])
            result = item in world.unavailable
        elif name == "goal" and len(args) >= 2:
            item = transform_item_name(args[0])
            required = self._parse_int(args[1])
            result = world.goals.get(item) == required
        elif name == "know_recipe" and len(args) >= 1:
            item = transform_item_name(args[0])
            result = item in recipe_outputs
        
        

        if result is None:
            return None
        if negated:
            result = not result
        return result

    def _preconditions_satisfied(self, preconditions: List[str]) -> bool:
        if not preconditions:
            return True
        world = World(predicates=self.current_facts)
        recipe_outputs = self._get_recipe_output_items()
        for cond in preconditions:
            result = self._condition_satisfied(cond, world, recipe_outputs)
            if result is not True:
                return False
        return True

    def _subgoal_preconditions_satisfied(self, subgoal: Optional[str], subgoals_decomposition: Optional[List[Dict]]) -> bool:
        if not subgoal or subgoal == "none":
            return True
        if not subgoals_decomposition:
            return False
        for entry in subgoals_decomposition:
            if entry.get("subgoal") != subgoal:
                continue
            for solution in entry.get("solutions", []) or []:
                preconditions = solution.get("preconditions", []) or []
                if self._preconditions_satisfied(preconditions):
                    return True
        return False

    def _refresh_subgoals_decomposition_status(
        self, subgoals_decomposition: Optional[List[Dict]]
    ) -> Optional[List[Dict]]:
        if not subgoals_decomposition:
            return subgoals_decomposition
        for entry in subgoals_decomposition:
            if entry.get("status") != "blocked":
                subgoal = entry.get("subgoal")
                if subgoal:
                    entry["status"] = "satisfied" if self._preconditions_satisfied([subgoal]) else "unsatisfied"
                
            for solution in entry.get("solutions", []) or []:
                if solution.get("status") == "blocked":
                    continue
                preconditions = solution.get("preconditions", []) or []
                solution["status"] = "satisfied" if self._preconditions_satisfied(preconditions) else "unsatisfied"
        return subgoals_decomposition

    def _build_skill_summary_for_planning(self) -> str:
        lines = []
        for name, sj in self.skill_json_dict.items():
            params = ", ".join(sj.get("parameters", []))
            desc = sj.get("description", "")
            lines.append(f"- {name}({params}): {desc}")
        return "\n".join(lines) if lines else "none"

    def _get_failure_memory_text(self) -> str:
        parts = []
        for item in self.subgoal_failure_memory:
            if item.get("last_failure"):
                parts.append(f"- {item['subgoal']}: {item['last_failure']} (count={item.get('count', 1)})")
        return "\n".join(parts) if parts else "none"

    # ===================== Online Skill Evolution =====================

    def _update_routing_rules(self, pattern: Dict, episode_idx: int) -> None:
        """Extract a planner routing rule from a planner_routing pattern."""
        from .prompts import ROUTING_RULE_EXTRACTION_PROMPT

        failing_skill = pattern.get("failing_skill", "")
        skill_json = self.skill_json_dict.get(failing_skill, {})

        # Build skill descriptions
        skill_desc_parts = []
        for sname, sj in self.skill_json_dict.items():
            params = ", ".join(sj.get("parameters", []))
            desc = sj.get("description", "")
            skill_desc_parts.append(f"- {sname}({params}): {desc}")

        prompt = ROUTING_RULE_EXTRACTION_PROMPT.format(
            failing_skill=failing_skill,
            failing_skill_description=skill_json.get("description", ""),
            failure_diagnostic=pattern.get("failure_diagnostic", ""),
            recovery_segment=pattern.get("recovery_segment", ""),
            recovery_skill_used=pattern.get("recovery_skill_used", "none"),
            skill_descriptions="\n".join(skill_desc_parts),
        )

        response = llm_response(prompt, self.llm, temperature=0.0, stop_strs=[])
        self._append_agent_log("=" * 20 + " routing rule extraction " + "=" * 20 + f"\n{prompt}\n{response}\n")

        rule = json.loads(repair_json(response))
        rule["source_episode"] = episode_idx
        rule["failing_skill"] = failing_skill
        rule["pattern_type"] = pattern.get("pattern_type", "")

        # Dedup: don't add if a rule with same failing_skill + pattern_type exists
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
        """Log a new_skill pattern for future consideration (no auto-processing)."""
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
        """Refine skill parameter guidelines by distilling accumulated anti_patterns into concise rules."""
        from .prompts import PARAMETER_GUIDELINE_REFINEMENT_PROMPT

        ANTI_PATTERN_THRESHOLD = 5  # Only refine when enough failures accumulated

        for skill_name, skill_json in self.skill_json_dict.items():
            guidelines = skill_json.get("parameter_bindings_guidelines", {})
            anti_patterns = guidelines.get("anti_patterns", [])

            # Count raw failure dumps (they start with "FAILED with bindings")
            raw_failures = [a for a in anti_patterns if a.startswith("FAILED with bindings")]
            if len(raw_failures) < ANTI_PATTERN_THRESHOLD:
                continue

            print(f"[HONING] Refining {skill_name} guidelines ({len(raw_failures)} raw failures)")

            # Sample crafting commands for context
            craft_sample = self.commands if hasattr(self, "commands") and self.commands else ""

            prompt = PARAMETER_GUIDELINE_REFINEMENT_PROMPT.format(
                skill_name=skill_name,
                parameters=", ".join(skill_json.get("parameters", [])),
                description=skill_json.get("description", ""),
                parameter_roles=json.dumps(guidelines.get("parameter_roles", {}), indent=2),
                anti_patterns="\n".join(f"- {a}" for a in anti_patterns),
                crafting_commands_sample=craft_sample[:2000],
            )

            response = llm_response(prompt, self.llm, temperature=0.0, stop_strs=[])
            self._append_agent_log("=" * 20 + f" guideline refinement {skill_name} " + "=" * 20 + f"\n{response}\n")

            result = json.loads(repair_json(response))

            # Update guidelines
            if result.get("refined_parameter_roles"):
                guidelines["parameter_roles"] = result["refined_parameter_roles"]
            if result.get("refined_anti_patterns"):
                guidelines["anti_patterns"] = result["refined_anti_patterns"]
            if result.get("refined_checklist"):
                guidelines["checklist_before_call"] = result["refined_checklist"]

            skill_json["parameter_bindings_guidelines"] = guidelines
            self._save_online_skills()
            print(f"[HONING] Refined {skill_name}: {len(raw_failures)} raw failures → {len(result.get('refined_anti_patterns', []))} rules")
            print(f"[HONING] Analysis: {result.get('analysis', '')[:200]}")

    def _get_routing_rules_text(self) -> str:
        """Format routing rules for injection into the planner prompt."""
        if not self.routing_rules:
            return "none"
        parts = []
        for rule in self.routing_rules:
            summary = rule.get("rule_summary", "")
            if not summary:
                summary = f"{rule.get('condition', '')} → {rule.get('action', '')}"
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

    def _auto_depth_graft(
        self, skill_name: str, failed_item: str, recipe_line: str,
        last_graft_get: str, pattern_type: str,
    ) -> None:
        """Auto-create a depth_composition graft when a graft's GET fails
        and the item has a recipe. Adds a craft branch after the last graft's
        final check node."""
        import re as _re_g
        from .infoflow2graph import graft_subgraph_at_node, extract_graft_node_ids, parse_mermaid_with_info

        # Dedup: check if depth_composition graft already exists
        normalized_pt = pattern_type.replace(" ", "_")
        for graft_info in self.evolution_ledger.values():
            existing_pt = graft_info.get("pattern_type", "").replace(" ", "_")
            if (graft_info.get("skill_name") == skill_name
                    and existing_pt == normalized_pt
                    and graft_info.get("status") in ("tentative", "solidified")):
                print(f"[ONLINE] Auto depth graft skipped: {pattern_type} already exists")
                return

        original_mermaid = self.skill_json_dict[skill_name]["mermaid_code"]

        # Find the CHECK_SUCCESS node of the last graft (the one that edges to FAILURE_END)
        graft_prefix_match = _re_g.match(r'^(G\d+)_', last_graft_get)
        if not graft_prefix_match:
            return
        last_prefix = graft_prefix_match.group(1)

        # Find the node from last graft that edges to FAILURE_END
        graft_check_node = None
        for line in original_mermaid.splitlines():
            line_s = line.strip()
            m = _re_g.match(r'(' + _re_g.escape(last_prefix) + r'_\w+)\s*-->.*?FAILURE_END', line_s)
            if m:
                graft_check_node = m.group(1)
                break
        if not graft_check_node:
            print(f"[ONLINE] Auto depth graft: can't find {last_prefix}_* → FAILURE_END edge")
            return

        # Determine next graft prefix
        new_prefix = self._next_graft_prefix(skill_name)

        # Build a simple depth_composition graft mermaid that tries to craft the item
        # Parse recipe: "craft 2 diorite using 2 quartz, 2 cobblestone"
        recipe_parts = recipe_line.split(" using ", 1)
        if len(recipe_parts) != 2:
            print(f"[ONLINE] Auto depth graft: can't parse recipe '{recipe_line}'")
            return

        # Node names must NOT include the prefix — graft_subgraph_at_node adds it
        graft_mmd = f"""flowchart TD
    GRAFT_ENTRY["DataOp: <br>writes GLOBAL: (G_TARGET_ITEM: ItemName:={{{{CURRENT_INGREDIENT}}}}, G_NEED_COUNT: Count:={{{{COUNT_TO_GET}}}})"]:::DataOp
    GRAFT_INVENTORY["PrimitiveAction: <br>(action: 'inventory')<br>out: (executed: Bool)"]:::PrimitiveAction
    GRAFT_FETCH_COUNT["DataOp: <br>writes GLOBAL: (G_CURRENT_COUNT: Count = current_count_of_item({{{{G_TARGET_ITEM}}}}))<br>local in: (G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::DataOp
    GRAFT_CHECK_ENOUGH["Check: <br>(inventory({{{{G_TARGET_ITEM}}}}, {{{{G_CURRENT_COUNT}}}}) and {{{{G_CURRENT_COUNT}}}} >= {{{{G_NEED_COUNT}}}})<br>local in: (G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}}, G_NEED_COUNT: Count = {{{{COUNT_TO_GET}}}}, G_CURRENT_COUNT: Count = {{{{G_CURRENT_COUNT}}}})"]:::Check
    GRAFT_COMPUTE_MISSING["DataOp: <br>writes GLOBAL: (G_COUNT_TO_GET: Count = max({{{{G_NEED_COUNT}}}} - {{{{G_CURRENT_COUNT}}}}, 0))<br>local in: (G_NEED_COUNT: Count = {{{{COUNT_TO_GET}}}}, G_CURRENT_COUNT: Count = {{{{G_CURRENT_COUNT}}}})"]:::DataOp
    GRAFT_GET["PrimitiveAction: <br>(action: 'get {{{{G_COUNT_TO_GET}}}} {{{{G_TARGET_ITEM}}}}')<br>local in: (G_COUNT_TO_GET: Count = {{{{G_COUNT_TO_GET}}}}, G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::PrimitiveAction
    GRAFT_RECHECK_COUNT["DataOp: <br>writes GLOBAL: (G_CURRENT_COUNT: Count = current_count_of_item({{{{G_TARGET_ITEM}}}})) <br>local in: (G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::DataOp
    GRAFT_FINAL_CHECK["Check: <br>(inventory({{{{G_TARGET_ITEM}}}}, {{{{G_CURRENT_COUNT}}}}) and {{{{G_CURRENT_COUNT}}}} >= {{{{G_NEED_COUNT}}}})<br>local in: (G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}}, G_NEED_COUNT: Count = {{{{COUNT_TO_GET}}}}, G_CURRENT_COUNT: Count = {{{{G_CURRENT_COUNT}}}})"]:::Check
    GRAFT_CRAFT["PrimitiveAction: <br>(action: 'craft {{{{action_item_count}}}} {{{{action_item_name}}}}')<br>local in: (action_item_count: Count = {{{{G_COUNT_TO_GET}}}}, action_item_name: ItemName = {{{{CURRENT_INGREDIENT}}}})"]:::PrimitiveAction
    GRAFT_CHECK_CRAFT_SUCCESS["Check: <br>(inventory({{{{G_TARGET_ITEM}}}}, {{{{G_CURRENT_COUNT}}}}) and {{{{G_CURRENT_COUNT}}}} >= {{{{G_NEED_COUNT}}}})<br>local in: (G_TARGET_ITEM: ItemName = {{{{CURRENT_INGREDIENT}}}}, G_NEED_COUNT: Count = {{{{COUNT_TO_GET}}}}, G_CURRENT_COUNT: Count = {{{{G_CURRENT_COUNT}}}})"]:::Check

    GRAFT_ENTRY --> GRAFT_INVENTORY
    GRAFT_INVENTORY --> GRAFT_FETCH_COUNT
    GRAFT_FETCH_COUNT --> GRAFT_CHECK_ENOUGH
    GRAFT_CHECK_ENOUGH -->|Yes| GRAFT_CONTINUE
    GRAFT_CHECK_ENOUGH -->|No| GRAFT_COMPUTE_MISSING
    GRAFT_COMPUTE_MISSING --> GRAFT_GET
    GRAFT_GET --> GRAFT_RECHECK_COUNT
    GRAFT_RECHECK_COUNT --> GRAFT_FINAL_CHECK
    GRAFT_FINAL_CHECK -->|Yes| GRAFT_CONTINUE
    GRAFT_FINAL_CHECK -->|No| GRAFT_CRAFT
    GRAFT_CRAFT --> GRAFT_CHECK_CRAFT_SUCCESS
    GRAFT_CHECK_CRAFT_SUCCESS -->|Yes| GRAFT_CONTINUE
    GRAFT_CHECK_CRAFT_SUCCESS -->|No| GRAFT_FAIL
"""

        try:
            combined = graft_subgraph_at_node(
                base_mermaid=original_mermaid,
                graft_mermaid=graft_mmd,
                remove_edge_src=graft_check_node,
                remove_edge_dst="FAILURE_END",
                continue_target="LOOP_FOR_INGREDIENTS",
                graft_prefix=new_prefix,
            )
            parse_mermaid_with_info(combined)  # validate
        except Exception as e:
            print(f"[ONLINE] Auto depth graft failed: {e}")
            return

        graft_id = f"{skill_name}__{normalized_pt}"
        graft_node_ids = list(extract_graft_node_ids(combined, new_prefix))

        self.skill_json_dict[skill_name]["mermaid_code"] = combined
        self.evolution_ledger[graft_id] = {
            "graft_id": graft_id,
            "status": "tentative",
            "skill_name": skill_name,
            "graft_point_src": graft_check_node,
            "graft_point_dst": "FAILURE_END",
            "pattern_type": normalized_pt,
            "episodes_seen": [],
            "graft_node_ids": graft_node_ids,
            "original_mermaid": original_mermaid,
            "success_count": 0,
            "fail_count": 0,
        }

        graft_path = os.path.join(self.online_evolved_dir, "grafts", f"{graft_id}.mmd")
        with open(graft_path, "w") as f:
            f.write(graft_mmd)
        self._save_online_skills()
        print(f"[ONLINE] Auto-grafted {graft_id} ({new_prefix}) → {skill_name}")

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

    def reveiew_episode(self, success_flag: bool = False, **kwargs) -> None:
        """Post-episode online evolution analysis.

        Called after each episode. When online_mode is enabled, analyzes the
        episode trajectory to find valuable failure→recovery patterns, induces
        sub-graph fragments, verifies them deductively, and grafts them onto
        failure points as tentative extensions.
        """
        if not self.online_mode:
            return

        episode_idx = kwargs.get("episode_idx", -1)
        goal = kwargs.get("goal", "")
        depth = kwargs.get("depth", 0)

        # Phase 0: Refine parameter guidelines if anti_patterns accumulated
        # (runs on both success and failure — failures produce anti_patterns too)
        try:
            self._refine_parameter_guidelines(episode_idx)
        except Exception as e:
            print(f"[ONLINE] Guideline refinement failed: {e}")
            traceback.print_exc()

        # Only analyze successful episodes — failed episodes have incomplete
        # recovery trajectories and cannot produce reliable graph inductions.
        if not success_flag:
            print(f"[ONLINE] Episode {episode_idx}: task failed, skipping evolution")
            return

        # Phase 1: Skip if no skill failures occurred (nothing to learn from)
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

        print(f"[ONLINE] Episode {episode_idx} ({goal}, depth={depth}): "
              f"{len(failed_calls)} skill failures, analyzing...")

        # Phase 2: LLM recovery analysis
        try:
            valuable_patterns = self._analyze_episode_for_recovery(
                traj_text, failed_calls, success_flag, goal, depth,
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

            # Correction: when craft_item_from_ingredients fails internally
            # (ingredient missing) and recovery is in-scope (no external skill),
            # treat as skill_graft even if LLM classified as planner_routing.
            # This ensures depth_composition grafts (G1/G2) are created.
            failing_skill = pattern.get("failing_skill", "")
            recovery_skill = pattern.get("recovery_skill_used", "none")
            if (evo_type == "planner_routing"
                    and failing_skill == "craft_item_from_ingredients"
                    and (recovery_skill in ("none", "", None, "craft_item_from_ingredients"))):
                evo_type = "skill_graft"
                print(f"[ONLINE] Corrected {failing_skill} depth pattern: planner_routing → skill_graft")

            try:
                if evo_type == "skill_graft":
                    print(f"[ONLINE] Pattern → skill_graft: {pattern.get('failing_skill')}/{pattern.get('pattern_type')}")
                    self._process_recovery_pattern(pattern, episode_idx)
                elif evo_type == "planner_routing":
                    print(f"[ONLINE] Pattern → planner_routing: {pattern.get('failing_skill')} → {pattern.get('recovery_skill_used', '?')}")
                    self._update_routing_rules(pattern, episode_idx)
                elif evo_type == "new_skill":
                    print(f"[ONLINE] Pattern → new_skill opportunity logged")
                    self._log_new_skill_opportunity(pattern, episode_idx)
                else:
                    print(f"[ONLINE] Unknown evolution_type '{evo_type}', defaulting to skill_graft")
                    self._process_recovery_pattern(pattern, episode_idx)
            except Exception as e:
                print(f"[ONLINE] Failed to process {evo_type} pattern: {e}")
                traceback.print_exc()
                continue

        # Save episode analysis for debugging
        analysis_path = os.path.join(
            self.online_evolved_dir, "episode_analysis",
            f"ep_{episode_idx}_analysis.json"
        )
        with open(analysis_path, "w") as f:
            json.dump({
                "episode_idx": episode_idx,
                "goal": goal,
                "depth": depth,
                "success_flag": success_flag,
                "num_failures": len(failed_calls),
                "valuable_patterns": valuable_patterns,
            }, f, indent=2, ensure_ascii=False)

    def _analyze_episode_for_recovery(
        self, traj_text: str, failed_calls: List[Dict],
        success_flag: bool, goal: str, depth: int,
    ) -> List[Dict]:
        """Phase 2: LLM analyzes trajectory to find valuable failure→recovery patterns."""
        from .prompts import EPISODE_RECOVERY_ANALYSIS_PROMPT

        skill_exec_summary = []
        for call in self._skill_execution_log:
            skill_exec_summary.append({
                "skill_name": call.get("skill_name", ""),
                "parameter_bindings": call.get("parameter_bindings", {}),
                "success": call.get("success", False),
                "diagnostic": call.get("diagnostic", ""),
            })

        # Build skill description summary for classification context
        skill_desc_parts = []
        for sname, sj in self.skill_json_dict.items():
            params = ", ".join(sj.get("parameters", []))
            desc = sj.get("description", "")
            skill_desc_parts.append(f"- {sname}({params}): {desc}")
        skill_descriptions = "\n".join(skill_desc_parts) if skill_desc_parts else "none"

        prompt = EPISODE_RECOVERY_ANALYSIS_PROMPT.format(
            trajectory=traj_text,
            skill_execution_log=json.dumps(skill_exec_summary, indent=2),
            crafting_commands=self.commands if hasattr(self, "commands") else "",
            task_goal=goal,
            episode_success=success_flag,
            skill_descriptions=skill_descriptions,
        )

        response = llm_response(prompt, self.llm, temperature=0.0, stop_strs=[])
        self._append_agent_log("=" * 20 + " recovery analysis prompt " + "=" * 20 + f"\n{prompt}\n")
        self._append_agent_log("=" * 20 + " recovery analysis response " + "=" * 20 + f"\n{response}\n")

        result = json.loads(repair_json(response))
        patterns = result.get("patterns", [])
        return [p for p in patterns if p.get("value_assessment") == "high"]

    def _next_graft_prefix(self, skill_name: str) -> str:
        """Return the next unused graft prefix (G0_, G1_, ...) for a skill."""
        existing = [g for g in self.evolution_ledger.values() if g.get("skill_name") == skill_name]
        return f"G{len(existing)}_"

    def _process_recovery_pattern(self, pattern: Dict, episode_idx: int) -> None:
        """Phase 3-4: Clean trajectory, induce sub-graph, verify, graft."""
        from .prompts import TRAJECTORY_CLEANING_PROMPT, GRAFT_INDUCTION_PROMPT

        skill_name = pattern.get("failing_skill", "")
        if skill_name not in self.skill_json_dict:
            print(f"[ONLINE] Unknown skill {skill_name}, skipping")
            return

        pattern_type = pattern.get("pattern_type", "unknown").replace(" ", "_")

        # Dedup: skip if an active graft already exists for this skill + pattern_type
        for graft_info in self.evolution_ledger.values():
            existing_pt = graft_info.get("pattern_type", "").replace(" ", "_")
            if (graft_info.get("skill_name") == skill_name
                    and existing_pt == pattern_type
                    and graft_info.get("status") in ("tentative", "solidified")):
                # Track that another episode triggered the same pattern
                seen = graft_info.setdefault("episodes_seen", [])
                if episode_idx not in seen:
                    seen.append(episode_idx)
                    self._save_online_skills()
                print(f"[ONLINE] Skip: {skill_name} already has active {pattern_type} graft "
                      f"({graft_info['graft_id']}, seen in episodes {seen})")
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
            crafting_commands=self.commands if hasattr(self, "commands") else "",
        )
        clean_response = llm_response(clean_prompt, self.llm, temperature=0.0, stop_strs=[])
        cleaned_trajectory = json.loads(repair_json(clean_response)).get("cleaned_trajectory", "")

        if not cleaned_trajectory:
            print(f"[ONLINE] Empty cleaned trajectory, skipping")
            return

        # Step B: Induce sub-graph
        # Detect referenced skills in recovery segment
        import re as _re
        referenced_skills = set()
        combined_text = recovery_segment + "\n" + cleaned_trajectory
        for ref_name in self.skill_json_dict:
            if ref_name != skill_name and ref_name in combined_text:
                referenced_skills.add(ref_name)
        if _re.search(r'\bcraft \d+', combined_text, _re.IGNORECASE):
            if "craft_item_from_ingredients" in self.skill_json_dict:
                # Allow self-reference for depth_composition: the skill needs to
                # see its own graph to inline craft logic when get fails.
                referenced_skills.add("craft_item_from_ingredients")
        if _re.search(r'\bget \d+', combined_text, _re.IGNORECASE):
            if "check_and_fetch_item" in self.skill_json_dict:
                referenced_skills.add("check_and_fetch_item")

        ref_graphs_text = ""
        if referenced_skills:
            for ref_name in sorted(referenced_skills):
                ref_sj = self.skill_json_dict[ref_name]
                ref_graphs_text += f"## Skill: {ref_name}\n"
                ref_graphs_text += f"Parameters: {ref_sj.get('parameters', [])}\n"
                ref_graphs_text += f"```mermaid\n{ref_sj['mermaid_code']}\n```\n\n"
        else:
            ref_graphs_text = "(none — recovery used only primitive actions)"

        # Determine scope variables based on skill
        if skill_name == "craft_item_from_ingredients":
            scope_vars = "CURRENT_INGREDIENT, COUNT_TO_GET, TARGET_COUNT, AVAILABLE_COUNT, TARGET_ITEM, NEED_COUNT, INGREDIENTS, INGREDIENTS_COUNT"
        elif skill_name == "check_and_fetch_item":
            scope_vars = "TARGET_ITEM, NEED_COUNT, CURRENT_COUNT, COUNT_TO_GET"
        else:
            scope_vars = "TARGET_ITEM, NEED_COUNT"

        graft_prefix = self._next_graft_prefix(skill_name)

        # Resolve actual graft point: if the original edge (e.g. D_COMPUTE_MISSING → FAILURE_END)
        # no longer exists (already replaced by a prior graft), find the current edge to FAILURE_END
        # from the latest graft's fail branch → chain graft.
        default_src = "D_COMPUTE_MISSING"
        default_dst = "FAILURE_END"
        actual_graft_src = default_src
        if f"{default_src} --> {default_dst}" not in original_mermaid and default_dst in original_mermaid:
            import re as _re2
            # Find the last graft node that edges into FAILURE_END
            for line in original_mermaid.splitlines():
                m = _re2.match(r'\s*(G\d+_\w+)\s*-->.*?' + _re2.escape(default_dst), line.strip())
                if m:
                    actual_graft_src = m.group(1)
            if actual_graft_src != default_src:
                print(f"[ONLINE] Chain graft: original edge gone, using {actual_graft_src} → {default_dst}")

        graft_mermaid = None
        for attempt in range(3):
            induce_prompt = GRAFT_INDUCTION_PROMPT.format(
                original_mermaid=original_mermaid,
                failure_node="FAILURE_END",
                failure_diagnostic=failure_diagnostic,
                cleaned_trajectory=cleaned_trajectory,
                pattern_type=pattern_type,
                scope_variables=scope_vars,
                crafting_commands=self.commands if hasattr(self, "commands") else "",
                referenced_skill_graphs=ref_graphs_text,
            )
            induce_response = llm_response(induce_prompt, self.llm, temperature=0.0, stop_strs=[])
            induced = json.loads(repair_json(induce_response))
            graft_mermaid = induced.get("graft_mermaid", "")

            # Override graft point with resolved actual source
            graft_src = actual_graft_src if actual_graft_src != default_src else induced.get("graft_point_src", default_src)
            graft_dst = induced.get("graft_point_dst", default_dst)

            if graft_mermaid:
                # Validate: try parsing the combined graph
                try:
                    from .infoflow2graph import graft_subgraph_at_node
                    combined = graft_subgraph_at_node(
                        base_mermaid=original_mermaid,
                        graft_mermaid=graft_mermaid,
                        remove_edge_src=graft_src,
                        remove_edge_dst=graft_dst,
                        continue_target=induced.get("continue_target", "LOOP_FOR_INGREDIENTS"),
                        graft_prefix=graft_prefix,
                    )
                    # Try parsing to validate
                    parse_mermaid_with_info(combined)
                    break
                except Exception as e:
                    print(f"[ONLINE] Graft validation failed (attempt {attempt+1}): {e}")
                    graft_mermaid = None

        if not graft_mermaid:
            print(f"[ONLINE] Could not produce valid graft after 3 attempts")
            return

        # Step C: Deductive verify — check the grafted graph is structurally sound
        # and that the induced graft nodes are reachable from the graft point.
        # Full trajectory replay (like reharsal_with_memory) requires a
        # trajectory_memory_graph node chain which we don't have for the recovery
        # segment. Instead, we verify:
        #   1. The combined graph parses without error (done above)
        #   2. The graft entry node is reachable from graft_point_src
        #   3. The graft has both a CONTINUE and FAIL exit path
        try:
            combined_graph = parse_mermaid_with_info(combined)
            graft_entry = induced.get("graft_point_src", "D_COMPUTE_MISSING")
            # Check edges from graft_point_src reach graft nodes
            reachable = set()
            for edge in combined_graph.edges:
                if edge.src == graft_entry:
                    reachable.add(edge.dst)
            has_graft_reach = any("G0_" in n for n in reachable) or any("G" in n for n in reachable)
            # Check graft has exit to continue_target and to failure
            continue_tgt = induced.get("continue_target", "LOOP_FOR_INGREDIENTS")
            fail_tgt = induced.get("graft_point_dst", "FAILURE_END")
            all_dsts = {e.dst for e in combined_graph.edges}
            has_continue = continue_tgt in all_dsts
            has_fail = fail_tgt in all_dsts

            if not has_graft_reach:
                print(f"[ONLINE] Deductive verify failed: graft nodes not reachable from {graft_entry}")
                return
            if not has_continue:
                print(f"[ONLINE] Deductive verify failed: no path to {continue_tgt}")
                return
            if not has_fail:
                print(f"[ONLINE] Deductive verify warning: no fallback path to {fail_tgt}")

            print(f"[ONLINE] Deductive verify passed for {skill_name} ({pattern_type})")
        except Exception as e:
            print(f"[ONLINE] Deductive verify failed: {e}")
            return

        # Step D: Graft + persist (reuse `combined` from Step B/C validation)
        from .infoflow2graph import extract_graft_node_ids

        combined_mermaid = combined  # already computed and validated above

        graft_id = f"{skill_name}__{pattern_type}"
        graft_node_ids = list(extract_graft_node_ids(combined_mermaid, graft_prefix))

        self.skill_json_dict[skill_name]["mermaid_code"] = combined_mermaid
        self.evolution_ledger[graft_id] = {
            "graft_id": graft_id,
            "status": "tentative",
            "skill_name": skill_name,
            "graft_point_src": graft_src,
            "graft_point_dst": graft_dst,
            "pattern_type": pattern_type,
            "episodes_seen": [episode_idx],
            "graft_node_ids": graft_node_ids,
            "original_mermaid": original_mermaid,
            "success_count": 0,
            "fail_count": 0,
        }

        # Save graft fragment
        graft_path = os.path.join(self.online_evolved_dir, "grafts", f"{graft_id}.mmd")
        with open(graft_path, "w") as f:
            f.write(graft_mermaid)

        self._save_online_skills()
        print(f"[ONLINE] Grafted {graft_id} (tentative) → {skill_name}")

    # ===================== End Online Skill Evolution =====================

    def _plan(self, sub_goal_feedback: str = "") -> Optional[dict]:
        """Layer 1: Merged planning (regression + approach recommendation)."""
        self.re_plan_count += 1
        self.re_plan_count_per_query += 1

        prompt = PLANNING_PROMPT.format(
            task_goal=self.task,
            facts=json.dumps(self.current_facts, indent=2),
            crafting_commands=self.commands,
            history_memory=sub_goal_feedback if sub_goal_feedback else "Initial step.",
            failure_memory=self._get_failure_memory_text(),
            skill_summary=self._build_skill_summary_for_planning(),
            routing_rules=self._get_routing_rules_text(),
            hints="",
        )

        response = llm_response(prompt, self.llm, temperature=self.temperature, stop_strs=[])

        if self.verbose:
            print("=" * 20 + " planning prompt " + "=" * 20)
            print(prompt)
            print("=" * 20 + " planning response " + "=" * 20)
            print(response)

        self._append_agent_log("=" * 20 + "planning prompt" + "=" * 20 + "\n{}\n".format(prompt))
        self._append_agent_log("=" * 20 + "planning response" + "=" * 20 + "\n{}\n".format(response))

        plan_json = json.loads(repair_json(response))
        self.current_regression_json = plan_json
        return plan_json

    def _update_task_guideline_from_regression(self, sub_goal_feedback: str = "Empty") -> None:
        
        self.re_plan_count += 1
        self.re_plan_count_per_query += 1
        
        max_planning_rounds = 5
        subgoals_decomposition = None

        if self.current_env_type not in self.task_level_guidelines:
            base_guideline = ""
            for task_type_i in self.task_level_guidelines:
                guideline_i = self.transform_task_guideline_to_text(self.task_level_guidelines[task_type_i], task_type_i)
                base_guideline += guideline_i
        else:
            base_guideline = self.transform_task_guideline_to_text(self.task_level_guidelines[self.current_env_type], task_type_i)
        if self.current_regression_json:
            subgoals_decomposition = self.current_regression_json["subgoals_decomposition"]
            subgoals_decomposition = self._refresh_subgoals_decomposition_status(subgoals_decomposition)
        
        hints = ""
        
        for planning_round in range(max_planning_rounds):
            prompt = REGRESSION_TOPOLOGY_PROMPT.format(
                task_goal=self.task,
                facts=json.dumps(self.current_facts, indent=2),
                crafting_commands=self.commands,
                history_memory=sub_goal_feedback,
                subgoal_traces=self._format_subgoal_traces_for_regression(),
                subgoals_decomposition=json.dumps(subgoals_decomposition, indent=2),
                hints=hints,
            )
            response = llm_response(prompt, self.llm, temperature=self.temperature, stop_strs=[])
            regression_json = json.loads(repair_json(response))
            updated_decomposition = regression_json.get("subgoals_decomposition")
            updated_decomposition = self._filter_decomposition_solutions_by_subgoal(updated_decomposition)
            
            # subgoals_decomposition = updated_decomposition
            if subgoals_decomposition:
                previous_sub_goal_name_dict = {
                    sub_goal_item["subgoal"]: sub_goal_item
                    for sub_goal_item in subgoals_decomposition
                }
            else:
                previous_sub_goal_name_dict = {}
            
            if updated_decomposition:
                update_sub_goal_name_dict = {
                    sub_goal_item["subgoal"]: sub_goal_item
                    for sub_goal_item in updated_decomposition
                }
            else:
                update_sub_goal_name_dict = {}

            new_decomposition = []


            for sub_goal_name, sub_goal_item in previous_sub_goal_name_dict.items():
                if sub_goal_name not in update_sub_goal_name_dict:
                    pass
                else:
                    if sub_goal_item["status"] in ["unsatisfied", "satisfied"]:
                        sub_goal_item["status"] = update_sub_goal_name_dict[sub_goal_name]["status"]

                        previous_solution_dict = {
                            solution_i["name"]: solution_i
                            for solution_i in sub_goal_item["solutions"]
                        }
                        update_solution_dict = {
                            solution_i["name"]: solution_i
                            for solution_i in update_sub_goal_name_dict[sub_goal_name]["solutions"]
                        }

                        new_solution = []

                        for sol_name_i, sol_i in previous_solution_dict.items():
                            if sol_name_i not in update_solution_dict:
                                pass
                            else:
                                if sol_i["status"] in ["unsatisfied", "satisfied"]:
                                    sol_i["status"] = update_solution_dict[sol_name_i]["status"]
                                    sol_i["evidence"] = update_solution_dict[sol_name_i]["evidence"]
                                else:
                                    # this solution has been blocked
                                    pass
                            new_solution.append(sol_i)

                        for sol_name_i, sol_i in update_solution_dict.items():
                            if sol_name_i not in previous_solution_dict:
                                new_solution.append(sol_i)
                        
                        sub_goal_item["solutions"] = new_solution
                    
                new_decomposition.append(sub_goal_item)




            for sub_goal_name, sub_goal_item in update_sub_goal_name_dict.items():
                if sub_goal_name not in previous_sub_goal_name_dict:
                    new_decomposition.append(sub_goal_item)
            
            if self.verbose:
                print("regression planning, update sub-goals")
                print(json.dumps(subgoals_decomposition, indent=4))  
                print("-"*20+">"*20)
                print(json.dumps(new_decomposition, indent=4))  


            subgoals_decomposition = new_decomposition
            # subgoals_decomposition = self._refresh_subgoals_decomposition_status(subgoals_decomposition)


            regression_json["subgoals_decomposition"] = subgoals_decomposition 

            if self.verbose:
                print("-" * 50 + "regression prompt" + "-"* 50)
                print(prompt)
                print("-" * 50 + "regression response" + "-"* 50)
                print(response)

            next_subgoal = regression_json.get("decision", {}).get("next_subgoal")
            if self.verbose:
                print("next sub-goal: ", next_subgoal)
                for entry in subgoals_decomposition:
                    if entry.get("subgoal") == next_subgoal:
                        print(json.dumps(entry, indent=4))
                print("current facts: ", self.current_facts)
                
            missing_next_sub_goal_flag = True
            for entry in subgoals_decomposition:
                if entry.get("subgoal") == next_subgoal:
                    if len(entry.get("subgoal", [])) > 0:
                        missing_next_sub_goal_flag = False
            
            if missing_next_sub_goal_flag:
                hints = "Note: The next sub-goal {} is missing or has no corresponding solution in the analyzed sub-goal decomposition.\n".format(next_subgoal)
                hints += "Refine the next-sub-goal decision or propose new solutions if any are viable.ss"

            else:
                if self._subgoal_preconditions_satisfied(next_subgoal, subgoals_decomposition):

                    # print(json.dumps(regression_json, indent=4))
                    # print("next sub-goal", next_subgoal)
                    # print(subgoals_decomposition["next_subgoal"])
                    # exit(0)
                    self.current_regression_topology = regression_json
                    self.current_task_guideline = base_guideline
                    self.current_task_guideline_json = {
                        "base_guideline": base_guideline,
                        "regression": regression_json,
                    }
                    self.current_regression_json = regression_json

                    self._append_agent_log(
                        "=" * 20 + "regression prompt" + "=" * 20 + "\n{content}\n".format(content=prompt)
                    )
                    self._append_agent_log(
                        "=" * 20 + "regression response" + "=" * 20 + "\n{content}\n".format(content=response)
                    )
                    self._append_agent_log(
                        "=" * 20 + "updated sub-goal decomposition" + "=" * 20 + "\n{content}\n".format(content=json.dumps(new_decomposition, indent=4))
                    )
                    return
                
                hints = "Note: All proposed solutions for sub-goal {} fail to satisfy their preconditions under the current world facts {}.\n".format(next_subgoal, self.current_facts)
                hints += "Refine the next-sub-goal decision or propose new solutions if any are viable.ss"
            sub_goal_feedback = "previous planning step produced a next_subgoal without satisfied preconditions; extend decomposition."

        self.current_regression_topology = regression_json
        self.current_task_guideline = base_guideline
        self.current_task_guideline_json = {
            "base_guideline": base_guideline,
            "regression": regression_json,
        }
        self.current_regression_json = regression_json

        # self._append_agent_log(
        #     "=" * 20 + "regression prompt" + "=" * 20 + "\n{content}\n".format(content=prompt)
        # )
        # self._append_agent_log(
        #     "=" * 20 + "regression response" + "=" * 20 + "\n{content}\n".format(content=response)
        # )
        self._append_agent_log(
            "Can not find a valid next sub-goal. Rollback to the final task-goal."
        )
        self.current_regression_json = None
        
        
        
    
    def perceive(self, obs, current_facts):

        obs_str = f"obs: {obs.strip()}\n"
        self.interaction_history += obs_str.strip() + "\n"

        prompt = OBSERVATION_TO_PREDICATES_PROMPT.format(facts=current_facts, 
                                                        new_action=self.previous_action, new_observation=obs)

        response = llm_response(prompt, self.llm, temperature=self.temperature)

        # print(response)
        response_json = json.loads(repair_json(response))
            # print(response_json)
        
        if self.verbose:
            print("="*20 + "state prompt" + "="*20)
            print(prompt.split("\nFollow exactly this schema for each new observation.\n")[1])
            print("="*20 + "state response" + "="*20)
            print(response)

        self._append_agent_log(
            "="*20 + "state prompt" + "="*20 + "\n{content}\n".format(
                content=prompt.split("\nFollow exactly this schema for each new observation.\n")[1]
            )
        )
        self._append_agent_log("="*20 + "state response" + "="*20 + "\n{content}\n".format(content=response))

        current_facts += response_json["add_facts"]
        for fact in response_json["remove_facts"]:
            if fact in current_facts:
                current_facts.remove(fact)
            else:
                pass
                # raise ValueError(f"Fact {fact} to be removed is not in current facts.")
        # self.interaction_history += f"facts: {self.current_facts}\n"

        # if "inventory(stick, 8)" in current_facts or "inventory(stick, 4)" in current_facts:
        #     exit(0)
        
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
        """Generate action: Layer 1 (planning) → Layer 2 (execution)."""

        self.re_plan_count = 0
        if self.preceived_flag:
            self.preceived_flag = False
        else:
            self.current_facts = self.perceive(obs, self.current_facts)

        # Track last action failure as temporary memory (refreshes each step)
        obs_str = str(obs).strip()
        if obs_str.startswith("Could not"):
            prev = getattr(self, 'previous_action', '') or ''
            self._last_action_failure = f"'{prev}' failed: {obs_str}"
            # For craft failures, append recipe + inventory so planner can diagnose
            if prev.startswith("craft "):
                inv = [f for f in self.current_facts if f.startswith("inventory(")]
                self._last_action_failure += f" | recipe: {prev} | inventory: {inv} | EXACT name match required — find alternative recipe."
                # Fix: detect repeated craft failures on the same action
                _prev_failed_craft = getattr(self, '_last_failed_craft', '')
                if prev == _prev_failed_craft:
                    self._craft_repeat_count = getattr(self, '_craft_repeat_count', 0) + 1
                    if self._craft_repeat_count >= 2:
                        self._record_subgoal_failure(
                            prev,
                            f"{prev} failed {self._craft_repeat_count}+ times: {obs_str}. "
                            "Ingredients are insufficient. MUST craft missing ingredients FIRST.",
                            "",
                        )
                else:
                    self._craft_repeat_count = 1
                self._last_failed_craft = prev
            # Persist get failures (item permanently unavailable)
            if prev.startswith("get "):
                # Strip leading count if present (e.g., "1 diorite" → "diorite")
                _tail = prev.split("get ", 1)[-1].strip()
                _parts = _tail.split(" ", 1)
                failed_item = (_parts[1] if len(_parts) > 1 and _parts[0].isdigit() else _tail).strip().rstrip("s")
                recipe_hint = ""
                if hasattr(self, "commands") and self.commands:
                    def _norm(s: str) -> str:
                        return s.lower().replace("_", " ").strip().rstrip("s")
                    needle = _norm(failed_item)
                    for ln in self.commands.splitlines():
                        line = ln.strip()
                        m = re.match(r'craft\s+\d+\s+(.+?)\s+using\b', line, re.IGNORECASE)
                        if not m:
                            continue
                        if _norm(m.group(1)) == needle:
                            recipe_hint = f" Recipe exists: '{line}'. Craft it instead of fetching."
                            break
                self._record_subgoal_failure(
                    prev,
                    f"{prev}: item unavailable in environment.{recipe_hint}"
                    + (" MUST use craft_item_from_ingredients instead." if recipe_hint else ""),
                    "",
                )
        else:
            self._last_action_failure = None

        precondition_error_msg, error_action = None, None
        cumulative_error_message = "At the current step, the following actions are not valid. \n"

        # ========== Layer 1: Planning ==========
        last_fail = getattr(self, '_last_action_failure', None) or ""
        feedback = f"Last action FAILED: {last_fail}. Do NOT repeat it — try a different approach." if last_fail else ""

        # Track consecutive failures on same subgoal → escalate feedback
        current_sg = getattr(self, 'sub_goal_instruction', '')
        if last_fail and current_sg:
            self._consecutive_subgoal_failures[current_sg] = \
                self._consecutive_subgoal_failures.get(current_sg, 0) + 1
            fail_count = self._consecutive_subgoal_failures[current_sg]
            if fail_count >= 3:
                feedback += (
                    f"\nCRITICAL: subgoal '{current_sg}' has failed {fail_count} consecutive times. "
                    "STOP retrying the same approach. You MUST: "
                    "1) Check if you have enough of EACH ingredient (count them in Facts); "
                    "2) Craft missing intermediate ingredients FIRST; "
                    "3) Try a completely different recipe chain if needed."
                )
        elif current_sg and current_sg in self._consecutive_subgoal_failures:
            # Success on this subgoal — reset counter
            del self._consecutive_subgoal_failures[current_sg]

        plan = self._plan(sub_goal_feedback=feedback if feedback else "No recent failure.")

        if self.re_plan_count >= 10:
            return "think: task failed! Too much re-planing (10) for produce a valid action. ", "textual action"

        next_subgoal = plan.get("next_subgoal")

        if next_subgoal in (None, "none"):
            return "think: The final goal has been achieved. ", "textual action"

        reasoning = plan.get("reasoning", "")
        sub_goal_guideline = "current sub-goal: " + str(next_subgoal) + "\n"
        sub_goal_guideline += "reasoning: " + reasoning + "\n"
        self.sub_goal_instruction = str(next_subgoal)

        with open(self.traj_path, "a") as f:
            f.write("[sub-goal] " + self.sub_goal_instruction + "\n")

        # Format approach recommendation for Layer 2
        approach = plan.get("approach", {"type": "none", "name": "", "why": ""})
        routing_recommendation = json.dumps({"recommendation": approach}, indent=4)

        # ========== Build detailed skill info for Layer 2 ==========
        skill_info = ""
        for task_type in ENV_TYPES:
            if task_type not in self.skill_memory_blocks_with_task_type:
                continue
            for sub_goal_i, calling_node_list in self.skill_memory_blocks_with_task_type[task_type].items():
                skill_json = self.skill_json_dict[sub_goal_i]
                skill_expression = "## " + sub_goal_i + "(" + ", ".join(skill_json["parameters"]) + ")"
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

        # ========== Layer 2: Execution (detailed skill info disclosure) ==========
        for try_i in range(3):
                
            
            if try_i > 0:
                cumulative_error_message += f"action: {error_action}\n"
                cumulative_error_message += f"reason: {precondition_error_msg}\n"
                error_message = cumulative_error_message
            else:
                error_message = ""

            prompt = EXECUTION_WITH_ROUTING_PROMPT.format(
                examples=self.example_trace, 
                task_message=self.interaction_history, 
                action_check_error = error_message,
                skill_info = skill_info,
                sub_goal_guideline = sub_goal_guideline,
                routing_recommendation = routing_recommendation,
                facts_current = json.dumps(self.current_facts, indent=2),
                crafting_commands=self.commands,
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

            self._append_agent_log("="*20 + "planning recommendation" + "="*20 + "\n{content}\n".format(content=routing_recommendation))

            try:
                prompt_content_for_log = prompt.split("\n\n\n# Here is the task.")[1]
            except Exception:
                prompt_content_for_log = prompt

            self._append_agent_log("="*20 + "action prompt" + "="*20 + "\n{content}\n".format(content=prompt_content_for_log))
            self._append_agent_log("="*20 + "action response" + "="*20 + "\n{content}\n".format(content=response))


            response_json = json.loads(repair_json(response))

            # Bug fix: guard against null/invalid skill + null action deadlock
            _skill = response_json.get("skill")
            _action = response_json.get("action")
            if not _action and (not _skill or _skill == "none"):
                precondition_error_msg = (
                    "Executor returned no action and no skill — "
                    "preconditions for the recommended approach are unmet. "
                    "Re-plan with a different decomposition."
                )
                error_action = str(response_json)
                continue

            if response_json["action"]:
                

                if response_json["action"].startswith("craft"):
                    action_text = response_json["action"].strip()
                    # Allow tag substitution: don't reject craft commands where
                    # a specific variant (e.g. "oak planks") replaces a generic
                    # tag name (e.g. "planks") — the env accepts these.
                    pass

                self.previous_action = response_json["action"]

                self.trajectory_memory_graph.nodes[self.trajectory_memory_node_id].action = response_json["action"].strip()

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
                else:
                    _param_value = param_value
                formated_param_bindings[param_name] = _param_value
            
                    
            precondition_satisfied = True
            precondition_error_msg = ""
            for precondition_expression in precondition_expression_list:
                params = formated_param_bindings
                converted_expr = precondition_expression.format(**params)
                

                current_world = World(predicates=self.current_facts)

                env = _build_predicate_env(executor, current_world, self.current_facts)
                condition_value = bool(eval(converted_expr, env, env))

                if self.verbose:
                    print("pre-condition checking:")
                    print("convert expr: ", precondition_expression, " --> ", converted_expr)
                    print("value: ", condition_value)

                if not condition_value:
                    precondition_satisfied = False
                    precondition_error_msg += f"Precondition {converted_expr} is not satisfied.\n"
            
            if precondition_satisfied:
                # self.previous_action = response_json
                return response_json, "skill action"
            else:
                error_message, error_action = precondition_error_msg, response_json
            
        # All 3 retries exhausted without a valid action/skill.
        # Re-plan: record the failure and ask the planner to decompose further.
        failed_approach = response_json.get("skill") or response_json.get("action") or "unknown"
        replan_feedback = (
            f"Executor failed 3 times on subgoal '{self.sub_goal_instruction}' "
            f"(last attempt: {failed_approach}). Preconditions unmet: {precondition_error_msg}. "
            "Decompose the subgoal into smaller steps — craft missing intermediate ingredients first."
        )
        self._record_subgoal_failure(
            str(failed_approach),
            f"{failed_approach} action failed: preconditions unmet for subgoal {self.sub_goal_instruction}",
            "",
        )
        return f"think: re-planning — {replan_feedback}", "textual action"


from concepts.dsl.dsl_functions import Function, FunctionTyping
from concepts.dsl.dsl_types import BOOL, ObjectType, ObjectConstant, ListType
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.dsl.expression import (
    ConstantExpression, 
    BoolExpression,
    BoolOpType,
    FunctionApplicationExpression,
    ObjectConstantExpression,
    QuantificationExpression,
    QuantificationOpType,
    VariableExpression,
)
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser
from concepts.dsl.value import Value
from concepts.dsl.expression_utils import flatten_expression


from .infoflow2graph import parse_mermaid_with_info, validate_edge_payloads, validate_node_inputs
from .infoflow2graph import reharsal_with_info, reharsal_online_with_info
from .infoflow2graph import extract_start_inputs

import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union
import ast
import re



ENV_TYPES = {
'pick_and_place': 'put',
'pick_clean_then_place': 'clean',
'pick_heat_then_place': 'heat',
'pick_cool_then_place': 'cool',
'look_at_obj': 'examine',
'pick_two_obj': 'puttwo'
}

ACTION_PATTERNS = [
    # action, regex pattern, post-process lambda returning dict of slots
    ("goto",  re.compile(r"^go to (?P<receptacle>.+)$"),                           lambda m: {"receptacle": canon(m["receptacle"])}),
    ("open",  re.compile(r"^open (?P<receptacle>.+)$"),                            lambda m: {"receptacle": canon(m["receptacle"])}),
    ("close", re.compile(r"^close (?P<receptacle>.+)$"),                           lambda m: {"receptacle": canon(m["receptacle"])}),
    ("take",  re.compile(r"^take (?P<item>.+) from (?P<src>.+)$"),             lambda m: {"item": canon(m["item"]), "receptacle": canon(m["src"])}),
    ("clean",  re.compile(r"^clean (?P<item>.+) with (?P<src>.+)$"),             lambda m: {"item": canon(m["item"]), "receptacle": canon(m["src"])}),
    ("cool",  re.compile(r"^cool (?P<item>.+) with (?P<src>.+)$"),             lambda m: {"item": canon(m["item"]), "receptacle": canon(m["src"])}),
    ("heat",  re.compile(r"^heat (?P<item>.+) with (?P<src>.+)$"),             lambda m: {"item": canon(m["item"]), "receptacle": canon(m["src"])}),
    ("put", re.compile(r"^put (?P<item>.+) (?:in(?:/on)?|on) (?P<receptacle>.+)$"), lambda m: {"item": canon(m["item"]), "receptacle": canon(m["receptacle"])}), 
    ("move",  re.compile(r"^move (?P<item>.+) to (?P<receptacle>.+)$"),              lambda m: {"item": canon(m["item"]), "receptacle": canon(m["receptacle"])}),
    ("use",re.compile(r"^use (?P<item>.+)$"),                          lambda m: {"item": canon(m["item"])})
    # add more patterns here as needed
]

PREDICATE_PATTERN = re.compile(r"^(?P<name>[^()]+)\((?P<args>.*)\)$")
def canon(text: str) -> str:
    """Turn 'apple 3' → 'apple_3', strip articles, lower-case."""
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.replace(" ", "_")

class World:
    def __init__(self, predicates: List[str] = None):
        self.predicates = predicates if predicates is not None else []

        self.all_items = set()
        self.all_receptacles = set()
        self.all_items_types = set()
        self.all_receptacles_types = set()

        for p in self.predicates:
            m = PREDICATE_PATTERN.match(p)

            receptacle_i, item_i = None, None
            func_name, func_args = m.groupdict()["name"], m.groupdict()["args"]
            if func_name == "locate":
                receptacle_i = func_args.strip()
            elif func_name == "reachable":
                receptacle_i = func_args.strip()
            elif func_name == "contains":
                receptacle_i = func_args.split(",")[0].strip()
                item_i = func_args.split(",")[1].strip()
            elif func_name == "holding":
                item_i = func_args
            elif func_name == "is_open":
                receptacle_i = func_args.strip()
            elif func_name == "is_closed":
                receptacle_i = func_args.strip()
            elif func_name == "is_cleaned":
                item_i = func_args.strip()
            elif func_name == "is_cooled":
                item_i = func_args.strip()
            elif func_name == "is_heated":
                item_i = func_args.strip()
            elif func_name == "is_turned_on":
                item_i = func_args.strip()
            
            if item_i is not None:
                normalized_item = transform_item_name(item_i)
                self.all_items.add(normalized_item)
                self.all_items_types.add(normalized_item.split("_")[0])

            if receptacle_i is not None:
                normalized_receptacle = transform_item_name(receptacle_i)
                self.all_receptacles.add(normalized_receptacle)
                self.all_receptacles_types.add(normalized_receptacle.split("_")[0])
        
        # print(self.predicates)
        
        # print(self.all_items)
        # print(self.all_receptacles)
        # print(self.all_items_types)
        # print(self.all_receptacles_types)
        
        # exit(0)

    def is_item_name(self, item: str) -> bool:
        if not item:
            return False
        return transform_item_name(item) in self.all_items

    def ensure_item_name(self, item: str) -> str:
        normalized = transform_item_name(item)
        if normalized not in self.all_items:
            known_items = ", ".join(sorted(self.all_items)) or "<none>"
            raise ValueError(
                f"Unknown item name '{item}' (normalized as '{normalized}'). Known items: {known_items}"
            )
        return normalized


T_AGENT = ObjectType("Agent")
T_RECEPTACLE = ObjectType("Receptacle")
T_ITEM = ObjectType("Item")
T_TYPE = ObjectType("TypeName")
T_ITEM_LIST = ListType(T_ITEM) 
T_RECEPTACLE_LIST = ListType(T_RECEPTACLE) 


def transform_item_name(item_name: str) -> str:
    # return item_name
    """
    Transform the item name in the expression to the format used in the skill workflow.
    """
    transformed_name, if_trans = re.subn(r'([A-Za-z]+)\s+(\d+)', r'\1_\2', item_name)
    return transformed_name if if_trans else item_name


def _normalize_list_argument(raw: Union[str, Iterable[str], None]) -> List[str]:
    """
    Parameters arrive either as Python lists or as stringified lists (after str.format).
    Normalize to a list of snake_case names so downstream predicates receive consistent inputs.
    """
    if raw is None:
        return []

    candidate: Iterable[str]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            candidate = [raw]
        else:
            if isinstance(parsed, (list, tuple, set)):
                candidate = parsed
            else:
                candidate = [parsed]
    elif isinstance(raw, (list, tuple, set)):
        candidate = raw
    else:
        candidate = [raw]  # best-effort fallback

    normalized: List[str] = []
    for value in candidate:
        if value is None:
            continue
        normalized.append(transform_item_name(str(value)))
    return normalized


def _exists_in_collection(values: List[str], predicate: Callable[[str], bool]) -> bool:
    for value in values:
        try:
            if bool(predicate(value)):
                
                #print(predicate, value, predicate(value),  "True")
                return True
            else:
                # print(predicate, value, predicate(value), "False")
                pass
        except Exception as e:
            print(f"Error checking predicate for value {value}: {e}")
            continue
    return False


def _forall_in_collection(values: List[str], predicate: Callable[[str], bool]) -> bool:
    for value in values:
        try:
            if not bool(predicate(value)):
                return False
        except Exception:
            return False
    return True


def exists_item_in_list(items: Union[str, Iterable[str], None], predicate: Callable[[str], bool]) -> bool:
    """
    Check if any item inside the provided list satisfies the predicate.
    Items can be provided as actual Python lists or stringified lists.
    """
    return _exists_in_collection(_normalize_list_argument(items), predicate)


def forall_item_in_list(items: Union[str, Iterable[str], None], predicate: Callable[[str], bool]) -> bool:
    """
    Check if every item in the provided list satisfies the predicate (vacuously true for empty lists).
    """
    return _forall_in_collection(_normalize_list_argument(items), predicate)


def exists_receptacle_in_list(receptacles: Union[str, Iterable[str], None], predicate: Callable[[str], bool]) -> bool:
    """
    Check if any receptacle in the list satisfies the predicate.
    """
    return _exists_in_collection(_normalize_list_argument(receptacles), predicate)


def forall_receptacle_in_list(receptacles: Union[str, Iterable[str], None], predicate: Callable[[str], bool]) -> bool:
    """
    Check if every receptacle in the list satisfies the predicate (vacuously true for empty lists).
    """
    return _forall_in_collection(_normalize_list_argument(receptacles), predicate)
    

def _build_domain() -> FunctionDomain:
    domain = FunctionDomain("alfworld")
    for object_type in (T_AGENT, T_RECEPTACLE, T_ITEM, T_TYPE):
        domain.define_type(object_type)

    domain.define_function(Function("locate", FunctionTyping[BOOL](T_RECEPTACLE)))
    domain.define_function(Function("reachable", FunctionTyping[BOOL](T_RECEPTACLE)))
    domain.define_function(Function("contains", FunctionTyping[BOOL](T_RECEPTACLE, T_ITEM)))
    domain.define_function(Function("holding", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("is_open", FunctionTyping[BOOL](T_RECEPTACLE)))
    domain.define_function(Function("is_closed", FunctionTyping[BOOL](T_RECEPTACLE)))
    domain.define_function(Function("is_cleaned", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("is_cooled", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("is_heated", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("is_turned_on", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("is_item_of_type", FunctionTyping[BOOL](T_ITEM, T_TYPE)))
    domain.define_function(Function("is_receptacle_of_type", FunctionTyping[BOOL](T_RECEPTACLE, T_TYPE)))


    



    return domain


DOMAIN = _build_domain()

def can_goto(receptacle: "Receptacle") -> bool:
    return reachable(receptacle) and not locate(receptacle)


def can_open(receptacle: "Receptacle") -> bool:
    return (
        locate(receptacle) 
        and is_closed(receptacle)
        and not is_open(receptacle)
    )


def can_take(receptacle: "Receptacle", item: "Item") -> bool:
    return (
        locate(receptacle)
        and contains(receptacle, item)
        and not is_closed(receptacle)
        and not holding(item)
    )


def can_put(receptacle: "Receptacle", item: "Item") -> bool:
    return (
        locate(receptacle) 
        and holding(item) 
        and not is_closed(receptacle)
    )

def can_clean(receptacle: "Receptacle", item: "Item") -> bool:
            return (
                locate(receptacle) 
                and holding(item)
                and not is_cleaned(item)
        )
def can_cool(receptacle: "Receptacle", item: "Item") -> bool:
            return (
                locate(receptacle) 
                and holding(item)
                and not is_cooled(item)
        )
def can_heat(receptacle: "Receptacle", item: "Item") -> bool:
            return (
                locate(receptacle) 
                and holding(item)
                and not is_heated(item)
        )
def can_turn_on(item: "Item") -> bool:
    return (
        exists("Receptacle", lambda receptacle: locate(receptacle) and contains(receptacle, item))
        and not is_turned_on(item)
    )



DERIVED_FUNCTION_BUILDERS = (
    can_goto,
    can_open,
    can_take,
    can_put,
    can_clean,
    can_cool,
    can_heat,
    can_turn_on,
)


def _register_derived_functions(domain: FunctionDomain) -> List[Tuple[str, str]]:
    parser = FOLPythonParser(domain, inplace_definition=True)
    registry: List[Tuple[str, str]] = list()
    for builder in DERIVED_FUNCTION_BUILDERS:
        fn = parser.parse_function(inspect.getsource(builder))
        domain.define_function(fn)
        registry.append((fn.name, str(fn.derived_expression)))
    return registry

DERIVED_SUMMARY = _register_derived_functions(DOMAIN)

class ALFWorldExecutor(FunctionDomainExecutor):
    def locate(self, receptacle: str) -> bool:  # type: ignore[override]
        predicate_str = f"locate({transform_item_name(receptacle)})"
        
        return predicate_str in self.grounding.predicates
    
    def reachable(self, receptacle: str) -> bool:  # type: ignore[override]
        predicate_str = f"reachable({transform_item_name(receptacle)})"

        # print("check reachable...: ", predicate_str, "\n", receptacle)
        # print(self.grounding.predicates)
        # print(predicate_str in self.grounding.predicates)
        return predicate_str in self.grounding.predicates

    def contains(self, receptacle: str, item: str) -> bool:  # type: ignore[override]
        predicate_str = f"contains({transform_item_name(receptacle)}, {transform_item_name(item)})"
        return predicate_str in self.grounding.predicates

    def holding(self, item: str) -> bool:  # type: ignore[override]

        # print(item)
        predicate_str = f"holding({transform_item_name(item)})"

        # print("check holding...: ", predicate_str, "\n", item)
        # print(predicate_str in self.grounding.predicates)

        return predicate_str in self.grounding.predicates

    def is_open(self, receptacle: str) -> bool:  # type: ignore[override]
        predicate_str = f"is_open({transform_item_name(receptacle)})"
        not_predicate_str = f"is_closed({transform_item_name(receptacle)})"
        return (predicate_str in self.grounding.predicates) or (not not_predicate_str in self.grounding.predicates)
    def is_closed(self, receptacle: str) -> bool:  # type: ignore[override]
        predicate_str = f"is_closed({transform_item_name(receptacle)})"
        return (predicate_str in self.grounding.predicates)
    def is_cleaned(self, item: str) -> bool:  # type: ignore[override]
        item_name = self._ensure_known_item(item)
        predicate_str = f"is_cleaned({item_name})"
        return predicate_str in self.grounding.predicates
    def is_cooled(self, item: str) -> bool:  # type: ignore[override]
        item_name = self._ensure_known_item(item)
        predicate_str = f"is_cooled({item_name})"
        return predicate_str in self.grounding.predicates
    def is_heated(self, item: str) -> bool:  # type: ignore[override]
        item_name = self._ensure_known_item(item)
        predicate_str = f"is_heated({item_name})"
        return predicate_str in self.grounding.predicates
    def is_turned_on(self, item: str) -> bool:  # type: ignore[override]
        item_name = self._ensure_known_item(item)
        predicate_str = f"is_turned_on({item_name})"
        return predicate_str in self.grounding.predicates
    def is_item_of_type(self, item: str, item_type: str) -> bool:  # type: ignore[override]
        item_name = self._ensure_known_item(item)
        return item_name.startswith(item_type + "_")
    def is_receptacle_of_type(self, receptacle: str, receptacle_type: str) -> bool:  # type: ignore[override]
        receptacle_name = transform_item_name(receptacle)
        return receptacle_name.startswith(receptacle_type + "_")
    
    def _ensure_known_item(self, item: str) -> str:
        grounding = getattr(self, "grounding", None)
        if grounding is not None and hasattr(grounding, "ensure_item_name"):
            return grounding.ensure_item_name(item)

        item_name = transform_item_name(item)
        if grounding is None:
            return item_name

        known_items = getattr(grounding, "all_items", None)
        if isinstance(known_items, (set, list, tuple)):
            if item_name not in known_items:
                known_items_str = ", ".join(sorted(known_items)) or "<none>"
                raise ValueError(
                    f"Unknown item name '{item}' (normalized as '{item_name}'). Known items: {known_items_str}"
                )
        return item_name
    
    def _objects_of_type(self, obj_type: ObjectType) -> Set[str]:
        objects: Set[str] = set()
        for predicate in getattr(self.grounding, "predicates", []):
            match = PREDICATE_PATTERN.match(predicate)
            if not match:
                continue

            fn_name = match.group("name")
            args_str = match.group("args")
            function = self.domain.functions.get(fn_name)
            if function is None or not args_str:
                continue

            args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            for variable, arg in zip(function.ftype.arguments, args):
                if variable.dtype is obj_type:
                    objects.add(arg)
        return objects


    def _execute(self, expr):
        if isinstance(expr, BoolExpression):
            if expr.bool_op is BoolOpType.AND:
                value = all(self._execute(child).value for child in expr.arguments)
            elif expr.bool_op is BoolOpType.OR:
                value = any(self._execute(child).value for child in expr.arguments)
            elif expr.bool_op is BoolOpType.NOT:
                value = not self._execute(expr.arguments[0]).value
            elif expr.bool_op is BoolOpType.IMPLIES:
                left, right = (self._execute(child).value for child in expr.arguments)
                value = (not left) or right
            elif expr.bool_op is BoolOpType.XOR:
                value = sum(self._execute(child).value for child in expr.arguments) % 2 == 1
            else:
                raise NotImplementedError(f"Unsupported boolean operator: {expr.bool_op}")
            return Value(BOOL, value)

        if isinstance(expr, QuantificationExpression):
            obj_type = expr.variable.dtype
            candidates = self._objects_of_type(obj_type)

            if expr.quantification_op is QuantificationOpType.EXISTS:
                for candidate in candidates:
                    constant_expr = ObjectConstantExpression(ObjectConstant(candidate, obj_type))
                    flattened = flatten_expression(
                        expr.expression,
                        mappings={VariableExpression(expr.variable): constant_expr},
                    )
                    if self._execute(flattened).value:
                        return Value(BOOL, True)
                return Value(BOOL, False)

            if expr.quantification_op is QuantificationOpType.FORALL:
                for candidate in candidates:
                    constant_expr = ObjectConstantExpression(ObjectConstant(candidate, obj_type))
                    flattened = flatten_expression(
                        expr.expression,
                        mappings={VariableExpression(expr.variable): constant_expr},
                    )
                    if not self._execute(flattened).value:
                        return Value(BOOL, False)
                return Value(BOOL, True)

            raise NotImplementedError(f"Unsupported quantifier: {expr.quantification_op}")

        if isinstance(expr, FunctionApplicationExpression) and expr.function.is_derived:
            mappings = {
                VariableExpression(var): arg
                for var, arg in zip(expr.function.arguments, expr.arguments)
            }
            expanded = flatten_expression(expr.function.derived_expression, mappings=mappings)
            return self._execute(expanded)

        if isinstance(expr, ObjectConstantExpression):
            constant = expr.constant
            return Value(constant.dtype, constant.name)

        return super()._execute(expr)

executor = ALFWorldExecutor(DOMAIN)



def get_pre_condition(action_param, induced_skills = []):
    if action_param["type"] == "goto":
        return lambda d: d.f_can_goto(action_param["receptacle"]), f"The target {action_param['receptacle']} should be reachable and not the current location of agent."
    elif action_param["type"] == "open":
        return lambda d: d.f_can_open(action_param["receptacle"]), f"The target {action_param['receptacle']} should be openable and currently closed. The agent should locate at this {action_param['receptacle']}."
    elif action_param["type"] == "close":
        return lambda d: d.f_can_close(action_param["receptacle"]), f"The target {action_param['receptacle']} should be closeable and currently open. The agent should locate at this {action_param['receptacle']}."
    elif action_param["type"] == "take":
        return lambda d: d.f_can_take(action_param["receptacle"], action_param["item"]), f"The target item {action_param['item']} of take action should be contained at the given receptacle {action_param['receptacle']}. The agent should locate at this {action_param['receptacle']}. The agent currently holds nothing."
    elif action_param["type"] == "put":
        return lambda d: d.f_can_put(action_param["receptacle"], action_param["item"]), f"The agent should locate at this {action_param['receptacle']}. The agent should hold the item {action_param['item']}."
    elif action_param["type"] == "move":
        return lambda d: d.f_can_put(action_param["receptacle"], action_param["item"]), f"The agent should locate at this {action_param['receptacle']}. The agent should hold the item {action_param['item']}."
    elif action_param["type"] == "clean":
        return lambda d: d.f_can_clean(action_param["receptacle"], action_param["item"]), f"The agent should locate at this {action_param['receptacle']}. The agent should hold the item {action_param['item']}."
    elif action_param["type"] == "cool":
        return lambda d: d.f_can_cool(action_param["receptacle"], action_param["item"]), f"The agent should locate at this {action_param['receptacle']}. The agent should hold the item {action_param['item']}."
    elif action_param["type"] == "heat":
        return lambda d: d.f_can_heat(action_param["receptacle"], action_param["item"]), f"The agent should locate at this {action_param['receptacle']}. The agent should hold the item {action_param['item']}."
    elif action_param["type"] == "use":
        return lambda d: d.f_can_turn_on(action_param["item"]), f"The agent should locate at a receptacle which contains the item {action_param['item']}."
    
    else:
        for (sub_goal_i, skill_i, mermaid_i, params_i) in induced_skills:
            print(skill_i["name"])
            print(action_param["type"])
            print(skill_i)

            if skill_i["name"] == action_param["type"]:
                print(skill_i["pre_condition"])
                return skill_i["pre_condition"], skill_i["pre_condition_msg"]

    raise ValueError(f"Pre-condition for action {action_param} is not defined.")

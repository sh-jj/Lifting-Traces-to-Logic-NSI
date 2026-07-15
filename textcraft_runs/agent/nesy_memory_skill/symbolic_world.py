from concepts.dsl.dsl_functions import Function, FunctionTyping
from concepts.dsl.dsl_types import BOOL, INT64, ObjectType, ObjectConstant, ListType, Variable
from concepts.dsl.executors.function_domain_executor import FunctionDomainExecutor
from concepts.dsl.expression import (
    ConstantExpression,
    BoolExpression,
    BoolOpType,
    FunctionApplicationExpression,
    ObjectConstantExpression,
    CompareOpType,
    QuantificationExpression,
    QuantificationOpType,
    ValueCompareExpression,
    VariableExpression,
)
from concepts.dsl.expression_utils import flatten_expression
from concepts.dsl.function_domain import FunctionDomain
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser
from concepts.dsl.value import Value
from concepts.dsl.parsers.fol_python_parser import FOLPythonParser

import inspect
import ast
import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union


ACTION_PATTERNS = [
    ("inventory", re.compile(r"^inventory\b", flags=re.IGNORECASE), lambda _: {}),
    (
        "get",
        re.compile(r"^get\s+(?:(?P<count>\d+)\s+)?(?P<item>.+?)(?:\.)?$", flags=re.IGNORECASE),
        lambda m: {
            "item": canon(m.group("item")),
            "count": _parse_count(m.group("count")) if m.group("count") else 1,
        },
    ),
    (
        "craft",
        re.compile(r"^craft\s+(?:(?P<count>\d+)\s+)?(?P<item>.+?)\s+using\s+(?P<ingredients>.+)$", flags=re.IGNORECASE),
        lambda m: {
            "item": canon(m.group("item")),
            "count": _parse_count(m.group("count")) if m.group("count") else 1,
            "ingredients": _parse_ingredient_list(m.group("ingredients")),
        },
    ),
]

PREDICATE_PATTERN = re.compile(r"^(?P<name>[^()]+)\((?P<args>.*)\)$")


def canon(text: str) -> str:
    """Normalize item text to snake_case without namespace prefixes."""
    if text is None:
        return ""
    
    # print("text: ", text)

    if type(text) == int:
        return str(text)
    
    text = text.strip().lower().rstrip(".")
    text = text.replace("minecraft:", "")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[\s\-]+", " ", text)
    text = re.sub(r"([A-Za-z]+)\s+(\d+)", r"\1_\2", text)
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def transform_item_name(item_name: str) -> str:
    """Alias kept for compatibility with the Alfworld code path."""
    return canon(item_name)


def _parse_count(raw: Optional[Union[str, int]]) -> int:
    """Parse a count from string/int, defaulting to 0 on failure."""
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(str(raw)))
        except Exception:
            return 0


def _parse_ingredient_list(raw: str) -> List[Dict[str, Union[str, int]]]:
    ingredients: List[Dict[str, Union[str, int]]] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        match = re.match(r"(?P<count>\d+)\s+(?P<item>.+)", token)
        if match:
            count = _parse_count(match.group("count"))
            item = canon(match.group("item"))
        else:
            count = 1
            item = canon(token)
        ingredients.append({"item": item, "count": count})
    return ingredients


class World:
    def __init__(self, predicates: Optional[List[str]] = None):
        self.predicates = predicates if predicates is not None else []

        self.inventory: Dict[str, int] = {}
        self.goals: Dict[str, int] = {}
        self.unavailable: Set[str] = set()

        self.all_items: Set[str] = set()
        self.all_items_types: Set[str] = set()

        self.agent_id = "agent"

        for p in self.predicates:
            match = PREDICATE_PATTERN.match(p.strip())
            if not match:
                continue
            func_name = match.groupdict()["name"].strip()
            func_args = [a.strip() for a in match.groupdict()["args"].split(",") if a.strip()]
            if func_name == "inventory" and len(func_args) == 2:
                item_i = transform_item_name(func_args[0])
                count_i = _parse_count(func_args[1])
                self.inventory[item_i] = count_i
                self.all_items.add(item_i)
                self.all_items_types.add(item_i.split("_")[0])
            elif func_name == "goal" and len(func_args) == 2:
                item_i = transform_item_name(func_args[0])
                count_i = _parse_count(func_args[1])
                self.goals[item_i] = count_i
                self.all_items.add(item_i)
                self.all_items_types.add(item_i.split("_")[0])
            elif func_name == "unavailable" and len(func_args) >= 1:
                item_i = transform_item_name(func_args[0])
                self.unavailable.add(item_i)
                self.all_items.add(item_i)
                self.all_items_types.add(item_i.split("_")[0])

    def is_item_name(self, item: str) -> bool:
        if not item:
            return False
        return transform_item_name(item) in self.all_items

    def ensure_item_name(self, item: str, strict: bool = False) -> str:
        normalized = transform_item_name(item)
        if strict and self.all_items and normalized not in self.all_items:
            known_items = ", ".join(sorted(self.all_items)) or "<none>"
            raise ValueError(
                f"Unknown item name '{item}' (normalized as '{normalized}'). Known items: {known_items}"
            )
        return normalized

    def current_count(self, item: str) -> int:
        item_name = transform_item_name(item)
        return self.inventory.get(item_name, 0)

    def goal_count(self, item: str) -> Optional[int]:
        item_name = transform_item_name(item)
        return self.goals.get(item_name)


T_AGENT = ObjectType("Agent")
T_ITEM = ObjectType("Item")
T_TYPE = ObjectType("TypeName")
T_ITEM_LIST = ListType(T_ITEM)
T_INT = INT64
T_INT_LIST = ListType(T_INT)


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
                return True
        except Exception:
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
    """Check if any item inside the provided list satisfies the predicate."""
    return _exists_in_collection(_normalize_list_argument(items), predicate)


def forall_item_in_list(items: Union[str, Iterable[str], None], predicate: Callable[[str], bool]) -> bool:
    """Check if every item in the provided list satisfies the predicate (vacuously true for empty lists)."""
    return _forall_in_collection(_normalize_list_argument(items), predicate)


def _build_domain() -> FunctionDomain:
    domain = FunctionDomain("textcraft")
    for object_type in (T_AGENT, T_ITEM, T_TYPE):
        domain.define_type(object_type)

    domain.define_type(T_INT)

    domain.define_function(Function("goal", FunctionTyping[BOOL](T_ITEM, T_INT)))
    domain.define_function(Function("inventory", FunctionTyping[BOOL](T_ITEM, T_INT)))
    domain.define_function(Function("unavailable", FunctionTyping[BOOL](T_ITEM)))
    domain.define_function(Function("current_count_of_item", FunctionTyping[T_INT](T_ITEM)))
    domain.define_function(Function("count", FunctionTyping[T_INT](T_ITEM)))

    domain.define_function(Function("numerical_greater_than", FunctionTyping[BOOL](T_INT,T_INT)))
    domain.define_function(Function("numerical_greater_equal", FunctionTyping[BOOL](T_INT,T_INT)))
    domain.define_function(Function("numerical_less_than", FunctionTyping[BOOL](T_INT,T_INT)))
    domain.define_function(Function("numerical_less_equal", FunctionTyping[BOOL](T_INT,T_INT)))
    domain.define_function(Function("numerical_equal", FunctionTyping[BOOL](T_INT,T_INT)))
    domain.define_function(Function("greater_than", FunctionTyping[BOOL](T_INT, T_INT)))
    domain.define_function(Function("is_holding", FunctionTyping[BOOL](T_ITEM)))



    return domain


DOMAIN = _build_domain()
parser = FOLPythonParser(DOMAIN, inplace_definition=True)


def can_inventory() -> bool:
    return True


def has_enough(item: "Item", required: int) -> bool:
    return current_count_of_item(item) >= required


def needs_item(item: "Item", required: int) -> bool:
    return current_count_of_item(item) < required


def can_get(item: "Item", required: int) -> bool:
    return (not unavailable(item)) and needs_item(item, required)


def can_craft(item: "Item", required: int) -> bool:
    return (not unavailable(item)) and needs_item(item, required)


DERIVED_FUNCTION_BUILDERS = (
    can_inventory,
    has_enough,
    needs_item,
    can_get,
    can_craft,
)


def _register_derived_functions(domain: FunctionDomain) -> List[Tuple[str, str]]:
    parser = FOLPythonParser(domain, inplace_definition=True)
    registry: List[Tuple[str, str]] = []
    for builder in DERIVED_FUNCTION_BUILDERS:
        fn = parser.parse_function(inspect.getsource(builder))
        domain.define_function(fn)
        registry.append((fn.name, str(fn.derived_expression)))
    return registry


DERIVED_SUMMARY = _register_derived_functions(DOMAIN)


class TextCraftExecutor(FunctionDomainExecutor):
    def goal(self, item: str, count: Union[int, str]) -> bool:  # type: ignore[override]
        item_name = self._normalize_item(item)
        target = _parse_count(count)
        return self.grounding.goals.get(item_name) == target

    def inventory(self, item: str, count: Union[int, str]) -> bool:  # type: ignore[override]
        item_name = self._normalize_item(item)
        target = _parse_count(count)
        return self.grounding.inventory.get(item_name) == target

    def unavailable(self, item: str) -> bool:  # type: ignore[override]
        item_name = self._normalize_item(item)
        return item_name in self.grounding.unavailable

    def current_count_of_item(self, item: str) -> int:  # type: ignore[override]
        item_name = self._normalize_item(item)
        # print(f"symbolic predicate current_count_of_item({item_name})", self.grounding.inventory.get(item_name, 0))
        return self.grounding.inventory.get(item_name, 0)

    def _normalize_item(self, item: str) -> str:
        grounding = getattr(self, "grounding", None)
        if grounding is not None and hasattr(grounding, "ensure_item_name"):
            return grounding.ensure_item_name(item)
        return transform_item_name(item)

    def numerical_greater_than(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) > int(b)

    def numerical_greater_equal(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) >= int(b)
    def numerical_less_than(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) < int(b)

    def numerical_less_equal(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) <= int(b)

    def numerical_equal(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) == int(b)   

    def count(self, item: str) -> int:  # type: ignore[override]
        """Alias for current_count_of_item so we can express count(k) in derived expressions."""
        return self.current_count_of_item(item)

    def greater_than(self, a: Union[int, str], b: Union[int, str]) -> bool:  # type: ignore[override]
        return int(a) > int(b)

    def is_holding(self, item: str) -> bool:  # type: ignore[override]
        return self.count(item) > 0

    def _objects_of_type(self, obj_type: ObjectType) -> Set[str]:
        objects: Set[str] = set()
        if obj_type is T_AGENT:
            objects.add(getattr(self.grounding, "agent_id", "agent"))
        if obj_type is T_ITEM:
            objects.update(getattr(self.grounding, "all_items", set()) or set())
            objects.update(getattr(self.grounding, "inventory", {}).keys())
            objects.update(getattr(self.grounding, "goals", {}).keys())
        if obj_type is T_TYPE:
            objects.update(getattr(self.grounding, "all_items_types", set()) or set())
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

        if isinstance(expr, ValueCompareExpression):
            lhs = self._execute(expr.lhs)
            rhs = self._execute(expr.rhs)

            if expr.compare_op is CompareOpType.EQ:
                value = lhs.value == rhs.value
            elif expr.compare_op is CompareOpType.NEQ:
                value = lhs.value != rhs.value
            elif expr.compare_op is CompareOpType.LT:
                value = lhs.value < rhs.value
            elif expr.compare_op is CompareOpType.LEQ:
                value = lhs.value <= rhs.value
            elif expr.compare_op is CompareOpType.GT:
                value = lhs.value > rhs.value
            elif expr.compare_op is CompareOpType.GEQ:
                value = lhs.value >= rhs.value
            else:
                raise NotImplementedError(f"Unsupported comparison operator: {expr.compare_op}")

            return Value(BOOL, value)

        if isinstance(expr, VariableExpression):
            var = expr.variable
            # For object-typed variables, treat the name as the object identifier.
            if isinstance(var.dtype, ObjectType):
                return Value(var.dtype, var.name)
            # For value-typed variables, fall back to the raw name/string.
            return Value(var.dtype, var.name)

        if isinstance(expr, ObjectConstantExpression):
            constant = expr.constant
            return Value(constant.dtype, constant.name)

        if isinstance(expr, ConstantExpression):
            constant = expr.constant
            if isinstance(constant, Value):
                return constant

            # Derived expression constants may come through as TensorValue/ValueBase instead of Value.
            constant_value = getattr(constant, "value", None)
            if constant_value is None and hasattr(constant, "item"):
                try:
                    constant_value = constant.item()
                except Exception:
                    constant_value = constant
            dtype = getattr(constant, "dtype", expr.return_type)
            return Value(dtype, constant_value)

        return super()._execute(expr)


executor = TextCraftExecutor(DOMAIN)


def get_pre_condition(action_param, induced_skills=None):
    induced_skills = induced_skills or []
    action_type = action_param.get("type")
    if action_type == "inventory":
        return lambda d: d.f_can_inventory(), "Inventory can be issued at any time."
    if action_type == "get":
        target_count = _parse_count(action_param.get("count", 1))
        return (
            lambda d: d.f_can_get(action_param["item"], target_count),
            f"Item {action_param['item']} should not be marked unavailable.",
        )
    if action_type == "craft":
        target_count = _parse_count(action_param.get("count", 1))
        return (
            lambda d: d.f_can_craft(action_param["item"], target_count),
            f"Ensure {action_param['item']} is needed and not unavailable before crafting.",
        )

    for (sub_goal_i, skill_i, mermaid_i, params_i) in induced_skills:
        if skill_i["name"] == action_type:
            return skill_i["pre_condition"], skill_i["pre_condition_msg"]

    raise ValueError(f"Pre-condition for action {action_param} is not defined.")


def _parse_action_line(line: str) -> Dict[str, Union[str, int, List[Dict[str, Union[str, int]]]]]:
    """Convert a free-form action string into an action_param dict using ACTION_PATTERNS."""
    line = line.strip()
    for name, pattern, builder in ACTION_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        action_param: Dict[str, Union[str, int, List[Dict[str, Union[str, int]]]]] = {"type": name}
        action_param.update(builder(match))
        return action_param
    raise ValueError(f"Unsupported action string: {line}")


def main():
    """Lightweight manual test that exercises the derived predicates and pre-condition helpers."""
    demo_predicates = [
        "inventory(dark_oak_log, 0)",
        "inventory(plank, 1)",
        "inventory(stick, 3)",
        "goal(plank, 2)",
        "goal(torch, 1)",
        "unavailable(diamond_pickaxe)",
    ]
    demo_world = World(predicates=demo_predicates)
    print("Demo predicates:")
    for predicate in demo_predicates:
        print(f"  - {predicate}")

    predicate_checks = [
        ("needs_item(plank, 2)", DOMAIN.f_needs_item("plank", 2)),
        ("has_enough(plank, 1)", DOMAIN.f_has_enough("plank", 1)),
        ("can_get(plank, 2)", DOMAIN.f_can_get("plank", 2)),
        ("can_get(diamond_pickaxe, 1)", DOMAIN.f_can_get("diamond_pickaxe", 1)),
        ("can_craft(torch, 1)", DOMAIN.f_can_craft("torch", 1)),
        (
            'inventory("dark_oak_log", "0") and numerical_greater_equal("0", "2")',
            BoolExpression(
                BoolOpType.AND,
                [
                    DOMAIN.f_inventory("dark_oak_log", "0"),
                    DOMAIN.f_numerical_greater_equal("0", "2"),
                ],
            ),
        ),
    ]
    print("\nDerived predicate evaluation:")
    for label, expr in predicate_checks:
        result = executor.execute(expr, demo_world).value
        print(f"  {label}: {result}")

    count_expr = DOMAIN.f_current_count_of_item("plank")
    count_value = executor.execute(count_expr, demo_world).value
    print("\nNon-boolean function evaluation:")
    print(f"  current_count_of_item(plank): {count_value}")

    action_lines = [
        "inventory",
        "get 1 plank",
        "craft 1 torch using 1 plank, 1 stick",
        "get 1 diamond_pickaxe",
    ]
    print("\nAction pre-condition evaluation:")
    for action_line in action_lines:
        action_param = _parse_action_line(action_line)
        pre_fn, message = get_pre_condition(action_param)
        value = executor.execute(pre_fn(DOMAIN), demo_world).value
        print(f"  {action_line!r}: {value} -> {message}")

    # ------------------------------------------------------------------
    # Minimal runnable test for: greater_than(count(k1), count(k2)) and is_holding(k1)
    # ------------------------------------------------------------------
    test_world = World(predicates=["inventory(k1, 3)", "inventory(k2, 1)"])
    test_expr = BoolExpression(
        BoolOpType.AND,
        [
            DOMAIN.f_greater_than(DOMAIN.f_count("k1"), DOMAIN.f_count("k2")),
            DOMAIN.f_is_holding("k1"),
        ],
    )
    test_result = executor.execute(test_expr, test_world).value
    print("\nTest: greater_than(count(k1), count(k2)) and is_holding(k1)")
    print(
        f"  k1={test_world.inventory.get('k1', 0)}, "
        f"k2={test_world.inventory.get('k2', 0)} -> {test_result}"
    )

    parser = FOLPythonParser(DOMAIN, inplace_definition=True)
    expr = parser.parse_expression(
        'greater_than(count(k1), count(k2)) and is_holding(k1)',
        arguments=[Variable('k1', T_ITEM), Variable('k2', T_ITEM)]
    )

    # 2) 准备一个小世界
    world = World(predicates=['inventory(k1, 3)', 'inventory(k2, 1)'])

    # 3) 执行表达式
    result = executor.execute(expr, world).value

    expr_calc = 'count(k1) > count(k2)'
    expr = parser.parse_expression(
        expr_calc,
        arguments=[Variable('k1', T_ITEM), Variable('k2', T_ITEM)]
    )
    result = executor.execute(expr, world).value
    print(f"  {expr_calc} -> {result}")

    expr_calc = 'count(k1) < count(k2)'
    expr = parser.parse_expression(
        expr_calc,
        arguments=[Variable('k1', T_ITEM), Variable('k2', T_ITEM)]
    )
    result = executor.execute(expr, world).value
    print(f"  {expr_calc} -> {result}")



if __name__ == "__main__":
    main()

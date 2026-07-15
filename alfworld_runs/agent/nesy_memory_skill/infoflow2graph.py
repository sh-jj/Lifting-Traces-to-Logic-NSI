# -*- coding: utf-8 -*-
"""
Mermaid (with info-carrying edges) → traversable graph + static validation + condensation DAG.

Key features:
- Parse nodes (incl. local in/out signatures) and edges with payloads like  A -->|{x:a, y:"b"}| B
- Validate: for every edge A→B, payload keys == B.local_in keys (GLOBAL is implicit)
- Traverse with foreach loop semantics; on each transition, apply edge payload to ctx
- Build condensation DAG (SCC collapsed) for static topological inspection; keep original edges & payloads

Assumptions:
- GLOBAL is implicit (not carried on edges). Edges only carry target node's local in.
- Node labels may contain:
    local in: {k1, k2}
    out: {k3, k4}
- Loop header format (case-insensitive):
    "For <VAR> in <LIST>"
- Edge mid labels (branching): "Yes", "No", "body", "done" (case-insensitive normalized)

Author: you :)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
from collections import OrderedDict, deque, Counter
import re
import ast
import html
import inspect
import copy

class Action_Unmatched_Exception(Exception):
    """Raised when a replayed action does not match the recorded trajectory."""


# try:
#     from .flow2graph import convert_all_items
# except ImportError:  # pragma: no cover - allow running as script
#     from flow2graph import convert_all_items  # type: ignore

def convert_all_items(text: str) -> str:
    """
    Replace every occurrence of 'letters-space-digits' with 'letters_digits'
    in the given text.  All other characters (punctuation, extra spaces, line
    breaks, etc.) are left untouched. E.g. "fridge 1" -> "fridge_1"
    """
    return re.sub(r'\b([A-Za-z]+)\s+(\d+)\b', r'\1_\2', text)

def inverse_all_items(text: str) -> str:
    return re.sub(r'\b([A-Za-z]+)_(\d+)\b', r'\1 \2', text)

BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
}

# ===================== Data structures =====================

@dataclass
class Node:
    id: str
    kind: str                 # 'start'|'terminal'|'loop'|'check'|'action'|'interface'|'unknown'
    raw_label: str = ""
    local_in: List[str] = field(default_factory=list)
    local_out: List[str] = field(default_factory=list)
    local_in_types: Dict[str, Optional[str]] = field(default_factory=dict)
    local_out_types: Dict[str, Optional[str]] = field(default_factory=dict)
    # Optional local-input binding expressions (e.g., rec: Type = CURRENT_RECEPTACLE)
    local_in_exprs: Dict[str, Optional[str]] = field(default_factory=dict)
    loop_inputs: List[str] = field(default_factory=list)
    loop_input_types: Dict[str, Optional[str]] = field(default_factory=dict)
    # For check/action
    expr: Optional[str] = None
    action: Optional[str] = None
    args: Optional[str] = None
    # For loop
    loop_var: Optional[str] = None
    loop_list: Optional[str] = None
    loop_list_expr: Optional[str] = None
    loop_iter_is_range: bool = False
    loop_range_arg_exprs: List[str] = field(default_factory=list)
    global_writes: List[str] = field(default_factory=list)
    global_write_types: Dict[str, Optional[str]] = field(default_factory=dict)
    global_write_exprs: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class Edge:
    src: str
    dst: str
    mid_label: Optional[str] = None   # 'Yes'|'No'|'body'|'done'|None
    payload_raw: Optional[str] = None # e.g. {receptacle: receptacle_i}
    payload_keys: Tuple[str, ...] = field(default_factory=tuple)
    mid_labels: Tuple[str, ...] = field(default_factory=tuple)  # all labels parsed from the edge (comma-separated)
    loop_directive: Optional[str] = None  # 'Start_Loop'|'Continue_Loop'|None

@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]
    global_vars: Dict[str, Optional[object]] = field(default_factory=dict)
    global_types: Dict[str, Optional[str]] = field(default_factory=dict)
    _adj: Dict[str, List[Edge]] = field(default_factory=dict)

    def __post_init__(self):
        for e in self.edges:
            self._adj.setdefault(e.src, []).append(e)

    def outgoing(self, nid: str) -> List[Edge]:
        return self._adj.get(nid, [])

# ===================== Mermaid parser (with info) =====================

NODE_PATTERNS = [
    re.compile(r'^\s*([A-Za-z0-9_]+)\s*\{\{(.+)\}\}', flags=re.DOTALL),        # {{ ... }}
    re.compile(r'^\s*([A-Za-z0-9_]+)\s*\["(.+?)"\]', flags=re.DOTALL),          # [" ... "]
    re.compile(r'^\s*([A-Za-z0-9_]+)\s*\[(.+?)\]', flags=re.DOTALL),            # [ ... ]
    re.compile(r'^\s*([A-Za-z0-9_]+)\s*\((.+?)\)', flags=re.DOTALL),            # ( ... )
]

EDGE_RE = re.compile(
    r'^\s*'                                  # start
    r'([A-Za-z0-9_]+)\s*'                    # src
    r'(?:--\s*([^>|-]+?)\s*)?'               # optional mid label after --
    r'-->\s*'                                # -->
    r'(?:\|\s*([^|]+?)\s*\|\s*)?'            # optional |payload|
    r'([A-Za-z0-9_]+)\s*$'                   # dst
)

CLASS_DEF_RE = re.compile(r'^\s*class\s+([A-Za-z0-9_,\s]+)\s+([A-Za-z0-9_]+)\s*$')

def _strip_comment(line: str) -> str:
    i = line.find("%%")
    return line if i < 0 else line[:i]

def _find_label(rest: str) -> str:
    """Find bracketed label content."""
    pairs = [("{{","}}"), ("["," ]"), ("["," ]"), ("("," )")]
    # We'll just search common forms robustly:
    for opener, closer in [("{{","}}"), ("["," ]"), ("[","]"), ("("," )"), ("(",")")]:
        if opener in rest and closer in rest:
            start = rest.find(opener)
            end = rest.rfind(closer)
            if end > start:
                return rest[start+len(opener):end].strip()
    return ""

def _normalize_mid_label(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    t = s.strip().lower()
    if t.startswith("yes"):  return "Yes"
    if t.startswith("no"):   return "No"
    if t.startswith("body"): return "body"
    if t.startswith("done"): return "done"
    return s.strip()

def _normalize_loop_directive(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip().lower().replace(" ", "_")
    if t in {"start_loop", "startloop"}:
        return "Start_Loop"
    if t in {"continue_loop", "continueloop", "continue"}:
        return "Continue_Loop"
    return None

_BRANCH_LABELS = {"Yes", "No", "body", "done"}

def _looks_like_payload_token(tok: str) -> bool:
    stripped = tok.strip()
    if not stripped:
        return False
    for sep in (":=", "="):
        if sep in stripped:
            return True
    if ":" in stripped:
        return True
    if "{" in stripped or "}" in stripped:
        return True
    return False

def _is_label_token(tok: str) -> bool:
    stripped = tok.strip()
    if not stripped:
        return False
    if _looks_like_payload_token(stripped):
        return False
    if _normalize_mid_label(stripped) in _BRANCH_LABELS:
        return True
    if _normalize_loop_directive(stripped):
        return True
    return False

def _normalize_edge_labels(tokens: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    From a list of label tokens (comma-separated), return:
      - branch label (Yes/No/body/done) if present
      - loop directive (Start_Loop/Continue_Loop) if present
      - normalized label list preserving order
    """
    branch: Optional[str] = None
    loop_dir: Optional[str] = None
    normalized: List[str] = []
    for tok in tokens:
        stripped = tok.strip()
        loop_norm = _normalize_loop_directive(stripped)
        branch_norm = _normalize_mid_label(stripped)
        if loop_norm:
            normalized.append(loop_norm)
            if loop_dir is None:
                loop_dir = loop_norm
            continue
        if branch_norm in _BRANCH_LABELS:
            normalized.append(branch_norm)
            if branch is None:
                branch = branch_norm
            continue
        if stripped:
            normalized.append(stripped)
    return branch, loop_dir, normalized

def _clean_label(label: str) -> str:
    if label.startswith('/"') and label.endswith('"/'):
        return label[2:-2]
    if label.startswith('"') and label.endswith('"'):
        return label[1:-1]
    if label.startswith("'") and label.endswith("'"):
        return label[1:-1]
    return label

def _extract_local_sig(raw: str, key: str) -> List[str]:
    names, _ = _extract_local_sig_with_types(raw, key)
    return names


def _extract_local_sig_with_types(raw: str, key: str) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    Extract local signature like: 'local in: {a, b}' or 'out: (x, y)'
    """
    m = re.search(
        rf'{key}\s*:\s*[\(\[\{{]([^)\]\}}]*)[\)\]\}}]',
        raw,
        flags=re.IGNORECASE
    )
    if not m:
        return [], {}
    items = [html.unescape(s.strip()) for s in m.group(1).split(",") if s.strip()]
    names: List[str] = []
    types: Dict[str, Optional[str]] = {}
    for it in items:
        if ":" in it:
            name_part, type_part = it.split(":", 1)
            name = name_part.strip()
            type_str = type_part.strip() or None
            if name:
                names.append(name)
                types[name] = type_str
            continue
        cleaned = it.strip()
        if cleaned:
            names.append(cleaned)
    return names, types


def _extract_local_sig_bindings_with_types(raw: str, key: str) -> Tuple[List[str], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Extended signature extractor that also captures inline value bindings, e.g.:
      local in: (rec: ReceptacleName = CURRENT_RECEPTACLE, targetType: ObjectTypeName = TARGET_TYPE)

    Returns: (names, types, value_exprs)
    """
    pattern = re.compile(rf'\b{re.escape(key)}\b\s*:', flags=re.IGNORECASE)
    m = pattern.search(raw)
    if not m:
        return [], {}, {}
    rest = raw[m.end():].lstrip()
    if not rest:
        return [], {}, {}
    opener = rest[0]
    closer = BRACKET_PAIRS.get(opener)
    if not closer:
        return [], {}, {}
    depth = 0
    buf: List[str] = []
    in_str = False
    str_q = ''
    for ch in rest:
        buf.append(ch)
        if in_str:
            if ch == str_q:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_q = ch
            continue
        if ch == opener:
            depth += 1
            continue
        if ch == closer:
            depth -= 1
            if depth == 0:
                break
    # remove outer brackets
    body = ''.join(buf)[1:-1].strip()
    if not body:
        return [], {}, {}

    names: List[str] = []
    types: Dict[str, Optional[str]] = {}
    values: Dict[str, Optional[str]] = {}
    for part in _split_top_level_items(body):
        name, type_str, value_expr = _split_name_value(part)
        if not name:
            continue
        names.append(name)
        types[name] = type_str
        if value_expr is not None:
            values[name] = value_expr
    return names, types, values


def _extract_identifier_list(raw: str, key: str) -> List[str]:
    return list(_extract_named_values(raw, key).keys())


def _split_name_value(token: str) -> Tuple[str, Optional[str], Optional[str]]:
    token = html.unescape(token.strip())
    if not token:
        return "", None, None
    value_part: Optional[str] = None
    if ":=" in token:
        name_part, value_part = token.split(":=", 1)
    elif "=" in token:
        name_part, value_part = token.split("=", 1)
    else:
        name_part = token
    type_part: Optional[str] = None
    if ":" in name_part:
        name_part, type_part = name_part.split(":", 1)
    name = name_part.strip()
    type_str = html.unescape(type_part.strip()) if type_part is not None and type_part.strip() else None
    value = value_part.strip() if value_part is not None else None
    return name, type_str, value or None


def _extract_named_values(raw: str, key: str) -> Dict[str, Optional[str]]:
    values, _ = _extract_named_values_with_types(raw, key)
    return values


def _extract_named_values_with_types(raw: str, key: str) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    
    # print('_extract_named_values_with_types: ', raw, key)

    pattern = re.compile(rf'\b{re.escape(key)}\b\s*:', flags=re.IGNORECASE)
    m = pattern.search(raw)
    if not m:
        return {}, {}
    rest = raw[m.end():]
    rest = rest.strip()
    if rest:
        # Allow optional HTML line breaks immediately after the colon, e.g. "writes GLOBAL:<br>(...)".
        rest = re.sub(r'^(?:<br\s*/?>\s*)+', '', rest, flags=re.IGNORECASE).lstrip()
    if not rest:
        return {}, {}

    content = ""
    if rest and rest[0] in BRACKET_PAIRS:
        opener = rest[0]
        closer = BRACKET_PAIRS[opener]
        depth = 0
        collected: List[str] = []
        for idx, ch in enumerate(rest):
            collected.append(ch)
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    content = "".join(collected[1:-1])
                    break
        if depth != 0:
            content = "".join(collected[1:])
    else:
        content = re.split(r'<br\s*/?>', rest, maxsplit=1, flags=re.IGNORECASE)[0]

    content = html.unescape(content.strip())
    if content.startswith("{{") and content.endswith("}}"):
        content = content[1:-1].strip()
    if content.startswith("{") and content.endswith("}"):
        content = content[1:-1].strip()
    if not content:
        return {}, {}

    content = re.sub(r'<br\s*/?>', ',', content, flags=re.IGNORECASE)
    content = content.replace("\n", ",")

    result: Dict[str, Optional[str]] = {}
    types: Dict[str, Optional[str]] = {}
    for part in _split_top_level_items(content):
        name, type_str, value = _split_name_value(part)
        if name:
            result[name] = value
            types[name] = type_str
    return result, types


def _normalize_param_name(raw: str, type_hint: Optional[str] = None) -> str:
    name = (raw or "").strip()
    # Drop trailing _INPUT (case-insensitive)
    name = re.sub(r"_?input$", "", name, flags=re.IGNORECASE)
    # Normalize to snake_case (already underscored; just lowercase)
    name = name.replace(" ", "_").replace("-", "_")
    name = re.sub(r"__+", "_", name)
    name = name.strip("_").lower()
    # Heuristic rename for target_type → target_object_type when type hint suggests an object type
    if name == "target_type" and (type_hint or "").lower() in ("itemtypename", "objecttypename"):
        return "target_object_type"
    return name


def _normalize_type_name(type_raw: Optional[str]) -> str:
    t = (type_raw or "").strip()
    tl = t.lower()
    if not t:
        return "str"
    if tl == "int":
        return "int"
    if tl == "bool" or t == "Bool":
        return "bool"
    if tl.startswith("optional_"):
        # Treat optional T as base T for input type purposes
        base = t.split("_", 1)[1] if "_" in t else t
        return _normalize_type_name(base)
    if tl.startswith("list") or "list_" in tl:
        return "list of str"
    # Common symbolics treated as strings
    return "str"


def extract_start_inputs(mermaid_code: str) -> Dict[str, str]:
    """Parse the START node's Interface Inputs from a Mermaid workflow.

    Returns a dict mapping the exact input names to their exact type strings
    as they appear in the START node (no renaming or type normalization).

    Robust to typical label forms like:
      START(["Interface: <br> Inputs: <br>FOO_INPUT: List_ReceptacleName<br>BAR_INPUT: ItemTypeName"]):::Interface
    """
    if not isinstance(mermaid_code, str) or not mermaid_code.strip():
        return {}
    start_label: Optional[str] = None
    for line in mermaid_code.splitlines():
        line_stripped = _strip_comment(line).strip()
        if not line_stripped.startswith("START"):
            continue
        # Find the quoted label content
        q1 = line_stripped.find('"')
        q2 = line_stripped.rfind('"')
        if q1 != -1 and q2 != -1 and q2 > q1:
            start_label = line_stripped[q1 + 1:q2]
            break
    if not start_label:
        return {}
    # Split by <br> and cleanup
    parts = re.split(r"<br\s*/?>", start_label, flags=re.IGNORECASE)
    parts = [html.unescape(p).strip() for p in parts if p and p.strip()]
    # Find the index of the "Inputs:" marker
    idx = -1
    for i, p in enumerate(parts):
        if p.lower().startswith("inputs:"):
            idx = i
            break
    if idx == -1:
        # If no explicit Inputs marker, try to parse key:type pairs in whole label
        idx = 0
    entries = parts[idx + 1:]
    params: Dict[str, str] = OrderedDict()
    for ent in entries:
        if ":" not in ent:
            continue
        # NAME: TYPE (ignore defaults here)
        name_part, type_part = ent.split(":", 1)
        raw_name = html.unescape(name_part.strip())
        raw_type = html.unescape(type_part.strip())
        if raw_name:
            params[raw_name] = raw_type
    return params


def _parse_default_literal(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    v = html.unescape(value.strip())
    
    # print(value, v)
    
    if not v:
        return None
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _register_global(
    global_vars: Dict[str, Optional[object]],
    global_types: Dict[str, Optional[str]],
    name: str,
    default_str: Optional[str],
    type_str: Optional[str] = None
) -> None:
    if not name:
        return
    default_value = _parse_default_literal(default_str)
    if name not in global_vars:
        global_vars[name] = default_value
    elif default_str is not None and global_vars[name] is None:
        global_vars[name] = default_value
    if name not in global_types:
        global_types[name] = type_str
    elif type_str and not global_types[name]:
        global_types[name] = type_str


def normalize_object_ids(action: str) -> str:
    """
    "take apple_1 from fridge_1" -> "take apple 1 from fridge 1"
    """
    pattern = re.compile(r'\b([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(\d+)\b')

    def _replace(match: re.Match) -> str:
        name = match.group(1).replace('_', ' ')
        return f"{name} {match.group(2)}"

    return pattern.sub(_replace, action)

def _first_line(raw: str) -> str:
    head = re.split(r'<br\s*/?>', raw, maxsplit=1, flags=re.IGNORECASE)[0]
    return html.unescape(head.strip())


def _extract_check_expr(raw: str) -> Optional[str]:
    """Extract the natural-language or DSL payload after 'check:'.

    Supports forms like:
      - "check: expr"
      - "Check: <br> expr"
      - with or without quotes around expr.
    """
    # 1) Most common: allow an optional <br> right after the colon, then capture until next <br> or end.
    m = re.search(
        r'check\s*:\s*(?:<br\s*/?>\s*)?["\']?(.+?)["\']?\s*(?:<br|$)',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # 2) Fallback: use the first line and try to peel after ':' or find a trailing lambda.
    head = _first_line(raw)
    if not head:
        return None
    lambda_match = re.search(r'(lambda\b.+)$', head, flags=re.IGNORECASE)
    if lambda_match:
        return lambda_match.group(1).strip()
    if ":" in head:
        after = head.split(":", 1)[1].strip()
        if after:
            return after
    return head or None


def _extract_action_and_args(raw: str) -> Tuple[Optional[str], Optional[str]]:
    # Prefer a simple string form as the full action string.
    # e.g. PrimitiveOp<br/>(action: goto_receptacle: 'go to {RECEPTACLE}')
    m = re.search(r'action\s*:\s*([A-Za-z0-9_]+)\s*:\s*[\'"]([^\'"]*)[\'"]', raw, flags=re.IGNORECASE)
    if m:
        full = m.group(2).strip()
        return "", full if full else None
    # action: 'go to {receptacle}' (treat quoted content as the full action string)
    m = re.search(r'action\s*:\s*[\'"]([^\'"]*)[\'"]', raw, flags=re.IGNORECASE)
    if m:
        full = m.group(1).strip()
        return "", full if full else None
    # action: goto (rare)
    m = re.search(r'action\s*:\s*([A-Za-z0-9_]+)\b', raw, flags=re.IGNORECASE)
    if m:
        return m.group(1), None
    head = _first_line(raw)
    m = re.match(r'([A-Za-z0-9_]+)\s*\((.*?)\)\s*$', head)
    if m:
        act = m.group(1)
        args = m.group(2).strip()
        return act, args or None
    if head:
        return head, None
    return None, None

_BRACED_VAR_FULL = re.compile(r'^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$')
_BRACED_VAR_ANY = re.compile(r'\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}')
_SINGLE_BRACED_VAR_FULL = re.compile(r'^\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}$')
_SINGLE_BRACED_VAR_ANY = re.compile(r'\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}')
_SINGLE_BRACED_VAR_STRICT = re.compile(r'(?<!\{)\{(?!\{)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})')
_GLOBAL_REF_PATTERNS = [
    re.compile(r'\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}'),
    re.compile(r'\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}'),
]


def _strip_braces(identifier: str) -> str:
    ident = identifier.strip()
    m = _BRACED_VAR_FULL.match(ident)
    if m:
        return m.group(1)
    m = _SINGLE_BRACED_VAR_FULL.match(ident)
    if m:
        return m.group(1)
    return ident.strip("{} ")


def _candidate_variants(name: str) -> Set[str]:
    base = name.strip()
    if not base:
        return set()
    variants: Set[str] = {base}
    base_us = base.replace(" ", "_")
    variants.update({base_us})
    variants.update({base.upper(), base.lower(), base_us.upper(), base_us.lower()})
    return variants


def _normalize_type_name(type_name: Optional[str]) -> Optional[str]:
    if not type_name:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9]', '', type_name)
    cleaned = cleaned.strip().lower()
    return cleaned or None


def _extract_loop_header(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.search(
        r'for(?:\s+each)?\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^\n<]+)',
        raw,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None, None
    loop_var = m.group(1)
    loop_list_raw = m.group(2).strip()
    loop_list_raw = re.split(r'<br\s*/?>', loop_list_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    loop_list = _strip_braces(loop_list_raw)
    return loop_var, loop_list, loop_list_raw

_RANGE_CALL_PATTERN = re.compile(r'^range\s*\((.*)\)\s*$', flags=re.IGNORECASE)

def _split_mid_payload(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given payload text between pipes, split out optional branch label.
    Example: "Yes: x=y" -> ("Yes", "x=y"); "No" -> ("No", None); "a: b" (no branch label) -> (None, "a: b")
    """
    if text is None:
        return None, None
    s = text.strip()
    if not s:
        return None, None
    if ":" in s:
        prefix, rest = s.split(":", 1)
        norm = _normalize_mid_label(prefix)
        if norm in {"Yes", "No", "body", "done"}:
            # Treat prefix as branch label; rest is payload (may be empty)
            payload = rest.strip() or None
            return norm, payload
        # Otherwise colon belongs to payload (e.g., key:value)
    norm = _normalize_mid_label(s)
    if norm in {"Yes", "No", "body", "done"}:
        return norm, None
    return None, s


def _split_top_level_items(body: str) -> List[str]:
    items: List[str] = []
    depth = 0
    token = ""
    in_str = False
    str_q = ""
    for ch in body:
        if in_str:
            token += ch
            if ch == str_q:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_q = ch
            token += ch
            continue
        if ch in "({[":
            depth += 1
            token += ch
            continue
        if ch in ")}]":
            depth = max(0, depth - 1)
            token += ch
            continue
        if ch in (",", "，") and depth == 0:
            if token.strip():
                items.append(token.strip())
            token = ""
            continue
        token += ch
    if token.strip():
        items.append(token.strip())
    return items


def _parse_payload_assignments(payload_raw: Optional[str]) -> List[Tuple[str, str]]:
    if not payload_raw:
        return []
    body = payload_raw.strip()
    if not body:
        return []
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip()
    items = _split_top_level_items(body)
    assignments: List[Tuple[str, str]] = []
    for item in items:
        if not item:
            continue
        # Prefer :=, then =, then :
        for sep in (":=", "=", ":"):
            if sep in item:
                left, right = item.split(sep, 1)
                key = left.strip()
                value = right.strip()
                if key:
                    assignments.append((key, value))
                break
        else:
            tok = item.strip()
            if tok:
                assignments.append((tok, tok))
    return assignments


def _parse_payload_keys(payload_raw: Optional[str]) -> Tuple[str, ...]:
    if not payload_raw:
        return tuple()
    return tuple(key for key, _ in _parse_payload_assignments(payload_raw))

def parse_mermaid_with_info(src: str) -> Graph:
    lines = [ln.rstrip() for ln in src.splitlines()]
    nodes: Dict[str, Node] = {}
    classes: Dict[str, str] = {}
    global_vars: Dict[str, Optional[object]] = {}
    global_types: Dict[str, Optional[str]] = {}

    # Pass 1: collect node ids + raw labels; collect inline :::Class if present
    for raw in lines:
        line = _strip_comment(raw).strip()
        if not line or line.startswith("flowchart") or line.startswith("classDef"):
            continue
        # class START,SUCCESS_END Interface
        mclass = CLASS_DEF_RE.match(line)
        if mclass:
            ids, cname = mclass.groups()
            for nid in [s.strip() for s in ids.split(",") if s.strip()]:
                classes[nid] = cname
            continue
        # node?
        for pat in NODE_PATTERNS:
            m = pat.match(line)
            if m:
                nid, label = m.group(1), _clean_label(m.group(2))
                # inline class :::Name
                inline_cls = None
                if ":::":  # safe
                    parts = line.split(":::")
                    if len(parts) >= 2:
                        inline_cls = parts[-1].strip()
                cls = inline_cls or classes.get(nid)
                # classify
                kind = "unknown"
                if cls == "Check":
                    kind = "check"
                elif cls == "PrimitiveAction":
                    kind = "action"
                elif cls == "LoopControl":
                    kind = "loop"
                elif cls == "DataOp":
                    kind = "dataop"
                elif cls == "Interface":
                    if nid == "START":
                        kind = "start"
                    elif nid.endswith("_END"):
                        kind = "terminal"
                    else:
                        kind = "interface"
                else:
                    if nid == "START": kind = "start"
                    elif nid.endswith("_END"): kind = "terminal"
                nodes[nid] = Node(id=nid, kind=kind, raw_label=label)
                break

    # Pass 1.5: fill node fields (local in/out, expr/action, loop header)
    for n in nodes.values():
        in_names, in_types, in_values = _extract_local_sig_bindings_with_types(n.raw_label, "local in")
        out_names, out_types = _extract_local_sig_with_types(n.raw_label, "out")
        n.local_in = in_names
        n.local_in_types = in_types
        n.local_in_exprs = in_values
        if n.kind == "loop":
            n.loop_inputs = list(in_names)
            n.loop_input_types = dict(in_types)
        n.local_out = out_names
        n.local_out_types = out_types
        if n.kind == "check":
            n.expr = _extract_check_expr(n.raw_label)
        if n.kind == "action":
            act, args = _extract_action_and_args(n.raw_label)
            n.action, n.args = act, args
        if n.kind == "loop":
            n.loop_var, n.loop_list, n.loop_list_expr = _extract_loop_header(n.raw_label)
            if n.loop_list_expr:
                range_match = _RANGE_CALL_PATTERN.match(n.loop_list_expr.strip())
                if range_match:
                    n.loop_iter_is_range = True
                    arg_body = range_match.group(1).strip()
                    if arg_body:
                        n.loop_range_arg_exprs = [part.strip() for part in _split_top_level_items(arg_body)]
                    else:
                        n.loop_range_arg_exprs = []
            # print("node praser: ", n.id, n.loop_var, n.loop_list, n.loop_list_expr)
            # print(n.local_in, n.local_in_types, n.local_in_exprs)
        global_write_map_raw, global_write_types_raw = _extract_named_values_with_types(n.raw_label, "writes global")
        global_write_map: Dict[str, Optional[str]] = {}
        global_write_types: Dict[str, Optional[str]] = {}
        for raw_name, expr in global_write_map_raw.items():
            clean_name = _strip_braces(raw_name)
            global_write_map[clean_name] = expr
            global_write_types[clean_name] = global_write_types_raw.get(raw_name)
        
        # print(n.id, n.kind)
        # print('global_write_map:', global_write_map, global_write_types)

        n.global_writes = list(global_write_map.keys())
        n.global_write_types = global_write_types
        n.global_write_exprs = global_write_map
        for name in n.global_writes:
            _register_global(
                global_vars,
                global_types,
                name,
                global_write_map.get(name),
                global_write_types.get(name)
            )
        # print('global_vars:', global_vars, global_types)
        # print('global_write_exprs:', n.global_write_exprs)

        if n.id.upper() == "FLOW_SPEC":
            inputs_raw, input_types_raw = _extract_named_values_with_types(n.raw_label, "inputs")
            outputs_raw, output_types_raw = _extract_named_values_with_types(n.raw_label, "outputs")
            for name, value in inputs_raw.items():
                clean_name = _strip_braces(name)
                _register_global(global_vars, global_types, clean_name, value, input_types_raw.get(name))
            for name, value in outputs_raw.items():
                clean_name = _strip_braces(name)
                _register_global(global_vars, global_types, clean_name, value, output_types_raw.get(name))

    # exit(0)

    # Pass 2: edges with payload
    edges: List[Edge] = []
    for raw in lines:
        line = _strip_comment(raw).strip()
        if not line or "-->" not in line:
            continue
        m = EDGE_RE.match(line)
        if not m:
            continue
        src, mid, payload_text, dst = m.group(1), m.group(2), m.group(3), m.group(4)

        payload_tokens = _split_top_level_items(payload_text) if payload_text else []
        mid_tokens = _split_top_level_items(mid) if mid else []

        label_tokens: List[str] = []
        branch_label: Optional[str] = None
        loop_directive: Optional[str] = None

        label_only_payload = bool(payload_tokens) and all(_is_label_token(tok) for tok in payload_tokens)

        payload_clean: Optional[str] = None
        if payload_text is not None:
            if label_only_payload:
                label_tokens.extend(payload_tokens)
                payload_clean = None
            else:
                pipe_branch, pipe_payload = _split_mid_payload(payload_text)
                if pipe_branch:
                    branch_label = pipe_branch
                    payload_clean = pipe_payload
                else:
                    payload_clean = payload_text.strip() if payload_text else None
                # even if not label-only, collect loop directives inside payload tokens when present
                for tok in payload_tokens:
                    if _is_label_token(tok):
                        label_tokens.append(tok)

        # include labels from the explicit mid segment (between -- and -->)
        if mid_tokens:
            label_tokens.extend(mid_tokens)

        branch_from_labels, loop_from_labels, normalized_labels = _normalize_edge_labels(label_tokens)
        if branch_label is None:
            branch_label = branch_from_labels
        if loop_directive is None:
            loop_directive = loop_from_labels

        edges.append(Edge(
            src=src.strip(),
            dst=dst.strip(),
            mid_label=branch_label,
            payload_raw=payload_clean,
            payload_keys=_parse_payload_keys(payload_clean),
            mid_labels=tuple(normalized_labels),
            loop_directive=loop_directive
        ))

    return Graph(nodes=nodes, edges=edges, global_vars=global_vars, global_types=global_types)

# ===================== Payload evaluation & validation =====================



def _resolve_placeholders(expr: object, ctx_lookup: Dict[str, object], verbose: bool = True) -> object:
    # Log the attempt so we know which expression and keys are available for substitution.

    if verbose:
        print(f"resolve placeholders..., expr: {expr}, ctx_lookup: {ctx_lookup.keys()}")

    # Non-string inputs are returned unchanged because there is nothing to substitute.
    if not isinstance(expr, str):
        return expr

    # Try to resolve the entire token if it is wrapped in {{ }}.
    stripped = expr.strip()
    full_match = _BRACED_VAR_FULL.match(stripped)
    if full_match:
        key = full_match.group(1)
        if key in ctx_lookup:
            return ctx_lookup[key]
        key_lower = key.lower()
        return ctx_lookup.get(key_lower, expr)

    # Handle single-brace placeholders like {var}.
    single_full = _SINGLE_BRACED_VAR_FULL.match(stripped)

    # print(f"single_full: {single_full}")
    if single_full:
        key = single_full.group(1)
        if key in ctx_lookup:
            return ctx_lookup[key]
        key_lower = key.lower()
        return ctx_lookup.get(key_lower, expr)

    def _fmt(value: object) -> str:
        # Render values using repr so that strings keep quotes and other python literals stay valid.
        if isinstance(value, str):
            return repr(value)
        return repr(value) if value is not None else "None"

    # Replace all {{var}} occurrences inside the string.
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        if key in ctx_lookup:
            return _fmt(ctx_lookup[key])
        key_lower = key.lower()
        if key_lower in ctx_lookup:
            return _fmt(ctx_lookup[key_lower])
        key_upper = key.upper()
        if key_upper in ctx_lookup:
            return _fmt(ctx_lookup[key_upper])
        return match.group(0)

    expr = _BRACED_VAR_ANY.sub(_repl, expr)

    # Replace single-brace placeholders, but keep the token name when no binding exists.
    def _single_repl(match: re.Match) -> str:
        key = match.group(1)
        if key in ctx_lookup:
            return _fmt(ctx_lookup[key])
        key_lower = key.lower()
        if key_lower in ctx_lookup:
            return _fmt(ctx_lookup[key_lower])
        key_upper = key.upper()
        if key_upper in ctx_lookup:
            return _fmt(ctx_lookup[key_upper])
        return key

    return _SINGLE_BRACED_VAR_ANY.sub(_single_repl, expr)


def _eval_value_expr(value_expr: Optional[str], ctx: dict, global_ctx: dict = None, verbose: bool = True) -> object:
    if value_expr is None:
        return None

    value_expr = value_expr.replace("{{", "{").replace("}}", "}")
    ctx_lookup: Dict[str, object] = {}
    if global_ctx:
        ctx_lookup.update(global_ctx)
    if ctx:
        ctx_lookup.update(ctx)

    resolved = _resolve_placeholders(value_expr, ctx_lookup, verbose=verbose)
    if not isinstance(resolved, str):
        return resolved

    v = html.unescape(resolved.strip())

    if verbose:
        print(f"resolved after placeholder replacement: {value_expr} -> {resolved}")
    if not v:
        return None
    single_full = _SINGLE_BRACED_VAR_FULL.match(v)
    
    if single_full:
        key = single_full.group(1)
        if key in ctx_lookup:
            return ctx_lookup[key]
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        if any(ch in v for ch in (".", "e", "E")):
            return float(v)
        return int(v)
    except ValueError:
        pass
    if v.startswith(("'", '"')) and v.endswith(("'", '"')):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v.strip("'\"")
    if v in ctx_lookup:
        return ctx_lookup[v]
    try:
        call_ast = ast.parse(v, mode="eval").body
    except SyntaxError:
        return v
    
    if verbose:
        print("value_expr: ", value_expr)
    # print(ctx[func_name])

    if isinstance(call_ast, ast.Call) and isinstance(call_ast.func, ast.Name):

        func_name = call_ast.func.id

        if func_name in {"select_one", "select_all"} and len(call_ast.args) >= 2:
            kind_src = ast.get_source_segment(v, call_ast.args[0]) or ast.unparse(call_ast.args[0])
            cond_src = ast.get_source_segment(v, call_ast.args[1]) or ast.unparse(call_ast.args[1])
            kind_val = _eval_value_expr(kind_src, ctx_lookup, global_ctx, verbose=verbose)
            cond_str = cond_src.strip()
            handler = selection_one if func_name == "select_one" else selection_all
            selection_ctx: Dict[str, object] = {}
            if global_ctx:
                selection_ctx.update(global_ctx)
            selection_ctx.update(ctx_lookup)

            if verbose:
                print(f"dive into the selection function, kind_val: {kind_val}, cond_str: {cond_str}")
            return handler(kind_val, cond_str, selection_ctx)
        
        try:
            domain_result = _try_eval_domain_function(v, global_ctx=global_ctx)
            if domain_result is not _DOMAIN_FUNC_MISSING:
                return domain_result
        except Exception as e:
            print(e)
            
        arg_values: List[object] = []
        for arg in call_ast.args:
            arg_src = ast.get_source_segment(v, arg)
            if arg_src is None:
                arg_src = ast.unparse(arg)
            arg_values.append(_eval_value_expr(arg_src, ctx_lookup, global_ctx, verbose=verbose))

        if func_name in ctx_lookup and callable(ctx_lookup[func_name]):
            return ctx_lookup[func_name](*arg_values)


        raise ValueError(f"Unknown function call: {func_name}({', '.join(map(repr, arg_values))})") from e
    return v


class _FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


_CONTAINS_RE = re.compile(r"^contains\(([^,]+),\s*([^\)]+)\)$")
_HOLDING_RE = re.compile(r"^holding\(([^,]+),\s*([^\)]+)\)$")
_LOCATE_RE = re.compile(r"^locate\(([^,]+),\s*([^\)]+)\)$")
_REACHABLE_RE = re.compile(r"^reachable\(([^,]+),\s*([^\)]+)\)$")


def _format_with_ctx(template: object, ctx: dict) -> object:
    if not isinstance(template, str):
        return template
    safe_ctx = _FormatDict({k: ctx[k] for k in ctx if isinstance(k, str)})
    try:
        return template.format_map(safe_ctx)
    except Exception:
        return template


def _get_current_facts(ctx: dict) -> List[str]:

    if "current_traj_node" in ctx:
        return ctx["current_traj_node"].facts_current
    if "current_facts" in ctx:
        return ctx["current_facts"]

    raise Exception("current_facts not found in ctx")


def _get_executor_world(ctx: dict):
    execu = ctx.get("executor")
    world_builder = ctx.get("world_builder")
    if execu is None or world_builder is None:
        raise RuntimeError("executor/world_builder missing from context")
    facts = _get_current_facts(ctx)
    current_world = world_builder(predicates=facts)
    return execu, current_world, facts


def _collect_candidates(kind: str, facts: List[str]) -> List[str]:
    kind_lower = (kind or "").lower()
    items: Set[str] = set()
    receptacles: Set[str] = set()
    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue
        m = _CONTAINS_RE.match(fact)
        if m:
            receptacles.add(m.group(1).strip())
            items.add(m.group(2).strip())
            continue
        m = _HOLDING_RE.match(fact)
        if m:
            items.add(m.group(2).strip())
            continue
        m = _LOCATE_RE.match(fact)
        if m:
            receptacles.add(m.group(2).strip())
            continue
        m = _REACHABLE_RE.match(fact)
        if m:
            receptacles.add(m.group(2).strip())
            continue
    if "item" in kind_lower:
        return sorted(items)
    if "receptacle" in kind_lower:
        return sorted(receptacles)
    return sorted(items.union(receptacles))


def _build_predicate_env(execu, current_world, facts: Optional[List[str]] = None):
    def _wrap(fn_name):
        def _inner(*args):
            with execu.with_grounding(current_world):
                return getattr(execu, fn_name)(*args)
        return _inner

    env = {
        "reachable": _wrap("reachable"),
        "locate": _wrap("locate"),
        "contains": _wrap("contains"),
        "holding": _wrap("holding"),
        "is_open": _wrap("is_open"),
        "is_closed": lambda r: (not _wrap("is_open")(r)),
        "is_cleaned": _wrap("is_cleaned") if hasattr(execu, "is_cleaned") else (lambda x: False),
        "is_cooled": _wrap("is_cooled") if hasattr(execu, "is_cooled") else (lambda x: False),
        "is_heated": _wrap("is_heated") if hasattr(execu, "is_heated") else (lambda x: False),
        "is_turned_on": _wrap("is_turned_on") if hasattr(execu, "is_turned_on") else (lambda x: False),
        "is_item_of_type": _wrap("is_item_of_type"),
        "is_receptacle_of_type": _wrap("is_receptacle_of_type") if hasattr(execu, "is_receptacle_of_type") else (lambda r, t: str(r).startswith(str(t) + "_")),
        "Item": "Item",
        "Receptacle": "Receptacle",
        "Agent": "Agent",
        "agent": "agent",
    }
    from .symbolic_world import (
        exists_item_in_list,
        forall_item_in_list,
        exists_receptacle_in_list,
        forall_receptacle_in_list,
    )
    env.update({
        "exists_item_in_list": exists_item_in_list,
        "forall_item_in_list": forall_item_in_list,
        "exists_receptacle_in_list": exists_receptacle_in_list,
        "forall_receptacle_in_list": forall_receptacle_in_list,
    })
    env.update({
        "can_goto": lambda agent, receptacle: env["reachable"](agent, receptacle) and (not env["locate"](agent, receptacle)),
        "can_open": lambda agent, receptacle: env["locate"](agent, receptacle) and (not env["is_open"](receptacle)),
        "can_take": lambda agent, receptacle, item: (
            env["locate"](agent, receptacle)
            and env["is_open"](receptacle)
            and env["contains"](receptacle, item)
            and (not env["holding"](agent, item))
        ),
        "can_put": lambda agent, receptacle, item: (
            env["locate"](agent, receptacle)
            and env["is_open"](receptacle)
            and env["holding"](agent, item)
        ),
    })
    fact_list = facts or []

    def _pool(dtype_name):
        return _collect_candidates(str(dtype_name), fact_list)

    def _exists(dtype_name, fn):
        for obj in _pool(dtype_name):
            try:
                if bool(fn(obj)):
                    return True
            except Exception:
                continue
        return False

    def _forall(dtype_name, fn):
        for obj in _pool(dtype_name):
            try:
                if not bool(fn(obj)):
                    return False
            except Exception:
                return False
        return True

    env["exists"] = _exists
    env["forall"] = _forall
    return env

_DOMAIN_FUNC_MISSING = None

def _try_eval_domain_function(func_str: str, global_ctx) -> object:
    
    try:
        execu = global_ctx.get("executor")
        facts = _get_current_facts(global_ctx)
        current_world = global_ctx["world_builder"](predicates=facts)
        env = _build_predicate_env(execu, current_world, facts)
        condition_value = bool(eval(func_str, env, env))
        return condition_value
    except Exception as e:
        print("eval domain function failed: ")
        print("function calling: ", func_str)
        print(e)
        return _DOMAIN_FUNC_MISSING
    


def _selection_search(ctx: dict, kind: str, cond_str: object, return_all: bool) -> object:
    execu, current_world, facts = _get_executor_world(ctx)
    env = _build_predicate_env(execu, current_world, facts)
    cond_template = _format_with_ctx(cond_str if cond_str is not None else "True", ctx)
    cond_code = convert_all_items(str(cond_template))
    cond_fn = eval("lambda x: (" + cond_code + ")", env, env)

    print('selection_search: ')
    print('cond_template: ', cond_template)
    print('cond_code: ', cond_code)
    print('cond_fn: ', cond_fn)

    pool = _collect_candidates(kind, facts)
    results: List[str] = []
    for obj in pool:
        try:
            if bool(cond_fn(obj)):
                if return_all:
                    results.append(obj)
                else:
                    return obj
        except Exception:
            continue
    return results if return_all else None

def _build_cond_env(execu: ALFWorldExecutor, world: World):
    def _reachable(receptacle):
        with execu.with_grounding(world):
            return execu.reachable(receptacle)

    def _locate(receptacle):
        with execu.with_grounding(world):
            return execu.locate(receptacle)

    def _contains(receptacle, item):
        with execu.with_grounding(world):
            return execu.contains(receptacle, item)

    def _holding(item):
        with execu.with_grounding(world):
            return execu.holding(item)

    def _is_open(receptacle):
        with execu.with_grounding(world):
            return execu.is_open(receptacle)

    def _is_closed(receptacle):
        with execu.with_grounding(world):
            return not execu.is_open(receptacle)

    def _is_item_of_type(item, type_name):
        return isinstance(type_name, str) and str(item).startswith(type_name + "_")

    def _is_receptacle_of_type(receptacle, type_name):
        return isinstance(type_name, str) and str(receptacle).startswith(type_name + "_")

    def _exists(dtype_name, fn):
        pool = []
        if isinstance(dtype_name, str):
            t = dtype_name.lower()
            if "item" in t:
                pool = list(_all_items(world))
            elif "receptacle" in t:
                pool = list(_all_receptacles(world))
            elif "agent" in t:
                pool = [world.agent_id]
        for obj in pool:
            try:
                if bool(fn(obj)):
                    return True
            except Exception:
                continue
        return False

    def _forall(dtype_name, fn):
        pool = []
        if isinstance(dtype_name, str):
            t = dtype_name.lower()
            if "item" in t:
                pool = list(_all_items(world))
            elif "receptacle" in t:
                pool = list(_all_receptacles(world))
            
        for obj in pool:
            try:
                if not bool(fn(obj)):
                    return False
            except Exception:
                return False
        return True

    return {
        # predicates
        "reachable": _reachable,
        "locate": _locate,
        "contains": _contains,
        "holding": _holding,
        "is_open": _is_open,
        "is_closed": _is_closed,
        "is_item_of_type": _is_item_of_type,
        "is_receptacle_of_type": _is_receptacle_of_type,
        # quantifiers
        "exists": _exists,
        "forall": _forall,
        # common constants
        "agent": "agent",
    }

def _all_receptacles(world):
    return list(world.all_receptacles)  

def _all_items(world):
    return list(world.all_items)

def selection_one(kind: str, cond_str: str, selection_ctx: dict) -> Optional[str]:
    selection_ctx = selection_ctx or {}
    execu = selection_ctx.get("executor")
    facts = _get_current_facts(selection_ctx)
    world = selection_ctx["world_builder"](predicates=facts)

    # print("selection iota: kind =", kind, ", cond_str =", cond_str)
    # print("current facts:", facts)
    env = _build_cond_env(execu, world)
    env.update({k: v for k, v in selection_ctx.items() if isinstance(k, str)})
    
    # Normalize object names like "garbagecan 1" -> "garbagecan_1" before evaluation.
    converted_cond_str = convert_all_items(cond_str)
   
    cond = eval("lambda x: (" + converted_cond_str + ")", env, env)
    
    # print(f"condition str: {'lambda x: (' + converted_cond_str + ')'}")

    kind_parsed = kind.strip("'").strip("\"").lower()
    if kind_parsed == 'item':
        pool = _all_items(world)
    elif kind_parsed == 'receptacle':
        pool = _all_receptacles(world)
    else:
        raise ValueError(f"Unknown kind: {kind} -> {kind_parsed}")
    pool = sorted(pool)
    # print(pool)
    for x in pool:
        try:
            # print(x)
            # print(cond)
            # print(f"object: {x}, condtion: {cond(x)}")
            if bool(cond(x)):
                return x
        except Exception as e:
            print("raise exception in selection_one: ", e)
            continue
    return None


def selection_all(kind: str, cond_str: str, selection_ctx: dict) -> List[str]:
    selection_ctx = selection_ctx or {}
    execu = selection_ctx.get("executor")
    facts = _get_current_facts(selection_ctx)
    world = selection_ctx["world_builder"](predicates=facts)

    print("selection iota: kind =", kind, ", cond_str =", cond_str)

    env = _build_cond_env(execu, world)
    env.update({k: v for k, v in selection_ctx.items() if isinstance(k, str)})
    # Normalize object names like "garbagecan 1" -> "garbagecan_1" before evaluation.
    
    normalized_cond_str = convert_all_items(cond_str)
    # Important: pass env as globals so names resolve inside the lambda at call time.
    cond = eval("lambda x: (" + normalized_cond_str + ")", env, env)
    
    kind_parsed = kind.strip("'").strip("\"").lower()
    if kind_parsed == 'item':
        pool = _all_items(world)
    elif kind_parsed == 'receptacle':
        pool = _all_receptacles(world)
    else:
        raise ValueError(f"Unknown kind: {kind} -> {kind_parsed}")
    pool = sorted(pool)
    # print(pool)
    results = []
    for x in pool:
        try:
            if bool(cond(x)):
                results.append(x)
        except Exception as e:
            print(e)
            continue
    print("selected results: ", results)
    return results




def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)



def _parse_and_eval_payload(payload_raw: Optional[str], ctx: dict, verbose: bool = True) -> Dict[str, object]:
    """
    Convert payload strings (legacy "{k:v}" or new "a=b, c" forms) into dict and evaluate values against ctx.
    """
    if not payload_raw:
        return {}

    out: Dict[str, object] = {}
    for key, value_expr in _parse_payload_assignments(payload_raw):
        out[key] = _eval_value_expr(value_expr, ctx, ctx, verbose=verbose)
    return out

def _apply_node_global_writes(
    graph: Graph,
    node: Node,
    ctx: dict,
    local_scope: Dict[str, object],
    verbose: bool = True,
) -> None:
    if not node.global_write_exprs:
        return
    # combined_ctx = dict(ctx)
    # combined_ctx.update(local_scope)
    combined_ctx = dict(local_scope)

    local_ctx = dict(local_scope)
    # Allow node-level local-in bindings to participate in expression evaluation

    global_writing_log = "Applying global write: "
    for name, expr in node.global_write_exprs.items():
        value = _eval_value_expr(expr, local_ctx, ctx, verbose=verbose)
        
        if verbose:
            print(f"Applying global write {name} = {expr} -> {value}")

        global_writing_log += f"{name} = {expr} -> {value}, "

        ctx[name] = value
        if name in graph.global_vars:
            graph.global_vars[name] = value
        else:
            graph.global_vars[name] = value
        combined_ctx[name] = value
    global_writing_log = global_writing_log.strip(", ") + ". "
    return global_writing_log

def validate_edge_payloads(graph: Graph) -> List[str]:
    """
    Validate edge payloads against destination local inputs.
    - Extras in payload not in destination's local in -> error.
    - Missing keys are allowed if covered by node-level local_in bindings
      or by same-named globals.
    """
    errs: List[str] = []
    for e in graph.edges:
        dst = graph.nodes.get(e.dst)
        if not dst:
            errs.append(f"Edge {e.src}->{e.dst}: destination node missing")
            continue
        expected = set(dst.local_in)
        got = set(e.payload_keys or tuple())
        extras = got - expected
        if extras:
            errs.append(
                f"Edge {e.src}->{e.dst}: payload has unknown keys {sorted(list(extras))}; expected subset of {sorted(list(expected))}."
            )
            continue
        missing = expected - got
        unsatisfied = [k for k in missing if (k not in dst.local_in_exprs) and (k not in graph.global_vars)]
        if unsatisfied:
            errs.append(
                f"Edge {e.src}->{e.dst}: payload missing keys {unsatisfied} not covered by node bindings or globals."
            )
    return errs

# ===================== Traversal with info (and foreach) =====================

def _collect_expr_global_refs(expr: Optional[str], known_globals: Set[str]) -> Tuple[Set[str], Set[str]]:
    """
    Extract candidate global variable names referenced in a local-in binding expression.
    Supports both {VAR} and {{VAR}} placeholders, and also returns the raw identifiers encountered.
    """
    matched_refs: Set[str] = set()
    raw_refs: Set[str] = set()

    def _register_candidate(candidate: str) -> None:
        cand = candidate.strip()
        if not cand:
            return
        raw_refs.add(cand)
        for alt in _candidate_variants(cand):
            if alt in known_globals:
                matched_refs.add(alt)

    if not expr:
        return matched_refs, raw_refs
    for pattern in _GLOBAL_REF_PATTERNS:
        for match in pattern.findall(expr):
            _register_candidate(match)
    stripped = expr.strip()
    if stripped:
        cleaned = stripped.strip("{} ").strip()
        if cleaned:
            _register_candidate(cleaned)
    return matched_refs, raw_refs


def _collect_placeholder_names(expr: Optional[str]) -> Set[str]:
    """
    Return placeholder identifiers referenced in an expression using {name} or {{name}} conventions.
    """
    names: Set[str] = set()
    if not expr:
        return names
    for pattern in _GLOBAL_REF_PATTERNS:
        for match in pattern.findall(expr):
            cleaned = match.strip()
            if cleaned:
                names.add(cleaned)
    return names


def _collect_single_brace_placeholders(expr: Optional[str]) -> Set[str]:
    """
    Return placeholder identifiers that still use the single-brace form {name}.
    """
    names: Set[str] = set()
    if not expr:
        return names
    for match in _SINGLE_BRACED_VAR_STRICT.finditer(expr):
        candidate = match.group(1).strip()
        if candidate:
            names.add(candidate)
    return names


def _compute_reachable_global_writes(graph: Graph) -> Dict[str, Set[str]]:
    """
    Compute, for each node, the set of global variables that are guaranteed to have been
    written by some predecessor (or seeded via defaults) along at least one path prior to the node.
    """
    spec_inputs: Set[str] = set()
    for node in graph.nodes.values():
        if node.id.upper() == "FLOW_SPEC":
            inputs, _ = _extract_named_values_with_types(node.raw_label, "inputs")
            for key in inputs.keys():
                spec_inputs.add(_strip_braces(key))

    writes_by_node: Dict[str, Set[str]] = {
        nid: set(node.global_writes or []) for nid, node in graph.nodes.items()
    }
    available: Dict[str, Set[str]] = {nid: set() for nid in graph.nodes}
    incoming: Dict[str, Set[str]] = {nid: set() for nid in graph.nodes}
    for edge in graph.edges:
        incoming.setdefault(edge.dst, set()).add(edge.src)

    default_globals = {name for name, value in graph.global_vars.items() if value is not None}
    initial_globals = set(default_globals) | spec_inputs

    seed_nodes: Set[str] = {
        nid for nid in graph.nodes
        if graph.nodes[nid].kind == "start" or not incoming.get(nid)
    }
    for nid in seed_nodes:
        if initial_globals:
            available[nid].update(initial_globals)

    worklist: deque[str] = deque(graph.nodes.keys())
    in_queue: Set[str] = set(graph.nodes.keys())

    while worklist:
        nid = worklist.popleft()
        in_queue.discard(nid)
        propagated = available[nid] | writes_by_node.get(nid, set())
        for edge in graph.outgoing(nid):
            dst = edge.dst
            if not propagated:
                continue
            before = available[dst]
            if not propagated.issubset(before):
                available[dst] = before | propagated
                if dst not in in_queue:
                    worklist.append(dst)
                    in_queue.add(dst)
    return available


def validate_node_inputs(graph: Graph) -> List[str]:
    """
    Validate that each Check/Action node has its required local inputs satisfiable
    without relying on edge payloads. For each required input name:
      - OK if the node defines an inline binding in its label (local_in_exprs), or
      - OK if a same-named global exists in graph.global_vars, or
      - OK if any incoming edge provides that key via payload (backward compatible).
    Additionally enforce that runtime expressions only reference locals declared in
    the node and that all placeholders use the double-brace {{VAR}} convention.

    Returns a list of error strings. Empty list means all nodes are satisfiable.
    """
    errs: List[str] = []
    # Pre-compute incoming payload keys per node (union across incoming edges)
    incoming_payload_map: Dict[str, Set[str]] = {}
    for e in graph.edges:
        if e.payload_keys:
            incoming_payload_map.setdefault(e.dst, set()).update(e.payload_keys)

    for nid, node in graph.nodes.items():
        if node.kind not in ("check", "action", "loop", "dataop"):
            continue
        expected: List[str] = list(node.local_in or [])
        local_aliases: Set[str] = set()
        for name in expected:
            local_aliases.update(_candidate_variants(name))
        if expected:
            # Enforce lowercase naming for locals and ensure they don't clash with globals by case only
            for key in expected:
                if key != key.lower():
                    errs.append(
                        f"Node {nid} ({node.kind}) local input '{key}' must be lowercase to distinguish it from globals."
                    )
                if key.upper() in graph.global_vars:
                    if node.kind != "action":
                        errs.append(
                            f"Node {nid} ({node.kind}) local '{key}' collides with global '{key.upper()}'; rename the local variable '{key}' to avoid casing-only confusion."
                        )
                    else:
                        errs.append(
                            f"Node {nid} ({node.kind}) local '{key}' collides with global '{key.upper()}'; the local definition of action node is fixed, rename the global variable '{key.upper()}' to avoid casing-only confusion."
                        )
            incoming_keys = incoming_payload_map.get(nid, set())
            missing: List[str] = []
            for key in expected:
                if key in (node.local_in_exprs or {}):
                    continue
                if key in graph.global_vars:
                    continue
                if key in incoming_keys:
                    continue
                missing.append(key)
            if missing:
                errs.append(
                    f"Node {nid} ({node.kind}) missing inputs {missing}; add inline bindings like 'local in: ({', '.join(missing)} = GLOBAL)', with the defined globals."
                )

        def _enforce_double_braces(expr: Optional[str], context: str) -> None:
            singles = _collect_single_brace_placeholders(expr)
            if singles:
                raw_vars = sorted(singles)

                vars_with_double_braces = ""
                var_strs = ["'{{" + var + "}}'" for var in raw_vars]
                vars_with_double_braces = "[" + ", ".join(var_strs) + "]"
                errs.append(
                    f"Node {nid} ({node.kind}) {context} uses single-brace placeholders {raw_vars}; wrap them as {vars_with_double_braces}."
                )

        def _ensure_locals_only(expr: Optional[str], context: str) -> None:
            placeholders = _collect_placeholder_names(expr)
            if not placeholders:
                return
            invalid = []
            for placeholder in placeholders:
                if _candidate_variants(placeholder) & local_aliases:
                    continue
                invalid.append(placeholder)
            if invalid:
                errs.append(
                    f"Node {nid} ({node.kind}) {context} references {sorted(invalid)} "
                    f"not provided by 'local in'; bind needed globals to locals first."
                )

        # Always enforce placeholder style for local bindings
        for param, expr in (node.local_in_exprs or {}).items():
            _enforce_double_braces(expr, f"local input binding '{param}'")

        runtime_exprs: List[Tuple[str, Optional[str]]] = []
        if node.expr:
            runtime_exprs.append(("expression", node.expr))
        if node.args:
            runtime_exprs.append(("action args", node.args))
        if node.loop_list_expr:
            runtime_exprs.append(("loop iterable", node.loop_list_expr))
        for target, expr in (node.global_write_exprs or {}).items():
            runtime_exprs.append((f"global write '{target}'", expr))

        for context, expr in runtime_exprs:
            # _enforce_double_braces(expr, context)
            _ensure_locals_only(expr, context)
    return errs


def validate_node_inputs_with_prefix_node(graph: Graph) -> List[str]:
    """
    Ensure each node's local inputs are backed by global variables written on some
    predecessor path. For every local input binding, identify referenced globals
    and verify that each node can reach at least one write to that global before execution.
    """
    errs: List[str] = []
    if not graph.nodes:
        return errs

    known_globals: Set[str] = set(graph.global_vars.keys())
    reachable_before: Dict[str, Set[str]] = _compute_reachable_global_writes(graph)

    for nid, node in graph.nodes.items():
        if node.kind not in ("check", "action", "loop", "dataop"):
            continue
        if not node.local_in:
            continue
        available = set(reachable_before.get(nid, set()))
        available.difference_update(node.global_writes or [])
        expr_map = node.local_in_exprs or {}
        type_map = node.local_in_types or {}
        local_aliases: Set[str] = set()
        for name in node.local_in:
            local_aliases.update(_candidate_variants(name))
        for param in node.local_in:
            matched_refs, raw_refs = _collect_expr_global_refs(expr_map.get(param), known_globals)
            candidate_names: Set[str] = set(raw_refs)
            if not candidate_names:
                candidate_names = {param}
                guesses = _candidate_variants(param)
                matched_refs = {g for g in guesses if g in known_globals}
            required_names = matched_refs if matched_refs else candidate_names
            if not required_names:
                continue
            candidate_pool = matched_refs if matched_refs else candidate_names
            available_matches: Set[str] = set()
            for cand in candidate_pool:
                available_matches.update(_candidate_variants(cand) & available)
            if not available_matches:
                errs.append(
                    f"Node {nid} ({node.kind}) local '{param}' expects globals {sorted(required_names)} "
                    f"but previous global writes before node cover only {sorted(available)}."
                )
                continue
            local_type = type_map.get(param)
            normalized_local_type = _normalize_type_name(local_type)
            if normalized_local_type:
                compatible = False
                mismatched: List[str] = []
                missing_typed: List[str] = []
                for global_name in sorted(available_matches):
                    global_type = graph.global_types.get(global_name)
                    normalized_global_type = _normalize_type_name(global_type)
                    if normalized_global_type is None:
                        missing_typed.append(global_name)
                        continue
                    if normalized_global_type == normalized_local_type:
                        compatible = True
                    else:
                        mismatched.append(f"{global_name}: {global_type}")
                if not compatible:
                    if mismatched:
                        errs.append(
                            f"Node {nid} ({node.kind}) local '{param}' type {local_type} mismatched with globals {mismatched}."
                        )
                    elif missing_typed:
                        errs.append(
                            f"Node {nid} ({node.kind}) local '{param}' type {local_type} cannot be verified; "
                            f"globals {sorted(missing_typed)} lack type annotations."
                        )
        if node.global_write_exprs:
            for target_name, expr in node.global_write_exprs.items():
                placeholders = _collect_placeholder_names(expr)
                if not placeholders:
                    continue
                invalid = []
                for placeholder in placeholders:
                    if _candidate_variants(placeholder) & local_aliases:
                        continue
                    invalid.append(placeholder)
                if invalid:
                    errs.append(
                        f"Node {nid} ({node.kind}) global write '{target_name}' uses placeholders {sorted(invalid)} "
                        f"not provided by 'local in'; bind them via local inputs before writing."
                    )
    return errs


def validate_check_nodes_without_global_writes(graph: Graph) -> List[str]:
    """
    Check nodes must be pure predicates—reject any that declare `writes global`
    assignments so the author can relocate those side effects elsewhere.
    """
    errs: List[str] = []
    for nid, node in graph.nodes.items():
        if node.kind != "check":
            continue
        writes = list(node.global_writes or [])
        if not writes and node.global_write_exprs:
            writes = list(node.global_write_exprs.keys())
        if not writes:
            continue
        formatted: List[str] = []
        for name in writes:
            expr = (node.global_write_exprs or {}).get(name)
            formatted.append(f"{name} = {expr}" if expr else name)
        detail = "; ".join(formatted)
        errs.append(
            f"Node {nid} (check) attempts to write globals {writes}; move these writes globals into a DataOp node."
        )
    return errs


def validate_control_flow_outgoing_edges(graph: Graph) -> List[str]:
    """
    Enforce the control-flow edge guideline:
      - LoopControl nodes must expose exactly two outgoing edges labelled `body` and `done`.
      - Check nodes must branch with exactly one `Yes` edge and one `No` edge.
      - DataOp and PrimitiveAction nodes must have exactly one outgoing edge (unlabeled or otherwise).
    """
    errs: List[str] = []

    def _summarize_labels(edges: List[Edge]) -> str:
        if not edges:
            return "no outgoing edges"
        labels = [edge.mid_label if edge.mid_label else "<unlabeled>" for edge in edges]
        return ", ".join(labels)

    def _normalized_labels(edges: List[Edge]) -> Counter:
        labels = [(edge.mid_label or "").strip() for edge in edges]
        return Counter(labels)

    def _format_label_list(labels: List[str]) -> str:
        if not labels:
            return "[]"
        formatted = [lbl if lbl else "<unlabeled>" for lbl in labels]
        return "[" + ", ".join(formatted) + "]"

    for nid, node in graph.nodes.items():
        outs = graph.outgoing(nid)
        labels_summary = _summarize_labels(outs)
        label_counts = _normalized_labels(outs)
        total = sum(label_counts.values())
        if node.kind == "loop":
            required = {"body", "done"}
            missing = [lbl for lbl in required if label_counts.get(lbl, 0) == 0]
            unexpected = [lbl for lbl in label_counts if lbl not in required]
            if missing or total != 2 or unexpected:
                details = []
                if missing:
                    details.append(f"missing edges {_format_label_list(missing)}")
                if unexpected:
                    details.append(f"unexpected labels {_format_label_list(unexpected)}")
                if total != 2:
                    details.append(f"has {total} outgoing edges (expected 2)")
                detail_str = "; ".join(details) if details else labels_summary
                errs.append(
                    f"Node {nid} (LoopControl) must have exactly two outgoing edges labelled 'body' and 'done'; {detail_str or labels_summary}."
                )
        elif node.kind == "check":
            required = {"Yes", "No"}
            missing = [lbl for lbl in required if label_counts.get(lbl, 0) == 0]
            unexpected = [lbl for lbl in label_counts if lbl not in required]
            if missing or total != 2 or unexpected:
                details = []
                if missing:
                    details.append(f"missing edges {_format_label_list(missing)}")
                if unexpected:
                    details.append(f"unexpected labels {_format_label_list(unexpected)}")
                if total != 2:
                    if total == 0:
                        details.append(f"has {total} outgoing edges (expected 2), remove this node if it is not used in control flow.")
                    else:
                        details.append(f"has {total} outgoing edges (expected 2)")
                detail_str = "; ".join(details) if details else labels_summary
                errs.append(
                    f"Node {nid} (Check) must branch with exactly one 'Yes' edge and one 'No' edge; {detail_str or labels_summary}."
                )
        elif node.kind in ("dataop", "action"):
            if total != 1:
                errs.append(
                    f"Node {nid} ({node.kind}) must have exactly one outgoing edge but has {total}: {labels_summary}, remove this node if it is not used in control flow."
                )
    return errs


def validate_control_flow_node_definitions(graph: Graph) -> List[str]:
    """
    Ensure every edge declared under Control-Flow Edges references nodes that were
    defined earlier in the Mermaid spec. Missing definitions typically mean the
    node block was deleted or renamed without updating the edge list.
    """
    errs: List[str] = []
    if not graph.edges:
        return errs

    defined_nodes: Set[str] = set(graph.nodes.keys())

    def _describe_edge(edge: Edge) -> str:
        label = f"[{edge.mid_label}]" if edge.mid_label else ""
        return f"{edge.src} {label}-> {edge.dst}".strip()

    missing_usage: Dict[str, List[str]] = {}
    for edge in graph.edges:
        if edge.src not in defined_nodes:
            missing_usage.setdefault(edge.src, []).append(f"as source in '{_describe_edge(edge)}'")
        if edge.dst not in defined_nodes:
            missing_usage.setdefault(edge.dst, []).append(f"as destination in '{_describe_edge(edge)}'")

    for nid, contexts in missing_usage.items():
        locations = "; ".join(contexts)
        errs.append(
            f"Node '{nid}' is referenced {locations} inside the Control-Flow Edges section but is never defined earlier; "
            f"add a node block for '{nid}' before the edge list or remove the edge."
        )
    return errs


def validate_loop_entry_edges(graph: Graph) -> List[str]:
    """
    Enforce loop entry labels:
      - Every edge entering a LoopControl node must carry Start_Loop or Continue_Loop.
      - Each LoopControl node must have at least one incoming Start_Loop edge.
      - Loop directives (Start_Loop/Continue_Loop) may only appear on edges whose destination is a LoopControl node.
    """
    errs: List[str] = []
    loop_nodes: Set[str] = {nid for nid, node in graph.nodes.items() if node.kind == "loop"}
    if not graph.edges or not loop_nodes:
        return errs

    allowed = {"Start_Loop", "Continue_Loop"}

    # Validate that loop directives only target loop nodes
    for e in graph.edges:
        if e.loop_directive:
            if e.dst not in loop_nodes:
                dst_kind = graph.nodes.get(e.dst).kind if e.dst in graph.nodes else "unknown"
                errs.append(
                    f"Edge {e.src}->{e.dst} labelled {e.loop_directive} must point to a LoopControl node, "
                    f"but {e.dst} is kind '{dst_kind}'."
                )
            elif e.loop_directive not in allowed:
                errs.append(
                    f"Edge {e.src}->{e.dst} uses unsupported loop directive '{e.loop_directive}'; "
                    f"use Start_Loop or Continue_Loop."
                )

    # For each loop node, require proper incoming labels and at least one Start_Loop
    for loop_nid in loop_nodes:
        incoming = [e for e in graph.edges if e.dst == loop_nid]
        if not incoming:
            continue
        has_start = False
        for e in incoming:
            if e.loop_directive is None:
                errs.append(
                    f"Edge {e.src}->{loop_nid} entering LoopControl must be labelled Start_Loop or Continue_Loop."
                )
            elif e.loop_directive not in allowed:
                errs.append(
                    f"Edge {e.src}->{loop_nid} uses unsupported loop directive '{e.loop_directive}'."
                )
            elif e.loop_directive == "Start_Loop":
                has_start = True
        if not has_start:
            errs.append(
                f"LoopControl {loop_nid} has no incoming edge labelled Start_Loop; at least one is required to reset/enumerate the loop."
            )

    return errs


def _pick_edge(graph: Graph, src: str, want: Optional[str]) -> Optional[Edge]:
    outs = graph.outgoing(src)
    if want is None:
        # prefer a single unlabeled edge
        unlabeled = [e for e in outs if (e.mid_label or "") == ""]
        if len(unlabeled) == 1:
            return unlabeled[0]
        return outs[0] if outs else None
    want_lower = want.lower() if want else None
    for e in outs:
        if (e.mid_label or "") == want:
            return e
        if e.mid_labels and want_lower:
            lbls = [lbl.lower() for lbl in e.mid_labels]
            if want_lower in lbls:
                return e
    return None

def _pick_edge_pref(graph: Graph, src: str, prefer: str) -> Optional[Edge]:
    outs = graph.outgoing(src)
    prefer_lower = prefer.lower()
    for e in outs:
        if (e.mid_label or "").lower() == prefer.lower():
            return e
        if e.mid_labels:
            lbls = [lbl.lower() for lbl in e.mid_labels]
            if prefer_lower in lbls:
                return e
    unlabeled = [e for e in outs if not e.mid_label]
    if len(unlabeled) == 1:
        return unlabeled[0]
    return None


def _callable_accepts(fn: Optional[Callable], min_positional: int) -> bool:
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    return len(positional) >= min_positional


def _apply_edge_payload(
    graph: Graph,
    ctx: dict,
    current_locals: Dict[str, object],
    edge: Edge, 
    verbose: bool = True,
) -> Dict[str, object]:
    dst_node = graph.nodes.get(edge.dst)
    local_expected: List[str] = list(dst_node.local_in) if dst_node else []

    eval_ctx = dict(ctx)
    eval_ctx.update(current_locals)
    updates = _parse_and_eval_payload(edge.payload_raw, eval_ctx, verbose=verbose)

    # Resolve expected locals: prefer explicit payload; else node-level binding; else same-name global
    resolved: Dict[str, object] = {}
    for k, v in updates.items():
        resolved[k] = v
    if dst_node is not None:
        for name in local_expected:
            if name in resolved:
                continue
            bind_expr = dst_node.local_in_exprs.get(name) if hasattr(dst_node, 'local_in_exprs') else None
            if bind_expr is not None:
                resolved[name] = _eval_value_expr(bind_expr, eval_ctx, ctx, verbose=verbose)
                continue
            if name in ctx:
                resolved[name] = ctx[name]

    # Remove stale local values from ctx when not tracked as global vars
    for name in local_expected:
        if name not in graph.global_vars:
            ctx.pop(name, None)

    next_locals: Dict[str, object] = {}
    for key, value in resolved.items():
        if (key not in local_expected) or (key in graph.global_vars):
            ctx[key] = value
            if key in graph.global_vars:
                graph.global_vars[key] = value
        if key in local_expected:
            next_locals[key] = value

    return next_locals

def traverse_with_info(
    graph: Graph,
    decide: Callable[..., bool],
    act: Optional[Callable[..., None]],
    ctx: dict,
    start: str = "START",
    max_steps: int = 2000,
    verbose: bool = True, 
) -> List[Tuple[str, Optional[str]]]:
    """
    Like your previous traverse(), but:
      - when taking an edge, apply its payload to ctx (evaluated via _parse_and_eval_payload)
      - still supports loop foreach semantics ('For VAR in LIST' on LoopControl)
    """
    log: List[Tuple[str, Optional[str]]] = []
    node_id = start
    prev_id: Optional[str] = None
    prev_edge: Optional[Edge] = None
    steps = 0
    current_locals: Dict[str, object] = {}
    decide_uses_locals = _callable_accepts(decide, 3)
    act_uses_locals = _callable_accepts(act, 4) if act else False

    for gname, gdefault in graph.global_vars.items():
        if gname not in ctx:
            ctx[gname] = copy.deepcopy(gdefault)

    def _record_edge(transition: Edge) -> None:
        mid_label = transition.mid_label.strip() if transition.mid_label else None
        payload = transition.payload_raw.strip() if transition.payload_raw else None
        label_text = None
        if transition.mid_labels:
            label_text = ", ".join([lbl.strip() for lbl in transition.mid_labels if lbl is not None])
        elif mid_label:
            label_text = mid_label
        if label_text and payload:
            edge_repr = f"{transition.src} -->|{label_text}, {payload}| {transition.dst}"
        elif label_text:
            edge_repr = f"{transition.src} -->|{label_text}| {transition.dst}"
        elif payload:
            edge_repr = f"{transition.src} -->|{payload}| {transition.dst}"
        else:
            edge_repr = f"{transition.src} --> {transition.dst}"
        log.append(("EDGE", edge_repr))

    while steps < max_steps:
        steps += 1
        node = graph.nodes[node_id]
        ctx_for_render = dict(ctx)
        ctx_for_render.update(current_locals)

        if verbose:
            print(node.id, node.kind, node.expr)
        
        
        # LOOP: foreach
        if node.kind == "loop" and node.loop_var and node.loop_list:
            idx_key = f"__loop_idx_{node.id}"
            first_key = f"__loop_init_{node.id}"
            incoming_dir = prev_edge.loop_directive if prev_edge else None
            if incoming_dir == "Start_Loop":
                ctx.pop(first_key, None)
                ctx.pop(idx_key, None)
            if not ctx.get(first_key, False):
                ctx[first_key] = True
                ctx[idx_key] = 0
            else:
                if prev_id is not None and prev_id != node.id:
                    if incoming_dir == "Start_Loop":
                        ctx[idx_key] = 0
                    else:
                        ctx[idx_key] = int(ctx.get(idx_key, 0)) + 1

            loop_inputs: Dict[str, object] = {}
            combined_ctx = dict(ctx)
            combined_ctx.update(current_locals)

            for name in node.loop_inputs or []:
                value = None
                if name in node.local_in_exprs:
                    try:
                        value = _eval_value_expr(node.local_in_exprs[name], combined_ctx, ctx, verbose=verbose)
                    except Exception:
                        value = None
                elif name in current_locals:
                    value = current_locals[name]
                elif name in combined_ctx:
                    value = combined_ctx[name]

                loop_inputs[name] = value
                if value is not None:
                    ctx[name] = value
                    combined_ctx[name] = value

            items_expr_raw = node.loop_list_expr
            items = None
            items_source = node.loop_list
            numeric_range_items: Optional[List[int]] = None
            if getattr(node, "loop_iter_is_range", False):
                arg_exprs = list(getattr(node, "loop_range_arg_exprs", []) or [])
                if not arg_exprs and node.loop_list_expr:
                    arg_exprs = [node.loop_list_expr]
                range_args: List[int] = []
                range_failure = False
                for arg_expr in arg_exprs:
                    expr = (arg_expr or "").strip()
                    if not expr:
                        continue
                    try:
                        value = _eval_value_expr(expr, combined_ctx, ctx, verbose=verbose)
                    except Exception:
                        range_failure = True
                        break
                    if value is None:
                        range_failure = True
                        break
                    try:
                        range_args.append(int(value))
                    except (TypeError, ValueError):
                        range_failure = True
                        break
                if not range_failure:
                    try:
                        numeric_range_items = list(range(*range_args))
                    except Exception:
                        numeric_range_items = None
            if items is None and items_expr_raw:
                try:
                    evaluated = _eval_value_expr(items_expr_raw, combined_ctx, ctx, verbose=verbose)
                except Exception:
                    evaluated = None
                if isinstance(evaluated, (list, tuple)):
                    items = evaluated
                elif evaluated is not None:
                    items_source = _strip_braces(str(evaluated)) or items_source

            if items is None and numeric_range_items is not None:
                items = numeric_range_items

            if items is None and items_source:
                if items_source in loop_inputs and loop_inputs[items_source] is not None:
                    items = loop_inputs[items_source]
                elif items_source in combined_ctx:
                    items = combined_ctx[items_source]
                else:
                    items = ctx.get(items_source, [])
            if items is None:
                items = []

            if not isinstance(items, (list, tuple)):
                raise TypeError(f"ctx['{node.loop_list}'] must be a list/tuple")
            i = int(ctx.get(idx_key, 0))
            log.append((node.id, f"For {node.loop_var} in {node.loop_list} (i={i}/{len(items)})"))
            
            if verbose:
                print((node.id, f"For {node.loop_var} in {node.loop_list} (i={i}/{len(items)})"))

            if i < len(items):
                ctx[node.loop_var] = items[i]
                if node.loop_var in graph.global_vars:
                    graph.global_vars[node.loop_var] = ctx[node.loop_var]
                loop_scope = dict(current_locals)
                loop_scope[node.loop_var] = ctx[node.loop_var]
                _apply_node_global_writes(graph, node, ctx, loop_scope, verbose=verbose)
                e = _pick_edge_pref(graph, node.id, "body")
            else:
                e = _pick_edge_pref(graph, node.id, "done")
            if e is None:
                raise RuntimeError(f"Loop node {node.id} requires 'body' or 'done' edge")
            current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
            _record_edge(e)
            prev_id, prev_edge, node_id = node.id, e, e.dst
            continue
        
        # CHECK: branching
        if node.kind == "check":
            expr = (node.expr or "").strip()
            outcome = decide(expr, ctx, dict(current_locals))
            if isinstance(outcome, tuple) and len(outcome) == 2:
                decision, log_i = outcome
            else:
                decision = bool(outcome)
                log_i = f"CHECK {expr}"
            log.append((node.id, log_i))
            br = "Yes" if decision else "No"
            e = _pick_edge(graph, node.id, br)
            if e is None:
                raise RuntimeError(f"Check node {node.id} missing '{br}' edge")
            current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
            _record_edge(e)
            prev_id, prev_edge, node_id = node.id, e, e.dst
            continue

        # ACTION
        if node.kind == "action":
            act_name = node.action or "action"
            args = node.args or ""

            try:
                ctx["_current_node_id"] = node.id
                log_entry = act(act_name, args, ctx, dict(current_locals))
            except Action_Unmatched_Exception as e:
                log_entry = str(e)
                log.append((node.id, log_entry))
                return log 

            if log_entry is None:
                log_entry = f"ACTION {act_name}({args})"
            log.append((node.id, log_entry))

            e = _pick_edge(graph, node.id, None)
            if e is None:
                raise RuntimeError(f"Action node {node.id} has no outgoing edge")
            current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
            _record_edge(e)
            prev_id, prev_edge, node_id = node.id, e, e.dst
            continue

        # DATAOP: perform data operations (writes GLOBAL, bindings), then proceed
        if node.kind == "dataop":

            if verbose:
                print("DataOp:", node.id, node.kind, node.global_write_exprs)
                print("current locals: ", current_locals)

            text_log = _apply_node_global_writes(graph, node, ctx, dict(current_locals), verbose=verbose)

            # writes already applied at the top for non-loop nodes; this branch clarifies control flow/logging
            log.append((node.id, f"DATAOP. Effect: {text_log}"))
            
            e = _pick_edge(graph, node.id, None)
            if e is None:
                raise RuntimeError(f"DataOp node {node.id} has no outgoing edge")
            current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
            _record_edge(e)
            prev_id, prev_edge, node_id = node.id, e, e.dst
            continue

        # START / INTERFACE / UNKNOWN
        if node.kind in ("start", "interface", "unknown"):
            log.append((node.id, node.raw_label))
            e = _pick_edge(graph, node.id, None)
            if e is None:
                raise RuntimeError(f"Node {node.id} has no outgoing edge")
            current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
            _record_edge(e)
            prev_id, prev_edge, node_id = node.id, e, e.dst
            continue

        # TERMINAL
        if node.kind == "terminal":
            log.append((node.id, node.raw_label))
            break

        # Fallback
        log.append((node.id, node.raw_label))
        e = _pick_edge(graph, node.id, None)
        if e is None:
            raise RuntimeError(f"Node {node.id} has no outgoing edge")
        current_locals = _apply_edge_payload(graph, ctx, current_locals, e, verbose=verbose)
        _record_edge(e)
        prev_id, prev_edge, node_id = node.id, e, e.dst

    return log

# ===================== Condensation DAG (SCC) =====================

def _build_orig_adj(g: Graph) -> Dict[str, List[str]]:
    all_nodes = set(g.nodes.keys())
    for e in g.edges:
        all_nodes.add(e.src); all_nodes.add(e.dst)
    adj = {n: [] for n in all_nodes}
    for e in g.edges:
        adj[e.src].append(e.dst)
    return adj

def _tarjan_scc(adj: Dict[str, List[str]]) -> List[List[str]]:
    idx = 0
    indices, low, stack, onstack = {}, {}, [], set()
    out: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal idx
        indices[v] = idx; low[v] = idx; idx += 1
        stack.append(v); onstack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w); low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop(); onstack.discard(w)
                comp.append(w)
                if w == v: break
            out.append(comp)

    for v in list(adj.keys()):
        if v not in indices:
            strongconnect(v)
    return out

@dataclass
class CNode:
    cid: int
    members: List[str]

@dataclass
class CEdge:
    src: int
    dst: int
    originals: List[Edge]   # original edges folded into this condensed edge

@dataclass
class Condensed:
    nodes: Dict[int, CNode]
    edges: List[CEdge]
    adj: Dict[int, List[int]]

def condense_to_dag(g: Graph) -> Condensed:
    adj = _build_orig_adj(g)
    sccs = _tarjan_scc(adj)
    nid2cid: Dict[str, int] = {}
    for cid, comp in enumerate(sccs):
        for v in comp:
            nid2cid[v] = cid

    cnodes = {cid: CNode(cid, sorted(comp)) for cid, comp in enumerate(sccs)}
    seen: Set[Tuple[int, int]] = set()
    cedges: List[CEdge] = []
    cadj: Dict[int, List[int]] = {cid: [] for cid in cnodes}

    # collect originals per condensed edge
    edge_map: Dict[Tuple[int, int], List[Edge]] = {}
    for e in g.edges:
        a, b = nid2cid[e.src], nid2cid[e.dst]
        if a == b:  # intra-scc
            continue
        if (a, b) not in edge_map:
            edge_map[(a, b)] = [e]
            cadj[a].append(b)
        else:
            edge_map[(a, b)].append(e)

    for (a, b), lst in edge_map.items():
        cedges.append(CEdge(src=a, dst=b, originals=lst))

    return Condensed(cnodes, cedges, cadj)




def reharsal_with_info(skill_workflow: Graph, ctx: dict) -> None:
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
                for alias in _candidate_variants(key):
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
            expr_fmt = expr_src.replace("{", "\"{").replace("}", "}\"")
            render_env = _build_render_context(local_args)
            rendered = expr_fmt.format(**render_env)
            converted_expr = convert_all_items(rendered)

            execu = ctx["executor"]
            facts = _get_current_facts(ctx)
            # print(facts)
            
            try:
                current_world = ctx["world_builder"](predicates=facts)
                env = _build_predicate_env(execu, current_world, facts)
                condition_value = bool(eval(converted_expr, env, env))
            except Exception as e:
                print(f"Error evaluating expression: {converted_expr}")
                print(f"Exception: {e}")

            print(f"step {ctx['current_step']}, {ctx['history'][ctx['current_step']]['obs']}")
            print(f"step {ctx['current_step']}, decide: {converted_expr}, value: {condition_value}")

            return condition_value, f"CHECK {converted_expr}"

        if raw_expr == "terminated?":
            return ctx["current_step"] >= len(ctx["history"])  # type: ignore
        
        
        print("here is a check node: ", raw_expr)

        if re.search(r"\b(exists|forall)\s*\(", raw_expr, flags=re.IGNORECASE):
            return _eval_bool_with_env(raw_expr)

        expr_q = raw_expr.replace("{", "\"{").replace("}", "}\"")
        expr_lambda = "lambda d: d.f_" + expr_q
        render_env = _build_render_context(local_args)
        converted_expr = convert_all_items(expr_lambda.format(**render_env))
        eval_locals = dict(ctx)
        eval_locals.update(local_args)
        eval_locals.setdefault("ctx", ctx)

        print("converted_expr: ", converted_expr)
        expr_fn = eval(converted_expr, globals(), eval_locals)

        facts = _get_current_facts(ctx)
        
        print(facts)
        try:
            current_world = ctx["world_builder"](predicates=facts)
            condition_value = ctx["executor"].execute(expr_fn(ctx["domain"]), current_world).value
        except Exception as e:
            print(f"Error evaluating expression: {converted_expr}")
            print(f"Exception: {e}") 
        print(condition_value)
        
        print(f"step {ctx['current_step']}, {ctx['history'][ctx['current_step']]['obs']}")
        print(f"step {ctx['current_step']}, decide: {converted_expr}, value: {condition_value}")

        return condition_value, f"CHECK {converted_expr}"


    def act(act_name: str, args: str, ctx: dict, local_args: dict) -> None:

        # print("action node: ", action, args, ctx.keys())
        while ctx["history"][ctx["current_step"]]["action"].startswith("think") and ctx["current_step"] < len(ctx["history"]):
            print(f"step {ctx['current_step']}, action: {ctx['history'][ctx['current_step']]['action']}")
            ctx["current_step"] += 1
        
        if ctx["current_step"] >= len(ctx["history"]):
            raise RuntimeError("No more steps in history")

        template = (args or "").strip()
        if not template and act_name:
            template = act_name
        render_env = _build_render_context(local_args)
        action_str = template.format(**render_env)

        action_str = inverse_all_items(action_str)


        # print(template)
        # print(local_args)

        if ctx["history"][ctx["current_step"]]["action"] == action_str:
            # action matches, move to next step
            pass
        else:
            # print("action: ", action)
            # print("args: ", args)
            # print("local_args: ", local_args)
            # print(action.format(**local_args))
            
            raise RuntimeError(f"Action `{action_str}` does not match history `{ctx['history'][ctx['current_step']]['action']}`")
        
        print(f"step {ctx['current_step']}, action: {ctx['history'][ctx['current_step']]['action']}")
        ctx["current_step"] += 1
        # No-op (you could log/update ctx)
        return f"ACTION {act_name}({action_str})"
    
    trace = traverse_with_info(skill_workflow, decide, act, ctx)
    for nid, info in trace:
        print(f"{nid}: {info}")
    if trace[-1][0] == "SUCCESS_END":
        return trace, True
    return trace, False


def reharsal_online_with_info(skill_workflow: Graph, ctx: dict) -> Tuple[List[Tuple[str, Optional[str]]], bool, dict]:
    """
    Online rehearsal variant that mirrors flow2graph.reharsal_online while respecting
    local/global payload propagation during traversal.
    """

    def _render_expr(expr: str, local_args: dict) -> str:
        expr_with_quotes = expr.replace("{", "\"{").replace("}", "}\"")
        render_globals = dict(skill_workflow.global_vars)
        for name, value in list(render_globals.items()):
            if ctx.get(name) is not None:
                render_globals[name] = ctx[name]
        render_globals.update(local_args)
        try:
            formatted = expr_with_quotes.format(**render_globals)
        except KeyError as e:
            print(f"KeyError: {e}")
            formatted = expr_with_quotes

        print(expr_with_quotes, " >>> ", formatted)
        return convert_all_items(formatted)

    def decide(expr: str, ctx_env: dict, local_args: dict) -> Tuple[bool, str]:
        print("decide node:", expr, ctx_env.keys())

        if expr == "terminated?":
            return False, "CHECK terminated?"

        # Helper to evaluate quantifier/boolean expressions directly
        def _eval_bool_with_env(expr_src: str) -> Tuple[bool, str]:
            converted_expr = _render_expr(expr_src.strip(), local_args)
            execu = ctx_env["executor"]
            assert isinstance(ctx_env["current_facts"], list)
            current_world = ctx_env["world_builder"](predicates=ctx_env["current_facts"])

            print(converted_expr)

            def _wrap(fn_name):
                def _inner(*args):
                    with execu.with_grounding(current_world):
                        return getattr(execu, fn_name)(*args)
                return _inner

            env = {
                "reachable": _wrap("reachable"),
                "locate": _wrap("locate"),
                "contains": _wrap("contains"),
                "holding": _wrap("holding"),
                "is_open": _wrap("is_open"),
                "is_closed": _wrap("is_closed"),
                "is_cleaned": _wrap("is_cleaned") if hasattr(execu, "is_cleaned") else (lambda x: False),
                "is_cooled": _wrap("is_cooled") if hasattr(execu, "is_cooled") else (lambda x: False),
                "is_heated": _wrap("is_heated") if hasattr(execu, "is_heated") else (lambda x: False),
                "is_turned_on": _wrap("is_turned_on") if hasattr(execu, "is_turned_on") else (lambda x: False),
                "is_item_of_type": _wrap("is_item_of_type"),
                "is_receptacle_of_type": _wrap("is_receptacle_of_type") if hasattr(execu, "is_receptacle_of_type") else (lambda r, t: str(r).startswith(str(t) + "_")),
                "Item": "Item",
                "Receptacle": "Receptacle",
                "Agent": "Agent",
            }
            env.update({
                "can_goto": lambda agent, receptacle: env["reachable"](agent, receptacle) and (not env["locate"](agent, receptacle)),
                "can_open": lambda agent, receptacle: env["locate"](agent, receptacle) and (not env["is_open"](receptacle)),
                "can_take": lambda agent, receptacle, item: (
                    env["locate"](agent, receptacle)
                    and env["is_open"](receptacle)
                    and env["contains"](receptacle, item)
                    and (not env["holding"](agent, item))
                ),
                "can_put": lambda agent, receptacle, item: (
                    env["locate"](agent, receptacle)
                    and env["is_open"](receptacle)
                    and env["holding"](agent, item)
                ),
            })

            def _pool(dtype_name):
                dn = str(dtype_name).lower()
                if "item" in dn:
                    objs = set()
                    for f in ctx_env.get("current_facts", []):
                        m = re.match(r"contains\(([^,]+),\s*([^\)]+)\)", f)
                        if m:
                            objs.add(m.group(2).strip())
                    return list(objs)
                if "receptacle" in dn:
                    recs = set()
                    for f in ctx_env.get("current_facts", []):
                        m1 = re.match(r"contains\(([^,]+),", f)
                        if m1:
                            recs.add(m1.group(1).strip())
                        m2 = re.match(r"locate\([^,]+,\s*([^\)]+)\)", f)
                        if m2:
                            recs.add(m2.group(1).strip())
                        m3 = re.match(r"reachable\([^,]+,\s*([^\)]+)\)", f)
                        if m3:
                            recs.add(m3.group(1).strip())
                    return list(recs)
                if "agent" in dn:
                    return ["agent"]
                return []

            def _exists(dtype_name, fn):
                for obj in _pool(dtype_name):
                    try:
                        if bool(fn(obj)):
                            return True
                    except Exception:
                        continue
                return False

            def _forall(dtype_name, fn):
                for obj in _pool(dtype_name):
                    try:
                        if not bool(fn(obj)):
                            return False
                    except Exception:
                        return False
                return True

            env["exists"] = _exists
            env["forall"] = _forall

            decision = bool(eval(converted_expr, env, env))
            
            print(f"decide: {converted_expr}, value: {decision}")

            return decision, f"CHECK {converted_expr}"

        if re.search(r"\b(exists|forall)\s*\(", expr, flags=re.IGNORECASE):
            return _eval_bool_with_env(expr)

        # default: function predicate path via domain function lambda
        expr_lambda = "lambda d: d.f_" + expr.strip()
        print(local_args)
        converted_expr = _render_expr(expr_lambda, local_args)
        eval_locals = dict(ctx_env)
        eval_locals.update(local_args)
        eval_locals.setdefault("ctx", ctx_env)
        expr_fn = eval(converted_expr, globals(), eval_locals)

        assert isinstance(ctx_env["current_facts"], list)
        current_world = ctx_env["world_builder"](predicates=ctx_env["current_facts"])
        decision = ctx_env["executor"].execute(expr_fn(ctx_env["domain"]), current_world).value

        print(f"decide: {converted_expr}, value: {decision}")
        return decision, f"CHECK {converted_expr}"

    def act(action: str, args: str, ctx_env: dict, local_args: dict) -> str:
        print("action node:", action, args, ctx_env.keys())


        template = (args or "").strip()
        if not template and action:
            template = action
        try:
            rendered_args = template.format(**local_args)
        except KeyError:
            rendered_args = template
        action_str = normalize_object_ids(rendered_args)

        observation, reward, done, info = ctx_env["env"].step([action_str])
        obs = ctx_env["process_obs"](observation[0])

        print("action:", action_str)
        print("obs:", obs)

        ctx_env["agent"].interaction_history += f"> {action_str.strip()}\n"

        ctx_env["current_facts"] = ctx_env["agent"].perceive(obs, ctx_env["current_facts"])
        print("current_facts:", ctx_env["current_facts"])
        ctx_env["agent"].previous_action = action_str

        ctx_env["history"].append({
            "action": action_str,
            "obs": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "facts": ctx_env["current_facts"],
        })

        if done and ctx_env.get("reason"):
            print("termination reason:", ctx_env["reason"])

        return f"ACTION {action}({action_str})"

    trace = traverse_with_info(skill_workflow, decide, act, ctx)

    for nid, info in trace:
        print(f"{nid}: {info}")
    success = bool(trace) and trace[-1][0] == "SUCCESS_END"
    return trace, success, ctx


# ===================== Demo / Example =====================


    
if __name__ == "__main__":

    from pathlib import Path

    demo_path = (
        Path(__file__).resolve().parents[3]
        / "skill_examples"
        / "alfworld"
        / "transfer_item_to_receptacle.mmd"
    )
    with demo_path.open(encoding="utf-8") as f:
        DEMO = f.read()
    # 1) Parse
    G = parse_mermaid_with_info(DEMO)

    # 2) Static validation
    edge_errors = validate_edge_payloads(G)
    if edge_errors:
        print("Validation errors [parameters mismatch]:")
        for e in edge_errors:
            print("  -", e)
    else:
        print("Validation passed: all edge payloads match destination 'local in'.")

    node_errors = validate_node_inputs(G)
    if node_errors:
        print("Validation errors [parameters mismatch]:")
        for e in node_errors:
            print("  -", e)
    else:
        print("Validation passed: all node payloads match destination 'local in'.")

    prefix_errors = validate_node_inputs_with_prefix_node(G)
    if prefix_errors:
        print("Validation errors [missing predecessor global writes]:")
        for e in prefix_errors:
            print("  -", e)
    else:
        print("Validation passed: all required globals are written before each node.")

    # 3) Dynamic traverse with info propagation
    ctx = {
        "receptacle_candidates": ["fridge 1", "cabinet 1", "diningtable 1"],
        "target_type": "apple",
    }

    def decide(expr: str, ctx: dict, local_args: dict) -> bool:
        # Simple demo: closed if fridge 1; exists in cabinet 1
        merged = dict(ctx)
        merged.update(local_args)
        if expr.lower().startswith("f_is_closed") or expr.lower().startswith("is_closed"):
            return merged.get("receptacle") == "fridge 1"
        if expr.lower().startswith("f_find_item_of_type") or expr.lower().startswith("exists"):
            return merged.get("receptacle") == "cabinet 1"
        return False

    def act(action: str, args: str, ctx: dict, local_args: dict) -> None:
        # no-op; you can log or mutate ctx if you need
        pass

    trace = traverse_with_info(G, decide, act, ctx)
    print("\nExecution trace:")
    for nid, info in trace:
        print(f"  {nid}: {info}")
    print("\nFinal ctx:", ctx)

    # 4) Condensation DAG (for static topological inspection)
    C = condense_to_dag(G)
    print("\nCondensation DAG nodes (SCCs):")
    for cid, cn in sorted(C.nodes.items()):
        print(f"  C{cid}: {cn.members}")
    print("Condensation DAG edges:")
    for ce in C.edges:
        forms = [f"{e.src} --{e.mid_label or ''}-->|{e.payload_raw or '{}'}| {e.dst}" for e in ce.originals]
        print(f"  C{ce.src} -> C{ce.dst}  via {forms}")


# ===================== Graft Utilities (Online Evolution) =====================

def graft_subgraph_at_node(
    base_mermaid: str,
    graft_mermaid: str,
    remove_edge_src: str,
    remove_edge_dst: str,
    continue_target: str,
    graft_prefix: str = "G0_",
) -> str:
    """Graft a sub-graph into a base mermaid flowchart by replacing an edge.

    Replaces the edge ``remove_edge_src --> remove_edge_dst`` with the graft
    sub-graph. The graft's ``GRAFT_CONTINUE`` exit maps to *continue_target*
    and ``GRAFT_FAIL`` maps back to the original *remove_edge_dst*.

    All graft node IDs are prefixed with *graft_prefix* to avoid collisions.

    Args:
        base_mermaid: Full mermaid flowchart string of the base skill.
        graft_mermaid: Mermaid fragment defining new nodes and edges.
            Must use ``GRAFT_ENTRY`` as the first node,
            ``GRAFT_CONTINUE`` for the success exit, and
            ``GRAFT_FAIL`` for the failure exit.
        remove_edge_src: Source node of the edge to replace.
        remove_edge_dst: Destination node of the edge to replace.
        continue_target: Node ID where ``GRAFT_CONTINUE`` should connect.
        graft_prefix: Prefix added to all graft node IDs.

    Returns:
        Combined mermaid string.
    """
    import re as _re

    # --- Step 1: Prefix graft node IDs --------------------------------
    graft_node_ids: set = set()
    for line in graft_mermaid.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%") or stripped.startswith("classDef"):
            continue
        m = _re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if m:
            graft_node_ids.add(m.group(1))
        for m2 in _re.finditer(r"-->\s*\|?[^|]*\|?\s*([A-Za-z_][A-Za-z0-9_]*)", stripped):
            graft_node_ids.add(m2.group(1))

    graft_node_ids.discard("GRAFT_ENTRY")
    graft_node_ids.discard("GRAFT_CONTINUE")
    graft_node_ids.discard("GRAFT_FAIL")

    prefixed_graft = graft_mermaid
    for nid in sorted(graft_node_ids, key=len, reverse=True):
        prefixed_graft = _re.sub(r'\b' + _re.escape(nid) + r'\b', graft_prefix + nid, prefixed_graft)

    prefixed_graft = prefixed_graft.replace("GRAFT_ENTRY", graft_prefix + "ENTRY")
    prefixed_graft = _re.sub(
        r'\b' + _re.escape("GRAFT_CONTINUE") + r'\b',
        continue_target,
        prefixed_graft,
    )
    prefixed_graft = _re.sub(
        r'\b' + _re.escape("GRAFT_FAIL") + r'\b',
        remove_edge_dst,
        prefixed_graft,
    )

    # --- Step 2: Remove the old edge from base mermaid ----------------
    edge_pattern = _re.compile(
        r'^(\s*)' + _re.escape(remove_edge_src) + r'\s*-->.*?' + _re.escape(remove_edge_dst) + r'\s*$',
        _re.MULTILINE,
    )
    edge_match = edge_pattern.search(base_mermaid)
    if not edge_match:
        raise ValueError(
            f"Cannot graft: edge '{remove_edge_src} --> {remove_edge_dst}' not found "
            f"in base mermaid. The skill graph does not have this edge."
        )

    # Extract the edge label (e.g., |No|, |done|) from the original edge
    original_edge_line = edge_match.group(0)
    label_match = _re.search(r'\|([^|]+)\|', original_edge_line)
    edge_label = f"|{label_match.group(1)}| " if label_match else ""

    new_base = edge_pattern.sub('', base_mermaid)

    # --- Step 3: Add the connection from remove_edge_src to graft entry
    entry_edge = f"    {remove_edge_src} -->{edge_label}{graft_prefix}ENTRY"

    # --- Step 4: Combine ------------------------------------------------
    graft_section = (
        f"\n    %% ===================== Grafted Sub-Graph ({graft_prefix}) =====================\n"
        f"{entry_edge}\n"
    )
    for line in prefixed_graft.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%") or stripped.startswith("flowchart"):
            continue
        graft_section += f"    {stripped}\n"

    combined = new_base.rstrip() + "\n" + graft_section
    return combined


def extract_graft_node_ids(mermaid_str: str, prefix: str = "G0_") -> set:
    """Return the set of node IDs in *mermaid_str* whose names start with *prefix*."""
    import re as _re
    ids: set = set()
    pattern = r'\b(' + _re.escape(prefix) + r'[A-Z][A-Za-z0-9_]*)\b'
    for m in _re.finditer(pattern, mermaid_str):
        ids.add(m.group(1))
    return ids


def revert_graft(skill_json_dict: dict, skill_name: str, original_mermaid: str) -> None:
    """Restore a skill's mermaid_code to its pre-graft state."""
    if skill_name in skill_json_dict:
        skill_json_dict[skill_name]["mermaid_code"] = original_mermaid

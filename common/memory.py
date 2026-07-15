
from typing import Dict, Iterable, List, Optional, Set, Tuple
import re


INTENT_EMBEDDING_MODEL = "text-embedding-3-small"
def _get_embedding_client():
    """Return an OpenAI client, using project default if available."""
    
    import openai  # type: ignore
    return openai.OpenAI()



def _embed_text_openai(text: str, model: str = INTENT_EMBEDDING_MODEL) -> list:
    """
    Get an embedding vector from OpenAI Embeddings API for the given text.
    On failure, raises the underlying exception.
    """
    client = _get_embedding_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def generate_embeddings(
                        text: str,
                        model: str = "text-embedding-3-small",
                        on_error: str = "warn") -> None:

    if not text:
        text = ""
    try:
        embedding = _embed_text_openai(text, model=model)
    except Exception as e:
        if on_error == 'raise':
            raise
        print(f"[embedding] warn: failed to embed text {text}: {e}")
    return embedding



def anonymuize_intent(intent: str) -> str:
    
    ITEM_WRAP_PATTERN = re.compile(r"\$(.+?)\$")
    RECEPTACLE_WRAP_PATTERN = re.compile(r"\&(.*?)\&")

    if not intent:
        return ""

    def _replace_item(match: re.Match) -> str:
        token = match.group(1).strip()
        if any(ch.isdigit() for ch in token):
            return "<ItemName>"
        return "<ItemType>"

    def _replace_receptacle(match: re.Match) -> str:
        token = match.group(1).strip()
        if any(ch.isdigit() for ch in token):
            return "<ReceptacleName>"
        return "<ReceptacleType>"

    intent = ITEM_WRAP_PATTERN.sub(_replace_item, intent)
    intent = RECEPTACLE_WRAP_PATTERN.sub(_replace_receptacle, intent)
    return intent
    


    
class TrajectoryMemoryNode:
    def __init__(self, observation_current, facts_current, action, action_remove_fact, action_add_fact, 
                 task_instruction=None, 
                 intent_for_action=None, important_predicates_for_action=None,
                 intent_embedding=None, embedding_model=INTENT_EMBEDDING_MODEL,
                 verbose=False):


        self.task_instruction = task_instruction
        self.observation_current = observation_current
        self.facts_current = facts_current
        self.action = action
        self.action_remove_fact = action_remove_fact
        self.action_add_fact = action_add_fact

        self.intent_for_action = intent_for_action
        self.intent_anonymized = anonymuize_intent(intent_for_action)
        self.important_predicates_for_action = important_predicates_for_action

        # Optional embedding fields
        self.intent_embedding = intent_embedding  # list[float] or None
        self.embedding_model = embedding_model

        self.id = None
        self.trajectory_next_node_id = None
        self.trajectory_prev_node_id = None

        if self.action is not None:
            self.action_type = self.action.split(" ")[0]
            
            if self.action.startswith("think:"):
                self.action_type = "think"
            elif self.action.startswith("go to"):
                self.action_type = "go to"
        else:
            self.action_type = None
        
        
        self.verbose = verbose
        if self.verbose:
            print("Initialize a TrajectoryMemoryNode:")
            print("observation_current:", self.observation_current)
            print("facts_current:", self.facts_current)
            print("action:", self.action)
            print("action_remove_fact:", self.action_remove_fact)
            print("action_add_fact:", self.action_add_fact)
            print("action_type:", self.action_type)
            print("intent_for_action:", self.intent_for_action)
            print("important_predicates_for_action:", self.important_predicates_for_action)
            print("intent_embedding_dim:", len(self.intent_embedding) if isinstance(self.intent_embedding, (list, tuple)) else None)
        
    def set_intent(self, intent: str):
        self.intent_for_action = intent
        # remove % and & in the intent string
        self.intent_for_action = intent.replace("$", "").replace("&", "")
        self.intent_anonymized = anonymuize_intent(intent)

    def show(self):
        # print("TrajectoryMemoryNode:")
        # print("id:", self.id)
        # print("task_instruction:", self.task_instruction)
        # print("observation_current:", self.observation_current)
        # print("facts_current:", self.facts_current)
        # print("action:", self.action)
        # print("action_type:", self.action_type)
        # print("intent_for_action:", self.intent_for_action)
        # print("important_predicates_for_action:", self.important_predicates_for_action)
        # print("intent_embedding_dim:", len(self.intent_embedding) if isinstance(self.intent_embedding, (list, tuple)) else None)

        print("-" * 50 + "\n"
            "    Node {node_id}: \n"
            "    obs: {obs} \n action: {action}".format(
                node_id=self.id,
                obs=self.observation_current,
                action=self.action,
            )
        )
        print("    action_intent:", self.intent_for_action)
        print("-" * 50)

    def to_dict(self):
        """
        Serialize this node into a JSON-friendly dictionary.
        """
        return {
            "id": self.id,
            "task_instruction": self.task_instruction,
            "observation_current": self.observation_current,
            "facts_current": list(self.facts_current) if self.facts_current is not None else [],
            "action": self.action,
            "action_type": self.action_type,
            "action_remove_fact": list(self.action_remove_fact) if self.action_remove_fact is not None else [],
            "action_add_fact": list(self.action_add_fact) if self.action_add_fact is not None else [],
            "intent_for_action": self.intent_for_action,
            "intent_anonymized": self.intent_anonymized,
            "important_predicates_for_action": (
                list(self.important_predicates_for_action)
                if self.important_predicates_for_action is not None else None
            ),
            "embedding_model": self.embedding_model,
            "intent_embedding": list(self.intent_embedding) if self.intent_embedding is not None else None,
            "trajectory_next_node_id": self.trajectory_next_node_id,
            "trajectory_prev_node_id": self.trajectory_prev_node_id,
        }


    @classmethod
    def from_dict(cls, data, verbose=False):
        """
        Reconstruct a TrajectoryMemoryNode from its serialized dictionary form.
        """
        important_predicates = data.get("important_predicates_for_action")
        if important_predicates is not None:
            important_predicates = list(important_predicates)

        intent_anonymized = data.get("intent_anonymized")
        intent_original = data.get("intent_for_action")
        if intent_original is None:
            intent_original = intent_anonymized

        node = cls(
            observation_current=data.get("observation_current"),
            facts_current=list(data.get("facts_current")) if data.get("facts_current") is not None else [],
            action=data.get("action"),
            action_remove_fact=list(data.get("action_remove_fact")) if data.get("action_remove_fact") is not None else [],
            action_add_fact=list(data.get("action_add_fact")) if data.get("action_add_fact") is not None else [],
            task_instruction=data.get("task_instruction"),
            intent_for_action=intent_original,
            important_predicates_for_action=important_predicates,
            intent_embedding=list(data.get("intent_embedding")) if data.get("intent_embedding") is not None else None,
            embedding_model=data.get("embedding_model"),
            verbose=verbose,
        )

        if intent_anonymized is not None:
            node.intent_anonymized = intent_anonymized
        if intent_original is None and intent_anonymized is not None:
            node.intent_for_action = intent_anonymized

        node.id = data.get("id")
        node.trajectory_next_node_id = data.get("trajectory_next_node_id")
        node.trajectory_prev_node_id = data.get("trajectory_prev_node_id")

        action_type = data.get("action_type")
        if action_type:
            node.action_type = action_type

        return node
    
    # --- Public API: add embeddings for nodes ---
                



class TrajectoryMemoryGraph:
    def __init__(self, verbose=False):

        self.verbose = verbose
        self.nodes = {}

        self.trajectory_head_list = []

    
    def add_node(self, node):

        node.id = self.node_count() + 1
        self.nodes[node.id] = node

        if self.verbose:
            print(f"Add node {node.id} to the graph.")
            print(f"Node {node.id} has action {node.action} with intent {node.intent_for_action} and important predicates {node.important_predicates_for_action}.")
        
        return node.id
    
    def add_edge(self, node_id1, node_id2, edge_type='trajectory'):

        if self.verbose:
            print(f"Add edge with type {edge_type}, node {node_id1} --> node {node_id2}.")

        if edge_type == 'trajectory':
            self.nodes[node_id1].trajectory_next_node_id = node_id2
            self.nodes[node_id2].trajectory_prev_node_id = node_id1
        
        else:
            raise Exception(f"Edge type {edge_type} not supported yet.")
        
    def node_count(self):
        return len(self.nodes)
    
    def show(self):

        for query in self.trajectory_head_list:

            node_id = query
            
            print(f"Trajectory head {node_id}: {self.nodes[node_id].task_instruction}")

            while node_id is not None:
                
                # print("-" * 100)
                # print("Node {node_id}:\n{obs}\nintent: {intent}\nintent anonymized: {intent_anonymized}\nimportant predicates: {important_predicates}\naction: {action}"
                #       .format(node_id=node_id, obs=self.nodes[node_id].observation_current, 
                #               intent=self.nodes[node_id].intent_for_action,
                #               intent_anonymized=self.nodes[node_id].intent_anonymized,
                #               important_predicates=self.nodes[node_id].important_predicates_for_action,
                #               action=self.nodes[node_id].action,
                #               ))
                
                # print("-" * 100)

                self.nodes[node_id].show()
                node_id = self.nodes[node_id].trajectory_next_node_id

    def to_dict(self):
        """
        Convert the trajectory memory graph into a dictionary that can be serialized.
        """
        graph_dict = {
            "trajectory_heads": list(self.trajectory_head_list),
            "nodes": {},
            "node_count": self.node_count(),
        }

        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            graph_dict["nodes"][node_id] = node.to_dict()

        return graph_dict

    @classmethod
    def from_dict(cls, data, verbose=False):
        """
        Reconstruct a TrajectoryMemoryGraph from its serialized dictionary form.
        """
        graph = cls(verbose=verbose)
        graph.load_from_dict(data)
        return graph
   
    def load_from_dict(self, data):
        """
        Load the trajectory memory graph from a dictionary into the current instance.
        """
        self.nodes = {}
        self.trajectory_head_list = list(data.get("trajectory_heads", []))

        nodes_data = data.get("nodes", {})
        for node_data in nodes_data.values():
            node = TrajectoryMemoryNode.from_dict(node_data, verbose=self.verbose)
            if node.id is None:
                raise ValueError("Serialized node is missing an 'id' field.")
            self.nodes[node.id] = node

        return self


class SkillCallingMemoryNode:
    """
    Node that stores the contextual summary generated when calling a skill for a
    particular subgoal. The node keeps the subgoal name, where it was executed in
    the trajectory graph, and the evidence collected from the segmentation prompt.
    """

    def __init__(
        self,
        subgoal_name: str,
        problem_node: int,
        trajectory_start_node: Optional[int],
        trajectory_end_node: Optional[int],
        subtrajectory_mapping: Iterable[Dict],
        parameter_bindings: Dict,
        related_crafting_commands: Optional[Iterable[str]],
        start_condition_evidence: Iterable[Dict],
        success_condition_evidence: Iterable[Dict],
        pre_span_summary: str,
        subgoal_initiation_intent: str,
        verbose: bool = False,
    ):
        self.subgoal_name = subgoal_name
        self.problem_node = problem_node
        self.trajectory_start_node = trajectory_start_node
        self.trajectory_end_node = trajectory_end_node
        self.subtrajectory_mapping = list(subtrajectory_mapping)
        self.parameter_bindings = dict(parameter_bindings)
        self.related_crafting_commands = list(related_crafting_commands)
        self.start_condition_evidence = list(start_condition_evidence)
        self.success_condition_evidence = list(success_condition_evidence)
        self.pre_span_summary = pre_span_summary
        self.subgoal_initiation_intent = subgoal_initiation_intent

        self.id: Optional[int] = None
        self.verbose = verbose

        if self.verbose:
            print(
                "Initialize SkillCallingMemoryNode:",
                {
                    "subgoal_name": self.subgoal_name,
                    "problem_node": self.problem_node,
                    "trajectory_start_node": self.trajectory_start_node,
                    "trajectory_end_node": self.trajectory_end_node,
                    "parameter_bindings": self.parameter_bindings,
                    "related_crafting_commands": self.related_crafting_commands,
                },
            )
        self.success_flag = False
        
    def set_success_flag(self, success_flag: bool):
        self.success_flag = success_flag

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "subgoal_name": self.subgoal_name,
            "problem_node": self.problem_node,
            "trajectory_start_node": self.trajectory_start_node,
            "trajectory_end_node": self.trajectory_end_node,
            "subtrajectory_mapping": list(self.subtrajectory_mapping),
            "parameter_bindings": dict(self.parameter_bindings),
            "related_crafting_commands": list(self.related_crafting_commands),
            "start_condition_evidence": list(self.start_condition_evidence),
            "success_condition_evidence": list(self.success_condition_evidence),
            "pre_span_summary": self.pre_span_summary,
            "subgoal_initiation_intent": self.subgoal_initiation_intent,
        }

    @classmethod
    def from_dict(cls, data: Dict, verbose: bool = False) -> "SkillCallingMemoryNode":
        node = cls(
            subgoal_name=data["subgoal_name"],
            problem_node=data["problem_node"],
            trajectory_start_node=data.get("trajectory_start_node"),
            trajectory_end_node=data.get("trajectory_end_node"),
            subtrajectory_mapping=data.get("subtrajectory_mapping", []),
            parameter_bindings=data.get("parameter_bindings", {}),
            related_crafting_commands=data.get("related_crafting_commands", []),
            start_condition_evidence=data.get("start_condition_evidence", []),
            success_condition_evidence=data.get("success_condition_evidence", []),
            pre_span_summary=data.get("pre_span_summary", ""),
            subgoal_initiation_intent=data.get("subgoal_initiation_intent", ""),
            verbose=verbose,
        )
        node.id = data.get("id")
        return node
    
    def show(self):
        print("-" * 50 + "\n"
            "  Node {node_id}: problem={problem}, "
            "traj_span=({start},{end}), params={params}".format(
                node_id=self.id,
                problem=self.problem_node,
                start=self.trajectory_start_node,
                end=self.trajectory_end_node,
                params=self.parameter_bindings,
            )
        )
        print("    pre_span_summary:", self.pre_span_summary)
        print("    related_crafting_commands:", self.related_crafting_commands)
        print("    subgoal_initiation_intent:", self.subgoal_initiation_intent)
        print("-" * 50)
    
    def expand_raw_trajectory(self, trajectory_graph):
        traj_info = ""
        current_traj_node = self.trajectory_start_node
        while current_traj_node != None:
            traj_info += trajectory_graph.nodes[current_traj_node].observation_current.strip() + "\n"
            traj_info += trajectory_graph.nodes[current_traj_node].action.strip() + "\n"
            current_traj_node = trajectory_graph.nodes[current_traj_node].trajectory_next_node_id

            if current_traj_node == self.trajectory_end_node:
                break
        
        return traj_info


class SkillCallingMemoryGraph:
    """
    Lightweight graph that stores skill-calling summaries per task/problem node.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.nodes: Dict[int, SkillCallingMemoryNode] = {}
        self.nodes_by_problem: Dict[int, List[int]] = {}
        self.nodes_by_subgoal: Dict[str, List[int]] = {}

    def node_count(self) -> int:
        return len(self.nodes)

    def add_node(self, node: SkillCallingMemoryNode) -> int:
        node.id = self.node_count() + 1
        self.nodes[node.id] = node
        self.nodes_by_problem.setdefault(node.problem_node, []).append(node.id)
        self.nodes_by_subgoal.setdefault(node.subgoal_name, []).append(node.id)

        if self.verbose:
            print(
                f"Add SkillCallingMemoryNode {node.id} for problem {node.problem_node} "
                f"subgoal {node.subgoal_name}"
            )

        return node.id
    def add_node_from_calling_summary(self, calling_summary: dict) -> int:
        node = SkillCallingMemoryNode(
                            subgoal_name=calling_summary["subgoal"],
                            problem_node=calling_summary["problem_node"],
                            trajectory_start_node=calling_summary["trajectory_start_node"],
                            trajectory_end_node=calling_summary["trajectory_end_node"],
                            subtrajectory_mapping=calling_summary["subtrajectory_mapping"],
                            parameter_bindings=calling_summary["parameter_bindings"],
                            related_crafting_commands=calling_summary["related_crafting_commands"],
                            start_condition_evidence=calling_summary["start_condition_evidence"],
                            success_condition_evidence=calling_summary["success_condition_evidence"],
                            pre_span_summary=calling_summary["pre_span_summary"],
                            subgoal_initiation_intent=calling_summary["subgoal_initiation_intent"],
                        )
        node.id = self.node_count() + 1
        self.nodes[node.id] = node
        self.nodes_by_problem.setdefault(node.problem_node, []).append(node.id)
        self.nodes_by_subgoal.setdefault(node.subgoal_name, []).append(node.id)

        if self.verbose:
            print(
                f"Add SkillCallingMemoryNode {node.id} for problem {node.problem_node} "
                f"subgoal {node.subgoal_name}"
            )

        return node.id

    def get_nodes_for_problem(self, problem_node: int) -> List[SkillCallingMemoryNode]:
        ids = self.nodes_by_problem.get(problem_node, [])
        return [self.nodes[node_id] for node_id in ids]
    
    def get_nodes_for_subgoal(self, subgoal_name: str) -> List[SkillCallingMemoryNode]:
        ids = self.nodes_by_subgoal.get(subgoal_name, [])
        return [self.nodes[node_id] for node_id in ids]

    def to_dict(self) -> Dict:
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "nodes_by_problem": {
                problem: list(node_ids) for problem, node_ids in self.nodes_by_problem.items()
            },
            "nodes_by_subgoal": {
                subgoal: list(node_ids) for subgoal, node_ids in self.nodes_by_subgoal.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict, verbose: bool = False) -> "SkillCallingMemoryGraph":
        graph = cls(verbose=verbose)
        graph.load_from_dict(data)
        return graph

    def load_from_dict(self, data: Dict) -> "SkillCallingMemoryGraph":
        self.nodes = {}
        self.nodes_by_problem = {}
        self.nodes_by_subgoal = {}



        for node_id_str, node_dict in data.get("nodes", {}).items():
            node = SkillCallingMemoryNode.from_dict(node_dict, verbose=self.verbose)
            if node.id is None:
                node.id = int(node_id_str)
            self.nodes[node.id] = node

        for problem, node_ids in data.get("nodes_by_problem", {}).items():
            self.nodes_by_problem[int(problem)] = list(node_ids)
        
        for subgoal, node_ids in data.get("nodes_by_subgoal", {}).items():
            self.nodes_by_subgoal[subgoal] = list(node_ids)



        return self
    
    def show(self):
        """
        Pretty-print all stored skill calling nodes grouped by problem node.
        """
        # for problem_node, node_ids in sorted(self.nodes_by_problem.items()):
        #     print(f"Problem node {problem_node}:")
        #     for node_id in node_ids:
        #         node = self.nodes.get(node_id)
        #         if node is None:
        #             continue
        #         print(
        #             "  Node {node_id}: subgoal={subgoal}, "
        #             "traj_span=({start},{end}), params={params}".format(
        #                 node_id=node_id,
        #                 subgoal=node.subgoal_name,
        #                 start=node.trajectory_start_node,
        #                 end=node.trajectory_end_node,
        #                 params=node.parameter_bindings,
        #             )
        #         )
        #         print("    pre_span_summary:", node.pre_span_summary)
        #         print("    subgoal_initiation_intent:", node.subgoal_initiation_intent)
        for subgoal_name, node_ids in sorted(self.nodes_by_subgoal.items()):
            print(f"Subgoal {subgoal_name}:")
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                node.show()

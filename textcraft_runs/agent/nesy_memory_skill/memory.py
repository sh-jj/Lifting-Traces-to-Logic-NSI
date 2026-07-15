

import re

from common.memory import TrajectoryMemoryNode, TrajectoryMemoryGraph
from common.memory import SkillCallingMemoryNode, SkillCallingMemoryGraph


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

class TrajectoryMemoryNode4TextCraft(TrajectoryMemoryNode):
    def __init__(self, observation_current, facts_current, action, action_remove_fact, action_add_fact, 
                 task_instruction=None, 
                 crafting_commands=None,
                 intent_for_action=None, important_predicates_for_action=None,
                 intent_embedding=None, embedding_model="",
                 verbose=False):
        
        self.crafting_commands = crafting_commands
        
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

    def to_dict(self):
        """
        Extend base serialization with crafting commands.
        """
        node_dict = super().to_dict()
        node_dict["crafting_commands"] = self.crafting_commands
        return node_dict

    @classmethod
    def from_dict(cls, data, verbose=False):
        """
        Reconstruct a TrajectoryMemoryNode4TextCraft from its serialized form,
        including crafting commands.
        """
        node = super().from_dict(data, verbose=verbose)
        node.crafting_commands = data.get("crafting_commands")
        return node




class TrajectoryMemoryGraph4TextCraft(TrajectoryMemoryGraph):
    def __init__(self, verbose=False):
    
        self.verbose = verbose
        self.nodes = {}

        self.trajectory_head_list = []

    def load_from_dict(self, data):
        """
        Load graph using TextCraft-specific nodes so crafting commands persist.
        """
        self.nodes = {}
        self.trajectory_head_list = list(data.get("trajectory_heads", []))

        nodes_data = data.get("nodes", {})
        for node_data in nodes_data.values():
            node = TrajectoryMemoryNode4TextCraft.from_dict(node_data, verbose=self.verbose)
            if node.id is None:
                raise ValueError("Serialized node is missing an 'id' field.")
            self.nodes[node.id] = node

        return self
    def show(self):
    
        for query in self.trajectory_head_list:

            node_id = query
            
            print(f"Trajectory head {node_id}: {self.nodes[node_id].task_instruction}")
            print(f"{self.nodes[node_id].crafting_commands}")
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
    

class SkillCallingMemoryNode4TextCraft(SkillCallingMemoryNode):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)


class SkillCallingMemoryGraph4TextCraft(SkillCallingMemoryGraph):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

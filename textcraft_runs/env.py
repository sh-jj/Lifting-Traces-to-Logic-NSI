import random
import re
from textcraft.utils import ActionFailed, ItemTag, ItemTagWithCount, Recipe, item_id_to_str
from textcraft.crafting_tree import CraftingTree
from typing import List
# to access data locally
import importlib.resources
from textcraft import TextCraft

class WrappedTextCraft(TextCraft):

    def __init__(self, split, minecraft_dir=None):
        if minecraft_dir is None:
            minecraft_dir = str(importlib.resources.files('textcraft') / 'data')

        
        self.inventory = {}
        self.action_regexes = {
            "craft": r"craft (.*) using (.*)",
            "get": r"get ([0-9]+) (.*)",
            "inventory": r"inventory",
        }
        self.count_regex = r"([0-9]+) (.*)"
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)

        item_depth_list = list(self.crafting_tree.item_recipes_min_depth(2))

        # print(item_depth_list)
        self.goal_list_with_depth = sorted(item_depth_list, key=lambda x: (x[1], x[0]))
        # print(self.goal_list_with_depth)
        # print(len(self.goal_list_with_depth), " tasks ")

        def print_goal_depth_distribution():
            depth_2_goals = [x[0] for x in self.goal_list if x[1] == 2]
            # print(depth_2_goals)
            depth_3_goals = [x[0] for x in self.goal_list if x[1] == 3]
            # print(depth_3_goals)
            depth_4_goals = [x[0] for x in self.goal_list if x[1] == 4]
            # print(depth_4_goals)

            print(len(depth_2_goals), " depth 2 goals")
            print(len(depth_3_goals), " depth 3 goals")
            print(len(depth_4_goals), " depth 4 goals")
        
        depth_2_goals = [(x[0], x[1]) for x in self.goal_list_with_depth if x[1] == 2]
        depth_3_goals = [(x[0], x[1]) for x in self.goal_list_with_depth if x[1] == 3]
        depth_4_goals = [(x[0], x[1]) for x in self.goal_list_with_depth if x[1] == 4]
        test_goals = depth_2_goals[-77:] + depth_3_goals + depth_4_goals
        val_goals = depth_2_goals[:-77]

        # print(len(test_goals), " test goals")
        # print(len(val_goals), " val goals")

        if split == "val":
            self.goal_list = val_goals

            print("There are {} val tasks in val set".format(len(self.goal_list)))
            print_goal_depth_distribution()

        else:
            self.goal_list = test_goals
            print("There are {} test tasks in test set".format(len(self.goal_list)))
            print_goal_depth_distribution()

        # print(self.goal_list_with_depth)
        # for tar, depth in self.goal_list_with_depth:
        #     if "dark_oak_logs" in tar:
        #         print(tar, depth)
        # dark oak logs
        # exit(0)

    def set_item_domain(self, item_domain):
        self.item_domain = item_domain
    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        try:
            for action_type, regex in self.action_regexes.items():
                match = re.match(regex, action)
                if match:
                    if action_type == "craft":
                        recipe = self.extract_recipe(
                            match.group(1), match.group(2))
                        if recipe is None:
                            raise ActionFailed(
                                "Could not extract a valid recipe from the given action, {}".format(action))
                        if not self.has_items(recipe.input_items):
                            # print("Could not find enough items to craft {}".format(recipe.output_item.item_tag.item_id))
                            item_name = recipe.output_item.item_tag.item_id.split("minecraft:")[1].replace("_", " ")
                            raise ActionFailed(
                                "Could not find enough items to craft {}".format(item_name))
                        output_itemtag_count = self.crafting_tree.craft(recipe)
                        if output_itemtag_count is None:
                            item_name = recipe.output_item.item_tag.item_id.split("minecraft:")[1].replace("_", " ")
                            item_count = recipe.output_item.count
                            raise ActionFailed(
                                "Could not find a valid recipe for {} {}".format(item_count, item_name))
                        self.remove_items(recipe.input_items)
                        self.add_item(output_itemtag_count.item_tag, output_itemtag_count.count)
                        observation = "Crafted {} {}".format(output_itemtag_count.count,
                                                            output_itemtag_count.item_tag.item_id)
                        if output_itemtag_count.item_tag.item_id == self.goal:
                            reward = 1
                            terminated = True
                    elif action_type == "get":
                        (item, amt) = match.group(2), int(match.group(1))
                        item_obj = self.item_str_to_obj(item)
                        if self.crafting_tree.is_craftable(item_obj.name):
                            raise ActionFailed("Could not find {}".format(item))
                        if self.crafting_tree.is_tag(item_obj.item_id) or \
                            item_obj.item_id is None:
                            raise ActionFailed("Could not find {}".format(item))
                        if not self.crafting_tree.is_valid_item(item_obj.item_id):
                            raise ActionFailed("Could not find {}".format(item))
                        self.add_item(item_obj, amt)
                        transformed_obj_name = item_obj.item_id.split("minecraft:")[1].replace("_", " ")
                        
                        # observation = "Got {} {}".format(amt, item)
                        observation = "Got {} {}".format(amt, transformed_obj_name)
                        
                    elif action_type == "inventory":
                        observation = "Inventory: "
                        if not len(self.inventory.items()): observation += 'You are not carrying anything.'
                        for item, amt in self.inventory.items():
                            observation += "[{}] ({}) ".format(item_id_to_str(item), amt)
                        # observation = observation.rstrip(', ')
                    else:
                        raise NotImplementedError(
                            "Action type {} not implemented".format(action_type))
            if observation is None:
                raise ActionFailed("Could not execute {}".format(action))

        except ActionFailed as e:
            observation = "{}".format(e.args[0])
            reward = 0
            info = {}

        return (observation, reward, terminated, truncated, info)
    


        
    def has_items(self, items:List[ItemTagWithCount]):
        for itemtag_count in items:
            if itemtag_count.item_tag.item_id not in self.inventory or \
                self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count:
                return False
        return True
    
    def add_item(self, item_tag: ItemTag, amt: int):
        if item_tag.item_id not in self.inventory:
            self.inventory[item_tag.item_id] = 0
        self.inventory[item_tag.item_id] += amt

    def remove_items(self, items: List[ItemTagWithCount]):
        for itemtag_amts in items:
            self.inventory[itemtag_amts.item_tag.item_id] -= itemtag_amts.count
            if self.inventory[itemtag_amts.item_tag.item_id] == 0:
                del self.inventory[itemtag_amts.item_tag.item_id]

    def extract_recipe(self, output_item_str, input_items_str) -> Recipe:
        # check if there is a number in the output item
        m = re.match("([0-9]+) (.*)", output_item_str)
        if m:
            output_item =  self.item_str_to_obj(m.group(2))
            output_item_count = int(m.group(1))
        else:
            output_item = self.item_str_to_obj(output_item_str)
            output_item_count = 1
        output_item_count = ItemTagWithCount(output_item, output_item_count)
        input_items = []
        for input_item_count in input_items_str.split(","):
            match = re.match(self.count_regex, input_item_count.strip())
            if match:
                count = int(match.group(1))
                item_str = match.group(2)
                input_item_obj = self.item_str_to_obj(item_str)
                input_items.append(ItemTagWithCount(input_item_obj, count))
            else:
                return None
                raise ActionFailed("Wrong item format: {}".format(input_item_count.strip()))
        return Recipe(input_items=input_items, output_item=output_item_count)
    
    def item_str_to_obj(self, item):
        item_name = item.strip()
        if hasattr(self, "item_domain"):
            normalized_item = re.sub(r"\s+", " ", item_name.strip().lower())
            normalized_domain = {
                re.sub(r"\s+", " ", name.strip().lower())
                for name in self.item_domain
                if isinstance(name, str)
            }
            if normalized_item not in normalized_domain:
                candidates = []
                if normalized_item.endswith("s"):
                    candidates.append(normalized_item[:-1])
                else:
                    candidates.append(normalized_item + "s")
                matched = next(
                    (candidate for candidate in candidates if candidate in normalized_domain),
                    None,
                )
                if matched is None:
                    raise ActionFailed("Could not find {}".format(item))
                normalized_item = matched
            item_name = normalized_item
            
        item_id = "minecraft:" + item_name.replace(" ", "_")
        if self.crafting_tree.is_tag(item_id):
            return ItemTag(tag=item_id)
        else:
            return ItemTag(item_id=item_id)
    
    def reset(self, env_index, seed=19260817, verbose=True):
        super().reset(seed=seed)
        # clean inventory
        self.inventory = {}
        random.seed(seed)
        


        # self.goal = "minecraft:dark_oak_sign"
        self.goal, self.depth = self.goal_list[env_index]
        if verbose:
            print("Index {} Goal: {} with depth: {}".format(env_index, self.goal, self.depth))
        
        recipes_set = set()
        distractor_set = set()
        max_distractor = 10
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal)
        for recipe in recipes:
            recipes_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipes_set:
                distractor_set.add(distractor.recipe_str)

        recipes_list = list(recipes_set) + random.sample(list(distractor_set),
                                                         min(len(distractor_set), max_distractor))
        random.shuffle(recipes_list)
        # print(recipes_list)
        # print(self.goal)

        return "Crafting commands:\n{}\n\nGoal: craft {}.".format("\n".join(recipes_list), 
                                                        item_id_to_str(self.goal)), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == "__main__":


    env = WrappedTextCraft(split="val")
    

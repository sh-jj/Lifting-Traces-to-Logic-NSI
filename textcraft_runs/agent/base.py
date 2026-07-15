
from .utils import llm_response, get_price, get_token_counts, reset_token_counts

class BaseAgent:
    def __init__(self):
        pass

    def act(self, observation):
        pass

    def reset(self):
        pass
    
    def reveiew_episode(self, **kwargs):
        pass

    def update_history(self, action):
        self.interaction_history += action.strip() + "\n"
    
    def update_skill_history(self, skill_i, function_calling, history):
        pass
    
    def reset_token_counts(self):
        reset_token_counts()
        
    def get_price(self):
        """
        Calculate the price of the LLM interactions.
        """
        prompt_tokens, completion_tokens = get_token_counts()
        return get_price(self.llm, prompt_tokens, completion_tokens)
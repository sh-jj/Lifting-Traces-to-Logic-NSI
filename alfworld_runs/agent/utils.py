
import os
import sys
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

completion_tokens = prompt_tokens = 0
llm_call_count = 0
Model = Literal["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-4o", "gpt-3.5-turbo-0125"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n = 1, drop_arrow: bool = False) -> str:
    global completion_tokens, prompt_tokens, llm_call_count
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        n=n,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    llm_call_count += 1
    if n > 1:
        if drop_arrow:
            responses = [choice.text.replace('>', '').strip() for choice in response.choices]
        else:
            responses = [choice.text.strip() for choice in response.choices]
        return responses
    if drop_arrow:
        return response.choices[0].text.replace('>', '').strip()
    return response.choices[0].text.strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, messages = None, n = 1, drop_arrow: bool = False) -> str:
    global completion_tokens, prompt_tokens, llm_call_count
    assert model != "text-davinci-003"
    if messages is None:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        n=n,
        temperature=temperature,
    )
    
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    llm_call_count += 1
    if n > 1:
        if drop_arrow:
            responses = [choice.message.content.replace('>', '').strip() for choice in response.choices]
        else:
            responses = [choice.message.content.strip() for choice in response.choices]
        return responses
    if drop_arrow:
        return response.choices[0].message.content.replace('>', '').strip()
    return response.choices[0].message.content.strip()

def llm_response(prompt, model: Model, temperature: float = 0.0, max_tokens: int = 4096, stop_strs: Optional[List[str]] = None, n=1) -> str:
    
    
    # print("llm prompt: ", prompt)
    if isinstance(prompt, str):
        if model == 'gpt-3.5-turbo-instruct':
            content = get_completion(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, n=n)
        else:
            content = get_chat(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, n=n)
    else:
        messages = prompt
        prompt = prompt[1]['content']
        if model == 'gpt-3.5-turbo-instruct':
            content = get_completion(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, n=n)
        else:
            content = get_chat(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, messages=messages, n=n)
    # print("llm response: ", content)
    return content

def get_token_counts():
    global prompt_tokens, completion_tokens
    return prompt_tokens, completion_tokens

def get_llm_call_count():
    global llm_call_count
    return llm_call_count

def reset_token_counts():
    global prompt_tokens, completion_tokens, llm_call_count
    prompt_tokens = 0
    completion_tokens = 0
    llm_call_count = 0

def get_price(model_name, input_tokens=prompt_tokens, output_tokens=completion_tokens):
    if model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"]:
        return input_tokens * 1e-6 * 0.5 + output_tokens * 1e-6 * 1.5
    elif model_name == "gpt-4o":
        # gpt-4o (2024-08-06 snapshot): $2.5 / $10 per 1M input/output tokens.
        return input_tokens * 1e-6 * 2.5 + output_tokens * 1e-6 * 10
    elif model_name == "gpt-4o-mini":
        return input_tokens * 1e-6 * 0.15 + output_tokens * 1e-6 * 0.6
    elif model_name == "gpt-4":
        return 0.00003 * input_tokens + 0.00006 * output_tokens
    else:
        # Unknown model: contribute 0 to the cost estimate rather than aborting
        # the run mid-evaluation. Override get_price() for exact pricing.
        return 0.0
"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import os
import re
import time 
import json
from pathlib import Path
from openai import AzureOpenAI, OpenAI

from utils import *
from openai_cost_logger import DEFAULT_LOG_PATH
from persona.prompt_template.openai_logger_singleton import OpenAICostLogger_Singleton
from persona.prompt_template import env_file
from persona.prompt_template import llm_trace

config_path = Path("../../openai_config.json")
with open(config_path, "r") as f:
    openai_config = json.load(f) 

# The routed remote model's key lives in an environment variable named by the
# config, never in the config (RB-A of 006-IP). Read the .env before anything
# asks for a key; an already-set variable is left alone.
env_file.load()


class BackendTransportError(Exception):
  """The backend did not answer: a 402, a 429, a timeout, a refused
  connection. Deliberately NOT the same thing as a model answering badly.

  Before 006-IP, GPT_request answered a transport failure with the string
  "TOKEN LIMIT EXCEEDED" and ChatGPT_request with "ChatGPT ERROR", and handed
  it to the caller's validator. The validators for the closed-set decisions
  accept any non-empty string, so on the sector and arena prompts that string
  was a well-formed answer: a dead backend read as a live simulation making
  bad choices. This exception is what makes the two distinguishable, and the
  retry loops re-raise it rather than spend an attempt on it (RB-C).
  """

  @property
  def reason(self):
    """What the trace files this under. The wrapper's own type says nothing;
    the type underneath says whether it was a 402, a 429 or a DNS error."""
    cause = self.__cause__ or self.__context__
    return "transport:%s" % (type(cause).__name__ if cause else "unknown")


def _resolve_key(config, key_field, env_field):
  """The key for one client: from the environment when the config names a
  variable, from the config otherwise.

  Naming a variable is a commitment: if it is unset, this raises rather than
  fall through to `key_field` and send an unauthenticated request whose 401
  the caller would read as an answer.
  """
  env_name = str(config.get(env_field) or "").strip()
  if not env_name:
    return config.get(key_field)
  if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
    raise ValueError(
        'openai_config.json: "%s" should be the NAME of an environment '
        "variable (e.g. OPEN_ROUTER_API), not the key itself. The key goes "
        'in "%s", or better, in the environment.' % (env_field, key_field))
  value = os.environ.get(env_name, "").strip()
  if not value:
    raise ValueError(
        'openai_config.json names "%s": "%s", and %s is not set in the '
        "environment%s. Set it, or put it in a .env beside the fork or the "
        "lab." % (env_field, env_name, env_name,
                  " (read: %s)" % ", ".join(env_file.loaded())
                  if env_file.loaded() else " (no .env was found)"))
  return value


def provider_pin():
  """The `provider` block sent with every request to the router, or None.

  One entry in `order` with `allow_fallbacks: false` is what makes a remote
  run one condition rather than an average over whoever was cheapest that
  minute: the weights, the quantization and the serving stack all move when
  the provider does. It is sent only when configured, so the Ollama path
  keeps sending exactly the body it sent before this plan (law 2).
  """
  pin = openai_config.get("model-provider")
  return pin if isinstance(pin, dict) and pin else None


def _extra_body():
  """What goes beside the OpenAI-shaped fields, or None when nothing does."""
  pin = provider_pin()
  return {"provider": pin} if pin else None

def setup_client(type: str, config: dict):
  """Setup the OpenAI client.

  Args:
      type (str): the type of client. Either "azure" or "openai".
      config (dict): the configuration for the client.

  Raises:
      ValueError: if the client is invalid.

  Returns:
      The client object created, either AzureOpenAI or OpenAI.
  """
  if type == "azure":
    client = AzureOpenAI(
        azure_endpoint=config["endpoint"],
        api_key=config["key"],
        api_version=config["api-version"],
    )
  elif type == "openai":
    # base_url is optional: None -> api.openai.com. Set it to point at any
    # OpenAI-compatible server (Ollama: http://localhost:11434/v1, vLLM, ...).
    client = OpenAI(
        api_key=config["key"],
        base_url=config.get("base_url"),
    )
  else:
    raise ValueError("Invalid client")
  return client

if openai_config["client"] == "azure":
  client = setup_client("azure", {
      "endpoint": openai_config["model-endpoint"],
      "key": openai_config["model-key"],
      "api-version": openai_config["model-api-version"],
  })
elif openai_config["client"] == "openai":
  client = setup_client("openai", {
      "key": _resolve_key(openai_config, "model-key", "model-key-env"),
      "base_url": openai_config.get("model-base-url"),
  })
else:
  raise ValueError("Invalid client")

if openai_config["embeddings-client"] == "azure":  
  embeddings_client = setup_client("azure", {
      "endpoint": openai_config["embeddings-endpoint"],
      "key": openai_config["embeddings-key"],
      "api-version": openai_config["embeddings-api-version"],
  })
elif openai_config["embeddings-client"] == "openai":
  embeddings_client = setup_client("openai", {
      "key": _resolve_key(openai_config, "embeddings-key",
                          "embeddings-key-env"),
      "base_url": openai_config.get("embeddings-base-url"),
  })
else:
  raise ValueError("Invalid embeddings client")

cost_logger = OpenAICostLogger_Singleton(
  experiment_name = openai_config["experiment-name"],
  log_folder = DEFAULT_LOG_PATH,
  cost_upperbound = openai_config["cost-upperbound"]
)


# The prompts sent through GPT_request were written for the legacy Completions
# API (they end mid-sentence: "... Activity: Isabella is"). Chat models, and
# open models in particular (llama3.1 via Ollama), answer them as a chatbot
# ("Here is the completed schedule:") unless told to behave as a completion
# engine. Override with the optional "completion-system-prompt" key in
# openai_config.json (set it to "" to disable).
DEFAULT_COMPLETION_SYSTEM_PROMPT = (
  "You are a text-completion engine, not a chat assistant. The user message is "
  "the beginning of a document. Continue it from exactly where it stops. Output "
  "only the continuation text, with no preamble, no explanation, no quotes and "
  "no restating of the input."
)
completion_system_prompt = openai_config.get("completion-system-prompt", DEFAULT_COMPLETION_SYSTEM_PROMPT)


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def ChatGPT_single_request(prompt): 
  temp_sleep()
  started = time.time()
  try:
    completion = client.chat.completions.create(
      model=openai_config["model"],
      messages=[{"role": "user", "content": prompt}],
      extra_body=_extra_body(),
    )
  except Exception as e:
    llm_trace.record_call("ChatGPT_single_request", openai_config["model"],
                          prompt, latency=time.time() - started, error=e)
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e
  llm_trace.record_call("ChatGPT_single_request", openai_config["model"], prompt,
                        response=completion, latency=time.time() - started)
  cost_logger.update_cost(completion, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
  return completion.choices[0].message.content


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  started = time.time()
  try: 
    completion = client.chat.completions.create(
    model=openai_config["model"],
    messages=[{"role": "user", "content": prompt}],
    extra_body=_extra_body(),
    )
    llm_trace.record_call("ChatGPT_request", openai_config["model"], prompt,
                          response=completion, latency=time.time() - started)
    cost_logger.update_cost(completion, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
    return completion.choices[0].message.content
  
  except Exception as e: 
    llm_trace.record_call("ChatGPT_request", openai_config["model"], prompt,
                          latency=time.time() - started, error=e)
    print(f"Error: {e}")
    # Was: return "ChatGPT ERROR". A string is an answer, and the closed-set
    # validators accept any non-empty one (RB-C).
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  label = llm_trace.caller_label()
  started = time.time()
  for i in range(repeat): 

    curr_gpt_response = None
    llm_trace.set_attempt(i + 1)
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        cleaned_response = func_clean_up(curr_gpt_response, prompt=prompt)
        llm_trace.record_outcome("ChatGPT_safe_generate_response", i + 1,
                                 False, True, latency=time.time() - started,
                                 label=label)
        return cleaned_response
      
      llm_trace.record_reject("validate_false", curr_gpt_response, label=label)
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except BackendTransportError as e:
      # Not an attempt: nothing was answered, so there is nothing to retry
      # against and nothing the next two attempts would learn (RB-C).
      llm_trace.record_reject(e.reason, curr_gpt_response, label=label)
      raise
    except: 
      llm_trace.record_reject(llm_trace.reject_reason_from_exception(),
                              curr_gpt_response, label=label)
      pass

  llm_trace.record_outcome("ChatGPT_safe_generate_response", repeat, True,
                           False, latency=time.time() - started, label=label)
  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  label = llm_trace.caller_label()
  started = time.time()
  for i in range(repeat): 
    curr_gpt_response = None
    llm_trace.set_attempt(i + 1)
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        cleaned_response = func_clean_up(curr_gpt_response, prompt=prompt)
        llm_trace.record_outcome("ChatGPT_safe_generate_response_OLD", i + 1,
                                 False, True, latency=time.time() - started,
                                 label=label)
        return cleaned_response
      llm_trace.record_reject("validate_false", curr_gpt_response, label=label)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except BackendTransportError as e:
      llm_trace.record_reject(e.reason, curr_gpt_response, label=label)
      raise
    except: 
      llm_trace.record_reject(llm_trace.reject_reason_from_exception(),
                              curr_gpt_response, label=label)
      pass
  llm_trace.record_outcome("ChatGPT_safe_generate_response_OLD", repeat, True,
                           False, latency=time.time() - started, label=label)
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  started = time.time()
  try: 
    # The prompt goes as "user" (a system-only conversation makes llama3.1 echo
    # the role token "assistant"); the completion-engine instruction goes as
    # "system" so the model continues the text instead of chatting about it.
    messages = []
    if completion_system_prompt:
      messages.append({"role": "system", "content": completion_system_prompt})
    messages.append({"role": "user", "content": prompt})
    # The "engine" in gpt_parameter is a legacy hardcoded name
    # ("gpt-35-turbo-0125" in run_gpt_prompt.py); the model actually used is
    # always the one from openai_config.json so any backend works.
    response = client.chat.completions.create(
                model=openai_config["model"],
                messages=messages,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],
                extra_body=_extra_body(),)
    llm_trace.record_call("GPT_request", openai_config["model"], prompt,
                          response=response, latency=time.time() - started)
    cost_logger.update_cost(response=response, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
    return response.choices[0].message.content
  except Exception as e:
    llm_trace.record_call("GPT_request", openai_config["model"], prompt,
                          latency=time.time() - started, error=e)
    print(f"Error: {e}")
    # Was: return "TOKEN LIMIT EXCEEDED" -- a string the retry loop spent an
    # attempt on and the validator, in five of the paper's closed-set
    # prompts, accepted (RB-C).
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  label = llm_trace.caller_label()
  started = time.time()
  for i in range(repeat): 
    llm_trace.set_attempt(i + 1)
    try:
      curr_gpt_response = GPT_request(prompt, gpt_parameter)
    except BackendTransportError as e:
      llm_trace.record_reject(e.reason, None, label=label)
      raise
    try:
      if func_validate(curr_gpt_response, prompt=prompt): 
        cleaned_response = func_clean_up(curr_gpt_response, prompt=prompt)
        llm_trace.record_outcome("safe_generate_response", i + 1, False, True,
                                 latency=time.time() - started, label=label)
        return cleaned_response
      llm_trace.record_reject("validate_false", curr_gpt_response, label=label)
      if verbose: 
        print ("---- repeat count: ", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")
    except:
      llm_trace.record_reject(llm_trace.reject_reason_from_exception(),
                              curr_gpt_response, label=label)
      pass
  # The fail-safe is what makes a broken LLM look like a live simulation
  # (a plausible "yes" or 4), so every activation has to show up in the trace.
  llm_trace.record_outcome("safe_generate_response", repeat, True, False,
                           latency=time.time() - started, label=label)
  return fail_safe_response


def get_embedding(text, model=openai_config["embeddings"]):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  started = time.time()
  try:
    response = embeddings_client.embeddings.create(input=[text], model=model)
  except Exception as e:
    llm_trace.record_call("embedding", model, text,
                          latency=time.time() - started, error=e)
    raise
  llm_trace.record_call("embedding", model, text, response=response,
                        latency=time.time() - started)
  cost_logger.update_cost(response=response, input_cost=openai_config["embeddings-costs"]["input"], output_cost=openai_config["embeddings-costs"]["output"])
  return response.data[0].embedding


if __name__ == '__main__':
  gpt_parameter = {"engine": openai_config["model"], "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)



















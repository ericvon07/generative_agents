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
from openai import (AzureOpenAI, OpenAI, RateLimitError, APITimeoutError,
                    APIConnectionError)

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


def _traced_pin():
  """The pin as the trace records it, or nothing at all.

  A pin is what the request ASKED for; the provider on the response is what
  answered. Both go on the call so a reader can see them disagree (RB-E).
  When there is no pin the key is absent rather than null, so a trace from a
  local run looks exactly as it did before this plan.
  """
  pin = provider_pin()
  return {"provider_pin": pin} if pin else None

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
    # "timeout" is set for the generator only (011-IP, the night of the
    # windows): the SDK's default is 600 s, and a request the router left
    # hanging held a run for five minutes before it came back empty.
    kwargs = {"timeout": config["timeout"]} if config.get("timeout") else {}
    client = OpenAI(
        api_key=config["key"],
        base_url=config.get("base_url"),
        **kwargs
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
      "timeout": openai_config.get("model-timeout"),
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


def _charge(response, costs=None):
  """Book one response against the run's budget.

  RB-B: when the backend reports what the call actually cost -- OpenRouter
  puts it in usage.cost, at the provider that actually served it, after
  whatever discount applied -- that number is the ledger's, and the price
  table is not consulted. "model-costs" stays for a backend that reports
  nothing, which is every local one, and there it is correctly 0.

  The ledger prices per million tokens, so a reported cost goes in as the
  blended rate that reproduces it exactly (cost = rate x total / 1e6). That
  rate is not a guess: it is what this call was charged at, which is the
  honest thing for the breakdown to show.

  Deliberately outside the try that wraps the request. update_cost raises
  when the run passes its cost-upperbound, and the bill guard firing is not
  a transport failure -- wrapping it would stop the run under the wrong name
  and write a second call record saying the backend errored when it did not.
  """
  costs = costs or openai_config["model-costs"]
  reported = llm_trace.response_facts(response).get("cost")
  total = getattr(getattr(response, "usage", None), "total_tokens", None)
  if reported is not None and total:
    rate = reported * 1e6 / total
    cost_logger.update_cost(response, input_cost=rate, output_cost=rate)
    return
  cost_logger.update_cost(response, input_cost=costs["input"],
                          output_cost=costs["output"])


# --- the ride-out -----------------------------------------------------------
# SV-A of 008-IP. RB-C says a transport failure raises rather than become a
# fail-safe, and rtrA-llm showed what that costs when the failure is transient:
# two 429s from DeepInfra, 24 seconds apart, ended two stages of a 32-minute
# run, and only the automatic runner's own restart carried the window to its
# end. RB-C's reopening clause named the successor, and this is it -- a bounded
# wait, every wait in the trace, and the SAME exception at the end of it.
#
# Deliberately not the paper's retry loop. That loop retries the PARSING of an
# answer that arrived, five times, and spends an attempt each time. This
# retries a request that was never served, spends no attempt, and is invisible
# to every validator.

_RATE_LIMIT_DEFAULTS = {"attempts": 5, "base-delay": 2.0, "max-delay": 60.0}


def _rate_limit_policy(config):
  """How long the run is willing to wait out a provider's bad minute.

  Five waits of 2, 4, 8, 16 and 32 seconds is just over a minute of patience,
  against a provider whose own message says "please retry shortly". Past that
  the limit is not a bad minute and the run should stop and say so.
  """
  policy = dict(_RATE_LIMIT_DEFAULTS)
  policy.update({k: v for k, v in (config.get("rate-limit") or {}).items()
                 if v is not None})
  return {"attempts": int(policy["attempts"]),
          "base_delay": float(policy["base-delay"]),
          "max_delay": float(policy["max-delay"])}


rate_limit_policy = _rate_limit_policy(openai_config)

# Swapped by the tests. The wait is the whole point of this code, so it cannot
# be tested by waiting.
_sleep = time.sleep


def _status_of(e):
  """The HTTP status behind an exception, or None. The OpenAI client puts it
  on the exception; anything else is asked for its response."""
  status = getattr(e, "status_code", None)
  if status is None:
    status = getattr(getattr(e, "response", None), "status_code", None)
  try:
    return None if status is None else int(status)
  except Exception:
    return None


def _is_rate_limit(e):
  """A 429, by the client's type or by the status underneath it. Both, because
  the type is the client's reading of the status and a router that answers 429
  through some other path is still a router saying wait."""
  return isinstance(e, RateLimitError) or _status_of(e) == 429


class EmptyResponse(Exception):
  """The router answered 200 with no choices and no usage.

  Seen twice in the first hour of 011-IP's typed window, each after exactly
  ~300 s: the router's own cut of a request the provider left hanging. Nothing
  came back that anyone could read, so it is ridden out like a 429 -- the
  paper's code used to reach the cost ledger with it and die there.
  """


# The statuses no amount of waiting fixes: a malformed request, a key, a
# balance, a permission, a model that does not exist. Everything else a
# provider can say is ridden out -- the night of 011-IP's windows Crusoe
# answered 429, 502 and 412 ("no available servers") in turn, and each one
# ended a window the first time it was met.
_PERMANENT = (400, 401, 402, 403, 404)


def _is_transient(e):
  """A request that got no answer, or an answer that says come back later:
  the router's empty body, the client's timeout, a dropped connection, and
  any status that is not in _PERMANENT. Ridden out from the same budget as
  a 429 and recorded the same way; when the budget runs out it raises,
  unchanged, exactly as a 429 does."""
  if isinstance(e, (EmptyResponse, APITimeoutError, APIConnectionError)):
    return True
  status = _status_of(e)
  return status is not None and status not in _PERMANENT


def _retry_after(e):
  """The provider's own instruction, when it sends one. Preferred over the
  backoff: a number the provider chose beats a number we guessed."""
  try:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    value = headers.get("retry-after", headers.get("Retry-After"))
    return None if value is None else max(0.0, float(value))
  except Exception:
    return None


class _Attempt(object):
  """How long the request that finally answered actually took.

  The ride-out can put a minute of sleeping between the first request and the
  one that answers. A latency_ms that included it would turn a provider's bad
  minute into a model that got slow -- and the mean latency of a call is the
  unit every wall-clock estimate in this lab is built from (rtrA-llm: 1.17 s
  over 1,218 calls). The waits are their own records; this is the request
  alone.
  """

  def __init__(self):
    self.latency = None
    self._started = None

  def start(self):
    self._started = time.time()

  def stop(self):
    if self._started is not None:
      self.latency = time.time() - self._started


def _send(path, attempt, create):
  """Make one request, riding out a rate limit.

  Returns whatever <create> returns. Raises whatever it raised, unchanged,
  once the ride-out's budget is spent -- so the caller's BackendTransportError
  and the reject that reads "transport:RateLimitError" are exactly the ones
  006-IP wrote, and no fail-safe reaches the world on any path (RB-C, S1.3).

  The backoff is deterministic: no jitter, because the run seeds `random` for
  reproducibility and a wait is not allowed to draw from that stream.
  """
  policy = rate_limit_policy
  waits = 0
  while True:
    attempt.start()
    try:
      response = create()
      attempt.stop()
      if not getattr(response, "choices", None):
        raise EmptyResponse("the router answered with no choices")
      return response
    except Exception as e:
      attempt.stop()
      if (not (_is_rate_limit(e) or _is_transient(e))
          or waits >= policy["attempts"]):
        raise
      waits += 1
      delay = _retry_after(e)
      if delay is None:
        delay = min(policy["max_delay"],
                    policy["base_delay"] * (2 ** (waits - 1)))
      llm_trace.record_rate_limit(path, status=_status_of(e) or
                                  (429 if _is_rate_limit(e) else None),
                                  attempt=waits, of=policy["attempts"],
                                  delay=delay, error=e)
      print("rate limit on %s: waiting %.1fs (%d of %d)"
            % (path, delay, waits, policy["attempts"]))
      _sleep(delay)


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def ChatGPT_single_request(prompt): 
  temp_sleep()
  attempt = _Attempt()
  try:
    completion = _send(
        "ChatGPT_single_request", attempt,
        lambda: client.chat.completions.create(
            model=openai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            extra_body=_extra_body(),
        ))
  except Exception as e:
    llm_trace.record_call("ChatGPT_single_request", openai_config["model"],
                          prompt, latency=attempt.latency, error=e)
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e
  llm_trace.record_call("ChatGPT_single_request", openai_config["model"], prompt,
                        response=completion, latency=attempt.latency,
                        extra=_traced_pin())
  _charge(completion)
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
  attempt = _Attempt()
  # Only the request is inside the try. Anything after it -- the ledger, the
  # bill guard, reading the text off the response -- is this process's
  # business, not the backend's, and must not be reported as the backend
  # having errored.
  try: 
    completion = _send(
        "ChatGPT_request", attempt,
        lambda: client.chat.completions.create(
            model=openai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            extra_body=_extra_body(),
        ))
  except Exception as e: 
    llm_trace.record_call("ChatGPT_request", openai_config["model"], prompt,
                          latency=attempt.latency, error=e)
    print(f"Error: {e}")
    # Was: return "ChatGPT ERROR". A string is an answer, and the closed-set
    # validators accept any non-empty one (RB-C).
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e
  llm_trace.record_call("ChatGPT_request", openai_config["model"], prompt,
                        response=completion, latency=attempt.latency,
                        extra=_traced_pin())
  _charge(completion)
  return completion.choices[0].message.content


# Deviation D11 (011-IP, GT-8). gemma-4-31b-it wraps its JSON in a markdown
# fence, and the parser below cuts at the last "}" but never at the first "{",
# so a correct answer is thrown away -- 213 of 213 responses on the 011-IP
# shakedown, 0 of 76 once this sentence was added. The parser's own repair
# belongs to 009-IP; this asks the model instead.
#
# Off by default, so the paper's prompt is byte-identical without it. When on,
# it is on for EVERY prompt on this path, in both arms -- the generator's own
# prompts, and on the llm arm the decisions that travel this path (the three
# poignancy scores). The typed backend never sees it: a jev decision sends a
# state, not this prompt.
NO_FENCE_SENTENCE = ("Reply with the JSON object only: no markdown, "
                     "no code fence.")


def no_fence_enabled():
  """`"json-no-fence": {"enabled": true}` in the config, or JSON_NO_FENCE=1
  in the environment. Off when neither says otherwise."""
  flag = os.environ.get("JSON_NO_FENCE")
  if flag is not None:
    return flag.strip().lower() in ("1", "true", "yes", "on")
  block = openai_config.get("json-no-fence") or {}
  return bool(block.get("enabled", False))


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
  if no_fence_enabled():
    prompt += NO_FENCE_SENTENCE + "\n"
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
      
      llm_trace.record_reject(llm_trace.reject_reason_for_validation(),
                              curr_gpt_response, label=label)
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
      llm_trace.record_reject(llm_trace.reject_reason_for_validation(),
                              curr_gpt_response, label=label)
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
  attempt = _Attempt()
  # The prompt goes as "user" (a system-only conversation makes llama3.1 echo
  # the role token "assistant"); the completion-engine instruction goes as
  # "system" so the model continues the text instead of chatting about it.
  messages = []
  if completion_system_prompt:
    messages.append({"role": "system", "content": completion_system_prompt})
  messages.append({"role": "user", "content": prompt})
  # Only the request is inside the try: see ChatGPT_request.
  try: 
    # The "engine" in gpt_parameter is a legacy hardcoded name
    # ("gpt-35-turbo-0125" in run_gpt_prompt.py); the model actually used is
    # always the one from openai_config.json so any backend works.
    response = _send(
        "GPT_request", attempt,
        lambda: client.chat.completions.create(
            model=openai_config["model"],
            messages=messages,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"],
            extra_body=_extra_body(),))
  except Exception as e:
    llm_trace.record_call("GPT_request", openai_config["model"], prompt,
                          latency=attempt.latency, error=e)
    print(f"Error: {e}")
    # Was: return "TOKEN LIMIT EXCEEDED" -- a string the retry loop spent an
    # attempt on and the validator, in five of the paper's closed-set
    # prompts, accepted (RB-C).
    raise BackendTransportError(
        "%s: %s" % (type(e).__name__, e)) from e
  llm_trace.record_call("GPT_request", openai_config["model"], prompt,
                        response=response, latency=attempt.latency,
                        extra=_traced_pin())
  _charge(response)
  return response.choices[0].message.content


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


# --- the composed state (011-IP, TS-A) --------------------------------------
# A typed question hands the backend a `state`. Until now that state was the
# dictionary the call site read out of prompt_input, and the audit of
# 2026-09-23 found that five of the eight typed questions carry keys their
# instruction never names -- facts the paper states in a sentence of its own
# and the dictionary hands over as a bare key. This is the other mechanism
# 011-IP weighs against descriptive criteria: give the backend the paper's own
# sentences, rendered from the same prompt_input the prompt is built from.
#
# Two things are deliberately dropped:
#
#   the few-shot examples   jev_questions.py already gives the reason -- they
#                           teach a format a typed answer does not have, and
#                           they are where "kitchen" came from.
#   the answer cue          the trailing "Answer: {", "{", "Reasoning:" or
#                           "Reasoning: Let's think step by step." that tells
#                           an LLM where to write. A Choice writes nowhere.
#
# What is left is a SUBSTRING of the prompt the llm backend would have sent
# from the same prompt_input, character for character (S2.1). That is the
# whole point: the audit that checks this wording is a string comparison, not
# a reading. The split is done on the TEMPLATE, before substitution, so a
# rendered value that happens to contain a "---" line cannot move the seam.

# Longest first: "Answer: {" must win over "{", and "Reasoning: Let's think
# step by step." over "Reasoning:".
_ANSWER_CUES = (
    "Reasoning: Let's think step by step.",
    'Answer in "yes" or "no":',
    "Answer: {",
    "Reasoning:",
    "{",
)


def last_block(prompt_lib_file):
  """The template's live block: what follows its last few-shot separator.

  Returns the block with its !<INPUT n>! slots still in it, so the caller
  substitutes exactly as generate_prompt does.
  """
  with open(prompt_lib_file, "r") as f:
    prompt = f.read()
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return re.split(r"(?m)^-{3,}[ \t]*$", prompt)[-1]


def strip_answer_cue(block):
  """Remove the one trailing cue that tells an LLM where to write, if any."""
  text = block.rstrip()
  for cue in _ANSWER_CUES:
    if text.endswith(cue):
      return text[:-len(cue)].rstrip()
  return text


def composed_state(curr_input, prompt_lib_file):
  """The paper's own sentences about THIS state, as a string.

  Same arguments as generate_prompt, and built by the same substitution in
  the same order, so the two cannot drift: this is that function's last
  block with the answer cue taken off.

  ARGS:
    curr_input: the prompt_input the call site built (a list, or one string).
    prompt_lib_file: the path to the same template generate_prompt is given.
  RETURNS:
    a str -- a substring of generate_prompt(curr_input, prompt_lib_file).
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  block = last_block(prompt_lib_file)
  for count, i in enumerate(curr_input):
    block = block.replace(f"!<INPUT {count}>!", i)
  return strip_answer_cue(block).strip()


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
      llm_trace.record_reject(llm_trace.reject_reason_for_validation(),
                              curr_gpt_response, label=label)
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
  _charge(response, openai_config["embeddings-costs"])
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



















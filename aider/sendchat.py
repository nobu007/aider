import hashlib
import json

import backoff
from aider.dump import dump  # noqa: F401
from aider.llm import litellm

CACHE_PATH = "~/.aider.send.cache.v1"
CACHE = None

RETRY_TIMEOUT = 60


def retry_exceptions():
    import httpx

    return (
        # httpx
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        # litellm
        litellm.exceptions.BadRequestError,
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.UnprocessableEntityError,
        litellm.exceptions.RateLimitError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.ContentPolicyViolationError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.APIError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.Timeout,
    )


def lazy_litellm_retry_decorator(func):
    def wrapper(*args, **kwargs):
        decorated_func = backoff.on_exception(
            backoff.expo,
            retry_exceptions(),
            max_time=RETRY_TIMEOUT,
            on_backoff=lambda details: print(
                f"{details.get('exception', 'Exception')}\nRetry in {details['wait']:.1f} seconds."
            ),
        )(func)
        return decorated_func(*args, **kwargs)

    return wrapper


def send_completion(
    model_name,
    messages,
    functions,
    stream,
    temperature=0,
    extra_params=None,
):
    from aider.llm import litellm

    kwargs = dict(
        model=model_name,
        messages=messages,
        stream=stream,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    if functions is not None:
        function = functions[0]
        kwargs["tools"] = [dict(type="function", function=function)]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": function["name"]}}

    if extra_params is not None:
        kwargs.update(extra_params)

    key = json.dumps(kwargs, sort_keys=True).encode()
    hash_object = hashlib.sha1(key)

    if not stream and CACHE is not None and key in CACHE:
        return hash_object, CACHE[key]

    res = litellm.completion(**kwargs)

    if not stream and CACHE is not None:
        CACHE[key] = res

    return hash_object, res


@lazy_litellm_retry_decorator
def simple_send_with_retries(model_name, messages, extra_params=None, fallback_models=["claude-3-5-sonnet-20240620"]):
    if fallback_models is None:
        fallback_models = []

    models_to_try = [model_name] + fallback_models

    for model in models_to_try:
        try:
            kwargs = {
                "model_name": model_name,
                "messages": messages,
                "functions": None,
                "stream": False,
                "extra_params": extra_params,
            }
            _hash, response = send_completion(**kwargs)
            return response.choices[0].message.content
        except (AttributeError, litellm.exceptions.BadRequestError) as e:
            print(f"Error with model {model}: {str(e)}")
            if model == models_to_try[-1]:
                print("All models failed. Returning None.")
                return None
            print(f"Retrying with next model: {models_to_try[models_to_try.index(model) + 1]}")

    return None

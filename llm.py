from typing import Union
import json
from data import LLM_cohere, openai
import pickle
import hashlib
import os
import numpy as np


def embed(texts: Union[list[str], str], model="cohere"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]

    response = LLM_cohere.embed(
        texts=texts,
        model="embed-english-v2.0",
    )
    embeddings = response.embeddings
    return embeddings


def gpt(
    messages,
    model="gpt-4",  # "gpt-3.5-turbo-0613",
    temperature=0,
    stop=None,
    st=None,
) -> str:
    stream = True if st else False
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop,
        stream=stream,
    )
    args = ([msg["content"] for msg in messages], model, temperature, stop)
    result = check_cache(args)
    # result = None
    if result:
        return result

    else:
        if stream:
            with st.sidebar:
                with st.chat_message("assistant", avatar="👨‍🏫"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

        else:
            full_response = response.choices[0].message.content

        write_to_cache(full_response, args)

        return full_response


def check_cache(*args):
    hash_object = hashlib.md5(str(args).encode())
    filename = hash_object.hexdigest()
    cache_dir = "cache"

    if filename in os.listdir(cache_dir):
        with open(os.path.join(cache_dir, filename), "rb") as f:
            return pickle.load(f)
    else:
        return None


def write_to_cache(data, *args):
    hash_object = hashlib.md5(str(args).encode())
    filename = hash_object.hexdigest()
    cache_dir = "cache"

    with open(os.path.join(cache_dir, filename), "wb") as f:
        pickle.dump(data, f)

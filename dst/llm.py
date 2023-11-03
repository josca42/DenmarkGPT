from typing import Union
import json
from dst.data import LLM_cohere, openai
import pickle
import hashlib
import os
import numpy as np
import time
from dst.db import crud, models
from tenacity import retry, wait_random_exponential, stop_after_attempt


def embed(
    texts: Union[list[str], str], lang, small=False, input_type="search_document"
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    if small:
        model = (
            "embed-english-light-v3.0"
            if lang == "en"
            else "embed-multilingual-light-v3.0"
        )
    else:
        model = "embed-english-v2.0" if lang == "en" else "embed-multilingual-v3.0"
    response = LLM_cohere.embed(
        texts=texts,
        model=model,
        input_type=input_type if not model == "embed-english-v2.0" else None,
    )
    embeddings = response.embeddings
    return embeddings


# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def gpt(
    messages,
    model="gpt-4",  # "gpt-3.5-turbo-0613",
    temperature=0,
    stop=None,
    st=None,
    setting_info={},
) -> str:
    stream = True if st else False
    # Cache responses
    if setting_info:
        cached_response = check_db(setting_info)
    else:
        cached_response = None

    if cached_response is None:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stop=stop,
            stream=stream,
        )
    else:
        response = cached_response

    if stream:
        with st.sidebar:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                if cached_response:
                    response_chunks = [
                        cached_response[i : i + 10]
                        for i in range(0, len(cached_response), 10)
                    ]
                    full_response = ""
                    for chunk in response_chunks:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    message_placeholder.markdown(full_response)
                else:
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
    else:
        full_response = (
            response if cached_response else response.choices[0].message.content
        )

    if cached_response is None and setting_info:
        write_to_db(full_response, setting_info)

    return full_response


def check_db(settings_info):
    if settings_info["lang"] == "en":
        model_llm, crud_llm = models.LLM_EN, crud.llm_en
    else:
        model_llm, crud_llm = models.LLM_DA, crud.llm_da

    settings_info_without_lang = {k: v for k, v in settings_info.items() if k != "lang"}
    model_obj = model_llm(**settings_info_without_lang)
    db_row = crud_llm.get(model_obj)
    return db_row.response if db_row else None
    # hash_object = hashlib.md5(str(args).encode())
    # filename = hash_object.hexdigest()
    # cache_dir = "cache"

    # if filename in os.listdir(cache_dir):
    #     with open(os.path.join(cache_dir, filename), "rb") as f:
    #         return pickle.load(f)
    # else:
    #     return None


def write_to_db(full_response, settings_info):
    if settings_info["lang"] == "en":
        model_llm, crud_llm = models.LLM_EN, crud.llm_en
    else:
        model_llm, crud_llm = models.LLM_DA, crud.llm_da

    settings_info_without_lang = {k: v for k, v in settings_info.items() if k != "lang"}
    model_obj = model_llm(response=full_response, **settings_info_without_lang)
    crud_llm.create(model_obj)

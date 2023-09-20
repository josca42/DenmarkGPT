from typing import Union
import json
from data import LLM_cohere, openai


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

    return full_response


if __name__ == "__main__":
    t = match_table("How has unemployment evolved?")

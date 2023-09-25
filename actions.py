import json
from jinja2 import Template
from llm import gpt
from query import find_table
from data import DST_SUBJECTS_INDEX_0_1


def match_action(user_input):
    msgs = [
        dict(role="system", content=ACTION_SYS_MSG),
        dict(role="user", content=user_input),
    ]
    response_txt = gpt(messages=msgs, model="gpt-3.5-turbo", temperature=0)
    return int(response_txt)


def explore_dst_data(user_input, st=None):
    _, table_ids = find_table(query=user_input, k=8)
    msgs = [
        dict(
            role="system",
            content=RESEARCH_SYS_MSG.render(
                subjects=json.dumps(DST_SUBJECTS_INDEX_0_1, ensure_ascii=False)
            ),
        ),
        dict(role="user", content=user_input),
    ]
    response_txt = gpt(messages=msgs, model="gpt-4", temperature=1, st=st)
    return response_txt, table_ids


def update_api_request_specs():
    # Dashboard state
    ...


###   Prompts   ###
ACTION_SYS_MSG = """Your job is to decide the type of request made by the user.

The request can be one of 3 types. 

1 - Specific question or query that should be answered.
2 - Exploratory request about the tables or information available at Denmark statistics. Input to how to go about exploring a topic using the data available.
3 - Update or change to existing request.

Output the request type number."""

RESEARCH_SYS_MSG = Template(
    """Your are a world research assistant from Denmarks statistics. Your task is to help users get started with finding answers to their questions.

In general Denmarks statistics contain information about the following highlevel areas:
{{ subjects }}

During your message bring the users attention to the tree graph to the right that shows all the different subjects and their associated tables that the user has access to. The user can checkbox the tables that should be used when asking new questions. Also tell the user that you have highlighted a few tables that might contain useful information in the below tree graph.

Do not advice the user on what he or she should or should not do"""
)

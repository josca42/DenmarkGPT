import json
from jinja2 import Template
from llm import gpt
from query import find_table
from data import DST_SUBJECTS_INDEX_0_1


def match_action(user_input, prev_table_descr, prev_api_request, lang, setting_info):
    sys_msg = dict(role="system", content=ACTION_SYS_MSG.render(lang=lang))
    example_msgs = ACTION_MSG_EN_EXAMPLE if lang == "en" else ACTION_MSG_DA_EXAMPLE
    user_msg = dict(
        role="user",
        content=ACTION_USER_MSG.render(
            table_description=prev_table_descr,
            api_request=prev_api_request,
            user_request=user_input,
        ),
    )
    response_txt = gpt(
        messages=[sys_msg] + example_msgs + [user_msg],
        model="gpt-3.5-turbo",
        temperature=0,
        setting_info=setting_info,
    )
    # print(response_txt)
    result = response_txt.split("-")
    if len(result) == 1:
        table_description = ""
    else:
        table_description = result[1].strip()
    action_type = int(result[0].strip())
    return action_type, table_description


def explore_dst_data(user_input, lang, setting_info, st=None):
    _, table_ids = find_table(query=user_input, lang=lang, k=8)
    msgs = [
        dict(
            role="system",
            content=RESEARCH_SYS_MSG.render(
                subjects=json.dumps(DST_SUBJECTS_INDEX_0_1, ensure_ascii=False),
                lang=lang,
            ),
        ),
        dict(role="user", content=user_input),
    ]
    response_txt = gpt(
        messages=msgs, model="gpt-4", temperature=1, st=st, setting_info=setting_info
    )
    return response_txt, table_ids


def update_api_request_specs():
    # Dashboard state
    ...


###   Prompts   ###
# FIXME
ACTION_SYS_MSG = Template(
    """Your job is to decide the type of request made by the user. The requests are made in order to extract information from tables at Denmarks Statistics.

The request can be one of 3 types. 

1 - Specific question or query that should be answered.
2 - Exploratory request about the tables or information available at Denmark statistics. Input to how to go about exploring a topic using the data available.
3 - Update or change to existing request.

To help you decide the type of request you are provided with the previous API request and the table that API request was made to. If no previous request has been made then the input is an empty string

Output the request type number. If the request type is of type 1 or 2 then add an ideal table description. An example of a table description is: Full-time unemployed persons by region, country of origin, sex and time.
{% if lang == "da" %}
Write the table description in danish.
{% endif %}"""
)

ACTION_MSG_EN_EXAMPLE = [
    dict(
        role="user",
        content="""Previous table description: 
Previous API request:
User request: Propose how to go about analysing unemployment""",
    ),
    dict(
        role="assistant",
        content="2 - Unemployment rate by region, age group, education level, and time.",
    ),
]

ACTION_MSG_DA_EXAMPLE = [
    dict(
        role="user",
        content="""Previous table description: 
Previous API request:
User request: Foreslå, hvordan man kan analysere udviklingen i antallet af fuldtidsledige""",
    ),
    dict(
        role="assistant",
        content="2 - Fuldtidsledige i pct. af arbejdsstyrken efter område, alder, køn og tid",
    ),
]

ACTION_USER_MSG = Template(
    """Previous table description: {{ table_description }}
Previous API request: {{ api_request }}
User request: {{ user_request }}"""
)

RESEARCH_SYS_MSG = Template(
    """Your are a world class research assistant from Denmarks statistics. Your task is to help users get started with finding answers to their questions.

In general Denmarks statistics contain information about the following highlevel areas:
{{ subjects }}

During your message bring the users attention to the tree graph to the right that shows all the different subjects and their associated tables that the user has access to. The user can checkbox the tables that should be used when asking new questions. Also tell the user that you have highlighted a few tables that might contain useful information in the tree graph to the right.

Do not advice the user on what he or she should or should not do.
{% if lang == "da" %}
Write the message in danish.
{% endif %}"""
)

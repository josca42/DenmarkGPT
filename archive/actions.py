def explore_dst_data(user_input, lang, setting_info, st=None):
    _, table_ids = find_table_candidates(query=user_input, lang=lang, k=8)
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


RESEARCH_SYS_MSG = Template(
    """Your are a world class research assistant from Denmarks statistics. Your task is to help users get started with finding answers to their questions.

In general Denmarks statistics contain information about the following highlevel areas:
{{ subjects }}

Mention you have highlighted a few subjects and tables in the tree graph to the right that gives an overview of all the available tables in Denmarks Statistics. Mention that the user can checkbox the tables that should be used when asking new questsion. In order for the selections to take effect the user must click the "use selected tables" button.

Do not advice the user on what he or she should or should not do.
{% if lang == "da" %}
Write the message in danish.
{% endif %}"""
)

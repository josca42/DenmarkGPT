import pandas as pd
from jinja2 import Template
import plotly.express as px
from llm import gpt
import ast


def create_px_plot(df, prompt, data_descr, variables, st=None):
    if st.session_state.new_prompt:
        # Create plotly express code
        msgs = [
            dict(role="system", content=PX_PLOT_SYS_MSG),
            dict(
                role="user",
                content=PX_PLOT_USER_MSG.render(
                    question=prompt, description=data_descr, variables=variables
                ).strip(),
            ),
        ]
        response_txt = gpt(messages=msgs, model="gpt-4", temperature=0, st=st)
    else:
        response_txt = st.session_state.plot_code_str

    if "TID" in df.columns:
        df.sort_values("TID", inplace=True)

    # Use ast to execute the code and extract the variable `fig`
    node = ast.parse(response_txt)
    local_namespace = {"df": df, "px": px}
    exec(compile(node, "<ast>", "exec"), local_namespace)
    fig = local_namespace.get("fig")

    return fig, response_txt


###   Prompts  ###
PX_PLOT_SYS_MSG = """You are a world-class data scientist. You code in Python and use plotly express to create data vizualisations. 

When writing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.

You job is to create a data visualisation that helps answer a question with a provided dataset with the name df.

You get the user question and information about the dataset that you have access to, on the following form:

Question: "User question"
Dataset: "Dataset description"
Variables: [{"name": "variable name", "text": variable description}, ... ]

Assume plotly.express is imported as px and that you have access to the dataset in the pandas dataframe, df. The y variable in the plots will always be named "y" and it contains the relevant values. If the dataset should be sorted by certain variables make sure to do so. Never subset or filter the dataset df.
Do not write fig.show()."""

PX_PLOT_USER_MSG = Template(
    """Question: {{ question }}
Dataset: {{ description }}
Variables: {{ variables }}"""
)

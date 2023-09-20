import pandas as pd
from jinja2 import Template
import plotly.express as px
from llm import gpt
import ast

# df = pd.read_parquet("./data/subjects.parquet")


# fig = go.Figure(
#     go.Icicle(
#         labels=df["id"],
#         parents=df["parent"],
#         text=df["description"],
#         # maxdepth=,
#     )
# )
# fig = fig.update_layout(margin=dict(b=0, l=0, r=0))


def create_px_plot(df, question, data_descr, variables):
    # Create plotly express code
    msgs = [
        dict(role="system", content=PX_PLOT_SYS_MSG),
        dict(
            role="user",
            content=PX_PLOT_USER_MSG.render(
                question=question, description=data_descr, variables=variables
            ).strip(),
        ),
    ]
    # response_txt = gpt(messages=msgs, model="gpt-4", temperature=0)
    df["y"] = df["INDHOLD"]
    response_txt = """fig = px.bar(df, x='OMRÅDE', y='y', title='Deficit of Municipalities', labels={'OMRÅDE':'Municipality', 'y':'Deficit'})
fig.update_layout(xaxis={'categoryorder':'total descending'})"""

    # Use ast to execute the code and extract the variable `fig`
    node = ast.parse(response_txt)
    local_namespace = {"df": df, "px": px}
    exec(compile(node, "<ast>", "exec"), local_namespace)
    fig = local_namespace.get("fig")

    return fig


###   Prompts  ###
PX_PLOT_SYS_MSG = """You are a world-class data scientist. You code in Python and use plotly express to create data vizualisations. 

When writing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.

You job is to create a data visualisation that helps answer a question with a provided dataset with the name df.

You get the user question and information about the dataset that you have access to, on the following form:

Question: "User question"
Dataset: "Dataset description"
Variables: [{"name": "variable name", "text": variable description}, ... ]

Assume plotly.express is imported as px and that you have access to the dataset in the pandas dataframe, df. The y variable in the plots will always be named "y" and it contains the relevant values.
Do not write fig.show()."""

PX_PLOT_USER_MSG = Template(
    """Question: {{ question }}
Dataset: {{ description }}
Variables: {{ variables }}"""
)

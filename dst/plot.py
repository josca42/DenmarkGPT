import pandas as pd
from jinja2 import Template
import plotly.express as px
from dst.llm import gpt
import ast
import matplotlib.pyplot as plt
from colorspacious import cspace_converter
import numpy as np
import bisect
import pydeck as pdk
from dst.data import gdf, KOMMUNER_ID, REGIONER_ID


def px_fig(df, prompt, metadata_df, variables, setting_info, st):
    setting_info["prev_request_table"] = ""
    setting_info["prev_request_api"] = ""
    setting_info["query_type"] = 4

    filters = []
    for var, vals in metadata_df["specs"].items():
        if vals != ["*"]:
            id2text = {v["id"]: v["text"] for v in metadata_df[var]["values"]}
            vals = [id2text[val] for val in vals if val in id2text]
            filters.append(
                f"{var} in {tuple(vals)}" if len(vals) > 1 else f"{var} = '{vals[0]}'"
            )
    filters_stmt = "WHERE " + " AND ".join(filters)

    for var in variables:
        var["n_unique"] = df[var["name"]].nunique()

    setting_info["action_type"] = 4
    if st.session_state.new_prompt:
        msgs = [
            dict(role="system", content=PX_PLOT_SYS_MSG),
            dict(
                role="user",
                content=PX_PLOT_USER_MSG.render(
                    question=prompt,
                    description=metadata_df["description"],
                    variables=variables,
                    filters=filters_stmt,
                ).strip(),
            ),
        ]
        response_txt = gpt(
            messages=msgs,
            model="gpt-4",
            temperature=0,
            setting_info=setting_info,
        )
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


def map(df, metadata, geo_type, color_theme="plasma", n_colors=20):
    def subset_df_to_one_val_pr_geo(df, metadata):
        df = df.copy()
        geo_var = metadata["geo"]["var"]
        if "TID" in metadata:
            if len(df["TID"].unique()) > 1:
                df = df[df["TID"] == df["TID"].max()]
            grouping_cols = [geo_var, "TID"]
        else:
            grouping_cols = [geo_var]

        df = df.groupby(grouping_cols)["y"].sum().reset_index()
        return df

    def merge_data_onto_map(df, metadata):
        text2id = metadata["geo"]["id2text_mapping"]
        df["geo_id"] = df[metadata["geo"]["var"]].map(text2id)
        merge_cols = ["geo_id", "y", "TID"] if "TID" in df.columns else ["geo_id", "y"]
        gdf_plot = gdf.merge(df[merge_cols], on="geo_id", how="inner")
        return gdf_plot

    df = subset_df_to_one_val_pr_geo(df, metadata)
    gdf_plot = merge_data_onto_map(df, metadata)

    if gdf_plot.empty:
        return None

    if geo_type == "kommuner":
        gdf_plot = gdf_plot[gdf_plot["geo_id"].isin(KOMMUNER_ID)]
    else:
        gdf_plot = gdf_plot[gdf_plot["geo_id"].isin(REGIONER_ID)]

    gdf_plot["color"] = generate_color_values(gdf_plot["y"], color_theme, n_colors)

    if "TID" in gdf_plot.columns:
        tooltip = {
            "html": "Area: {navn} <br/>" "Value: {y} <br/>" "Time: {TID} <br/>",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        columns = ["navn", "color", "geometry", "y", "TID"]
    else:
        tooltip = {
            "html": "Area: {navn} <br/>" "Value: {y} <br/>",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }
        columns = ["navn", "color", "geometry", "y"]

    view_state = pdk.ViewState(
        latitude=56,
        longitude=10.87,
        zoom=6,
    )
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf_plot[columns],
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_line_color=[0, 0, 0],
        get_fill_color="color",
        auto_highlight=True,
        pickable=True,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip=tooltip,
    )


def generate_color_values(values, color_theme, n_colors):
    color_list = generate_color_list(color_theme, n_colors)
    color_intervals = np.linspace(min(values), max(values), n_colors - 1)
    return [
        assign_color_group(
            colors=color_list, color_intervals=color_intervals, value=val
        )
        for val in values
    ]


def generate_color_list(color_theme, n_colors):
    """
    Generate list of rgb colors.

    color_theme: matplotlib color theme
    N: number of colors in color list i.e. level of granularity.
    """
    x = np.linspace(0, 1, n_colors)
    rgb = plt.get_cmap(color_theme)(x)[np.newaxis, :, :3]
    lab = cspace_converter("sRGB1", "sRGB255")(rgb)
    return lab.round(2).squeeze()


def assign_color_group(colors, color_intervals, value):
    color_pos = bisect.bisect_right(color_intervals, value)
    return list(colors[color_pos])


###   Prompts  ###
# FIXME: Give ability to aggregate measures.
PX_PLOT_SYS_MSG = """You are a world-class data scientist. You code in Python and use plotly express to create data vizualisations. 

When writing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.

Your job is to create a data visualisation that helps answer a user question. To do so you are provided with a dataset, df.

You get the user question and information about the dataset on the following form:

Question: "User question"
Dataset: "Dataset description"
Variables: [{"name": "variable name", "text": "variable description", "n_unique": "Number of unique values the variable has"}, ... ]
Filters: "SQL WHERE statement that was used to filter the dataset"

Assume plotly.express is imported as px and that you have access to the dataset in the pandas dataframe, df. The y variable in the plots will always be named "y" and it contains the relevant values. If the dataset should be sorted by certain variables make sure to do so. Never subset or filter the dataset df.
Do not write fig.show()."""

PX_PLOT_USER_MSG = Template(
    """Question: {{ question }}
Dataset: {{ description }}
Variables: {{ variables }}
Filters: {{ filters }}"""
)

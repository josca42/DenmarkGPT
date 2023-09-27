import matplotlib.pyplot as plt
from colorspacious import cspace_converter
import numpy as np
import bisect
import pydeck as pdk
from data import gdf, KOMMUNER_ID, REGIONER_ID


def plot_map(df, metadata, geo_type, color_theme="plasma", n_colors=20):
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
    gdf_plot = gdf.merge(df[["geo_id", "y"]], on="geo_id", how="inner")
    return gdf_plot


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

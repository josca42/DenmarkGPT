import matplotlib.pyplot as plt
from colorspacious import cspace_converter
import numpy as np
import bisect
import pydeck as pdk
from data import gdf, KOMMUNER_ID, REGIONER_ID


def plot_map(df, metadata, color_theme="plasma", n_colors=20):
    gdf_plot = merge_data_onto_map(df, metadata)
    gdf_plot["color"] = generate_color_values(
        gdf_plot["INDHOLD"], color_theme, n_colors
    )

    if contains_both_regioner_and_kommuner(gdf_plot):
        gdf_plot = gdf_plot[gdf_plot["geo_id"].isin(KOMMUNER_ID)]

    view_state = pdk.ViewState(
        latitude=56,
        longitude=10.87,
        zoom=6,
    )
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=gdf_plot[["navn", "color", "geometry", "INDHOLD"]],
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

    tooltip = {
        "html": "Country: {country} <br/>" "Count: {count} <br/>",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        # tooltip=tooltip,
    )


def merge_data_onto_map(df, metadata):
    text2id = metadata["geo"]["id2text_mapping"]
    df["geo_id"] = df["OMRÅDE"].map(text2id)
    gdf_plot = gdf.merge(df[["geo_id", "INDHOLD"]], on="geo_id", how="inner")
    return gdf_plot


def contains_both_regioner_and_kommuner(gdf_plot):
    # Determine if data contains both regions and municipalities
    geo_ids = gdf_plot["geo_id"].unique()
    contains_regioner = np.any([_id in geo_ids for _id in REGIONER_ID])
    contains_kommuner = np.any([_id in geo_ids for _id in KOMMUNER_ID])
    return contains_regioner and contains_kommuner


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

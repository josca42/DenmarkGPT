from data import KOMMUNER_ID, REGIONER_ID, ALL_GEO_IDS


def apply_filters(df, select_multi, metadata_df):
    # Apply filters selected in multi select boxes
    vars_processed = ["y"]
    for var, values in select_multi.items():
        if values:
            if df[var].dtype != "object":
                df = df[df[var].astype(str).isin(values)]
            else:
                df = df[df[var].isin(values)]
            vars_processed.append(var)

    # Remove totals unless only totals are in data
    for var in df.columns:
        if var not in vars_processed and var != "geo_id":
            totals = metadata_df[var]["totals"]
            if totals:
                n_unique = len(df[var].unique())
                if n_unique > 1:
                    df = df[~df[var].isin(totals)]

    return df


def create_filter_boxes(df, metadata_df, st):
    if "geo" in metadata_df and df[[metadata_df["geo"]["var"]]].nunique()[0] > 1:
        select_geo_type = st.selectbox(
            "Region type", metadata_df["geo"]["geo_types"], 0
        )
        text2id = metadata_df["geo"]["id2text_mapping"]
        df["geo_id"] = df[metadata_df["geo"]["var"]].map(text2id)
    else:
        select_geo_type = None

    select_multi = {}
    filters = {}
    variables = []
    for var, select in metadata_df["specs"].items():
        if (
            "geo" in metadata_df
            and var == metadata_df["geo"]["var"]
            and select_geo_type in ["kommuner", "regioner"]
        ):
            if df["geo_id"].isin(ALL_GEO_IDS).any():
                geo_ids = KOMMUNER_ID if select_geo_type == "kommuner" else REGIONER_ID
                values = [
                    v["text"] for v in metadata_df[var]["values"] if v["id"] in geo_ids
                ]
                df = df[df[var].isin(values)]
            else:
                values = []
        elif select == ["*"]:
            values = [v["text"] for v in metadata_df[var]["values"]]
        else:
            values = select

        if len(values) > 1:
            select_multi[var] = st.multiselect(var[0].upper() + var[1:].lower(), values)

        variables.append({"name": var, "text": metadata_df[var]["text"]})

    return df, select_geo_type, select_multi, variables

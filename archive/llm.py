def match_action(msg):
    query_embedding = embed([msg])
    D, I = action_index.search(np.array(query_embedding).astype("float32"), 1)
    return action_ids[I[0][0]]

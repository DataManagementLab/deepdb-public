if (relevantScope[{node_scope}]) {{
    // notNanPerNode[{node_id}] = true;
    {floating_data_type} probsNode{node_id}[] = {{ {node_p} }};

    //not null condition
    if (nullValueIdx{node_scope} != -1) {{
        nodeIntermediateResult[{node_id}] = 1 - probsNode{node_id}[nullValueIdx{node_scope}];
    }} else {{
        for (int &idx: possibleValues{node_scope}) {{
            nodeIntermediateResult[{node_id}] += probsNode{node_id}[idx];
        }}
    }}
    {final_assert}
}}
{floating_data_type} spn{spn_id}(vector<bool> relevantScope, vector<bool> featureScope, {method_params}){{
    {floating_data_type} resultValue = 0.0;
    // bool notNanPerNode[{node_count}] = {{ false }};
    {floating_data_type} nodeIntermediateResult[{node_count}] = {{ 0 }};

{method_body}

    return resultValue;
}}
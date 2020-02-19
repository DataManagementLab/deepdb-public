if (relevantScope[{node_scope}]) {{
    if (featureScope[{node_scope}]) {{
        if (inverse{node_scope}) {{
            nodeIntermediateResult[{node_id}] = {inverted_mean};
        }} else {{
            nodeIntermediateResult[{node_id}] = {mean};
        }}
    }} else {{

        vector<{floating_data_type}> uniqueVals{node_id}{{ {unique_values} }};
        vector<{floating_data_type}> probSum{node_id}{{ {prob_sum} }};

        // search right and left bounds via binary search
        int leftIdx{node_id} = 0;
        if (!leftMinusInf{node_scope}) {{
            vector<{floating_data_type}>::iterator leftBoundIdx{node_id};
            leftBoundIdx{node_id} = std::lower_bound(uniqueVals{node_id}.begin(), uniqueVals{node_id}.end(), leftCondition{node_scope});
            leftIdx{node_id} = leftBoundIdx{node_id} - uniqueVals{node_id}.begin();
        }}

        int rightIdx{node_id} = uniqueVals{node_id}.size();
        if (!rightMinusInf{node_scope}) {{
            vector<{floating_data_type}>::iterator rightBoundIdx{node_id};
            rightBoundIdx{node_id} = std::upper_bound(uniqueVals{node_id}.begin(), uniqueVals{node_id}.end(), rightCondition{node_scope});
            rightIdx{node_id} = rightBoundIdx{node_id} - uniqueVals{node_id}.begin();
        }}

        nodeIntermediateResult[{node_id}] = probSum{node_id}[rightIdx{node_id}] - probSum{node_id}[leftIdx{node_id}];

        // exclude null value if it was included before
        if (((leftMinusInf{node_scope} || leftCondition{node_scope} < nullValue{node_scope}) && (rightMinusInf{node_scope} || rightCondition{node_scope} > nullValue{node_scope})) ||
            (!leftMinusInf{node_scope} && (nullValue{node_scope} == leftCondition{node_scope}) && leftIncluded{node_scope}) ||
            (!rightMinusInf{node_scope} && (nullValue{node_scope} == rightCondition{node_scope}) && rightIncluded{node_scope})) {{
            nodeIntermediateResult[{node_id}] -= {null_value_prob}; // null value prob
        }}

        // left value should not be included in interval
        if (!leftIncluded{node_scope} && !leftMinusInf{node_scope} && leftCondition{node_scope} == uniqueVals{node_id}[leftIdx{node_id}]) {{
            nodeIntermediateResult[{node_id}] -= probSum{node_id}[leftIdx{node_id} + 1] - probSum{node_id}[leftIdx{node_id}];
        }}

        //same for right
        if (!rightIncluded{node_scope} && !rightMinusInf{node_scope} && rightCondition{node_scope} == uniqueVals{node_id}[rightIdx{node_id}-{node_id}] && leftCondition{node_scope} != rightCondition{node_scope}) {{
            nodeIntermediateResult[{node_id}] -= probSum{node_id}[rightIdx{node_id}] - probSum{node_id}[rightIdx{node_id} - 1];
        }}
    }}
    {final_assert}
}}
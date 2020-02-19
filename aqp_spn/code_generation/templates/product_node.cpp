if ({scope_check}) {{
{subtree_code}
    nodeIntermediateResult[{node_id}] = 1.0;
    {result_calculation}
    {final_assert}
}}
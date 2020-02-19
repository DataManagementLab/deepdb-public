import copy


def print_conditions(conditions, seperator='Λ'):
    """Pretty prints a set of conditions with a custom seperator."""
        
    formula = ""
    for i, (table, condition) in enumerate(conditions):
        formula += table + "." + condition
        if i < len(conditions) - 1:
            formula += ' ' + seperator + ' '

    return formula


def gen_full_join_query(schema_graph, relationship_set, table_set, join_type):
    """
    Creates the full outer join to for a relationship set for join_type FULL OUTER JOIN or JOIN
    """

    from_clause = ""
    if len(relationship_set) == 0:
        assert(len(table_set) == 1)

        from_clause = list(table_set)[0]
    
    else:
        included_tables = set()
        relationships = copy.copy(relationship_set)
        
        while relationships:
            # first relation to be included
            if len(included_tables) == 0:
                relationship = relationships.pop()
                relationship_obj = schema_graph.relationship_dictionary[relationship]
                included_tables.add(relationship_obj.start)
                included_tables.add(relationship_obj.end)
                from_clause += relationship_obj.start + " " + join_type + " " + relationship_obj.end + " ON " + relationship
            else:
                # search in suitable relations
                relationship_to_add = None
                for relationship in relationships:
                    relationship_obj = schema_graph.relationship_dictionary[relationship]
                    if (relationship_obj.start in included_tables and relationship_obj.end not in included_tables) or \
                            (relationship_obj.end in included_tables and relationship_obj.start not in included_tables):
                        relationship_to_add = relationship
                if relationship_to_add is None:
                    raise ValueError("Query not a tree")
                # add it to where formula
                relationship_obj = schema_graph.relationship_dictionary[relationship_to_add]
                if (relationship_obj.start in included_tables and relationship_obj.end not in included_tables):
                    from_clause += " " + join_type + " " + relationship_obj.end + " ON " + relationship_to_add
                    included_tables.add(relationship_obj.end)
                    relationships.remove(relationship_to_add)
                elif (relationship_obj.end in included_tables and relationship_obj.start not in included_tables):
                    from_clause += " " + join_type + " " + relationship_obj.start + " ON " + relationship_to_add
                    included_tables.add(relationship_obj.start)
                    relationships.remove(relationship_to_add)
    
    return "SELECT {} FROM " + from_clause + " {}"
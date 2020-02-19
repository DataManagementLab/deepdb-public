import random


def create_random_join(schema, no_relationships):
    assert no_relationships >= 0, "No_relationships must be greater equal 0"

    start_tables = list(schema.tables)
    random.shuffle(start_tables)
    start_table_obj = start_tables[0]

    merged_tables = {start_table_obj.table_name}
    relationships = set()

    for i in range(no_relationships):

        possible_next_relationships = list()

        for relationship_obj in schema.relationships:
            # already in random relationships
            if relationship_obj.identifier in relationships:
                continue

            if relationship_obj.start in merged_tables and \
                    relationship_obj.end not in merged_tables:
                possible_next_relationships.append((relationship_obj.identifier, relationship_obj.end))

            elif relationship_obj.end in merged_tables and \
                    relationship_obj.start not in merged_tables:
                possible_next_relationships.append((relationship_obj.identifier, relationship_obj.start))

        random.shuffle(possible_next_relationships)
        if len(possible_next_relationships) == 0:
            return list(relationships), merged_tables

        relationship, table = possible_next_relationships[0]
        merged_tables.add(table)
        relationships.add(relationship)

    return list(relationships), merged_tables

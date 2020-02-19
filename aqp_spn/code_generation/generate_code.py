import logging
from enum import Enum
from time import perf_counter

import numpy as np
from spn.structure.Base import assign_ids, Product, get_number_of_nodes
from spn.structure.StatisticalTypes import MetaType

from aqp_spn.aqp_leaves import Categorical, IdentityNumericLeaf, Sum
from ensemble_compilation.spn_ensemble import read_ensemble

import os

logger = logging.getLogger(__name__)


class TemplatePath(Enum):
    current_file_path = __file__
    current_file_dir = os.path.dirname(__file__)
    MASTER = os.path.join(current_file_dir, 'templates/master.cpp')
    CATEGORICAL = os.path.join(current_file_dir, 'templates/categorical_leave.cpp')
    IDENTITY = os.path.join(current_file_dir, 'templates/identity_leave.cpp')
    PRODUCT = os.path.join(current_file_dir, 'templates/product_node.cpp')
    SUM = os.path.join(current_file_dir, 'templates/sum_node.cpp')
    METHOD_MASTER = os.path.join(current_file_dir, 'templates/method_master.cpp')
    REGISTRATION_MASTER = os.path.join(current_file_dir, 'templates/registration_master.cpp')


def replace_template(template_path, value_dictionary, depth):
    with open(template_path.value, 'r') as ftemp:
        templateString = ftemp.read()

    code_string = templateString.format(**value_dictionary)
    padding = ''.join(['    '] * depth)
    return ''.join([padding + line for line in code_string.splitlines(True)])


def comma_seperated_list(value_list):
    return ', '.join([str(v) for v in value_list])


def generate_scope_check(scope):
    return ' || '.join([f'relevantScope[{node_scope}]' for node_scope in scope])


def generate_categorical_node(node, root_node, floating_data_type, depth):
    value_dictionary = {
        'node_id': node.id,
        'node_scope': node.scope[0],
        'node_p': comma_seperated_list(node.p),
        'final_assert': f'resultValue = nodeIntermediateResult[{node.id}];' if root_node == node else '',
        'floating_data_type': floating_data_type
    }
    return replace_template(TemplatePath.CATEGORICAL, value_dictionary, depth)


def nan_replacement(value):
    if np.isnan(value):
        return 0
    else:
        return value


def generate_identity_node(node, root_node, floating_data_type, depth):
    value_dictionary = {
        'node_id': node.id,
        'node_scope': node.scope[0],
        'null_value_prob': node.null_value_prob,
        'unique_values': comma_seperated_list(node.unique_vals),
        'prob_sum': comma_seperated_list(node.prob_sum),
        'mean': nan_replacement(node.mean * (1 - node.null_value_prob)),
        'inverted_mean': nan_replacement(node.inverted_mean * (1 - node.null_value_prob)),
        'floating_data_type': floating_data_type,
        'final_assert': f'resultValue = nodeIntermediateResult[{node.id}];' if root_node == node else ''
    }
    return replace_template(TemplatePath.IDENTITY, value_dictionary, depth)


def generate_product_node(node, root_node, floating_data_type, depth):
    # if ({scope_check}) {{
    #     {subtree_code}
    #     nodeIntermediateResult[{node_id}] = 1.0
    #     {result_calculation}
    # }}

    result_calculation_lines = []
    for child in node.children:
        result_calculation_lines += [f'if ({generate_scope_check(child.scope)}) '
                                     f'{{nodeIntermediateResult[{node.id}] *= nodeIntermediateResult[{child.id}];}}']

    value_dictionary = {
        'node_id': node.id,
        'scope_check': generate_scope_check(node.scope),
        'subtree_code': '\n'.join(
            [generate_method_body(child, root_node, floating_data_type, depth) for child in node.children]),
        'result_calculation': '\n    '.join(result_calculation_lines),
        'final_assert': f'resultValue = nodeIntermediateResult[{node.id}];' if root_node == node else ''
    }
    return replace_template(TemplatePath.PRODUCT, value_dictionary, depth)


def generate_sum_node(node, root_node, floating_data_type, depth):
    # if ({scope_check}) {{
    # {subtree_code}
    #     {result_calculation}
    #     {final_assert}
    # }}

    result_calculation_lines = []
    for i, child in enumerate(node.children):
        result_calculation_lines += [f'nodeIntermediateResult[{child.id}] * {node.weights[i]}']

    value_dictionary = {
        'scope_check': generate_scope_check(node.scope),
        'subtree_code': '\n'.join(
            [generate_method_body(child, root_node, floating_data_type, depth) for child in node.children]),
        'result_calculation': f'nodeIntermediateResult[{node.id}]=' + ' + '.join(result_calculation_lines) + ';',
        'final_assert': f'resultValue = nodeIntermediateResult[{node.id}];' if root_node == node else ''
    }
    return replace_template(TemplatePath.SUM, value_dictionary, depth)


def generate_method_body(node, root_node, floating_data_type, depth):
    if isinstance(node, Categorical):
        return generate_categorical_node(node, root_node, floating_data_type, depth + 1)
    elif isinstance(node, IdentityNumericLeaf):
        return generate_identity_node(node, root_node, floating_data_type, depth + 1)
    elif isinstance(node, Product):
        return generate_product_node(node, root_node, floating_data_type, depth + 1)
    elif isinstance(node, Sum):
        return generate_sum_node(node, root_node, floating_data_type, depth + 1)
    else:
        raise NotImplementedError


def generate_code(spn_id, spn, meta_types, floating_data_type):
    """
    Generates inference code for an SPN
    :param target_path: the path the generated C++ code is written to
    :param floating_data_type: data type floating numbers are represented in generated C++ code
    :param spn: root node of an SPN
    :return: code string
    """

    # make sure we have ids
    assign_ids(spn)

    # fill method body according to SPN structure
    method_body = generate_method_body(spn, spn, floating_data_type, 0)

    # build parameters used in generated c++ function
    method_params = []
    passed_params = []
    for i, type in enumerate(meta_types):
        if type == MetaType.DISCRETE:
            method_params += [f'vector <int> possibleValues{i}', f'int nullValueIdx{i}']
            passed_params += [f'py::arg("possibleValues{i}")', f'py::arg("nullValueIdx{i}")']
        elif type == MetaType.REAL:
            method_params += [f'bool inverse{i}', f'bool leftMinusInf{i}', f'float leftCondition{i}',
                              f'bool rightMinusInf{i}', f'float rightCondition{i}', f'bool leftIncluded{i}',
                              f'bool rightIncluded{i}', f'float nullValue{i}']
            passed_params += [f'py::arg("inverse{i}")', f'py::arg("leftMinusInf{i}")', f'py::arg("leftCondition{i}")',
                              f'py::arg("rightMinusInf{i}")', f'py::arg("rightCondition{i}")',
                              f'py::arg("leftIncluded{i}")', f'py::arg("rightIncluded{i}")', f'py::arg("nullValue{i}")']

    value_dictionary = {
        'spn_id': spn_id,
        'method_body': method_body,
        'method_params': ', '.join(method_params),
        'node_count': get_number_of_nodes(spn),
        'passed_params': ', '.join(passed_params),
        'floating_data_type': floating_data_type
    }
    generated_method = replace_template(TemplatePath.METHOD_MASTER, value_dictionary, 0)
    registrate_method = replace_template(TemplatePath.REGISTRATION_MASTER, value_dictionary, 0)

    return generated_method, registrate_method


def generate_ensemble_code(spn_ensemble, floating_data_type='float', ensemble_path=None):
    registrations = []
    methods = []
    logger.debug(f"Starting code generation")
    for i, spn in enumerate(spn_ensemble.spns):
        spn.id = i
        gen_start = perf_counter()
        generated_method, registrate_method = generate_code(i, spn.mspn, spn.meta_types, floating_data_type)
        registrations.append(registrate_method)
        methods.append(generated_method)
        gen_end = perf_counter()
        logger.debug(f"Generated code for SPN {i + 1}/{len(spn_ensemble.spns)} in {gen_end - gen_start:.2f}s.")

    value_dictionary = {
        'methods': '\n\n'.join(methods),
        'registration': '\n\t'.join(registrations)
    }
    generated_code = replace_template(TemplatePath.MASTER, value_dictionary, 0)

    if ensemble_path is not None:
        spn_ensemble.save(ensemble_path)

    with open('optimized_inference.cpp', 'w') as f:
        f.write(generated_code)

    logger.debug(f"Finished code generation.")

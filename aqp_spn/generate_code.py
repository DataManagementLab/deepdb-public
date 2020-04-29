import logging
from time import perf_counter

from rspn.code_generation.generate_code import generate_code, replace_template, TemplatePath

logger = logging.getLogger(__name__)


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

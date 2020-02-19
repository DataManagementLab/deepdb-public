from enum import Enum

from ensemble_compilation.utils import print_conditions


class FactorType(Enum):
    INDICATOR_EXP = 0
    EXPECTATION = 1


class IndicatorExpectation:
    """
    Represents E[1_{conditions} * 1/ denominator_multipliers].
    """

    def __init__(self, denominator_multipliers, conditions, nominator_multipliers=None, spn=None, inverse=False,
                 table_set=None):
        self.nominator_multipliers = nominator_multipliers
        if self.nominator_multipliers is None:
            self.nominator_multipliers = []
        self.denominator_multipliers = denominator_multipliers
        self.conditions = conditions
        self.spn = spn
        self.min_val = 0
        self.inverse = inverse
        self.table_set = table_set
        if table_set is None:
            self.table_set = set()
        if self.spn is not None:
            self.min_val = 1 / self.spn.full_join_size

    def contains_groupby(self, group_bys):
        for table, attribute in group_bys:
            for cond_table, condition in self.conditions:
                if cond_table == table and condition.startswith(attribute):
                    return True
        return False

    def matches(self, other_expectation, ignore_inverse=False, ignore_spn=False):
        if self.inverse != other_expectation.inverse and not ignore_inverse:
            return False
        if set(self.nominator_multipliers) != set(other_expectation.nominator_multipliers):
            return False
        if set(self.denominator_multipliers) != set(other_expectation.denominator_multipliers):
            return False
        if set(self.conditions) != set(other_expectation.conditions):
            return False
        if not ignore_spn and self.table_set != other_expectation.table_set:
            return False
        return True

    def __hash__(self):
        return hash((FactorType.INDICATOR_EXP, self.inverse, frozenset(self.nominator_multipliers),
                     frozenset(self.denominator_multipliers), frozenset(self.conditions), frozenset(self.table_set)))

    def is_inverse(self, other_expectation):
        return self.inverse != other_expectation.inverse and self.matches(other_expectation, ignore_inverse=True)

    def __str__(self):
        """
        Prints Expectation of multipliers for conditions.
        E(multipliers * 1_{c_1 Λ… Λc_n})
        """

        if self.inverse:
            formula = " / E("
        else:
            formula = " * E("

        for i, (table, normalizer) in enumerate(self.nominator_multipliers):
            formula += table + "." + normalizer
            if i < len(self.nominator_multipliers) - 1:
                formula += "*"
        if len(self.nominator_multipliers) == 0:
            formula += "1"

        if len(self.denominator_multipliers) > 0:
            formula += "/("

            # 1/multiplier
            for i, (table, normalizer) in enumerate(self.denominator_multipliers):
                formula += table + "." + normalizer
                if i < len(self.denominator_multipliers) - 1:
                    formula += "*"
            formula += ")"

        # |c_1 Λ… Λc_n
        if len(self.conditions) > 0:
            formula += "* 1_{"
            formula += print_conditions(self.conditions)
            formula += "}"
        formula += ")"

        return formula

    def print_conditions(self, seperator='Λ'):
        return print_conditions(self.conditions, seperator=seperator)


class Expectation:
    """
    Represents conditional expectation of feature with normalizing multipliers.
    """

    def __init__(self, features, normalizing_multipliers, conditions, spn=None):
        self.features = features
        self.normalizing_multipliers = normalizing_multipliers
        self.conditions = conditions
        self.spn = spn
        self.min_val = 1

    def matches(self, other_expectation, ignore_spn=False):
        if set(self.features) != set(other_expectation.features):
            return False
        if set(self.normalizing_multipliers) != set(other_expectation.normalizing_multipliers):
            return False
        if set(self.conditions) != set(other_expectation.conditions):
            return False
        if not ignore_spn and self.spn != other_expectation.spn:
            return False
        return True

    def __hash__(self):
        return hash((FactorType.EXPECTATION, frozenset(self.features), frozenset(self.normalizing_multipliers),
                     frozenset(self.conditions), self.spn))

    def __str__(self):
        """
        Prints Expectation of feature for conditions.
        E(feature | c_1 Λ… Λc_n) (norm by multipliers).
        """

        formula = " * E("
        # features
        for i, (table, multiplier) in enumerate(self.features):
            formula += table + "." + multiplier
            if i < len(self.features) - 1:
                formula += "*"

        # /(multipliers)
        if len(self.normalizing_multipliers) > 0:
            formula += " /("
            # 1/multiplier
            for i, (table, normalizer) in enumerate(self.normalizing_multipliers):
                formula += table + "." + normalizer
                if i < len(self.normalizing_multipliers) - 1:
                    formula += "*"
            formula += ")"

        # |c_1 Λ… Λc_n
        if len(self.conditions) > 0:
            formula += "| "
            formula += print_conditions(self.conditions)

        formula += ")"

        return formula

    def print_conditions(self, seperator='Λ'):
        return print_conditions(self.conditions, seperator=seperator)


class Probability:

    def __init__(self, conditions):
        self.conditions = conditions

    def matches(self, other_probability):
        if set(self.conditions) != set(other_probability.conditions):
            return False
        return True

    def __str__(self):
        """
        Prints Probability of conditions
        """

        formula = ""
        if len(self.conditions) > 0:
            formula += " * P("
            formula += print_conditions(self.conditions)
            formula += ")"

        return formula

    def print_conditions(self, seperator='Λ'):
        return print_conditions(self.conditions, seperator=seperator)

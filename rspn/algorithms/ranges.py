import numpy as np


class NominalRange:
    """
    This class specifies the range for a nominal attribute. It contains a list of integers which
    represent the values which are in the range.
    
    e.g. possible_values = [5,2] 
    """

    def __init__(self, possible_values, null_value=None, is_not_null_condition=False):
        self.is_not_null_condition = is_not_null_condition
        self.possible_values = np.array(possible_values, dtype=np.int64)
        self.null_value = null_value

    def is_impossible(self):
        return len(self.possible_values) == 0

    def get_ranges(self):
        return self.possible_values


class NumericRange:
    """
    This class specifies the range for a numeric attribute. It contains a list of intervals which
    represents the values which are valid. Inclusive Intervals specifies whether upper and lower bound are included.
    
    e.g. ranges = [[10,15],[22,23]] if valid values are between 10 and 15 plus 22 and 23 (bounds inclusive)
    """

    def __init__(self, ranges, inclusive_intervals=None, null_value=None, is_not_null_condition=False):
        self.is_not_null_condition = is_not_null_condition
        self.ranges = ranges
        self.null_value = null_value
        self.inclusive_intervals = inclusive_intervals
        if self.inclusive_intervals is None:
            self.inclusive_intervals = []
            for interval in self.ranges:
                self.inclusive_intervals.append([True, True])

    def is_impossible(self):
        return len(self.ranges) == 0

    def get_ranges(self):
        return self.ranges

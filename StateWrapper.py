from TSPClasses import *

class StateWrapper:

    def __init__(self, table, state_bound, src_path, route, depth):
        self._table = table
        self._state_bound = state_bound
        self._src_path = src_path
        self._route = route
        self._depth = depth

    def __lt__(self, other):
        if self._depth > other._depth:
            return True
        elif self._state_bound < other._state_bound:
            return True
        else:
            return False
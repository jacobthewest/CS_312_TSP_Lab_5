from TSPClasses import *

class StateWrapper:

    def __init__(self, table, state_bssf, src_path, route):
        self._table = table
        self._state_bssf = state_bssf
        self._src_path = src_path
        self._route = route

    def __lt__(self, other):
        return self._state_bssf < other._state_bssf
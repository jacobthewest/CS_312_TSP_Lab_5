from TSPClasses import *

class CityWrapper:

    def __init__(self, cost, city):
        self._cost = cost
        self._city = city

    def __lt__(self, other):
        return self._cost < other._cost
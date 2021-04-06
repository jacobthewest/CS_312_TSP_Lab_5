from TSPClasses import *

class CityWrapper:

    def __init__(self, cost, city, indexInCities):
        self._cost = cost
        self._city = city
        self._indexInCities = indexInCities

    def __lt__(self, other):
        return self._cost < other._cost
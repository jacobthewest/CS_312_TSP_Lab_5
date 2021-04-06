#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
# 	from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
from CityWrapper import *
import heapq
import itertools



class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour( self, time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation( ncities )
            route = []
            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count # Number of solutions discovered.
        results['soln'] = bssf # Object containing the route.
        results['max'] = None # Max size of the queue.
        results['total'] = None # Total states generated.
        results['pruned'] = None # Number of states pruned.
        return results


    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy( self,time_allowance=60.0 ):
        results = {}
        cities = self._scenario.getCities()
        visitedCities = {}

        startTime = time.time()
        route = []
        route.append(cities[0]) # Add starting city to the route
        visitedCities[cities[0]._name] = True  # Prevent us from visiting our starting city

        runningCost = 0

        # For every city, get the minimal cost to another city and add it to our route
        i = 0
        counter = 0
        while(counter < len(cities)):
            currCity = cities[i]
            heapFromCurrCity = []

            # Find costs from current city to every other city
            for j in range(len(cities)):
                tempCity = cities[j]
                try:
                    isVisited = visitedCities[tempCity._name] # Will not throw exception if city has
                                                              # been visited. Therefore, skip it.
                except:
                    cost = currCity.costTo(tempCity)
                    wrappedCity = CityWrapper(cost, tempCity, j)
                    heapq.heappush(heapFromCurrCity, wrappedCity)

            # Obtain the closest city, make it impossible to visit in the future, and add it to the route.
            if len(heapFromCurrCity) == 0:
                # currCity is the last city. We are done.
                break
            wrappedCity = heapq.heappop(heapFromCurrCity)
            closestCityToCurrentCity =  wrappedCity._city
            cost = wrappedCity._cost
            i = wrappedCity._indexInCities
            visitedCities[closestCityToCurrentCity._name] = True  # Prevent us from visiting the closest city
                                                                  # in future calculations
            route.append(closestCityToCurrentCity)
            runningCost += cost

            counter += 1

        endTime = time.time()
        bssf = TSPSolution(route)
        bssf.cost = math.inf
        if len(visitedCities.keys()) == len(cities):
            bssf.cost = runningCost

        timePassed = endTime - startTime

        results['time'] = timePassed
        results['cost'] = bssf.cost
        results['count'] = 1  # Number of solutions discovered. Will always be 1 for the greedy solution
        results['soln'] = bssf  # Object containing the route.
        results['max'] = None  # Max size of the queue. Will always be 1 for the greedy solution
        results['total'] = None  # Total states generated. Will always be 1 for the greedy solution
        results['pruned'] = None  # Number of states pruned. Will always be 0 for the greedy solution

        return results






    def jacobGreedy( self,time_allowance=60.0 ):
        # Let it be a copy of my greedy function
        pass



    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound( self, time_allowance=60.0 ):
        pass



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy( self,time_allowance=60.0 ):
        pass





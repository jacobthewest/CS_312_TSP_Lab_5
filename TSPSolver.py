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
from StateWrapper import *
import heapq
import itertools
INFINITY = math.inf


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
        results['cost'] = bssf.cost if foundTour else INFINITY
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
        bssf.cost = INFINITY
        if len(visitedCities.keys()) == len(cities):
            bssf.cost = runningCost

        timePassed = endTime - startTime

        results['time'] = timePassed
        results['cost'] = bssf.cost
        results['count'] = 1  # Number of solutions discovered. Will always be 1 for the greedy solution
        results['soln'] = bssf  # Object containing the route.
        results['max'] = 1  # Max size of the queue. Will always be 1 for the greedy solution
        results['total'] = 1  # Total states generated. Will always be 1 for the greedy solution
        results['pruned'] = 0  # Number of states pruned. Will always be 0 for the greedy solution

        return results


    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound( self, time_allowance=60.0):
        self.initResults()
        table = self.createParent()
        self._queue = []

        # Start the timer
        startTime = time.time()

        # Create the first states
        for row in range(len(table)):
            for col in range(len(table[row])):
                if table[row][col][0] != INFINITY: # 0 because that is where the cost to get to
                                                   # the city is stored
                    self.statify(row, col, table, [])

        # Perform branch and bound work on our newly created states
        while self._queue:

            # Pop off the state with the smallest bssf/bound thing in the queue
            state = heapq.heappop(self._queue)

            # Perform a pruning check
            # self._results['soln'].cost holds our current BSSF cost
            if state._state_bssf <= self._results['soln'].cost:

                # Get the row to expand into more states
                row = state._src_path[1] # 1 Is the col index. Our row we will
                # expand will come from the prev
                # state's column.
                table = state._table

                # Expand the cell in the row into a new state if it is not infinity
                for i in range(len(table[row])):
                    if table[row][i][0] != INFINITY: # 0 because that is where the cost to get to
                                                     # the city is stored
                        route = state._route
                        self.statify(row, i, table, route, state._state_bssf)
            else: # We need to prune this state
                self._results['pruned'] += 1

            # Do we have time to do more work?
            currTime = time.time()
            tempTimePassed = currTime - startTime
            totalTimePassed = tempTimePassed + self._results['time']
            if totalTimePassed >= time_allowance:
                break

        # Done processing all of our states
        # Set the time it took to perform the algorithm
        currTime = time.time()
        tempTimePassed = currTime - startTime
        self._results['time'] += tempTimePassed

        # Done
        return self._results


    # Performs operations on the parent table to update the bounds
    # from moving to a new city, and creates a table to show those
    # changes.
    def statify(self, row, col, table, route, parentBSSF = None):

        # A little logic trick to see if this is one of the first, undeveloped states
        # or if we need to do more processing on the state because it is well-developed
        bound = parentBSSF
        if not parentBSSF:
            bound = self._results['soln'].cost

        # Inifinitize the rows and columns in the table from the operation
        table, solnFound = self.infinitize(row, col, table)

        # Zero out the rows. Update the bound & table
        bound, table = self.zeroRows(bound, table)

        # Zero out the cols. Update the bound & table
        bound, table = self.zeroCols(bound, table)

        # Add the path taken to get to the city to the route
        route.append(table[row][col][1]) # 1 because that is the city index in the tuple in the table

        # Update things if we have found a solution and it is a better solution
        if bound < self._results['soln'].cost and solnFound:
            # Set the BSSF to help with future pruning.
            # Update the count of solutions discovered.
            self._results['count'] += 1
            self._results['soln'] = TSPSolution(route)
            self._results['soln'].cost = bound

        # Create variables to create a state to add to the queue
        srcPath = (row, col)
        # bssfCost = self._results['soln'].cost
        # returnState = StateWrapper(table, bssfCost, srcPath, route)
        returnState = StateWrapper(table, bound, srcPath, route)

        # Add the state object to the queue and see if our queue
        # is the biggest it has ever been
        heapq.heappush(self._queue, returnState)
        if self._results['max'] < len(self._queue):
            self._results['max'] = len(self._queue)

        # Update the our counter for tracking the number
        # of generated states
        self._results['total'] += 1


    # Subtracts the min value in every row from every element
    # to ensure we always have at least one zero in every row.
    # Adds the subtraction amount to the bound and returns it.
    def zeroRows(self, bound, table):

        for i in range(len(table)):
            minVal = INFINITY

            # Find row values
            for j in range(len(table[i])):
                if table[i][j][0] < minVal: # 0 because the table is of tuples (cost, city)
                    minVal = table[i][j][0]

            # If a minVal < INFINITY exists then we should update our table
            # Update the bound
            if minVal != INFINITY:
                bound += minVal

                # Update row values
                for j in range(len(table[i])):
                    tempVal = table[i][j][0]
                    tempVal -= minVal
                    table[i][j] = (tempVal, table[i][j][1])  # 0 because the table is of tuples (cost, city)

        return bound, table

    # Subtracts the min value in every col from every element
    # to ensure we always have at least one zero in every col.
    # Adds the subtraction amount to the bound and returns it.
    def zeroCols(self, bound, table):

        for i in range(len(table)):
            minVal = INFINITY

            # Find col values
            for j in range(len(table)):
                if table[j][i][0] < minVal: # 0 because the table is of tuples (cost, city)
                    minVal = table[j][i][0]

            # If a minVal < INFINITY exists then we should update our table
            # Update the bound
            if minVal != INFINITY:
                bound += minVal

                # Update col values
                for j in range(len(table[i])):
                    tempVal = table[j][i][0]
                    tempVal -= minVal
                    table[j][i] = (tempVal, table[j][i][1]) # 0 because the table is of tuples (cost, city)

        return bound, table

    # Updates the rows and cols to be infinity from.
    # Updates the backtrace to be infinity
    def infinitize(self, row, col, table):

        infinityCount = 0
        doneCount = len(table) * len(table)

        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j][0] == INFINITY: # 0 because the table is of tuples (cost, city)
                    infinityCount += 1
                elif row == i: # Make the row infinity (if qualifies)
                    table[i][j] = (INFINITY, table[i][j][1])
                elif col == j: # Make the col infinity (if qualifies)
                    table[i][j] = (INFINITY, table[i][j][1])

        # Infinitize the backtrace
        table[col][row] = (INFINITY, table[col][row][1]) # Handles the reverse of table[row][col]
        solnFound = False
        if infinityCount == doneCount:
            solnFound = True

        return table, solnFound



    def createParent(self):
        cities = self._scenario.getCities()
        numCities = len(cities)
        table = []

        # Create empty table to fill
        for city in cities:
            table.append([])

        # Populate the parent table
        for i in range(numCities):
            currCity = cities[i]
            for j in range(numCities):
                cost = currCity.costTo(cities[j])
                table[i].append((cost, cities[j]))
        return table



    def initResults(self):
        self._results = self.greedy()

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





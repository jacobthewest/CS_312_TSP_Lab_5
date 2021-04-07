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
import copy

INFINITY = math.inf


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
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

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        self._cities = self._scenario.getCities()
        ncities = len(self._cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(self._cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else INFINITY
        results['time'] = end_time - start_time
        results['count'] = count  # Number of solutions discovered.
        results['soln'] = bssf  # Object containing the route.
        results['max'] = None  # Max size of the queue.
        results['total'] = None  # Total states generated.
        results['pruned'] = None  # Number of states pruned.
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

    # Time complexity: O(n^2) because we compare every city to every other city when
    #                  we are trying to find the min cost.
    # Space complexity: O(n). Because of a map of size n called visitedCities,
    #                   a heapQueue that is of size n (worst case),
    #                   and a route list of size n.
    def greedy(self, time_allowance=60.0):

        results = {}
        self._cities = self._scenario.getCities()
        visitedCities = {}

        self._startTime = time.time()
        self._time_allowance = time_allowance
        route = []
        route.append(self._cities[0])  # Add starting city to the route
        visitedCities[self._cities[0]._name] = True  # Prevent us from visiting our starting city

        runningCost = 0

        # For every city, get the minimal cost to another city and add it to our route
        i = 0
        counter = 0
        # This is O(n^2) because of an n size loop inside of an n size loop
        while (counter < len(self._cities)):
            currCity = self._cities[i]
            heapFromCurrCity = []

            # Find costs from current city to every other city
            for j in range(len(self._cities)):
                tempCity = self._cities[j]
                try:
                    isVisited = visitedCities[tempCity._name]  # Will not throw exception if city has
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
            closestCityToCurrentCity = wrappedCity._city
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
        if len(visitedCities.keys()) == len(self._cities):
            bssf.cost = runningCost

        timePassed = endTime - self._startTime

        results['time'] = timePassed
        results['cost'] = bssf.cost
        results['count'] = 0  # Number of solutions discovered. Will always be 1 for the greedy solution
        results['soln'] = bssf  # Object containing the route.
        results['max'] = 0  # Max size of the queue. Will always be 1 for the greedy solution
        results['total'] = 0  # Total states generated. Will always be 1 for the greedy solution
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

    # Time Complexity: O(number of states I generate * n^4) Because I have a O(n^4) time
    #                  complex operation to generate and build each state, and then I
    #                  process every state in my queue.
    # Space Complexity: O(number of states I generate * n^2) Because I have a queue full of every
    #                   state that I generate. Each state object has a table of size O(n^2) inside
    #                   of it.
    def branchAndBound(self, time_allowance=60.0):
        # Start the timer
        self._startTime = time.time()
        self._timesUp = False

        self.initResults()
        bound, table = self.createParent()
        self._queue = []

        # Add the first state to the queue
        #             StateWrapper(matrix, bound, srcPath, route, depth)
        firstCity = self._cities[0]
        returnState = StateWrapper(table, bound, (0, 0), [firstCity], 0)
        heapq.heappush(self._queue, returnState)
        self._results['max'] += 1 # Because we now have our first state in the queue
        self._results['total'] += 1  # Because we have generated a state


        # Perform branch and bound work on our newly created states
        #
        # Time complexity: O(however many states we generate * n) because our queue is full of state objects
        # and for each state object we will look at every cell in its table.
        #
        # Space complexity: O(number of states I generate * n^2) Because I have a queue full of every
        #                   state that I generate. Each state object has a table of size O(n^2) inside
        #                   of it.
        while self._queue:

            # Pop off the state with the smallest bssf/bound thing in the queue
            state = heapq.heappop(self._queue)

            # Perform a pruning check
            # self._results['soln'].cost holds our current BSSF cost
            if state._state_bound < self._results['soln'].cost:

                # Get the row to expand into more states
                row = state._src_path[1]  # 1 Is the col index. Our row we will
                                          # expand will come from the prev
                                          # state's column.
                table = state._table

                startCity = state._route[0]

                # Expand the cell in the row into a new state if it is not infinity
                for i in range(len(table[row])):
                    if table[row][i][0] != INFINITY:  # 0 because that is where the cost to get to
                                                      # the city is stored
                        # Make sure we don't re-visit cities, but we are ok to revisit the start city
                        # if we are in the last city
                        if table[row][i][1] != startCity or len(state._route) == len(self._cities):
                            self.statify(row, i, table, state._route, state._depth, state._state_bound)
            else:  # We need to prune this state
                self._results['pruned'] += 1

            # Do we have time to do more work?
            # This check happens after we build states from every row in focus
            self.timeCheck()
            if self._timesUp:
                return self.wrapThingsUp()

        # Done processing all of our states
        return self.wrapThingsUp()

    def wrapThingsUp(self):
        # Set the time it took to perform the algorithm
        currTime = time.time()
        tempTimePassed = currTime - self._startTime
        self._results['time'] += tempTimePassed
        self._results['cost'] = self._results['soln'].cost
        self._results['pruned'] += len(self._queue)  # Add states that were never processed to the pruned count.
                                                     # The lab specs ask us to do this.

        # Done
        return self._results

    def timeCheck(self):
        currTime = time.time()
        tempTimePassed = currTime - self._startTime
        totalTimePassed = tempTimePassed + self._results['time']
        if totalTimePassed >= self._time_allowance:
            self._timesUp = True

    # Performs operations on the parent table to update the bounds
    # from moving to a new city, and creates a table to show those
    # changes.
    # Time Complexity: O(n^2) because our zeroRows(), zeroCols(),
    #                  and infinitize() functions are all O(n^2).
    # Space Complexity: O(n^2) because the table object is the
    #                   largest object at O(n^2) (n rows and n cols).
    def statify(self, row, col, table, route, depth, bound):

        # Create a table that matches the parent table
        tableMatch = []
        for i in range(len(table)):
            tableMatch.append([])
            for j in range(len(table)):
                tableMatch[i].append(table[i][j])

        # Inifinitize the rows and columns in the table from the operation
        updatedBound, tableMatch, solnFound = self.infinitize(row, col, tableMatch, bound)

        # Zero out the rows. Update the bound & table
        updatedBound, tableMatch = self.zeroRows(updatedBound, tableMatch)

        # Zero out the cols. Update the bound & table
        updatedBound, tableMatch = self.zeroCols(updatedBound, tableMatch)

        # Add the path taken to get to the city to the route
        updatedRoute = copy.deepcopy(route)
        updatedRoute.append(tableMatch[row][col][1])  # 1 because that is the city index in the tuple in the table

        # Update things if we have found a solution and it is a better solution
        if solnFound and updatedBound < self._results['soln'].cost:
            # Update the count of solutions discovered.
            self._results['count'] += 1
            # Set the BSSF to help with future pruning.
            self._results['soln'] = TSPSolution(route) # Use the original route so it doesn't have the first city twice.
            self._results['soln'].cost = updatedBound
            self._results['pruned'] -= 1 # Just because it is going to increment one above what
                                         # what it should be right after this on line 303.
            self._results['total'] -= 1 # Similar situation to the pruned problem above...

        # The updatedBound is < BSSF so it is worth pursuing this route
        if updatedBound < self._results['soln'].cost:
            # Create variables to create a state to add to the queue
            srcPath = (row, col)
            returnState = StateWrapper(tableMatch, updatedBound, srcPath, updatedRoute, depth + 1)

            # Add the state object to the queue and see if our queue
            # is the biggest it has ever been
            heapq.heappush(self._queue, returnState)
            if self._results['max'] < len(self._queue):
                self._results['max'] = len(self._queue)
        else:
            self._results['pruned'] += 1

        # Update the our counter for tracking the number
        # of generated states
        self._results['total'] += 1

    # Subtracts the min value in every row from every element
    # to ensure we always have at least one zero in every row.
    # Adds the subtraction amount to the bound and returns it.
    #
    # Time complexity: O(n^2) because we look at every cell in table.
    # Space Complexity: O(n^2) because we have a table of n rows and n columns.
    def zeroRows(self, bound, table):

        for i in range(len(table)):
            minVal = INFINITY

            # Find row values
            for j in range(len(table[i])):
                if table[i][j][0] < minVal:  # 0 because the table is of tuples (cost, city)
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
    #
    # Time complexity: O(n^2) because we look at every cell in table.
    # Space Complexity: O(n^2) because we have a table of n rows and n columns.
    def zeroCols(self, bound, table):

        for i in range(len(table)):
            minVal = INFINITY

            # Find col values
            for j in range(len(table)):
                if table[j][i][0] < minVal:  # 0 because the table is of tuples (cost, city)
                    minVal = table[j][i][0]

            # If a minVal < INFINITY exists then we should update our table
            # Update the bound
            if minVal != INFINITY:
                bound += minVal

                # Update col values
                for j in range(len(table[i])):
                    tempVal = table[j][i][0]
                    tempVal -= minVal
                    table[j][i] = (tempVal, table[j][i][1])  # 0 because the table is of tuples (cost, city)

        return bound, table

    # Updates the rows and cols to be infinity based on the city path taken to get there.
    # Updates the backtrace to be infinity
    #
    # Time Complexity: O(n^2). We have a double for loop to check every cell in the table.
    # Space Complexity: O(n^2). The table is of size O(n^2)
    def infinitize(self, row, col, table, bound):

        infinityCount = 0
        doneCount = len(table) * len(table)

        # Add the current cell's value to the bound
        bound += table[row][col][0]

        # O(n^2) loops
        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j][0] == INFINITY:  # 0 because the table is of tuples (cost, city)
                    infinityCount += 1
                elif row == i:  # Make the row infinity (if qualifies)
                    table[i][j] = (INFINITY, table[i][j][1])
                    infinityCount += 1
                elif col == j:  # Make the col infinity (if qualifies)
                    table[i][j] = (INFINITY, table[i][j][1])
                    infinityCount += 1

        # Infinitize the backtrace
        if table[col][row][0] != INFINITY:
            table[col][row] = (INFINITY, table[col][row][1])  # Handles the reverse of table[row][col]
            infinityCount += 1

        solnFound = False
        if infinityCount == doneCount:
            solnFound = True

        return bound, table, solnFound

    # Time Complexity: O(n^2) because we compare every city to every other city.
    # Space Complexity: O(n^2) because we create a table of n rows and n columns.
    def createParent(self):
        self._cities = self._scenario.getCities()
        numCities = len(self._cities)
        table = []

        # Create empty table to fill
        for city in self._cities:
            table.append([])

        # Populate the parent table
        for i in range(numCities):
            currCity = self._cities[i]
            for j in range(numCities):
                cost = currCity.costTo(self._cities[j])
                table[i].append((cost, self._cities[j]))

        # Find the bound and create the reduced cost matrix
        # This takes O(n^2) time complexity and O(n^2) space complexity
        bound, table = self.zeroRows(0, table)
        bound, table = self.zeroCols(0, table)

        return bound, table

    # Time Complexity: O(n^2) because my greedy solution is of O(n^2) complexity
    # Space Complexity: O(n^2) because my greedy solution creates a list of cities
    #                   in its route.
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

    def fancy(self, time_allowance=60.0):
        pass

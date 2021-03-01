# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import heapq
import operator
from collections import OrderedDict, defaultdict, deque
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
from queue import PriorityQueue


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    waypoints = maze.waypoints
    explored = []
    frontier = deque()
    frontier.append([start])
    while frontier:
        current = frontier.popleft()
        cell = current[-1] # a tuple (i, j)
        # print("current: ", current, "||", "cell: ", cell)
        if cell not in explored: #avoid loops
            neighbors = maze.neighbors(cell[0], cell[1])
            explored.append(cell)
            for neighbor in neighbors:
                next_step = list(current)
                next_step.append(neighbor)
                frontier.append(next_step)
                # print('next_step', next_step)
                if neighbor in waypoints:
                    return next_step

    
    
                
def manh_dist(goal, next):
    return abs(next[0] - goal[0]) + abs(next[1] - goal[1])
def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    goal = maze.waypoints[0]

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    g = {}
    came_from[start] = start
    g[start] = 0
    while frontier:
        current = frontier.get()
        if current == goal:
            reconst_path = []
            while came_from[current] != current:
                reconst_path.append(current)
                current = came_from[current]
            reconst_path.append(start)
            reconst_path.reverse()
            return reconst_path
        for next in maze.neighbors(current[0],current[1]):
            new_cost = g[current] + 1
            if next not in g or new_cost < g[next]:
                g[next] = new_cost
                f_n = new_cost + manh_dist(goal, next)
                frontier.put(next, f_n)
                came_from[next] = current
    return None

# the Kuskal MST from geeksforgeeks  
class MST:
    def __init__(self,vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self,u,v,w):
        self.graph.append([u,v,w])

    def find(self,parent,i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self,parent, rank,x,y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[xroot] = yroot
            rank[yroot] += 1

    def KruskalMST(self, current_state):
        Vertices = set(self.V.copy())
        Vertices.add(current_state)
        queue = []
        rank = {}
        parent = {}
        parent[current_state] = current_state
        rank[current_state] = 0
        for i in Vertices:
            parent[i] = i
            rank[i] = 0
            for j in Vertices:
                if i == j:
                    pass
                else:
                    heapq.heappush(queue,(manh_dist(i,j),i,j))

        while len(self.graph) + 1 < len(Vertices) :
            min = heapq.heappop(queue)
            x = self.find(parent, min[1])
            y = self.find(parent, min[2])
            if x == y:
                continue
            self.addEdge(min[1],min[2],manh_dist(min[2],min[1]))
            self.union(parent, rank, x, y)
        result = 0
        for i in self.graph:
            result += i[2]
        return result

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    queue = []
    start = maze.start
    goals = maze.waypoints
    start_state = (start, '0'*len(goals))        
    heapq.heappush(queue, (0, (start_state, [start])))    
    g_n = {start_state: 0}                

    explored = set()

    while queue:
        current_element = heapq.heappop(queue)
        current_state = current_element[1][0]
        current_path = current_element[1][1]
        current_node = current_path[-1]
        neighbors = maze.neighbors(current_node[0], current_node[1])
        explored.add(current_node)

        for neighbor in neighbors:
            neighbor_state = hashing(goals, neighbor, current_state)
            new_cost = g_n[current_state] + 1
            new_path = list(current_path)
            new_path.append(neighbor)
            if neighbor_state[1].find('0') == -1:
                return new_path

            if neighbor_state not in g_n or g_n[neighbor_state] > new_cost:
                remaining_goals = getRemainingGoals(goals, neighbor_state[1])
                mst = MST(remaining_goals)
                f_n = new_cost + mst.KruskalMST(neighbor_state[0])
                g_n[neighbor_state] = new_cost
                heapq.heappush(queue, (f_n, (neighbor_state, new_path)))


#
def hashing(waypoints, waypoint, parent_state):
    if waypoint in waypoints:
        index = waypoints.index(waypoint)
        hashmap = list(parent_state[1])
        hashmap[index] = '1'
        hashmap = ''.join(hashmap)
    else:
        hashmap = parent_state[1]

    return (waypoint, hashmap)

def getRemainingGoals(goals, hashmap):
    remaining_goals = list()
    for i, n in enumerate(list(hashmap)):
        if n == '0':
            remaining_goals.append(goals[i])
    return remaining_goals

def astar_multiple(maze):
    """    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    queue = []
    start = maze.start
    goals = maze.waypoints
    start_state = (start, '0'*len(goals))         
    heapq.heappush(queue, (0, (start_state, [start])))    
    g_n = {start_state: 0}                 

    explored = set()

    while queue:
        current_element = heapq.heappop(queue)
        current_state = current_element[1][0]
        current_path = current_element[1][1]
        current_node = current_path[-1]
        neighbors = maze.neighbors(current_node[0], current_node[1])
        explored.add(current_node)

        for neighbor in neighbors:
            neighbor_state = hashing(goals, neighbor, current_state)
            new_cost = g_n[current_state] + 1
            new_path = list(current_path)
            new_path.append(neighbor)
            if neighbor_state[1].find('0') == -1:
                return new_path

            if neighbor_state not in g_n or g_n[neighbor_state] > new_cost:
                remaining_goals = getRemainingGoals(goals, neighbor_state[1])
                mst = MST(remaining_goals)
                f_n = new_cost + mst.KruskalMST(neighbor_state[0])
                g_n[neighbor_state] = new_cost
                heapq.heappush(queue, (f_n, (neighbor_state, new_path)))

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            

import heapq
import math
import time

dx = [-1, 1, 0, 0]
dy = [0, 0, 1, -1]

dfs_counter = 0
bfs_counter = 0
astar_counter = 0
ucs_counter = 0
dls_counter = 0
ids_counter = 0
greedy_counter = 0

dfs_path = []
bfs_path = []
astar_path = []
ucs_path = []
dls_path = []
ids_path = []
greedy_path = []

dfs_cost = 0
bfs_cost = 0
astar_cost = 0
ucs_cost = 0
dls_cost = 0
ids_cost = 0
greedy_cost = 0

dfs_depth = 0
bfs_depth = 0
astar_depth = 0
ucs_depth = 0
dls_depth = 0
ids_depth = 0
greedy_depth = 0

time_dfs = 0
time_bfs = 0
time_astar = 0
time_ucs = 0
time_dls = 0
time_ids = 0
time_greedy = 0


def getStringRepresentation(x):
    if int(math.log10(x)) + 1 == 9:
        return str(x)
    else:
        return "0" + str(x)


def getChildren(state):
    children = []
    idx = state.index('0')
    i = int(idx / 3)
    j = int(idx % 3)
    for x in range(0, 4):
        nx = i + dx[x]
        ny = j + dy[x]
        nwIdx = int(nx * 3 + ny)
        if checkValid(nx, ny):
            listTemp = list(state)
            listTemp[idx], listTemp[nwIdx] = listTemp[nwIdx], listTemp[idx]
            children.append(''.join(listTemp))
    return children


def getPath(parentMap, inputState):
    path = []
    temp = 12345678
    while temp != inputState:
        path.append(temp)
        temp = parentMap[temp]
    path.append(inputState)
    path.reverse()
    return path


def printPath(path):
    for i in path:
        print(getStringRepresentation(i))


def goalTest(state):
    if state == 12345678:
        return True
    return False


def isSolvable(digit):
    count = 0
    for i in range(0, 9):
        for j in range(i, 9):
            if digit[i] > digit[j] and digit[i] != 9:
                count += 1
    return count % 2 == 0


def BFS(inputState):
    start_time = time.time()
    q = []
    explored = {}
    parent = {}
    parent_cost = {}
    integer_state = int(inputState)
    q.append(integer_state)
    cnt = 0
    global bfs_counter
    global bfs_path
    global bfs_cost
    global bfs_depth
    global time_bfs
    bfs_depth = 0
    parent_cost[integer_state] = 0
    while q:
        cnt += 1
        state = q.pop(0)
        explored[state] = 1
        bfs_depth = max(bfs_depth, parent_cost[state])
        if goalTest(state):
            path = getPath(parent, int(inputState))
            bfs_counter = cnt
            bfs_path = path
            bfs_cost = len(path) - 1
            time_bfs = float(time.time() - start_time)
            return 1
        children = getChildren(getStringRepresentation(state))
        for child in children:
            child_int = int(child)
            if child_int not in explored:
                q.append(child_int)
                parent[child_int] = state
                explored[child_int] = 1
                parent_cost[child_int] = 1 + parent_cost[state]
    bfs_path = []
    bfs_cost = 0
    bfs_counter = cnt
    time_bfs = float(time.time() - start_time)
    return 0


def DFS(inputState):
    start_time = time.time()
    stack = []
    explored = {}
    parent = {}
    parent_cost = {}
    integer_state = int(inputState)
    parent_cost[integer_state] = 0
    explored[integer_state] = 1
    stack.append(integer_state)
    cnt = 0
    global dfs_counter
    global dfs_path
    global dfs_cost
    global dfs_depth
    global time_dfs
    dfs_depth = 0
    while stack:
        cnt += 1
        state = stack[-1]
        stack.pop()
        dfs_depth = max(dfs_depth, parent_cost[state])
        if goalTest(state):
            path = getPath(parent, int(inputState))
            dfs_counter = cnt
            dfs_path = path
            dfs_cost = len(path) - 1
            time_dfs = float(time.time() - start_time)
            return 1
        children = getChildren(getStringRepresentation(state))
        for child in children:
            child_int = int(child)
            if child_int not in explored:
                stack.append(child_int)
                parent[child_int] = state
                explored[child_int] = 1
                parent_cost[child_int] = 1 + parent_cost[state]
    dfs_path = []
    dfs_cost = 0
    dfs_counter = cnt
    time_dfs = float(time.time() - start_time)
    return 0


def checkValid(i, j):
    if i >= 3 or i < 0 or j >= 3 or j < 0:
        return 0
    return 1


def getDistance(state):
    tot = 0
    for i in range(1, 9):
        goalX = int(i / 3)
        goalY = i % 3
        idx = state.index(str(i))
        itemX = int(idx / 3)
        itemY = idx % 3
        tot += (abs(goalX - itemX) + abs(goalY - itemY))
    return tot


def AStarSearch(inputState):
    start_time = time.time()
    integer_state = int(inputState)
    heap = []
    explored = {}
    parent = {}
    cost_map = {}
    heapq.heappush(heap, (getDistance(inputState), integer_state))
    cost_map[integer_state] = getDistance(inputState)
    heap_map = {}
    heap_map[integer_state] = 1
    global astar_counter
    global astar_path
    global astar_cost
    global astar_depth
    global time_astar
    astar_depth = 0
    while heap:
        node = heapq.heappop(heap)
        state = node[1]
        string_state = getStringRepresentation(state)
        parent_cost = node[0] - getDistance(string_state)
        if not state in explored:
            astar_depth = max(parent_cost, astar_depth)
        explored[state] = 1

        if goalTest(state):
            path = getPath(parent, int(inputState))
            astar_path = path
            astar_counter = (len(explored))
            astar_cost = len(path) - 1
            time_astar = float(time.time() - start_time)

            return 1

        children = getChildren(string_state)
        for child in children:
            new_cost = getDistance(child)
            child_int = int(child)
            if child_int not in explored and child not in heap_map:
                heapq.heappush(heap, (parent_cost + new_cost + 1, child_int))
                heap_map[child_int] = 1
                cost_map[child_int] = parent_cost + new_cost + 1
                parent[child_int] = state
            elif child_int in heap_map:
                if (new_cost + parent_cost + 1) < cost_map[child_int]:
                    parent[child_int] = state
                    cost_map[child_int] = new_cost + parent_cost + 1
                    heapq.heappush(heap, (parent_cost + 1 + new_cost, child_int))
    astar_cost = 0
    astar_path = []
    astar_counter = (len(explored))
    time_astar = float(time.time() - start_time)

    return 0


def uniform_cost_search(inputState):
    start_time = time.time()

    integer_state = int(inputState)
    
    frontier = [(0, integer_state)]
    came_from = {integer_state: None}
    cost_so_far = {integer_state: 0}

    global ucs_counter, ucs_path, ucs_cost, ucs_depth, time_ucs
    ucs_counter = 0
    ucs_depth = 0

    while frontier:
        current_cost, current_state = heapq.heappop(frontier)

        ucs_counter += 1

        if goalTest(current_state):
            ucs_path = getPath(came_from, integer_state)
            ucs_cost = len(ucs_path) - 1
            ucs_depth = max(ucs_depth, cost_so_far[current_state])
            time_ucs = time.time() - start_time
            return 1

        for next_state_str in getChildren(getStringRepresentation(current_state)):
            next_state = int(next_state_str)
            new_cost = cost_so_far[current_state] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current_state

    ucs_path = []
    ucs_cost = 0
    time_ucs = time.time() - start_time
    return 0


def depth_limited_search(inputState, limit=100):
    global dls_counter, dls_path, dls_cost, dls_depth, time_dls
    start_time = time.time()
    
    stack = [(inputState, 0)]
    parentMap = {inputState: None}
    visited = set([inputState])
    dls_counter = 0
    found = False

    while stack:
        state, depth = stack.pop()
        dls_counter += 1
        
        if goalTest(int(state)):
            found = True
            # Reconstruct the path
            dls_path = []
            while state is not None:
                dls_path.append(int(state))
                state = parentMap.get(str(state))
            dls_path.reverse()
            dls_cost = len(dls_path) - 1
            dls_depth = depth
            break
        
        if depth < limit:
            for child in getChildren(str(state)):
                if child not in visited:
                    visited.add(child)
                    parentMap[child] = str(state)
                    stack.append((child, depth + 1))

    time_dls = time.time() - start_time
    return 1 if found else 0


def iterative_deepening_search(inputState):
    global ids_counter, ids_path, ids_cost, ids_depth, time_ids
    start_time = time.time()
    
    ids_counter = 0
    found = False
    depth_limit = 0
    MAX_DEPTH = 20
    
    while not found and depth_limit <= MAX_DEPTH:
        stack = [(inputState, 0)]
        visited = set([inputState])
        
        while stack:
            current_state, current_depth = stack.pop()
            ids_counter += 1
            
            if goalTest(current_state):
                found = True
                ids_path = [current_state]
                ids_cost = current_depth
                ids_depth = current_depth
                break
            
            if current_depth < depth_limit:
                for child in getChildren(current_state):
                    if child not in visited:
                        visited.add(child)
                        stack.append((child, current_depth + 1))
        
        depth_limit += 1
    
    time_ids = time.time() - start_time
    return 1 if found else 0


def greedy_search(inputState):
    start_time = time.time()

    integer_state = int(inputState) if isinstance(inputState, str) else inputState
    
    frontier = [(getDistance(getStringRepresentation(integer_state)), integer_state)]
    came_from = {integer_state: None}
    
    global greedy_counter, greedy_path, greedy_cost, greedy_depth, time_greedy
    greedy_counter = 0
    greedy_depth = 0

    while frontier:
        current_heuristic, current_state = heapq.heappop(frontier)

        greedy_counter += 1

        if goalTest(current_state):
            greedy_path = getPath(came_from, integer_state)
            greedy_cost = len(greedy_path) - 1
            time_greedy = time.time() - start_time
            return 1

        for next_state_str in getChildren(getStringRepresentation(current_state)):
            next_state = int(next_state_str)
            if next_state not in came_from:
                heapq.heappush(frontier, (getDistance(getStringRepresentation(next_state)), next_state))
                came_from[next_state] = current_state

    time_greedy = time.time() - start_time
    return 0
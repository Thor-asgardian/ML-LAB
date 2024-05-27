import heapq

def best_first_search(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], start)]
    visited = set()
    parent = {start: None}

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)
        
        if current_node == goal:
            break

        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor))
                    parent[neighbor] = current_node

    path = []
    node = goal
    while node:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

heuristic = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 0,
    'E': 2,
    'F': 3,
    'G': 1
}

start = 'A'
goal = 'D'

path = best_first_search(graph, start, goal, heuristic)
print("Best First Search Path:", path)

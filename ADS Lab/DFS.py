class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u in self.graph:
            self.graph[u].append(v)
        else:
            self.graph[u] = [v]

    def dfs(self, start_vertex, visited):
        visited.add(start_vertex)
        print(start_vertex, end=" ")

        if start_vertex in self.graph:
            for neighbor in self.graph[start_vertex]:
                if neighbor not in visited:
                    self.dfs(neighbor, visited)

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

start_vertex = 2  # Starting vertex for DFS
visited = set()

print("Depth-First Traversal (starting from vertex 2):")
g.dfs(start_vertex, visited)

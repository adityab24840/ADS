from collections import deque

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, u, v):
        self.adj_matrix[u][v] = 1
        self.adj_matrix[v][u] = 1

    def bfs(self, start_vertex):
        visited = [False] * self.num_vertices
        queue = deque()
        queue.append(start_vertex)
        visited[start_vertex] = True

        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")

            for i in range(self.num_vertices):
                if self.adj_matrix[vertex][i] == 1 and not visited[i]:
                    queue.append(i)
                    visited[i] = True

# Example usage:
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

start_vertex = 2  # Starting vertex for BFS

print("Breadth-First Traversal (starting from vertex 2):")
g.bfs(start_vertex)

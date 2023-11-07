class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u in self.graph:
            self.graph[u].append(v)
        else:
            self.graph[u] = [v]

    def dfs(self, node, visited, component):
        visited.add(node)
        component.append(node)
        for neighbor in self.graph.get(node, []):
            if neighbor not in visited:
                self.dfs(neighbor, visited, component)

    def connected_components(self):
        visited = set()
        components = []
        for node in self.graph:
            if node not in visited:
                component = []
                self.dfs(node, visited, component)
                components.append(component)
        return components

# Example usage:
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(3, 4)

connected_components = g.connected_components()

print("Connected Components:")
for component in connected_components:
    print(component)

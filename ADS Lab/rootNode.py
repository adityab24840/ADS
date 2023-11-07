class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def count_leaf_nodes(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaf_nodes(node.left) + count_leaf_nodes(node.right)

def count_total_nodes(node):
    if node is None:
        return 0
    return 1 + count_total_nodes(node.left) + count_total_nodes(node.right)

def display_all_values(node):
    if node:
        display_all_values(node.left)
        print(node.value, end=" ")
        display_all_values(node.right)

# Example usage:
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

# a. Compute the number of leaf nodes
leaf_node_count = count_leaf_nodes(root)
print("Number of leaf nodes:", leaf_node_count)

# b. Compute the total number of nodes in the tree
total_node_count = count_total_nodes(root)
print("Total number of nodes in the tree:", total_node_count)

# c. Display all values of the nodes
print("Values of all nodes:")
display_all_values(root)

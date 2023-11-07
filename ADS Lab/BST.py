class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(root, key):
    if not root:
        return TreeNode(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root

def delete(root, key):
    if not root:
        return root
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        root.key = min_value_node(root.right)
        root.right = delete(root.right, root.key)
    return root

def min_value_node(node):
    while node.left:
        node = node.left
    return node.key

def level_order_traversal(root):
    if not root:
        return

    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.key, end=" ")

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

# Creating an empty binary search tree
root = None

# Inserting nodes into the BST
root = insert(root, 50)
root = insert(root, 30)
root = insert(root, 70)
root = insert(root, 20)
root = insert(root, 40)
root = insert(root, 60)
root = insert(root, 80)

print("Level Order Traversal (Breadth-First Traversal):")
level_order_traversal(root)

# Deleting a node (e.g., delete node with key 30)
root = delete(root, 30)

print("\nLevel Order Traversal after deleting node with key 30:")
level_order_traversal(root)

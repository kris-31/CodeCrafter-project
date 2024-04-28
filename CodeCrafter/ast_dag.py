import ast
from anytree import Node, RenderTree

# Function to recursively build the DAG
def build_dag(tree, parent=None, seen=None):
    if seen is None:
        seen = set()

    if isinstance(tree, ast.AST):
        node_name = str(type(tree).__name__)

        # Check if node has already been processed
        if node_name in seen:
            return
        seen.add(node_name)

        node_obj = Node(node_name)
        if parent:
            parent.children += (node_obj,)
        for _, value in ast.iter_fields(tree):
            if isinstance(value, list):
                for item in value:
                    build_dag(item, parent=node_obj, seen=seen)
            elif isinstance(value, ast.AST):
                build_dag(value, parent=node_obj, seen=seen)

# Read the code from the file
with open("inputcode.txt", "r") as file:
    code = file.read()

# Parse the code into an AST
tree = ast.parse(code)

# Initialize root node for the DAG
root_node = Node("Root")

# Build the DAG
build_dag(tree, parent=root_node)

# Print the DAG using RenderTree
for pre, fill, node in RenderTree(root_node):
    print("%s%s" % (pre, node.name))

# Function to recursively build the tree
def build_tree(node, parent=None):
    if isinstance(node, ast.AST):
        node_name = str(type(node).__name__)
        node_obj = Node(node_name, parent=parent)
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    build_tree(item, parent=node_obj)
            elif isinstance(value, ast.AST):
                build_tree(value, parent=node_obj)

# Build the AST tree
root_node = Node("Root")
build_tree(tree, parent=root_node)

# Print the tree
for pre, fill, node in RenderTree(root_node):
    print("%s%s" % (pre, node.name))

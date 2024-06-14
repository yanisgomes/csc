import networkx as nx
import matplotlib.pyplot as plt

def generate_tree(K, L):
    # Initialize the directed graph
    G = nx.DiGraph()
    
    # Add the root node with attributes for layer and position
    G.add_node(0, layer=0, pos=(0, 0))  # Root node with layer attribute and position
    
    # Last used node ID to ensure each node has a unique identifier
    last_node_id = 0
    
    # A list to track nodes at each layer
    current_layer = [(0, (0, 0))]  # Tuple (node_id, (x_position, y_position))
    
    # Build the tree layer by layer
    for layer in range(1, K + 1):
        next_layer = []
        width = 2 ** (K - layer)  # Adjust the spacing for each level
        for parent_id, parent_pos in current_layer:
            start_x = parent_pos[0] - (L - 1) * width / 2  # Center children under the parent
            for i in range(L):
                last_node_id += 1
                x_pos = start_x + i * width
                G.add_node(last_node_id, layer=layer, pos=(x_pos, -layer))
                G.add_edge(parent_id, last_node_id)
                next_layer.append((last_node_id, (x_pos, -layer)))
        current_layer = next_layer  # Move to the next layer
    
    return G

# Example usage
K = 4  # Number of layers
L = 4  # Number of branches per node

tree_graph = generate_tree(K, L)

# Configure the node positions for better visualization using the 'pos' attribute
pos = nx.get_node_attributes(tree_graph, 'pos')  # Directly access the pos attribute

nx.draw(tree_graph, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=700, font_size=12)
plt.title("Generated DiGraph with {} Layers and {} Branches per Node".format(K, L))
plt.show()

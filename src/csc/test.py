import networkx as nx
import matplotlib.pyplot as plt

def generate_tree(K, L):
    # Initialiser le graphe dirigé
    G = nx.DiGraph()
    
    # Ajouter le nœud racine avec un attribut de couche
    G.add_node(0, layer=0)  # Nœud racine avec attribut de couche
    
    # Dernier ID de nœud utilisé pour s'assurer que chaque nœud a un identifiant unique
    last_node_id = 0
    
    # Une liste pour suivre les nœuds à chaque couche
    current_layer = [0]
    
    # Construire l'arbre couche par couche
    for layer in range(1, K + 1):
        next_layer = []
        for j, node in enumerate(current_layer):
            # Ajouter L enfants pour chaque nœud dans la couche actuelle
            for i in range(1, L + 1):
                last_node_id += 1
                G.add_node(last_node_id, layer=layer)
                G.add_edge(node, last_node_id)
                next_layer.append(last_node_id)
        current_layer = next_layer  # Passer à la prochaine couche
    
    return G

# Exemple d'utilisation
K = 3  # Nombre de couches
L = 2  # Nombre de branches par nœud

tree_graph = generate_tree(K, L)

# Configurer la position des nœuds pour une meilleure visualisation en utilisant l'attribut 'layer'
pos = {}
for node, layer in nx.get_node_attributes(tree_graph, 'layer').items():
    pos[node] = ((node % (L**layer)) * (1 / L**layer), -layer)

nx.draw(tree_graph, pos, with_labels=True, node_color='skyblue', edge_color='black', node_size=700, font_size=12)
plt.title("Generated DiGraph with {} Layers and {} Branches per Node".format(K, L))
plt.show()

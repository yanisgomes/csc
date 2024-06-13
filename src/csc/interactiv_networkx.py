import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def on_click(event):
    # Coordonnées du point cliqué
    click_x, click_y = event.xdata, event.ydata
    if click_x is None or click_y is None:
        # Si le clic est en dehors des axes, ne rien faire
        return
    # Trouver le nœud le plus proche du clic
    distances = [(n, np.sqrt((pos[n][0] - click_x)**2 + (pos[n][1] - click_y)**2)) for n in pos]
    closest_node, min_distance = min(distances, key=lambda x: x[1])
    # Seuil de distance pour considérer qu'un nœud a été cliqué
    if min_distance < 0.1:  # Ajuster ce seuil si nécessaire
        ax2.clear()
        ax2.plot(data[closest_node], label=f'Data for node {closest_node}')
        ax2.legend()
        fig.canvas.draw()

# Données exemple pour le graphe
data = {
    0: [1, 2, 3],
    1: [3, 4, 5],
    2: [5, 6, 7],
    3: [2, 5, 9]
}

# Création du graphe
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

# Dessin du graphe
fig, (ax1, ax2) = plt.subplots(2, 1)
pos = nx.spring_layout(G)  # Positions des nœuds pour le layout "spring"
nx.draw(G, pos, ax=ax1, with_labels=True, node_color='skyblue')

# Connecter l'événement de clic avec la fonction on_click
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

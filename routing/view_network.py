import osmnx as ox
import matplotlib.pyplot as plt

print("Loading Chennai road network...")

G = ox.load_graphml("data/chennai_roads.graphml")

print("Network loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

fig, ax = ox.plot_graph(
    G,
    node_size=0,
    edge_linewidth=0.3,
    figsize=(16, 200),
    show=False,
    close=False
)
plt.show()
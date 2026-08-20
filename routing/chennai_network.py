import osmnx as ox

print("Downloading Chennai road network...")

G = ox.graph_from_place(
    "Chennai, Tamil Nadu, India",
    network_type="drive"
)

print("Download complete!")
print("Number of road nodes:", len(G.nodes))
print("Number of road edges:", len(G.edges))

ox.save_graphml(G, "data/chennai_roads.graphml")

print("Road network saved successfully!")
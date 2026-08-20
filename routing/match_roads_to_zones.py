import osmnx as ox
import json

from flood_zones import FLOOD_ZONES, get_zone_for_point


print("Loading Chennai road network...")

G = ox.load_graphml("data/chennai_roads.graphml")

print("Road network loaded!")
print("Total road nodes:", len(G.nodes))


zone_nodes = {}

for zone_id in FLOOD_ZONES:
    zone_nodes[zone_id] = []


print("\nMatching road nodes to flood zones...\n")


for node_id, node_data in G.nodes(data=True):

    latitude = float(node_data["y"])
    longitude = float(node_data["x"])

    zone_id = get_zone_for_point(latitude, longitude)

    if zone_id is not None:
        zone_nodes[zone_id].append(node_id)


print("========== RESULTS ==========")

for zone_id, nodes in zone_nodes.items():
    print(zone_id, "->", len(nodes), "road nodes")

print("=============================")


with open("data/flood_zone_nodes.json", "w") as file:
    json.dump(zone_nodes, file, indent=4)


print("\nFlood zone node mapping saved successfully!")
print("File: data/flood_zone_nodes.json")
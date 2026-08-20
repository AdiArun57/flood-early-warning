import osmnx as ox
import networkx as nx
import json


print("Loading Chennai road network...")

G = ox.load_graphml("data/chennai_roads.graphml")

print("Road network loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))


print("Loading flood zone data...")

with open("data/flood_zone_nodes.json", "r") as file:
    FLOOD_ZONE_NODES = json.load(file)

print("Flood zone data loaded!")


def get_flooded_nodes(zone_id):

    return set(FLOOD_ZONE_NODES.get(zone_id, []))


def remove_flooded_nodes(graph, flooded_nodes):

    safe_graph = graph.copy()

    for node in flooded_nodes:

        if node in safe_graph:
            safe_graph.remove_node(node)

    return safe_graph


def find_nearest_node(graph, latitude, longitude):

    return ox.distance.nearest_nodes(
        graph,
        longitude,
        latitude
    )


def find_safe_route(
    graph,
    start_node,
    destination_node,
    flooded_nodes
):

    safe_graph = remove_flooded_nodes(
        graph,
        flooded_nodes
    )

    if start_node not in safe_graph:
        print("Starting location is flooded!")
        return None

    if destination_node not in safe_graph:
        print("Destination is flooded!")
        return None

    try:

        route = nx.shortest_path(
            safe_graph,
            start_node,
            destination_node,
            weight="length"
        )

        return route

    except nx.NetworkXNoPath:

        print("No safe route found!")

        return None


print("Route engine ready!")

if __name__ == "__main__":

    print("\nTesting safe evacuation route...")

    flooded_nodes = get_flooded_nodes("CHN_01")

    print("Flooded nodes in CHN_01:", len(flooded_nodes))

    start_node = list(G.nodes)[1000]
    destination_node = list(G.nodes)[50000]

    route = find_safe_route(
        G,
        start_node,
        destination_node,
        flooded_nodes
    )

    if route:
        print("\nSAFE ROUTE FOUND!")
        print("Number of nodes:", len(route))
        print("First node:", route[0])
        print("Last node:", route[-1])
    else:
        print("\nNO SAFE ROUTE FOUND!")
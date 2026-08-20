import osmnx as ox


print("Loading Chennai road network...")

G = ox.load_graphml("data/chennai_roads.graphml")

print("Road network loaded!")
print("Nodes:", len(G.nodes))


def route_to_coordinates(route):

    coordinates = []

    for node_id in route:

        if node_id not in G:
            print("Node not found:", node_id)
            print("Node type:", type(node_id))
            print("First 10 graph nodes:", list(G.nodes)[:10])
            return []

        latitude = float(G.nodes[node_id]["y"])
        longitude = float(G.nodes[node_id]["x"])

        coordinates.append([latitude, longitude])

    return coordinates


print("Route coordinate converter ready!")
if __name__ == "__main__":

    print("\nTesting route coordinate conversion...")

    test_route = [
    list(G.nodes)[1000],
    list(G.nodes)[2000],
    list(G.nodes)[3000],
    list(G.nodes)[4000],
    list(G.nodes)[5000]
]

    coordinates = route_to_coordinates(test_route)

    print("\nRoute coordinates:")

    for coordinate in coordinates:
        print(coordinate)

    print("\nConversion successful!")
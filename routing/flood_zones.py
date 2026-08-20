from shapely.geometry import Polygon, Point


FLOOD_ZONES = {
    "CHN_01": {
        "polygon": [
            [80.210, 12.970],
            [80.230, 12.970],
            [80.230, 12.990],
            [80.210, 12.990],
            [80.210, 12.970]
        ],
        "risk_level": "HIGH",
        "risk_score": 0.8
    },

    "CHN_02": {
        "polygon": [
            [80.215, 13.010],
            [80.235, 13.010],
            [80.235, 13.030],
            [80.215, 13.030],
            [80.215, 13.010]
        ],
        "risk_level": "MODERATE",
        "risk_score": 0.5
    },

    "CHN_03": {
        "polygon": [
            [80.230, 13.030],
            [80.250, 13.030],
            [80.250, 13.050],
            [80.230, 13.050],
            [80.230, 13.030]
        ],
        "risk_level": "LOW",
        "risk_score": 0.2
    },

    "CHN_04": {
        "polygon": [
            [80.195, 13.000],
            [80.215, 13.000],
            [80.215, 13.020],
            [80.195, 13.020],
            [80.195, 13.000]
        ],
        "risk_level": "LOW",
        "risk_score": 0.2
    },

    "CHN_05": {
        "polygon": [
            [80.235, 13.015],
            [80.255, 13.015],
            [80.255, 13.035],
            [80.235, 13.035],
            [80.235, 13.015]
        ],
        "risk_level": "HIGH",
        "risk_score": 0.7
    }
}


def get_zone_polygon(zone_id):
    coordinates = FLOOD_ZONES[zone_id]["polygon"]
    return Polygon(coordinates)


def get_zone_for_point(latitude, longitude):
    point = Point(longitude, latitude)

    for zone_id, zone_data in FLOOD_ZONES.items():
        polygon = Polygon(zone_data["polygon"])

        if polygon.contains(point):
            return zone_id

    return None
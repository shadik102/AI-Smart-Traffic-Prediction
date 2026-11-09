# ============================================================
# AI-BASED SMART TRAFFIC CONGESTION PREDICTION
# AND ROUTE OPTIMIZATION
#
# Complete Streamlit Application
# ============================================================

import os
import json
import pickle
import warnings
from math import radians, sin, cos, sqrt, atan2, degrees

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

try:
    import joblib
except Exception:
    joblib = None

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

try:
    import osmnx as ox
    OSM_AVAILABLE = True
except Exception:
    OSM_AVAILABLE = False

import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ROUTES_DIR = os.path.join(RESULTS_DIR, "routes")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ROUTES_DIR, exist_ok=True)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "production_traffic_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "production_preprocessor.pkl"
)


# ============================================================
# CITY / LOCATION DATA
# ============================================================

CITY_DATA = {

    "Hyderabad": {
        "Madhapur": (17.4483, 78.3915),
        "Hitech City": (17.4435, 78.3772),
        "Gachibowli": (17.4401, 78.3489),
        "Kukatpally": (17.4849, 78.4138),
        "Banjara Hills": (17.4156, 78.4347),
        "Jubilee Hills": (17.4326, 78.4071),
        "Ameerpet": (17.4375, 78.4483),
        "Secunderabad": (17.4399, 78.4983),
        "Begumpet": (17.4448, 78.4663),
        "LB Nagar": (17.3457, 78.5522),
        "Mehdipatnam": (17.3960, 78.4370),
        "Charminar": (17.3616, 78.4747),
        "Koti": (17.3840, 78.4867),
        "Dilsukhnagar": (17.3687, 78.5247),
        "Uppal": (17.4058, 78.5591),
    },

    "Anantapur": {
        "Anantapur": (14.6819, 77.6006),
        "RTC Bus Stand": (14.6756, 77.5950),
        "Sapthagiri Circle": (14.6811, 77.6000),
        "Srinagar Colony": (14.6745, 77.5895),
        "Rudrampeta": (14.7040, 77.5905),
        "Bellary Road": (14.6925, 77.6035),
        "Ramachandra Nagar": (14.6665, 77.6070),
        "Old Town": (14.6775, 77.6025),
        "Georgepet": (14.6825, 77.5940),
        "Court Road": (14.6750, 77.6005),
    },

    "Puttaparthi": {
        "Puttaparthi": (14.1652, 77.8116),
        "Prasanthi Nilayam": (14.1656, 77.8114),
        "Gopuram": (14.1645, 77.8130),
        "Sai Nagar": (14.1710, 77.8150),
        "Chitravathi Road": (14.1575, 77.8080),
        "Bus Stand": (14.1615, 77.8085),
        "Hill View Stadium": (14.1685, 77.8180),
        "Kothacheruvu Road": (14.1740, 77.8030),
    },

    "Kadiri": {
        "Kadiri": (14.1110, 78.1590),
        "Kadiri Bus Stand": (14.1125, 78.1615),
        "Railway Station": (14.1135, 78.1570),
        "Court Road": (14.1085, 78.1600),
        "Gandhi Road": (14.1105, 78.1555),
        "Reddy Nagar": (14.1055, 78.1650),
        "YSR Nagar": (14.1150, 78.1680),
        "Kadiri Bypass": (14.0980, 78.1450),
    },

    "Hindupur": {
        "Hindupur": (13.8281, 77.4914),
        "Hindupur Bus Stand": (13.8295, 77.4940),
        "Railway Station": (13.8270, 77.4960),
        "Lepakshi Road": (13.8250, 77.4800),
        "Parigi Road": (13.8350, 77.4850),
        "RTC Colony": (13.8320, 77.5000),
        "Muddireddypalli": (13.8180, 77.4880),
        "Hindupur Bypass": (13.8400, 77.4700),
    },

    "Bengaluru": {
        "Whitefield": (12.9698, 77.7500),
        "Electronic City": (12.8452, 77.6602),
        "Koramangala": (12.9352, 77.6245),
        "Indiranagar": (12.9784, 77.6408),
        "MG Road": (12.9756, 77.6067),
        "Hebbal": (13.0358, 77.5970),
        "Yelahanka": (13.1007, 77.5963),
        "Marathahalli": (12.9591, 77.6974),
    },

    "Chennai": {
        "T Nagar": (13.0418, 80.2341),
        "Anna Nagar": (13.0850, 80.2101),
        "Adyar": (13.0012, 80.2565),
        "Guindy": (13.0067, 80.2206),
        "Velachery": (12.9815, 80.2180),
        "Tambaram": (12.9249, 80.1000),
        "Porur": (13.0381, 80.1565),
    },

    "Mumbai": {
        "Andheri": (19.1197, 72.8468),
        "Bandra": (19.0607, 72.8362),
        "Powai": (19.1176, 72.9060),
        "Dadar": (19.0178, 72.8478),
        "Colaba": (18.9067, 72.8147),
        "Thane": (19.2183, 72.9781),
    },

    "Delhi": {
        "Connaught Place": (28.6315, 77.2167),
        "Dwarka": (28.5921, 77.0460),
        "Rohini": (28.7495, 77.0565),
        "Saket": (28.5245, 77.2066),
        "Noida": (28.5355, 77.3910),
        "Gurugram": (28.4595, 77.0266),
    }
}


# ============================================================
# TRAFFIC SPEEDS
# ============================================================

TRAFFIC_SPEEDS = {
    "Low": 45,
    "Medium": 35,
    "High": 25,
    "Severe": 15
}


TRAFFIC_COLORS = {
    "Low": "green",
    "Medium": "orange",
    "High": "red",
    "Severe": "darkred"
}


# ============================================================
# DAY MAPPING
# ============================================================

DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "online_route": None,
    "online_distance": None,
    "online_eta": None,
    "online_nodes": None,
    "online_graph": None,
    "offline_route": None,
    "offline_distance": None,
    "offline_eta": None,
    "offline_graph": None,
    "voice_enabled": True,
    "last_prediction": None,
    "last_prediction_probability": None,
    "last_prediction_input": None,
    "navigation_steps": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two coordinates.
    Returns kilometers.
    """

    R = 6371.0

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    dlat = lat2 - lat1
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def format_km(distance):
    if distance is None:
        return "N/A"

    return f"{float(distance):.2f} km"


def format_minutes(minutes):
    if minutes is None:
        return "N/A"

    minutes = float(minutes)

    if minutes < 60:
        return f"{minutes:.1f} min"

    hours = int(minutes // 60)
    mins = int(round(minutes % 60))

    return f"{hours} hr {mins} min"


def calculate_eta(distance_km, traffic_level):
    speed = TRAFFIC_SPEEDS.get(
        traffic_level,
        TRAFFIC_SPEEDS["Medium"]
    )

    if speed <= 0:
        return None

    return (distance_km / speed) * 60


# ============================================================
# MODEL LOADING
# ============================================================

def load_serialized_file(path):

    if not os.path.exists(path):
        return None, f"File not found: {path}"

    try:

        with open(path, "rb") as file:
            obj = pickle.load(file)

        return obj, "Loaded successfully"

    except Exception as pickle_error:

        if joblib is not None:

            try:

                obj = joblib.load(path)

                return obj, "Loaded successfully with joblib"

            except Exception as joblib_error:

                return (
                    None,
                    "pickle error: "
                    + str(pickle_error)
                    + " | joblib error: "
                    + str(joblib_error)
                )

        return None, str(pickle_error)


@st.cache_resource
def load_model():

    return load_serialized_file(MODEL_PATH)


@st.cache_resource
def load_preprocessor():

    return load_serialized_file(PREPROCESSOR_PATH)


MODEL, MODEL_MESSAGE = load_model()
PREPROCESSOR, PREPROCESSOR_MESSAGE = load_preprocessor()


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_traffic_dataset():

    preferred_names = [
        "traffic_data.csv",
        "traffic_dataset.csv",
        "traffic.csv",
        "synthetic_traffic_data.csv",
        "traffic_data.parquet",
        "traffic_dataset.parquet"
    ]

    search_roots = [
        DATA_DIR,
        BASE_DIR,
        RESULTS_DIR
    ]

    candidates = []

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):

            # Skip virtual environment
            dirs[:] = [
                d for d in dirs
                if d not in {
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".git"
                }
            ]

            for file in files:

                lower = file.lower()

                if lower.endswith(".csv") or lower.endswith(".parquet"):

                    full_path = os.path.join(
                        current_root,
                        file
                    )

                    candidates.append(full_path)

    # Preferred filenames first
    for preferred in preferred_names:

        for candidate in candidates:

            if os.path.basename(candidate).lower() == preferred:
                return candidate

    # Traffic-like filename
    for candidate in candidates:

        name = os.path.basename(candidate).lower()

        if "traffic" in name:
            return candidate

    # Look at columns
    for candidate in candidates:

        try:

            if candidate.lower().endswith(".csv"):
                temp = pd.read_csv(candidate, nrows=5)
            else:
                temp = pd.read_parquet(candidate)

            columns = {
                str(c).lower()
                for c in temp.columns
            }

            traffic_columns = {
                "congestion_level",
                "vehicle_count",
                "average_speed_kmph"
            }

            if len(columns.intersection(traffic_columns)) >= 2:
                return candidate

        except Exception:
            continue

    return None


@st.cache_data
def load_traffic_dataset():

    path = find_traffic_dataset()

    if path is None:
        return None, None

    try:

        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            return None, path

        return df, path

    except Exception:
        return None, path


# ============================================================
# PREDICTION
# ============================================================

def build_prediction_input(
    location,
    hour,
    day_of_week,
    temperature_c,
    rainfall_mm,
    vehicle_count,
    average_speed_kmph,
    road_capacity,
    accident_reported,
    road_work,
    special_event,
    road_distance_km,
    estimated_travel_time_min,
    road_type="Urban",
    weather="Clear",
    traffic_signal_status="Working"
):

    return pd.DataFrame([{

        "location": location,

        "road_type": road_type,

        "weather": weather,

        "traffic_signal_status": traffic_signal_status,

        "hour": safe_int(hour),

        "day_of_week": safe_int(day_of_week),

        "temperature_c": safe_float(
            temperature_c,
            30.0
        ),

        "rainfall_mm": safe_float(
            rainfall_mm,
            0.0
        ),

        "vehicle_count": safe_int(
            vehicle_count,
            100
        ),

        "average_speed_kmph": safe_float(
            average_speed_kmph,
            35.0
        ),

        "road_capacity": safe_int(
            road_capacity,
            500
        ),

        "accident_reported": safe_int(
            accident_reported
        ),

        "road_work": safe_int(
            road_work
        ),

        "special_event": safe_int(
            special_event
        ),

        "road_distance_km": safe_float(
            road_distance_km
        ),

        "estimated_travel_time_min": safe_float(
            estimated_travel_time_min
        )

    }])


def predict_traffic_level(
    location,
    hour,
    day_of_week,
    temperature_c,
    rainfall_mm,
    vehicle_count,
    average_speed_kmph,
    road_capacity,
    accident_reported,
    road_work,
    special_event,
    road_distance_km,
    estimated_travel_time_min,
    road_type="Urban",
    weather="Clear",
    traffic_signal_status="Working"
):

    if MODEL is None:
        return None, None, "Traffic model could not be loaded."

    if PREPROCESSOR is None:
        return (
            None,
            None,
            "Preprocessor could not be loaded. "
            "Please check models/production_preprocessor.pkl"
        )

    try:

        raw_input = build_prediction_input(
            location=location,
            hour=hour,
            day_of_week=day_of_week,
            temperature_c=temperature_c,
            rainfall_mm=rainfall_mm,
            vehicle_count=vehicle_count,
            average_speed_kmph=average_speed_kmph,
            road_capacity=road_capacity,
            accident_reported=accident_reported,
            road_work=road_work,
            special_event=special_event,
            road_distance_km=road_distance_km,
            estimated_travel_time_min=estimated_travel_time_min,
            road_type=road_type,
            weather=weather,
            traffic_signal_status=traffic_signal_status
        )

        # ====================================================
        # CRITICAL FIX:
        # RAW DATA -> PREPROCESSOR -> MODEL
        # ====================================================

        processed = PREPROCESSOR.transform(raw_input)

        if hasattr(
            PREPROCESSOR,
            "get_feature_names_out"
        ):

            try:

                feature_names = (
                    PREPROCESSOR
                    .get_feature_names_out()
                )

                processed = pd.DataFrame(
                    processed,
                    columns=feature_names,
                    index=raw_input.index
                )

            except Exception:
                pass

        prediction = MODEL.predict(processed)[0]

        probability = None

        if hasattr(MODEL, "predict_proba"):

            try:

                probabilities = (
                    MODEL.predict_proba(processed)[0]
                )

                probability = (
                    float(np.max(probabilities))
                    * 100
                )

            except Exception:
                probability = None

        return prediction, probability, None

    except Exception as error:

        return (
            None,
            None,
            f"Prediction failed: {error}"
        )


# ============================================================
# OSM ONLINE ROUTING
# ============================================================

@st.cache_resource(
    show_spinner=False,
    max_entries=8
)
def download_osm_graph(
    center_lat,
    center_lon,
    radius
):

    if not OSM_AVAILABLE:
        raise RuntimeError(
            "OSMnx is not installed."
        )

    return ox.graph.graph_from_point(
        (
            center_lat,
            center_lon
        ),
        dist=radius,
        network_type="drive",
        simplify=True
    )


def calculate_graph_route_distance(
    graph,
    route
):

    total_meters = 0.0

    for index in range(len(route) - 1):

        u = route[index]
        v = route[index + 1]

        edge_data = graph.get_edge_data(
            u,
            v
        )

        if not edge_data:
            continue

        lengths = []

        for data in edge_data.values():

            try:
                length = float(
                    data.get(
                        "length",
                        0
                    )
                )

                lengths.append(length)

            except Exception:
                pass

        if lengths:
            total_meters += min(lengths)

    return total_meters / 1000.0


def get_route_coordinates(
    graph,
    route
):

    coordinates = []

    for node in route:

        data = graph.nodes[node]

        lat = data.get("y")
        lon = data.get("x")

        if lat is not None and lon is not None:

            coordinates.append(
                (
                    float(lat),
                    float(lon)
                )
            )

    return coordinates


def find_shortest_road_route(
    source_coords,
    target_coords
):

    if not OSM_AVAILABLE:

        return (
            None,
            None,
            None,
            "OSMnx is not installed."
        )

    try:

        source_lat, source_lon = source_coords
        target_lat, target_lon = target_coords

        straight_distance = haversine(
            source_lat,
            source_lon,
            target_lat,
            target_lon
        )

        # Give OSM enough area to find actual roads.
        radius = max(
            3000,
            int(
                straight_distance
                * 1000
                * 1.8
            )
        )

        radius = min(
            radius,
            50000
        )

        center_lat = (
            source_lat + target_lat
        ) / 2

        center_lon = (
            source_lon + target_lon
        ) / 2

        graph = download_osm_graph(
            center_lat,
            center_lon,
            radius
        )

        source_node = ox.distance.nearest_nodes(
            graph,
            X=source_lon,
            Y=source_lat
        )

        target_node = ox.distance.nearest_nodes(
            graph,
            X=target_lon,
            Y=target_lat
        )

        route = ox.routing.shortest_path(
            graph,
            source_node,
            target_node,
            weight="length"
        )

        if route is None:

            return (
                None,
                None,
                None,
                "No drivable road route was found."
            )

        if len(route) < 2:

            return (
                None,
                None,
                None,
                "The route contains insufficient road nodes."
            )

        distance_km = calculate_graph_route_distance(
            graph,
            route
        )

        return (
            graph,
            route,
            distance_km,
            None
        )

    except Exception as error:

        return (
            None,
            None,
            None,
            str(error)
        )


# ============================================================
# OFFLINE GRAPHML
# ============================================================

def find_graphml_files():

    files_found = []

    search_roots = [
        BASE_DIR,
        DATA_DIR,
        RESULTS_DIR
    ]

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):

            dirs[:] = [
                d for d in dirs
                if d not in {
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".git"
                }
            ]

            for file in files:

                if file.lower().endswith(
                    ".graphml"
                ):

                    files_found.append(
                        os.path.join(
                            current_root,
                            file
                        )
                    )

    return sorted(
        list(set(files_found))
    )


@st.cache_resource(
    show_spinner=False
)
def load_graphml(path):

    return ox.load_graphml(path)


def find_offline_route(
    graph,
    source_coords,
    target_coords
):

    try:

        source_lat, source_lon = source_coords
        target_lat, target_lon = target_coords

        source_node = ox.distance.nearest_nodes(
            graph,
            X=source_lon,
            Y=source_lat
        )

        target_node = ox.distance.nearest_nodes(
            graph,
            X=target_lon,
            Y=target_lat
        )

        route = nx.shortest_path(
            graph,
            source_node,
            target_node,
            weight="length"
        )

        distance_km = calculate_graph_route_distance(
            graph,
            route
        )

        return (
            route,
            distance_km,
            None
        )

    except Exception as error:

        return (
            None,
            None,
            str(error)
        )


# ============================================================
# MAP CREATION
# ============================================================

def create_route_map(
    graph,
    route,
    source_coords,
    target_coords,
    route_color="blue",
    route_name="Shortest road route",
    zoom=14
):

    if not FOLIUM_AVAILABLE:
        return None

    coordinates = get_route_coordinates(
        graph,
        route
    )

    if not coordinates:
        return None

    center_lat = (
        sum(
            point[0]
            for point in coordinates
        )
        / len(coordinates)
    )

    center_lon = (
        sum(
            point[1]
            for point in coordinates
        )
        / len(coordinates)
    )

    fmap = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=zoom,
        control_scale=True
    )

    folium.Marker(
        location=source_coords,
        tooltip="START",
        popup="<b>Starting Point</b>",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(fmap)

    folium.Marker(
        location=target_coords,
        tooltip="DESTINATION",
        popup="<b>Destination</b>",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(fmap)

    # ONLY THE SELECTED ROAD ROUTE
    # is drawn here.
    folium.PolyLine(
        locations=coordinates,
        color=route_color,
        weight=7,
        opacity=0.95,
        tooltip=route_name
    ).add_to(fmap)

    fmap.fit_bounds(
        coordinates,
        padding=(30, 30)
    )

    return fmap


def create_location_map(
    locations
):

    if not FOLIUM_AVAILABLE:
        return None

    if not locations:
        return None

    first = list(locations.values())[0]

    fmap = folium.Map(
        location=[
            first[0],
            first[1]
        ],
        zoom_start=12
    )

    for name, coords in locations.items():

        folium.Marker(
            location=coords,
            tooltip=name,
            popup=name,
            icon=folium.Icon(
                color="blue",
                icon="info-sign"
            )
        ).add_to(fmap)

    return fmap


# ============================================================
# NAVIGATION INSTRUCTIONS
# ============================================================

def bearing_between(
    point1,
    point2
):

    lat1 = radians(point1[0])
    lat2 = radians(point2[0])

    dlon = radians(
        point2[1] - point1[1]
    )

    x = (
        sin(dlon)
        * cos(lat2)
    )

    y = (
        cos(lat1)
        * sin(lat2)
        - sin(lat1)
        * cos(lat2)
        * cos(dlon)
    )

    bearing = (
        degrees(
            atan2(x, y)
        )
        + 360
    ) % 360

    return bearing


def bearing_direction(
    bearing
):

    directions = [
        "North",
        "North-East",
        "East",
        "South-East",
        "South",
        "South-West",
        "West",
        "North-West"
    ]

    index = int(
        (bearing + 22.5) / 45
    ) % 8

    return directions[index]


def create_navigation_steps(
    graph,
    route
):

    coordinates = get_route_coordinates(
        graph,
        route
    )

    if len(coordinates) < 2:
        return []

    steps = []

    total_distance = 0.0

    for index in range(
        len(coordinates) - 1
    ):

        point1 = coordinates[index]
        point2 = coordinates[index + 1]

        segment_distance = haversine(
            point1[0],
            point1[1],
            point2[0],
            point2[1]
        )

        total_distance += segment_distance

        bearing = bearing_between(
            point1,
            point2
        )

        direction = bearing_direction(
            bearing
        )

        if index == 0:

            text = (
                f"Start and continue "
                f"towards {direction}."
            )

        elif index == len(coordinates) - 2:

            text = (
                f"Continue for "
                f"{segment_distance:.2f} km "
                f"and arrive at the destination."
            )

        else:

            text = (
                f"Continue {direction} "
                f"for approximately "
                f"{segment_distance:.2f} km."
            )

        steps.append({
            "step": index + 1,
            "distance_km": segment_distance,
            "direction": direction,
            "text": text
        })

    return steps


# ============================================================
# VOICE NAVIGATION
# ============================================================

def speak_text(text):

    safe_text = (
        str(text)
        .replace("\\", "")
        .replace("'", "\\'")
        .replace("\n", " ")
    )

    components.html(
        f"""
        <script>
        const text = '{safe_text}';

        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();

            const speech =
                new SpeechSynthesisUtterance(text);

            speech.rate = 0.9;
            speech.pitch = 1.0;
            speech.volume = 1.0;

            window.speechSynthesis.speak(speech);
        }}
        </script>
        """,
        height=0
    )


def stop_voice():

    components.html(
        """
        <script>
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        </script>
        """,
        height=0
    )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():

    st.sidebar.title("🚦 Smart Traffic AI")

    st.sidebar.markdown(
        """
        ### Navigation

        Use the menu below to explore
        traffic prediction and route
        optimization.
        """
    )

    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "🚦 Traffic Prediction",
            "🗺️ Route Optimization",
            "🌍 Online OSM Routing",
            "💾 Offline GraphML Routing",
            "📊 Route Comparison",
            "📈 Traffic Analytics",
            "🔊 Voice Navigation",
            "ℹ️ About"
        ]
    )

    st.sidebar.divider()

    st.sidebar.subheader(
        "🔊 Voice Control"
    )

    voice = st.sidebar.toggle(
        "Voice Navigation ON/OFF",
        value=st.session_state.voice_enabled
    )

    st.session_state.voice_enabled = voice

    if voice:
        st.sidebar.success(
            "Voice navigation is ON"
        )
    else:
        st.sidebar.info(
            "Voice navigation is OFF"
        )

    st.sidebar.divider()

    st.sidebar.caption(
        "AI-Based Smart Traffic Congestion "
        "Prediction and Route Optimization"
    )

    return page


# ============================================================
# ROUTE SELECTOR
# ============================================================

def route_selector(
    key_prefix="route"
):

    city = st.selectbox(
        "🌆 Select City",
        list(CITY_DATA.keys()),
        key=f"{key_prefix}_city"
    )

    locations = CITY_DATA[city]

    location_names = list(
        locations.keys()
    )

    col1, col2 = st.columns(2)

    with col1:

        source_name = st.selectbox(
            "📍 From",
            location_names,
            key=f"{key_prefix}_source"
        )

    with col2:

        destination_options = [
            name
            for name in location_names
            if name != source_name
        ]

        destination_name = st.selectbox(
            "🏁 To",
            destination_options,
            key=f"{key_prefix}_destination"
        )

    source_coords = locations[source_name]
    destination_coords = locations[destination_name]

    return (
        city,
        source_name,
        destination_name,
        source_coords,
        destination_coords
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    st.title(
        "🏠 AI-Based Smart Traffic "
        "Congestion Prediction & Route Optimization"
    )

    st.markdown(
        """
        ### 🚦 Smart Traffic Management System

        This application combines machine learning,
        OpenStreetMap road networks, online routing,
        offline GraphML routing and voice navigation.
        """
    )

    # ========================================================
    # STATUS CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if MODEL is not None:
            st.success("🤖 Model Ready")
        else:
            st.error("🤖 Model Missing")

    with col2:

        if PREPROCESSOR is not None:
            st.success("⚙️ Preprocessor Ready")
        else:
            st.error("⚙️ Preprocessor Missing")

    with col3:

        if OSM_AVAILABLE:
            st.success("🌍 OSM Ready")
        else:
            st.warning("🌍 OSM Unavailable")

    with col4:

        if FOLIUM_AVAILABLE:
            st.success("🗺️ Maps Ready")
        else:
            st.warning("🗺️ Maps Unavailable")

    st.divider()

    st.subheader(
        "🗺️ Find Shortest Actual Road Route"
    )

    (
        city,
        source_name,
        destination_name,
        source_coords,
        destination_coords
    ) = route_selector(
        "dashboard"
    )

    traffic_level = st.selectbox(
        "🚦 Expected Traffic Level",
        [
            "Low",
            "Medium",
            "High",
            "Severe"
        ],
        key="dashboard_traffic"
    )

    if st.button(
        "🚀 Find Shortest Road Route",
        type="primary",
        use_container_width=True
    ):

        if not OSM_AVAILABLE:

            st.error(
                "OSMnx is not installed."
            )

        else:

            with st.spinner(
                "Finding actual road route from OpenStreetMap..."
            ):

                (
                    graph,
                    route,
                    distance_km,
                    error
                ) = find_shortest_road_route(
                    source_coords,
                    destination_coords
                )

            if error:

                st.error(
                    f"❌ Route failed: {error}"
                )

            else:

                eta = calculate_eta(
                    distance_km,
                    traffic_level
                )

                navigation_steps = (
                    create_navigation_steps(
                        graph,
                        route
                    )
                )

                st.session_state.online_graph = graph
                st.session_state.online_route = route
                st.session_state.online_distance = distance_km
                st.session_state.online_eta = eta
                st.session_state.navigation_steps = navigation_steps

                st.success(
                    "✅ Actual shortest road route found."
                )

    # ========================================================
    # ROUTE RESULTS
    # ========================================================

    if st.session_state.online_route is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "📏 Distance",
                format_km(
                    st.session_state.online_distance
                )
            )

        with col2:

            st.metric(
                "⏱️ Estimated Time",
                format_minutes(
                    st.session_state.online_eta
                )
            )

        with col3:

            st.metric(
                "🚦 Traffic",
                traffic_level
            )

        with col4:

            st.metric(
                "🛣️ Road Nodes",
                len(
                    st.session_state.online_route
                )
            )

        st.subheader(
            "🗺️ Shortest Road Route"
        )

        fmap = create_route_map(
            st.session_state.online_graph,
            st.session_state.online_route,
            source_coords,
            destination_coords,
            route_color="blue",
            route_name="Shortest actual road route"
        )

        if fmap is not None:

            st_folium(
                fmap,
                width=None,
                height=550,
                returned_objects=[]
            )

        else:

            st.warning(
                "Map could not be displayed."
            )

        st.subheader(
            "🧭 Navigation"
        )

        steps = st.session_state.navigation_steps

        if steps:

            for step in steps:

                st.write(
                    f"**{step['step']}.** "
                    f"{step['text']}"
                )

            if st.session_state.voice_enabled:

                if st.button(
                    "🔊 Speak Navigation",
                    key="dashboard_speak"
                ):

                    text = " ".join(
                        step["text"]
                        for step in steps
                    )

                    speak_text(text)


# ============================================================
# TRAFFIC PREDICTION PAGE
# ============================================================

def traffic_prediction():

    st.title(
        "🚦 Traffic Congestion Prediction"
    )

    st.markdown(
        """
        Enter traffic conditions and the trained
        machine learning model will predict the
        congestion level.
        """
    )

    if MODEL is None:

        st.error(
            f"Model unavailable.\n\n{MODEL_MESSAGE}"
        )

        return

    if PREPROCESSOR is None:

        st.error(
            f"Preprocessor unavailable.\n\n"
            f"{PREPROCESSOR_MESSAGE}"
        )

        return

    locations = []

    for city in CITY_DATA:

        locations.extend(
            list(
                CITY_DATA[city].keys()
            )
        )

    location = st.selectbox(
        "📍 Location",
        locations
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        hour = st.slider(
            "Hour",
            0,
            23,
            9
        )

    with col2:

        day_name = st.selectbox(
            "Day",
            list(
                DAY_MAPPING.keys()
            )
        )

    with col3:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=0.0
        )

    with col2:

        vehicle_count = st.number_input(
            "Vehicle Count",
            min_value=0,
            max_value=10000,
            value=500
        )

    with col3:

        average_speed = st.number_input(
            "Average Speed (km/h)",
            min_value=1.0,
            max_value=150.0,
            value=35.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        road_capacity = st.number_input(
            "Road Capacity",
            min_value=1,
            max_value=20000,
            value=1000
        )

    with col2:

        accident = st.selectbox(
            "Accident Reported",
            [0, 1]
        )

    with col3:

        road_work = st.selectbox(
            "Road Work",
            [0, 1]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        special_event = st.selectbox(
            "Special Event",
            [0, 1]
        )

    with col2:

        road_distance = st.number_input(
            "Road Distance (km)",
            min_value=0.0,
            max_value=500.0,
            value=5.0
        )

    with col3:

        travel_time = st.number_input(
            "Estimated Travel Time (min)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        road_type = st.selectbox(
            "Road Type",
            [
                "Urban",
                "Highway",
                "Residential",
                "Commercial",
                "Intersection"
            ]
        )

    with col2:

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Rain",
                "Cloudy",
                "Fog"
            ]
        )

    with col3:

        signal = st.selectbox(
            "Traffic Signal Status",
            [
                "Working",
                "Not Working"
            ]
        )

    if st.button(
        "🔮 Predict Traffic",
        type="primary",
        use_container_width=True
    ):

        prediction, probability, error = (
            predict_traffic_level(
                location=location,
                hour=hour,
                day_of_week=DAY_MAPPING[day_name],
                temperature_c=temperature,
                rainfall_mm=rainfall,
                vehicle_count=vehicle_count,
                average_speed_kmph=average_speed,
                road_capacity=road_capacity,
                accident_reported=accident,
                road_work=road_work,
                special_event=special_event,
                road_distance_km=road_distance,
                estimated_travel_time_min=travel_time,
                road_type=road_type,
                weather=weather,
                traffic_signal_status=signal
            )
        )

        if error:

            st.error(error)

        else:

            st.session_state.last_prediction = prediction
            st.session_state.last_prediction_probability = probability

            st.session_state.last_prediction_input = {
                "location": location,
                "hour": hour,
                "day": day_name
            }

            st.success(
                "Traffic prediction completed."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Congestion",
                    str(prediction)
                )

            with col2:

                if probability is not None:

                    st.metric(
                        "Model Confidence",
                        f"{probability:.2f}%"
                    )

                else:

                    st.metric(
                        "Model Confidence",
                        "N/A"
                    )

            level = str(prediction)

            if level in TRAFFIC_COLORS:

                st.info(
                    f"🚦 Traffic level: **{level}**"
                )


# ============================================================
# ROUTE OPTIMIZATION
# ============================================================

def route_optimization():

    st.title(
        "🗺️ Route Optimization"
    )

    st.markdown(
        """
        Find the shortest drivable road route using
        OpenStreetMap road geometry.
        """
    )

    (
        city,
        source_name,
        destination_name,
        source_coords,
        destination_coords
    ) = route_selector(
        "optimization"
    )

    traffic = st.select_slider(
        "🚦 Traffic Condition",
        options=[
            "Low",
            "Medium",
            "High",
            "Severe"
        ],
        value="Medium"
    )

    if st.button(
        "🛣️ Optimize Route",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Calculating shortest road route..."
        ):

            (
                graph,
                route,
                distance,
                error
            ) = find_shortest_road_route(
                source_coords,
                destination_coords
            )

        if error:

            st.error(error)

        else:

            eta = calculate_eta(
                distance,
                traffic
            )

            steps = create_navigation_steps(
                graph,
                route
            )

            st.session_state.online_graph = graph
            st.session_state.online_route = route
            st.session_state.online_distance = distance
            st.session_state.online_eta = eta
            st.session_state.navigation_steps = steps

            st.success(
                "Shortest actual road route calculated."
            )

    if st.session_state.online_route:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Distance",
                format_km(
                    st.session_state.online_distance
                )
            )

        with col2:

            st.metric(
                "ETA",
                format_minutes(
                    st.session_state.online_eta
                )
            )

        with col3:

            st.metric(
                "Traffic",
                traffic
            )

        fmap = create_route_map(
            st.session_state.online_graph,
            st.session_state.online_route,
            source_coords,
            destination_coords,
            route_color="blue"
        )

        if fmap:

            st_folium(
                fmap,
                width=None,
                height=550,
                returned_objects=[]
            )


# ============================================================
# ONLINE OSM ROUTING
# ============================================================

def online_osm_routing():

    st.title(
        "🌍 Online OSM Routing"
    )

    st.info(
        "This module downloads the current "
        "OpenStreetMap drivable-road network "
        "for the selected area."
    )

    if not OSM_AVAILABLE:

        st.error(
            "OSMnx is not installed."
        )

        return

    (
        city,
        source_name,
        destination_name,
        source_coords,
        destination_coords
    ) = route_selector(
        "osm"
    )

    if st.button(
        "🌍 Calculate OSM Road Route",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Downloading road network and calculating route..."
        ):

            (
                graph,
                route,
                distance,
                error
            ) = find_shortest_road_route(
                source_coords,
                destination_coords
            )

        if error:

            st.error(
                f"OSM routing failed: {error}"
            )

        else:

            eta = calculate_eta(
                distance,
                "Medium"
            )

            steps = create_navigation_steps(
                graph,
                route
            )

            st.session_state.online_graph = graph
            st.session_state.online_route = route
            st.session_state.online_distance = distance
            st.session_state.online_eta = eta
            st.session_state.navigation_steps = steps

            st.success(
                "✅ OSM route calculated successfully."
            )

    if st.session_state.online_route:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Road Distance",
                format_km(
                    st.session_state.online_distance
                )
            )

        with col2:

            st.metric(
                "Estimated Time",
                format_minutes(
                    st.session_state.online_eta
                )
            )

        fmap = create_route_map(
            st.session_state.online_graph,
            st.session_state.online_route,
            source_coords,
            destination_coords,
            route_color="blue",
            route_name="OSM shortest road route"
        )

        if fmap:

            st_folium(
                fmap,
                width=None,
                height=600,
                returned_objects=[]
            )


# ============================================================
# OFFLINE GRAPHML ROUTING
# ============================================================

def offline_graphml_routing():

    st.title(
        "💾 Offline GraphML Routing"
    )

    st.markdown(
        """
        Use a saved `.graphml` road network without
        downloading a new OSM network.
        """
    )

    graph_files = find_graphml_files()

    if not graph_files:

        st.warning(
            "No GraphML files were found in the project."
        )

        st.code(
            """
Place your GraphML file somewhere inside:

AI-Based Smart Traffic Congestion Prediction
and Route Optimization/

Example:
data/hyderabad.graphml
            """
        )

        return

    selected_graph = st.selectbox(
        "💾 Select Offline Graph",
        graph_files,
        format_func=lambda x: os.path.relpath(
            x,
            BASE_DIR
        )
    )

    try:

        graph = load_graphml(
            selected_graph
        )

        st.success(
            f"Graph loaded: "
            f"{len(graph.nodes):,} nodes | "
            f"{len(graph.edges):,} edges"
        )

    except Exception as error:

        st.error(
            f"Could not load GraphML: {error}"
        )

        return

    (
        city,
        source_name,
        destination_name,
        source_coords,
        destination_coords
    ) = route_selector(
        "offline"
    )

    if st.button(
        "💾 Find Offline Route",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Calculating offline route..."
        ):

            (
                route,
                distance,
                error
            ) = find_offline_route(
                graph,
                source_coords,
                destination_coords
            )

        if error:

            st.error(
                f"Offline routing failed: {error}"
            )

        else:

            eta = calculate_eta(
                distance,
                "Medium"
            )

            steps = create_navigation_steps(
                graph,
                route
            )

            st.session_state.offline_graph = graph
            st.session_state.offline_route = route
            st.session_state.offline_distance = distance
            st.session_state.offline_eta = eta
            st.session_state.navigation_steps = steps

            st.success(
                "Offline route calculated."
            )

    if st.session_state.offline_route:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Offline Distance",
                format_km(
                    st.session_state.offline_distance
                )
            )

        with col2:

            st.metric(
                "Estimated Time",
                format_minutes(
                    st.session_state.offline_eta
                )
            )

        fmap = create_route_map(
            st.session_state.offline_graph,
            st.session_state.offline_route,
            source_coords,
            destination_coords,
            route_color="purple",
            route_name="Offline GraphML route"
        )

        if fmap:

            st_folium(
                fmap,
                width=None,
                height=600,
                returned_objects=[]
            )


# ============================================================
# ROUTE COMPARISON
# ============================================================

def route_comparison():

    st.title(
        "📊 Route Comparison"
    )

    st.markdown(
        """
        Compare the online OSM route with an offline
        GraphML route when both are available.
        """
    )

    online_distance = (
        st.session_state.online_distance
    )

    offline_distance = (
        st.session_state.offline_distance
    )

    if (
        online_distance is None
        and offline_distance is None
    ):

        st.info(
            "First calculate an Online OSM route "
            "and/or Offline GraphML route."
        )

        return

    rows = []

    if online_distance is not None:

        rows.append({
            "Route": "Online OSM",
            "Distance (km)": round(
                online_distance,
                2
            ),
            "ETA (min)": round(
                st.session_state.online_eta or 0,
                2
            )
        })

    if offline_distance is not None:

        rows.append({
            "Route": "Offline GraphML",
            "Distance (km)": round(
                offline_distance,
                2
            ),
            "ETA (min)": round(
                st.session_state.offline_eta or 0,
                2
            )
        })

    comparison_df = pd.DataFrame(rows)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    if len(comparison_df) >= 2:

        online = comparison_df.iloc[0]
        offline = comparison_df.iloc[1]

        distance_difference = (
            abs(
                online["Distance (km)"]
                - offline["Distance (km)"]
            )
        )

        st.metric(
            "Distance Difference",
            f"{distance_difference:.2f} km"
        )

    if online_distance is not None:

        st.subheader(
            "🌍 Online OSM Route"
        )

        st.write(
            f"Distance: "
            f"**{online_distance:.2f} km**"
        )

    if offline_distance is not None:

        st.subheader(
            "💾 Offline GraphML Route"
        )

        st.write(
            f"Distance: "
            f"**{offline_distance:.2f} km**"
        )


# ============================================================
# TRAFFIC ANALYTICS
# ============================================================

def traffic_analytics():

    st.title(
        "📈 Traffic Analytics"
    )

    df, dataset_path = (
        load_traffic_dataset()
    )

    if df is None:

        st.warning(
            "Traffic dataset not found."
        )

        st.info(
            """
            The application automatically searches
            the project for CSV and Parquet files.

            Recommended location:

            data/traffic_data.csv
            """
        )

        st.code(
            "Get-ChildItem -Path . -Recurse "
            "-File -Include *.csv,*.parquet"
        )

        return

    st.success(
        "Traffic dataset loaded successfully."
    )

    st.caption(
        f"Dataset: "
        f"{os.path.relpath(dataset_path, BASE_DIR)}"
    )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    st.subheader(
        "📋 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        missing = int(
            df.isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            f"{missing:,}"
        )

    with col4:

        numeric_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        st.metric(
            "Numeric Columns",
            numeric_count
        )

    st.divider()

    # ========================================================
    # CONGESTION DISTRIBUTION
    # ========================================================

    congestion_column = None

    possible_columns = [
        "congestion_level",
        "congestion",
        "traffic_level",
        "traffic_congestion"
    ]

    for column in possible_columns:

        if column in df.columns:

            congestion_column = column
            break

    if congestion_column:

        st.subheader(
            "🚦 Congestion Distribution"
        )

        distribution = (
            df[congestion_column]
            .value_counts()
            .reset_index()
        )

        distribution.columns = [
            "Congestion Level",
            "Count"
        ]

        st.bar_chart(
            distribution.set_index(
                "Congestion Level"
            )
        )

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # VEHICLE COUNT
    # ========================================================

    if "vehicle_count" in df.columns:

        st.subheader(
            "🚗 Vehicle Count"
        )

        vehicle_data = (
            pd.to_numeric(
                df["vehicle_count"],
                errors="coerce"
            )
            .dropna()
        )

        if not vehicle_data.empty:

            st.line_chart(
                vehicle_data.reset_index(
                    drop=True
                )
            )

            st.write(
                f"Average vehicles: "
                f"**{vehicle_data.mean():.2f}**"
            )

    # ========================================================
    # AVERAGE SPEED
    # ========================================================

    if "average_speed_kmph" in df.columns:

        st.subheader(
            "🏎️ Average Speed"
        )

        speed_data = (
            pd.to_numeric(
                df["average_speed_kmph"],
                errors="coerce"
            )
            .dropna()
        )

        if not speed_data.empty:

            st.line_chart(
                speed_data.reset_index(
                    drop=True
                )
            )

            st.write(
                f"Average speed: "
                f"**{speed_data.mean():.2f} km/h**"
            )

    # ========================================================
    # LOCATION ANALYSIS
    # ========================================================

    if "location" in df.columns:

        st.subheader(
            "📍 Traffic by Location"
        )

        location_counts = (
            df["location"]
            .value_counts()
        )

        st.bar_chart(
            location_counts
        )

    # ========================================================
    # HOUR ANALYSIS
    # ========================================================

    if "hour" in df.columns:

        st.subheader(
            "⏰ Traffic by Hour"
        )

        hour_series = pd.to_numeric(
            df["hour"],
            errors="coerce"
        )

        hour_counts = (
            hour_series
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            hour_counts
        )

    # ========================================================
    # RAW DATA
    # ========================================================

    with st.expander(
        "🔎 View Dataset"
    ):

        st.dataframe(
            df.head(500),
            use_container_width=True
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "📊 Numerical Summary"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )


# ============================================================
# VOICE NAVIGATION PAGE
# ============================================================

def voice_navigation():

    st.title(
        "🔊 Voice Navigation"
    )

    st.markdown(
        """
        Use browser text-to-speech to hear the
        calculated navigation instructions.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        enabled = st.toggle(
            "🔊 Voice Navigation ON/OFF",
            value=st.session_state.voice_enabled
        )

        st.session_state.voice_enabled = enabled

    with col2:

        if enabled:

            st.success(
                "Voice is ON"
            )

        else:

            st.info(
                "Voice is OFF"
            )

    if st.button(
        "⛔ Stop Voice"
    ):

        stop_voice()

    steps = st.session_state.navigation_steps

    if not steps:

        st.info(
            "Calculate a route first."
        )

        return

    st.subheader(
        "🧭 Current Navigation Route"
    )

    for step in steps:

        st.write(
            f"**{step['step']}.** "
            f"{step['text']}"
        )

    if enabled:

        if st.button(
            "🔊 Speak Full Navigation",
            type="primary",
            use_container_width=True
        ):

            navigation_text = " ".join(
                step["text"]
                for step in steps
            )

            speak_text(
                navigation_text
            )


# ============================================================
# ABOUT PAGE
# ============================================================

def about():

    st.title(
        "ℹ️ About"
    )

    st.markdown(
        """
        # 🚦 AI-Based Smart Traffic Congestion
        # Prediction and Route Optimization

        This project combines **Artificial Intelligence,
        Machine Learning, OpenStreetMap and route
        optimization** to provide a smart traffic
        management system.

        ## 🎯 Main Objectives

        - Predict traffic congestion levels.
        - Calculate actual road routes.
        - Find shortest road-distance routes.
        - Compare online and offline routing.
        - Analyze traffic datasets.
        - Provide navigation instructions.
        - Provide browser-based voice navigation.

        ## 🤖 Machine Learning

        The application uses a trained traffic
        classification model.

        Prediction pipeline:

        **Raw traffic input**
        ↓
        **Production Preprocessor**
        ↓
        **Encoded / Scaled Features**
        ↓
        **Production Traffic Model**
        ↓
        **Traffic Congestion Level**

        ## 🌍 Online Routing

        Online routing uses:

        - OpenStreetMap
        - OSMnx
        - NetworkX

        The selected route is calculated using the
        actual drivable road network.

        ## 💾 Offline Routing

        Offline routing can use saved GraphML
        road-network files.

        ## 🔊 Voice Navigation

        The application uses browser speech synthesis
        for navigation instructions.

        ## 📊 Analytics

        The Analytics module automatically searches
        the project for traffic CSV and Parquet
        datasets.

        ## 🛠️ Technology Stack

        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - NetworkX
        - OSMnx
        - OpenStreetMap
        - Folium
        - Streamlit-Folium
        - GraphML
        """
    )

    st.divider()

    st.subheader(
        "📁 Project Paths"
    )

    st.write(
        f"Project directory: `{BASE_DIR}`"
    )

    st.write(
        f"Model: `{MODEL_PATH}`"
    )

    st.write(
        f"Preprocessor: `{PREPROCESSOR_PATH}`"
    )

    st.write(
        f"Data directory: `{DATA_DIR}`"
    )

    st.divider()

    st.subheader(
        "⚙️ System Status"
    )

    status_df = pd.DataFrame([
        {
            "Component": "Traffic Model",
            "Status": (
                "Available"
                if MODEL is not None
                else "Missing"
            )
        },
        {
            "Component": "Preprocessor",
            "Status": (
                "Available"
                if PREPROCESSOR is not None
                else "Missing"
            )
        },
        {
            "Component": "OSMnx",
            "Status": (
                "Available"
                if OSM_AVAILABLE
                else "Missing"
            )
        },
        {
            "Component": "Folium",
            "Status": (
                "Available"
                if FOLIUM_AVAILABLE
                else "Missing"
            )
        }
    ])

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    page = sidebar()

    if page == "🏠 Dashboard":

        dashboard()

    elif page == "🚦 Traffic Prediction":

        traffic_prediction()

    elif page == "🗺️ Route Optimization":

        route_optimization()

    elif page == "🌍 Online OSM Routing":

        online_osm_routing()

    elif page == "💾 Offline GraphML Routing":

        offline_graphml_routing()

    elif page == "📊 Route Comparison":

        route_comparison()

    elif page == "📈 Traffic Analytics":

        traffic_analytics()

    elif page == "🔊 Voice Navigation":

        voice_navigation()

    elif page == "ℹ️ About":

        about()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
import os
import time
import math
import joblib
import requests
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import folium
import osmnx as ox

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Smart Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "models/production_traffic_model.pkl"
PREPROCESSOR_FILE = "data/processed/preprocessor.pkl"
DATA_FILE = "data/raw/traffic_data.csv"


# ============================================================
# OSM SETTINGS
# ============================================================

OSM_USER_AGENT = "SmartTrafficAI/1.0"

# Maximum distance for detailed road-graph routing.
# This prevents huge downloads from public OSM services.
MAX_DETAILED_ROUTE_KM = 60

# Maximum graph radius.
MAX_GRAPH_RADIUS_METERS = 100000


# ============================================================
# LOCATION DATABASE
# ============================================================

LOCATION_DATA = {

    "Andhra Pradesh": {

        "Anantapur": [
            "Kamalanagar",
            "Ram Nagar",
            "Sapthagiri Circle",
            "Old Town",
            "RTC Bus Stand",
            "Georgepet",
            "Ramnagar",
            "Aravind Nagar"
        ],

        "Visakhapatnam": [
            "Maddilapalem",
            "MVP Colony",
            "Gajuwaka",
            "Dwaraka Nagar",
            "Seethammadhara",
            "Siripuram",
            "Rushikonda",
            "Akkayyapalem"
        ],

        "Vijayawada": [
            "Benz Circle",
            "Governorpet",
            "Patamata",
            "Moghalrajpuram",
            "Bhavanipuram",
            "Auto Nagar",
            "Poranki",
            "Ramavarappadu"
        ],

        "Tirupati": [
            "Tiruchanoor Road",
            "Alipiri",
            "Renigunta Road",
            "Kapila Theertham",
            "MR Palli",
            "Korlagunta",
            "Leela Mahal Circle",
            "RTC Bus Stand"
        ]
    },

    "Telangana": {

        "Hyderabad": [
            "Hitech City",
            "Madhapur",
            "Gachibowli",
            "Kukatpally",
            "Banjara Hills",
            "Jubilee Hills",
            "Ameerpet",
            "Begumpet",
            "Secunderabad",
            "LB Nagar",
            "Kondapur",
            "Miyapur",
            "Mehdipatnam",
            "Dilsukhnagar",
            "Uppal",
            "Kompally",
            "Hafeezpet",
            "Nallagandla"
        ],

        "Warangal": [
            "Hanamkonda",
            "Kazipet",
            "Warangal Fort",
            "Subedari",
            "Nakkalagutta",
            "Balasamudram",
            "Hunter Road",
            "Mulugu Road"
        ]
    },

    "Karnataka": {

        "Bengaluru": [
            "Whitefield",
            "Electronic City",
            "Koramangala",
            "Indiranagar",
            "Hebbal",
            "Marathahalli",
            "Yelahanka",
            "Jayanagar",
            "Malleshwaram",
            "BTM Layout",
            "Rajajinagar",
            "HSR Layout"
        ],

        "Mysuru": [
            "Vijayanagar",
            "Hebbal",
            "Kuvempunagar",
            "Saraswathipuram",
            "Nazarbad",
            "Jayalakshmipuram",
            "Siddhartha Layout",
            "Bannur Road"
        ]
    },

    "Tamil Nadu": {

        "Chennai": [
            "T Nagar",
            "Anna Nagar",
            "Adyar",
            "Velachery",
            "Guindy",
            "Tambaram",
            "OMR",
            "Porur",
            "Ambattur",
            "Egmore"
        ],

        "Coimbatore": [
            "Gandhipuram",
            "RS Puram",
            "Peelamedu",
            "Singanallur",
            "Saibaba Colony",
            "Ukkadam",
            "Saravanampatti",
            "Kuniyamuthur"
        ]
    },

    "Kerala": {

        "Kochi": [
            "Kakkanad",
            "Edappally",
            "Kaloor",
            "Vyttila",
            "Fort Kochi",
            "Palarivattom",
            "Aluva",
            "Marine Drive"
        ],

        "Thiruvananthapuram": [
            "Kowdiar",
            "Kazhakootam",
            "Pattom",
            "Vazhuthacaud",
            "Thampanoor",
            "Kesavadasapuram",
            "Sreekariyam",
            "Palayam"
        ]
    },

    "Puducherry": {

        "Puducherry": [
            "White Town",
            "Lawspet",
            "Muthialpet",
            "Reddiarpalayam",
            "Ariyankuppam",
            "Mudaliarpet",
            "Heritage Town",
            "Kalapet"
        ]
    }
}


# ============================================================
# TRAFFIC OPTIONS
# ============================================================

ROAD_TYPES = [
    "Highway",
    "Arterial",
    "Collector",
    "Local"
]

WEATHER_OPTIONS = [
    "Clear",
    "Cloudy",
    "Rainy",
    "Heavy Rain"
]

SIGNAL_OPTIONS = [
    "Working",
    "Not Working"
]

DAY_OPTIONS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    if not os.path.exists(PREPROCESSOR_FILE):

        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_FILE}"
        )

    return joblib.load(PREPROCESSOR_FILE)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# GEOCODING
# ============================================================

@st.cache_data(ttl=86400)
def geocode_location(
    area,
    city,
    state
):

    query = f"{area}, {city}, {state}, India"

    url = "https://nominatim.openstreetmap.org/search"

    headers = {
        "User-Agent": OSM_USER_AGENT
    }

    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        results = response.json()

        if not results:

            return None

        return {
            "lat": float(results[0]["lat"]),
            "lon": float(results[0]["lon"]),
            "display_name": results[0]["display_name"]
        }

    except Exception as e:

        st.error(
            f"Geocoding failed for {query}: {e}"
        )

        return None


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    d_phi = math.radians(
        lat2 - lat1
    )

    d_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(d_phi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(d_lambda / 2) ** 2
    )

    return (
        2
        * radius
        * math.asin(
            math.sqrt(a)
        )
    )


# ============================================================
# CREATE OVERVIEW MAP
# ============================================================

def create_overview_map(
    source_lat,
    source_lon,
    destination_lat,
    destination_lon,
    source_name,
    destination_name
):

    center_lat = (
        source_lat +
        destination_lat
    ) / 2

    center_lon = (
        source_lon +
        destination_lon
    ) / 2

    route_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # SOURCE
    folium.Marker(
        [
            source_lat,
            source_lon
        ],
        popup=f"Source: {source_name}",
        tooltip="🟢 Source",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(route_map)

    # DESTINATION
    folium.Marker(
        [
            destination_lat,
            destination_lon
        ],
        popup=f"Destination: {destination_name}",
        tooltip="🔴 Destination",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(route_map)

    # DIRECT CONNECTION LINE
    folium.PolyLine(
        [
            [
                source_lat,
                source_lon
            ],
            [
                destination_lat,
                destination_lon
            ]
        ],
        tooltip="Direct geographic distance",
        weight=4,
        opacity=0.7,
        dash_array="8,8"
    ).add_to(route_map)

    route_map.fit_bounds(
        [
            [
                source_lat,
                source_lon
            ],
            [
                destination_lat,
                destination_lon
            ]
        ],
        padding=(30, 30)
    )

    return route_map


# ============================================================
# DOWNLOAD OSM ROAD GRAPH
# ============================================================

@st.cache_resource(show_spinner=False)
def download_road_graph(
    center_lat,
    center_lon,
    distance_meters
):

    graph = ox.graph_from_point(
        (
            center_lat,
            center_lon
        ),
        dist=distance_meters,
        network_type="drive",
        simplify=True
    )

    # Add road speeds.
    graph = ox.add_edge_speeds(
        graph,
        fallback=40
    )

    # Add travel times.
    graph = ox.add_edge_travel_times(
        graph
    )

    # Make our own clean travel-time attribute.
    for u, v, key, data in graph.edges(
        keys=True,
        data=True
    ):

        speed = data.get(
            "speed_kph",
            40
        )

        try:

            speed = float(speed)

        except Exception:

            speed = 40

        speed = max(
            10,
            min(
                speed,
                130
            )
        )

        length = float(
            data.get(
                "length",
                0
            )
        )

        data["speed_kph"] = speed

        data["traffic_travel_time"] = (
            length /
            (speed * 1000 / 3600)
        )

    return graph


# ============================================================
# EDGE DATA
# ============================================================

def get_best_edge(
    graph,
    u,
    v
):

    edge_data = graph.get_edge_data(
        u,
        v
    )

    if edge_data is None:

        return None

    # MultiDiGraph
    if isinstance(edge_data, dict):

        if all(
            isinstance(value, dict)
            for value in edge_data.values()
        ):

            return min(
                edge_data.values(),
                key=lambda item: item.get(
                    "traffic_travel_time",
                    item.get(
                        "length",
                        float("inf")
                    )
                )
            )

        return edge_data

    return None


# ============================================================
# A* HEURISTIC
# ============================================================

def astar_heuristic(
    node_a,
    node_b,
    graph
):

    a = graph.nodes[node_a]
    b = graph.nodes[node_b]

    distance = haversine(
        a["y"],
        a["x"],
        b["y"],
        b["x"]
    )

    # 130 km/h is deliberately used as an upper-bound speed.
    # Therefore straight-line distance / 130 km/h
    # is a lower bound on travel time.
    max_speed_mps = (
        130 / 3.6
    )

    return (
        distance /
        max_speed_mps
    )


# ============================================================
# CALCULATE DIJKSTRA + A*
# ============================================================

def calculate_routes(
    graph,
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):

    source_node = ox.distance.nearest_nodes(
        graph,
        X=source_lon,
        Y=source_lat
    )

    destination_node = ox.distance.nearest_nodes(
        graph,
        X=destination_lon,
        Y=destination_lat
    )

    # --------------------------------------------------------
    # DIJKSTRA
    # --------------------------------------------------------

    dijkstra_route = nx.shortest_path(
        graph,
        source=source_node,
        target=destination_node,
        weight="traffic_travel_time",
        method="dijkstra"
    )

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    astar_route = nx.astar_path(
        graph,
        source=source_node,
        target=destination_node,
        heuristic=lambda a, b: astar_heuristic(
            a,
            b,
            graph
        ),
        weight="traffic_travel_time"
    )

    return (
        dijkstra_route,
        astar_route
    )


# ============================================================
# ROUTE STATISTICS
# ============================================================

def calculate_route_statistics(
    graph,
    route
):

    total_distance = 0
    total_time = 0

    for i in range(
        len(route) - 1
    ):

        u = route[i]
        v = route[i + 1]

        edge = get_best_edge(
            graph,
            u,
            v
        )

        if edge is None:
            continue

        total_distance += float(
            edge.get(
                "length",
                0
            )
        )

        total_time += float(
            edge.get(
                "traffic_travel_time",
                0
            )
        )

    distance_km = (
        total_distance /
        1000
    )

    time_minutes = (
        total_time /
        60
    )

    traffic_cost = (
        distance_km
        +
        time_minutes * 0.5
    )

    return {
        "distance_km": distance_km,
        "time_minutes": time_minutes,
        "traffic_cost": traffic_cost
    }


# ============================================================
# ROUTE MAP
# ============================================================

def create_route_map(
    graph,
    dijkstra_route,
    astar_route,
    source_lat,
    source_lon,
    destination_lat,
    destination_lon,
    source_name,
    destination_name
):

    center_lat = (
        source_lat +
        destination_lat
    ) / 2

    center_lon = (
        source_lon +
        destination_lon
    ) / 2

    route_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    folium.Marker(
        [
            source_lat,
            source_lon
        ],
        popup=f"Source: {source_name}",
        tooltip="🟢 Source",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(route_map)

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    folium.Marker(
        [
            destination_lat,
            destination_lon
        ],
        popup=f"Destination: {destination_name}",
        tooltip="🔴 Destination",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(route_map)

    # --------------------------------------------------------
    # DIJKSTRA
    # --------------------------------------------------------

    dijkstra_coordinates = []

    for node in dijkstra_route:

        node_data = graph.nodes[node]

        dijkstra_coordinates.append(
            [
                node_data["y"],
                node_data["x"]
            ]
        )

    folium.PolyLine(
        dijkstra_coordinates,
        tooltip="Dijkstra Route",
        weight=7,
        opacity=0.85
    ).add_to(route_map)

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    astar_coordinates = []

    for node in astar_route:

        node_data = graph.nodes[node]

        astar_coordinates.append(
            [
                node_data["y"],
                node_data["x"]
            ]
        )

    folium.PolyLine(
        astar_coordinates,
        tooltip="A* Route",
        weight=5,
        opacity=0.75,
        dash_array="10,10"
    ).add_to(route_map)

    # --------------------------------------------------------
    # FIT BOUNDS
    # --------------------------------------------------------

    route_map.fit_bounds(
        [
            [
                source_lat,
                source_lon
            ],
            [
                destination_lat,
                destination_lon
            ]
        ],
        padding=(30, 30)
    )

    return route_map


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🚦 AI-Based Smart Traffic Congestion Prediction"
)

st.markdown(
    """
    Traffic congestion prediction with **Machine Learning**
    and real **OpenStreetMap route optimization** using
    **Dijkstra and A\***.
    """
)


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:

    model = load_model()

    preprocessor = load_preprocessor()

    df = load_dataset()

except Exception as e:

    st.error(
        f"Application loading error: {e}"
    )

    st.stop()


# ============================================================
# TRAFFIC PREDICTION
# ============================================================

st.header(
    "🚗 Traffic Congestion Prediction"
)

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)


# ============================================================
# PREDICTION LOCATION
# ============================================================

with prediction_col1:

    prediction_state = st.selectbox(
        "State",
        list(
            LOCATION_DATA.keys()
        ),
        key="prediction_state"
    )

    prediction_city = st.selectbox(
        "City",
        list(
            LOCATION_DATA[
                prediction_state
            ].keys()
        ),
        key="prediction_city"
    )

    prediction_area = st.selectbox(
        "Area",
        LOCATION_DATA[
            prediction_state
        ][
            prediction_city
        ],
        key="prediction_area"
    )

    road_type = st.selectbox(
        "Road Type",
        ROAD_TYPES
    )

    weather = st.selectbox(
        "Weather",
        WEATHER_OPTIONS
    )


# ============================================================
# PREDICTION TRAFFIC
# ============================================================

with prediction_col2:

    traffic_signal = st.selectbox(
        "Traffic Signal Status",
        SIGNAL_OPTIONS
    )

    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day = st.selectbox(
        "Day",
        DAY_OPTIONS
    )

    vehicle_count = st.number_input(
        "Vehicle Count",
        min_value=0,
        max_value=10000,
        value=1200,
        step=50
    )

    average_speed = st.number_input(
        "Average Speed (km/h)",
        min_value=1.0,
        max_value=150.0,
        value=40.0,
        step=1.0
    )


# ============================================================
# PREDICTION ROAD DETAILS
# ============================================================

with prediction_col3:

    road_capacity = st.number_input(
        "Road Capacity",
        min_value=100,
        max_value=20000,
        value=2500,
        step=100
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=30.0,
        step=0.5
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0
    )

    accident_reported = st.selectbox(
        "Accident Reported",
        ["No", "Yes"]
    )

    road_work = st.selectbox(
        "Road Work",
        ["No", "Yes"]
    )

    special_event = st.selectbox(
        "Special Event",
        ["No", "Yes"]
    )


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "🔮 Predict Traffic Congestion",
    width="stretch"
):

    try:

        # IMPORTANT:
        # day_of_week remains a STRING.
        # The ColumnTransformer handles it as categorical.

        input_df = pd.DataFrame(
            [{
                "location": prediction_area,
                "road_type": road_type,
                "weather": weather,
                "traffic_signal_status": traffic_signal,
                "hour": hour,
                "day_of_week": day,
                "temperature_c": temperature,
                "rainfall_mm": rainfall,
                "vehicle_count": vehicle_count,
                "average_speed_kmph": average_speed,
                "road_capacity": road_capacity,
                "accident_reported": (
                    1
                    if accident_reported == "Yes"
                    else 0
                ),
                "road_work": (
                    1
                    if road_work == "Yes"
                    else 0
                ),
                "special_event": (
                    1
                    if special_event == "Yes"
                    else 0
                ),
                "road_distance_km": 0.0,
                "estimated_travel_time_min": 0.0,
                "congestion_percentage": 0.0
            }]
        )

        processed = preprocessor.transform(
            input_df
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        processed_df = pd.DataFrame(
            processed,
            columns=feature_names
        )

        # Remove leakage feature.
        if (
            "numeric__congestion_percentage"
            in processed_df.columns
        ):

            processed_df = processed_df.drop(
                columns=[
                    "numeric__congestion_percentage"
                ]
            )

        # Match production model.
        processed_df = processed_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(
            processed_df
        )[0]

        probabilities = model.predict_proba(
            processed_df
        )[0]

        confidence = float(
            probabilities.max()
        )

        estimated_time = (
            5
            +
            (
                vehicle_count /
                max(
                    road_capacity,
                    1
                )
            )
            * 15
        )

        st.success(
            f"Predicted Congestion Level: **{prediction}**"
        )

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "Congestion Level",
                str(prediction)
            )

        with result2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

        with result3:

            st.metric(
                "Estimated Travel Time",
                f"{estimated_time:.1f} min"
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# ROUTE OPTIMIZATION
# ============================================================

st.header(
    "🗺️ Real Road Route Optimization"
)

st.info(
    "The app automatically chooses between detailed "
    "road routing and a lightweight overview map based "
    "on the distance between the selected locations."
)


# ============================================================
# SOURCE
# ============================================================

source_col, destination_col = st.columns(2)

with source_col:

    st.subheader(
        "🟢 Source Location"
    )

    source_state = st.selectbox(
        "Source State",
        list(
            LOCATION_DATA.keys()
        ),
        key="source_state"
    )

    source_city = st.selectbox(
        "Source City",
        list(
            LOCATION_DATA[
                source_state
            ].keys()
        ),
        key="source_city"
    )

    source_area = st.selectbox(
        "Source Area",
        LOCATION_DATA[
            source_state
        ][
            source_city
        ],
        key="source_area"
    )


# ============================================================
# DESTINATION
# ============================================================

with destination_col:

    st.subheader(
        "🔴 Destination Location"
    )

    destination_state = st.selectbox(
        "Destination State",
        list(
            LOCATION_DATA.keys()
        ),
        key="destination_state"
    )

    destination_city = st.selectbox(
        "Destination City",
        list(
            LOCATION_DATA[
                destination_state
            ].keys()
        ),
        key="destination_city"
    )

    destination_area = st.selectbox(
        "Destination Area",
        LOCATION_DATA[
            destination_state
        ][
            destination_city
        ],
        key="destination_area"
    )


# ============================================================
# SELECTED JOURNEY
# ============================================================

st.subheader(
    "🚗 Selected Journey"
)

st.write(
    f"**FROM:** {source_area}, "
    f"{source_city}, {source_state}"
)

st.write(
    f"**TO:** {destination_area}, "
    f"{destination_city}, {destination_state}"
)


# ============================================================
# CALCULATE ROUTE
# ============================================================

if st.button(
    "🛣️ Calculate Route",
    width="stretch"
):

    # --------------------------------------------------------
    # GEOCODE SOURCE
    # --------------------------------------------------------

    with st.spinner(
        "Finding the source location..."
    ):

        source_location = geocode_location(
            source_area,
            source_city,
            source_state
        )

    # Respect Nominatim public service rate limit.
    time.sleep(1.1)

    # --------------------------------------------------------
    # GEOCODE DESTINATION
    # --------------------------------------------------------

    with st.spinner(
        "Finding the destination location..."
    ):

        destination_location = geocode_location(
            destination_area,
            destination_city,
            destination_state
        )

    # --------------------------------------------------------
    # CHECK GEOCODING
    # --------------------------------------------------------

    if source_location is None:

        st.error(
            "OpenStreetMap could not find the selected "
            "source area. Try another area."
        )

        st.stop()

    if destination_location is None:

        st.error(
            "OpenStreetMap could not find the selected "
            "destination area. Try another area."
        )

        st.stop()

    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    source_lat = source_location["lat"]
    source_lon = source_location["lon"]

    destination_lat = destination_location["lat"]
    destination_lon = destination_location["lon"]

    st.write(
        f"📍 Source coordinates: "
        f"`{source_lat:.6f}, {source_lon:.6f}`"
    )

    st.write(
        f"📍 Destination coordinates: "
        f"`{destination_lat:.6f}, {destination_lon:.6f}`"
    )

    # --------------------------------------------------------
    # GEOGRAPHIC DISTANCE
    # --------------------------------------------------------

    straight_distance_m = haversine(
        source_lat,
        source_lon,
        destination_lat,
        destination_lon
    )

    straight_distance_km = (
        straight_distance_m / 1000
    )

    st.metric(
        "Direct Geographic Distance",
        f"{straight_distance_km:.1f} km"
    )

    # ========================================================
    # FAR DISTANCE HANDLING
    # ========================================================

    if (
        straight_distance_km
        >
        MAX_DETAILED_ROUTE_KM
    ):

        st.warning(
            f"The selected locations are "
            f"{straight_distance_km:.1f} km apart. "
            f"A detailed OSM road graph would require "
            f"a very large download, so the application "
            f"automatically switches to overview mode."
        )

        st.subheader(
            "🗺️ Location Overview Map"
        )

        overview_map = create_overview_map(
            source_lat,
            source_lon,
            destination_lat,
            destination_lon,
            f"{source_area}, {source_city}",
            f"{destination_area}, {destination_city}"
        )

        st_folium(
            overview_map,
            width=None,
            height=600,
            returned_objects=[]
        )

        st.info(
            "This map shows the selected locations and "
            "their direct geographic connection. "
            "Dijkstra/A* road routing is not calculated "
            "for this long-distance pair."
        )

        st.markdown(
            """
            ### Why detailed routing is skipped

            Downloading a complete driving-road graph for
            a long-distance route can create a very large
            request to the public OpenStreetMap/Overpass
            infrastructure.

            For this project, detailed routing is therefore
            automatically used only when the selected points
            are close enough for a reasonable graph download.
            """
        )

    # ========================================================
    # DETAILED ROUTING
    # ========================================================

    else:

        center_lat = (
            source_lat +
            destination_lat
        ) / 2

        center_lon = (
            source_lon +
            destination_lon
        ) / 2

        # Calculate a graph radius that covers both points.
        graph_radius = max(
            8000,
            int(
                straight_distance_m * 0.70
                + 5000
            )
        )

        graph_radius = min(
            graph_radius,
            MAX_GRAPH_RADIUS_METERS
        )

        st.info(
            f"Detailed road routing enabled. "
            f"Downloading approximately "
            f"{graph_radius / 1000:.1f} km road-network radius."
        )

        try:

            with st.spinner(
                "Downloading OpenStreetMap road network..."
            ):

                graph = download_road_graph(
                    center_lat,
                    center_lon,
                    graph_radius
                )

            st.success(
                f"Road network loaded: "
                f"{len(graph.nodes):,} nodes and "
                f"{len(graph.edges):,} road edges."
            )

            # ------------------------------------------------
            # CALCULATE ROUTES
            # ------------------------------------------------

            with st.spinner(
                "Calculating Dijkstra and A* routes..."
            ):

                dijkstra_route, astar_route = (
                    calculate_routes(
                        graph,
                        source_lat,
                        source_lon,
                        destination_lat,
                        destination_lon
                    )
                )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            dijkstra_stats = (
                calculate_route_statistics(
                    graph,
                    dijkstra_route
                )
            )

            astar_stats = (
                calculate_route_statistics(
                    graph,
                    astar_route
                )
            )

            # ------------------------------------------------
            # MAP
            # ------------------------------------------------

            route_map = create_route_map(
                graph,
                dijkstra_route,
                astar_route,
                source_lat,
                source_lon,
                destination_lat,
                destination_lon,
                f"{source_area}, {source_city}",
                f"{destination_area}, {destination_city}"
            )

            st.subheader(
                "🗺️ Real OpenStreetMap Road Route"
            )

            st_folium(
                route_map,
                width=None,
                height=650,
                returned_objects=[]
            )

            # ------------------------------------------------
            # DIJKSTRA
            # ------------------------------------------------

            st.subheader(
                "🟢 Dijkstra Route"
            )

            d1, d2, d3 = st.columns(3)

            with d1:

                st.metric(
                    "Distance",
                    f"{dijkstra_stats['distance_km']:.2f} km"
                )

            with d2:

                st.metric(
                    "Estimated Time",
                    f"{dijkstra_stats['time_minutes']:.1f} min"
                )

            with d3:

                st.metric(
                    "Traffic Cost",
                    f"{dijkstra_stats['traffic_cost']:.2f}"
                )

            # ------------------------------------------------
            # A*
            # ------------------------------------------------

            st.subheader(
                "⭐ A* Route"
            )

            a1, a2, a3 = st.columns(3)

            with a1:

                st.metric(
                    "Distance",
                    f"{astar_stats['distance_km']:.2f} km"
                )

            with a2:

                st.metric(
                    "Estimated Time",
                    f"{astar_stats['time_minutes']:.1f} min"
                )

            with a3:

                st.metric(
                    "Traffic Cost",
                    f"{astar_stats['traffic_cost']:.2f}"
                )

            # ------------------------------------------------
            # COMPARISON
            # ------------------------------------------------

            st.subheader(
                "📊 Route Comparison"
            )

            comparison = pd.DataFrame(
                {
                    "Algorithm": [
                        "Dijkstra",
                        "A*"
                    ],
                    "Distance (km)": [
                        round(
                            dijkstra_stats[
                                "distance_km"
                            ],
                            2
                        ),
                        round(
                            astar_stats[
                                "distance_km"
                            ],
                            2
                        )
                    ],
                    "Estimated Time (min)": [
                        round(
                            dijkstra_stats[
                                "time_minutes"
                            ],
                            2
                        ),
                        round(
                            astar_stats[
                                "time_minutes"
                            ],
                            2
                        )
                    ],
                    "Traffic Cost": [
                        round(
                            dijkstra_stats[
                                "traffic_cost"
                            ],
                            2
                        ),
                        round(
                            astar_stats[
                                "traffic_cost"
                            ],
                            2
                        )
                    ]
                }
            )

            st.dataframe(
                comparison,
                width="stretch",
                hide_index=True
            )

            st.write(
                f"**Dijkstra route nodes:** "
                f"{len(dijkstra_route)}"
            )

            st.write(
                f"**A* route nodes:** "
                f"{len(astar_route)}"
            )

        except nx.NetworkXNoPath:

            st.error(
                "No drivable road path was found between "
                "the selected locations. Try another pair "
                "of nearby areas."
            )

        except Exception as e:

            st.error(
                f"Detailed route calculation failed: {e}"
            )

            st.info(
                "Try two areas within the same city. "
                "The public OpenStreetMap road-data service "
                "may not be suitable for very large graph downloads."
            )


# ============================================================
# DATASET ANALYTICS
# ============================================================

st.header(
    "📊 Traffic Dataset Analytics"
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Records",
        f"{len(df):,}"
    )

with metric2:

    st.metric(
        "Locations",
        f"{df['location'].nunique():,}"
    )

with metric3:

    st.metric(
        "Average Speed",
        f"{df['average_speed_kmph'].mean():.1f} km/h"
    )

with metric4:

    st.metric(
        "Average Vehicles",
        f"{df['vehicle_count'].mean():.0f}"
    )


# ============================================================
# CONGESTION DISTRIBUTION
# ============================================================

st.subheader(
    "📈 Congestion Distribution"
)

fig1, ax1 = plt.subplots(
    figsize=(9, 4)
)

df["congestion_level"].value_counts().reindex(
    [
        "Low",
        "Medium",
        "High",
        "Severe"
    ],
    fill_value=0
).plot(
    kind="bar",
    ax=ax1
)

ax1.set_xlabel(
    "Congestion Level"
)

ax1.set_ylabel(
    "Number of Records"
)

ax1.set_title(
    "Traffic Congestion Distribution"
)

plt.xticks(
    rotation=0
)

st.pyplot(
    fig1,
    clear_figure=True
)


# ============================================================
# VEHICLES BY HOUR
# ============================================================

st.subheader(
    "🚘 Average Vehicles by Hour"
)

hourly_data = (
    df.groupby("hour")[
        "vehicle_count"
    ]
    .mean()
    .reset_index()
)

fig2, ax2 = plt.subplots(
    figsize=(9, 4)
)

ax2.plot(
    hourly_data["hour"],
    hourly_data["vehicle_count"],
    marker="o"
)

ax2.set_xlabel(
    "Hour"
)

ax2.set_ylabel(
    "Average Vehicle Count"
)

ax2.set_title(
    "Average Traffic Volume by Hour"
)

ax2.grid(
    True,
    alpha=0.3
)

st.pyplot(
    fig2,
    clear_figure=True
)


# ============================================================
# WEATHER
# ============================================================

st.subheader(
    "🌦️ Average Congestion by Weather"
)

weather_data = (
    df.groupby("weather")[
        "congestion_percentage"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)

fig3, ax3 = plt.subplots(
    figsize=(9, 4)
)

weather_data.plot(
    kind="bar",
    ax=ax3
)

ax3.set_xlabel(
    "Weather"
)

ax3.set_ylabel(
    "Average Congestion %"
)

ax3.set_title(
    "Weather vs Congestion"
)

plt.xticks(
    rotation=0
)

st.pyplot(
    fig3,
    clear_figure=True
)


# ============================================================
# LOCATION SUMMARY
# ============================================================

st.subheader(
    "📍 Location Summary"
)

location_summary = (
    df.groupby("location")
    .agg(
        Average_Vehicles=(
            "vehicle_count",
            "mean"
        ),
        Average_Speed=(
            "average_speed_kmph",
            "mean"
        ),
        Average_Congestion=(
            "congestion_percentage",
            "mean"
        ),
        Records=(
            "location",
            "count"
        )
    )
    .sort_values(
        "Average_Congestion",
        ascending=False
    )
    .reset_index()
)

st.dataframe(
    location_summary,
    width="stretch",
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🚦 AI-Based Smart Traffic Congestion Prediction "
    "| Machine Learning + OpenStreetMap + Dijkstra + A*"
)

st.caption(
    "Map data © OpenStreetMap contributors."
)
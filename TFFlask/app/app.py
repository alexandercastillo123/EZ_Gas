from flask import Flask, render_template, request, jsonify, redirect, url_for
import networkx as nx
import numpy as np
from geopy.distance import geodesic
import csv
import requests
import math

app = Flask(__name__)

# Diccionario para almacenar los subgrafos de cada localidad
subgrafos = {}

def cargar_datos():
    """Carga los datos del CSV y genera subgrafos por localidad."""
    gas_stations = {}

    # Leer el archivo CSV
    with open('spanish_gas_stations.csv', mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latitud = float(row.get('latitud'))
            longitud = float(row.get('longitud'))
            localidad = row.get('localidad').title()  # Convertir a título
            municipio = row.get('municipio')

            if localidad not in gas_stations:
                gas_stations[localidad] = []

            gas_stations[localidad].append({
                "latitud": latitud,
                "longitud": longitud,
                "municipio": municipio,
                "localidad": localidad,
                "marca": row.get('rotulo', '').strip()
            })

    # Generar un subgrafo para cada localidad
    for localidad, estaciones in gas_stations.items():
        subgrafos[localidad] = generar_subgrafo(estaciones)

def generar_subgrafo(estaciones):
    """Genera un grafo para una localidad específica."""
    G = nx.Graph()

    # Crear nodos y conectar aquellos dentro del radio de 50 km
    for i, estacion1 in enumerate(estaciones):
        G.add_node(i, **estacion1)

        for j, estacion2 in enumerate(estaciones[i + 1:], start=i + 1):
            distancia = geodesic(
                (estacion1['latitud'], estacion1['longitud']),
                (estacion2['latitud'], estacion2['longitud'])
            ).kilometers

            if distancia <= 100:  # Conectar si la distancia es menor o igual a 100 km
                G.add_edge(i, j, weight=round(distancia, 2))

    return G

def a_star_algorithm(graph, start, end):
    """Implementación del algoritmo A*."""
    start_node = len(graph.nodes)  # Asignar un identificador único al nodo de inicio
    graph.add_node(start_node, **start)

    # Conectar el nodo de inicio a los nodos más cercanos
    for node, data in graph.nodes(data=True):
        if node != start_node:
            distance = geodesic((start['latitud'], start['longitud']), (data['latitud'], data['longitud'])).kilometers
            graph.add_edge(start_node, node, weight=distance)

    open_set = set([start_node])
    closed_set = set()
    g = {}
    parents = {}

    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None

        for v in open_set:
            if n is None or g[v] + heuristic(graph.nodes[v], graph.nodes[end]) < g[n] + heuristic(graph.nodes[n], graph.nodes[end]):
                n = v

        if n is None:
            print('Path does not exist!')
            graph.remove_node(start_node)
            return None

        if n == end:
            path = []

            while parents[n] != n:
                path.append(n)
                n = parents[n]

            path.append(start_node)
            path.reverse()

            # Eliminar el nodo de inicio temporal del grafo
            graph.remove_node(start_node)

            return path

        open_set.remove(n)
        closed_set.add(n)

        for m, edge_data in graph[n].items():
            weight = edge_data['weight']
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n

                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

    print('Path does not exist!')
    # Eliminar el nodo de inicio temporal del grafo
    graph.remove_node(start_node)
    return None

def heuristic(node, end):
    """Función heurística para el algoritmo A*."""
    return geodesic((node['latitud'], node['longitud']), (end['latitud'], end['longitud'])).kilometers

def calcular_combustible_estimado(distancia_km, consumo_litro_100km=7):
    """Calcula el combustible estimado necesario."""
    litros_necesarios = (distancia_km * consumo_litro_100km) / 100
    return round(litros_necesarios, 2)

@app.route('/')
def landing():
    """Página de aterrizaje."""
    return render_template('landing.html')

# Ruta /index eliminada - la landing va directamente a /ruta
# @app.route('/index')
# def index():
#     """Página principal con la lista de localidades."""
#     localidades = list(subgrafos.keys())
#     ordenar = request.args.get('ordenar')
#     if ordenar == 'alfabeticamente':
#         localidades.sort()
#     return render_template('index.html', localidades=localidades)

@app.route('/grafo/<localidad>')
def mostrar_grafo(localidad):
    """Muestra un mapa interactivo con las estaciones de la localidad."""
    localidad = localidad.title()  # Convertir a título
    if localidad not in subgrafos:
        return f"No hay datos para la localidad: {localidad}", 404

    # Extraer los datos de las estaciones en la localidad
    estaciones = [
        {"latitud": data['latitud'], "longitud": data['longitud'], "municipio": data['municipio']}
        for _, data in subgrafos[localidad].nodes(data=True)
    ]

    return render_template('graph.html', localidad=localidad, estaciones=estaciones)

@app.route('/buscar', methods=['POST'])
def buscar():
    """Ruta para buscar una localidad específica."""
    localidad = request.form.get('localidad').title()  # Convertir a título
    if localidad in subgrafos:
        return jsonify({"url": f"/grafo/{localidad}"})
    return jsonify({"error": "Localidad no encontrada"}), 404

@app.route('/sugerencias', methods=['GET'])
def sugerencias():
    """Ruta para obtener sugerencias de búsqueda."""
    query = request.args.get('q', '').lower()
    sugerencias = [loc for loc in subgrafos.keys() if loc.lower().startswith(query)]
    return jsonify(sugerencias)

@app.route('/ruta')
def mostrar_ruta():
    """Muestra un mapa interactivo para ingresar la ubicación del usuario y ver la ruta."""
    estaciones = []
    for localidad, subgrafo in subgrafos.items():
        estaciones.extend([
            {
                "latitud": data['latitud'],
                "longitud": data['longitud'],
                "municipio": data['municipio'],
                "localidad": localidad,
                "node_id": node,
                "marca": data.get('marca', '')
            }
            for node, data in subgrafo.nodes(data=True)
        ])

    return render_template('route.html', estaciones=estaciones)

@app.route('/calcular_ruta', methods=['POST'])
def calcular_ruta():
    """Calcula la mejor ruta desde la ubicación del usuario a la estación más cercana."""
    payload = request.get_json(force=True)
    localidad = str(payload.get('localidad', '')).title()
    origen = payload.get('origen')
    destino = payload.get('destino')
    start_lat = payload.get('start_lat')
    start_lng = payload.get('start_lng')

    if not localidad or destino is None:
        return jsonify({"error": "Faltan parámetros: localidad y destino"}), 400

    subgrafo = subgrafos.get(localidad)
    if subgrafo is None:
        return jsonify({"error": "Localidad no encontrada"}), 404

    try:
        destino = int(destino)
    except (TypeError, ValueError):
        return jsonify({"error": "Los identificadores de estación deben ser numéricos"}), 400

    if destino not in subgrafo.nodes:
        return jsonify({"error": "Estación de destino no encontrada"}), 404

    # Trabajar sobre una copia para no mutar el grafo base
    grafo_trabajo = subgrafo.copy()

    # Determinar el nodo de origen (estación o punto manual)
    if start_lat is not None and start_lng is not None:
        try:
            start_lat = float(start_lat)
            start_lng = float(start_lng)
        except (TypeError, ValueError):
            return jsonify({"error": "Coordenadas de inicio inválidas"}), 400

        origen = "punto_manual"
        grafo_trabajo.add_node(origen, latitud=start_lat, longitud=start_lng, municipio="Ubicación elegida", localidad=localidad)
        # Conectar solo a las 3 estaciones más cercanas para que use el grafo existente
        distancias = []
        for node_id, data in subgrafo.nodes(data=True):
            distancia = geodesic((start_lat, start_lng), (data['latitud'], data['longitud'])).kilometers
            distancias.append((node_id, distancia))
        distancias.sort(key=lambda x: x[1])
        for node_id, dist in distancias[:3]:  # Solo las 3 más cercanas
            grafo_trabajo.add_edge(origen, node_id, weight=round(dist, 2))
    else:
        if origen is None:
            return jsonify({"error": "Selecciona estación de origen o un punto en el mapa"}), 400
        try:
            origen = int(origen)
        except (TypeError, ValueError):
            return jsonify({"error": "Los identificadores de estación deben ser numéricos"}), 400
        if origen not in grafo_trabajo.nodes:
            return jsonify({"error": "Estación de origen no encontrada"}), 404

    try:
        path_nodes = nx.shortest_path(grafo_trabajo, source=origen, target=destino, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return jsonify({"error": "No se encontró una ruta"}), 404

    # Obtener coordenadas de los nodos intermedios
    waypoints = []
    for node in path_nodes:
        data = grafo_trabajo.nodes[node]
        waypoints.append(f"{data['longitud']},{data['latitud']}")
    
    # Usar OSRM para obtener la ruta real por carretera (GRATIS)
    tiempo_estimado = None
    distancia_km = None
    velocidad_promedio = None
    combustible_estimado = None
    
    try:
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{';'.join(waypoints)}"
        osrm_params = {'overview': 'full', 'geometries': 'geojson', 'annotations': 'true'}
        response = requests.get(osrm_url, params=osrm_params, timeout=10)
        
        if response.status_code == 200:
            osrm_data = response.json()
            if osrm_data.get('routes'):
                route = osrm_data['routes'][0]
                
                # Extraer coordenadas de la ruta real
                route_coords = route['geometry']['coordinates']
                path_coordinates = [{"latitud": coord[1], "longitud": coord[0]} for coord in route_coords]
                
                # Obtener tiempo estimado (en segundos) y distancia (en metros)
                tiempo_estimado = round(route['duration'] / 60, 1)  # Convertir a minutos
                distancia_km = round(route['distance'] / 1000, 2)  # Convertir a kilómetros
                
                # Calcular velocidad promedio
                if distancia_km > 0 and tiempo_estimado > 0:
                    velocidad_promedio = round(distancia_km / (tiempo_estimado / 60), 1)  # km/h
                
                # Calcular combustible estimado (consumo promedio 7L/100km)
                combustible_estimado = calcular_combustible_estimado(distancia_km)
            else:
                # Fallback a línea recta si OSRM falla
                path_coordinates = [{"latitud": grafo_trabajo.nodes[node]['latitud'], "longitud": grafo_trabajo.nodes[node]['longitud']} for node in path_nodes]
        else:
            # Fallback a línea recta si OSRM falla
            path_coordinates = [{"latitud": grafo_trabajo.nodes[node]['latitud'], "longitud": grafo_trabajo.nodes[node]['longitud']} for node in path_nodes]
    except Exception as e:
        # Fallback a línea recta en caso de error
        path_coordinates = [{"latitud": grafo_trabajo.nodes[node]['latitud'], "longitud": grafo_trabajo.nodes[node]['longitud']} for node in path_nodes]

    return jsonify({
        "path": path_coordinates,
        "tiempo_estimado_minutos": tiempo_estimado,
        "distancia_km": distancia_km,
        "velocidad_promedio": velocidad_promedio,
        "combustible_estimado_litros": combustible_estimado
    })

if __name__ == '__main__':
    # Cargar los datos del CSV al iniciar la aplicación
    cargar_datos()
    app.run(debug=True)
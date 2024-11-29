import itertools
import networkx as nx
import pandas as pd
import igraph as ig
import numpy as np
from sklearn.model_selection import ParameterGrid


def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# NODE DISTRIBUTION
#todo
# symetrycznosc grafu
# zachowanie communities blisko siebie
# kąt miedzy krawędziami

def node_distribution(posdf):
    total_sum = 0
    for (i, row_i), (j, row_j) in itertools.combinations(posdf.iterrows(), 2):
        dij = distance(row_i['X'], row_i['Y'], row_j['X'], row_j['Y'])
        if dij != 0:  # Avoid division by zero
            total_sum += 1 / dij**2
    return total_sum

# DISTANCE TO BORDERLINES

def distance_to_borderlines(posdf):
    total_sum = 0
    for index, row in posdf.iterrows():
        x, y = row['X'], row['Y']
        # Calculate distances to sides
        ri = 1 - x  
        li = 1 + x  
        ti = 1 - y  
        bi = 1 + y  
        # to avoid diving by zero
        ri += .01 if ri==0 else ri
        ti += .01 if ti==0 else ti
        li += .01 if li==0 else li
        bi += .01 if bi==0 else bi

        total_sum += (1 / (ri ** 2) + 1 / (ti ** 2) + 1 / (li ** 2) + 1 / (bi ** 2))
    return round(total_sum, 3)


# EDGE LENGTHS SUM

def edge_length_sum(graph, posdf):
    total_length = 0
    for u, v in graph.edges():
        x1, y1 = posdf.loc[u, ['X', 'Y']]
        x2, y2 = posdf.loc[v, ['X', 'Y']]
        edge_length = distance(x1, y1, x2, y2)
        total_length += edge_length**2
    return total_length


# NODE TO EDGE DISTANCE

def point_to_segment_distance(x, y, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_squared))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5

def edge_node_distance_contribution(G, pos_df, ):
    total_contribution = 0
    g_min = 5 #cannot be more, the plane is restricted to -1 to 1
    for node in G.nodes():
        x_node, y_node = pos_df.loc[node, 'X'], pos_df.loc[node, 'Y']
        for u, v in G.edges():
            #if node in the edge - skip 
            if node == u or node == v:
                continue
            else:
                x1, y1 = pos_df.loc[u, 'X'], pos_df.loc[u, 'Y']
                x2, y2 = pos_df.loc[v, 'X'], pos_df.loc[v, 'Y']
                distance = point_to_segment_distance(x_node, y_node, x1, y1, x2, y2)
                if distance < g_min:
                    g_min = distance
                contribution = 1 / (distance ** 2) if distance != 0 else 0  # Avoid division by zero
                total_contribution += contribution
    # lam4 = lam5/g_min**2
    lam4 = 1/g_min**2
    return (total_contribution, lam4)

#EDGE INTERSECTIONS

def intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate slopes
    slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    slope2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
    # Calculate intercepts
    intercept1 = y1 - slope1 * x1 if slope1 != float('inf') else x1
    intercept2 = y3 - slope2 * x3 if slope2 != float('inf') else x3
    # Check if segments are parallel
    if slope1 == slope2:
        return False
    # Calculate intersection point
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    # Check if intersection point lies within both line segments
    if min(x1, x2) <= x_intersect <= max(x1, x2) and \
       min(x3, x4) <= x_intersect <= max(x3, x4) and \
       min(y1, y2) <= y_intersect <= max(y1, y2) and \
       min(y3, y4) <= y_intersect <= max(y3, y4):
        return True
    return False

def count_edge_crossings(graph, pos_df):
    crossings = 0
    # lam4 = lam5/g_min**2

    for (u1, v1), (u2, v2) in itertools.combinations(graph.edges(), 2):
        if len(set([u1, v1, u2, v2])) != 4:
            continue
        else:
            line1 = (pos_df.loc[u1, 'X'], pos_df.loc[u1, 'Y'], pos_df.loc[v1, 'X'], pos_df.loc[v1, 'Y'])
            line2 = (pos_df.loc[u2, 'X'], pos_df.loc[u2, 'Y'], pos_df.loc[v2, 'X'], pos_df.loc[v2, 'Y'])
            if intersect(line1, line2):
                crossings += 1
                # print(u1, v1)
                # print(u2, v2)
                # print('crossed')
    return crossings**2  

from sklearn.metrics import silhouette_score

from scipy.spatial.distance import pdist, squareform


def intra_cluster_distance(G, posdf):
    """
    Calculate the average intra-cluster distance for each cluster and overall.
    
    :param positions: Array-like, positions of the points (coordinates)
    :param labels: List of cluster labels corresponding to each point
    :return: Dictionary containing intra-cluster distances for each cluster and overall average
    """

    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
    G_ig.add_vertices(list(missing_vertices))

    resolutions = np.linspace(0.1, 1.5, 10)  # Adjust the range as needed
    param_grid = {'resolution': resolutions}
    grid = ParameterGrid(param_grid)

    best_modularity = -np.inf
    best_partition = None
    for params in grid:
        partition = G_ig.community_leiden(objective_function="modularity", **params)
        modularity = G_ig.modularity(partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    list_comms = [i for i, com in enumerate(best_partition) for node in com]

    unique_labels = np.unique(list_comms)
    intra_cluster_distances = {}
    total_distance = 0

    for label in unique_labels:
        cluster_points = posdf[np.array(list_comms) == label]
        if len(cluster_points) > 1:
            # Calculate pairwise distances between all points in the cluster
            pairwise_distances = pdist(cluster_points)
            cluster_distance_sum = np.sum(pairwise_distances)
            intra_cluster_distances[label] = cluster_distance_sum
            total_distance += cluster_distance_sum
        else:
            # For clusters with a single point, distance is 0
            intra_cluster_distances[label] = 0.0

    intra_cluster_distances['overall_sum'] = total_distance
    return intra_cluster_distances

#using silhouette score
def measure_communities_closeness(G, posdf):
    G_ig = ig.Graph.TupleList(nx.to_edgelist(G), directed=False)
    missing_vertices = set(G.nodes()) - set(G_ig.vs['name'])
    G_ig.add_vertices(list(missing_vertices))

    resolutions = np.linspace(0.1, 1.5, 10)  # Adjust the range as needed
    param_grid = {'resolution': resolutions}
    grid = ParameterGrid(param_grid)

    best_modularity = -np.inf
    best_partition = None
    for params in grid:
        partition = G_ig.community_leiden(objective_function="modularity", **params)
        modularity = G_ig.modularity(partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    list_comms = [i for i, com in enumerate(best_partition) for node in com]
    sil_score = silhouette_score(posdf, list_comms)

    return sil_score

    #todo 
    #potraktowac list_comms jako true labels
    # ari_scores['Leiden'] = adjusted_rand_score(true_labels, list_comms)

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def sum_of_angles(G, posdf):
    total_angle = 0
    edges = list(G.edges())
    
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            
            p1 = posdf.loc[u1, ['X', 'Y']].values
            p2 = posdf.loc[v1, ['X', 'Y']].values
            p3 = posdf.loc[u2, ['X', 'Y']].values
            p4 = posdf.loc[v2, ['X', 'Y']].values
            vec1 = p2 - p1
            vec2 = p4 - p3
            
            angle = calculate_angle(vec1, vec2)
            angle = angle/360
            total_angle += min(angle, .5 - angle)
    return total_angle


# To measure the symmetry of a graph visualization, 
# you can use a metric based on the distribution of node positions. 
# One approach is to calculate the variance of the distances of nodes 
# from the center of the graph. A lower variance indicates higher symmetry.

def measure_graph_symmetry(G, posdf):
    # Calculate the center of the graph
    center_x = posdf['X'].mean()
    center_y = posdf['Y'].mean()
    
    # Calculate the distances of nodes from the center
    distances = np.sqrt((posdf['X'] - center_x)**2 + (posdf['Y'] - center_y)**2)
    
    # Calculate the variance of the distances
    variance = np.var(distances)
    
    # Invert the variance to get a symmetry score (lower variance means higher symmetry)
    symmetry_score = 1 / (1 + variance)
    
    return symmetry_score


def g_visualisation_metric(G, pos, lam1=.1, lam2=.001, lam3=.01, lam5=.1):
    posdf = pd.DataFrame.from_dict(pos, orient='index', columns=['X', 'Y'])
    pen1 = lam1 * node_distribution(posdf, lam1)
    pen2 = lam2 * distance_to_borderlines(posdf, lam2)
    pen3 = lam3 * edge_length_sum(G, posdf, lam3)
    (pen5, lam4) = lam5*edge_node_distance_contribution(G, posdf)
    pen4 = lam4 * count_edge_crossings(G, posdf)
    print(f'"Node distribution : {pen1}')
    print(f'Borderlines : {pen2}')
    print(f'Edge Lengths : {pen3}')
    print(f'Edge crossings : {pen4}')
    print(f'Edge Node distance : {pen5}')
    return pen1 + pen2 + pen3 + pen4 + pen5
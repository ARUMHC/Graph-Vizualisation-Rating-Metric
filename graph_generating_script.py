import random
import numpy as np   
import networkx as nx

def generate_G(sizes, inside_prob, outside_prob):
    probs = np.eye(len(sizes)) * inside_prob

    # Set the off-diagonal elements to the desired value (0.01)
    probs[probs == 0] = outside_prob
    true_labels=[]
    i=0
    for size in sizes:
        true_labels += ([i]*size)
        i += 1
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 200))
    for node, community in zip(G.nodes(), true_labels):
        G.nodes[node]['community'] = community
        
    return (G, true_labels)


def generate_G_randomized(n_vertex, n_comms, inside_prob, outside_prob, dispersion=.35):
    base_value = n_vertex / n_comms
    max_deviation = base_value * dispersion
    deviations = np.random.uniform(-max_deviation, max_deviation, n_comms)
    communities = (base_value + deviations).astype(int)
    
    # Adjust the sum to be close to n_vertex if necessary
    diff = n_vertex - np.sum(communities)
    while diff != 0:
        for i in range(abs(diff)):
            index = np.random.randint(0, n_comms)
            if diff > 0:
                if (communities[index] + 1) <= (base_value + max_deviation):
                    communities[index] += 1
                    diff -= 1
            elif diff < 0:
                if (communities[index] - 1) >= (base_value - max_deviation):
                    communities[index] -= 1
                    diff += 1

    sizes = communities.tolist()
    
    #Generate the graph part
    
    probs = np.eye(len(sizes)) * inside_prob
    probs[probs == 0] = outside_prob
    true_labels=[]
    i=0
    for size in sizes:
        true_labels += ([i]*size)
        i += 1
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 200))
    for node, community in zip(G.nodes(), true_labels):
        G.nodes[node]['community'] = community
        
    return (G, true_labels)


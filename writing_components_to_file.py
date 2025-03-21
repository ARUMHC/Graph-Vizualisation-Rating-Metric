import sys
from pathlib import Path

sys.path.insert(0,str(Path(r'C:\Users\ku51015\CHMURA\mystuff\Graph-Vizualisation-Rating-Metric-randoms\Graph-Vizualisation-Rating-Metric-randoms').resolve()))
# sys.path.insert(0,str(Path(r'C:\Users\Kinga\Desktop\MAGISTERKA\Graph-Vizualisation-Rating-Metric-3').resolve()))

from graph_generating_script import *
from graph_metric_script import *
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pickle
import seaborn as sns
from tqdm import tqdm



def write_calculated_components_to_file(annotations_path, category):
    # annotations = pd.read_excel('randoms_ER_annotations_scores.xlsx')
    annotations = pd.read_excel(annotations_path)
    import os 
    posdf_dir = 'Graph-Vizualisation-Rating-Metric-randoms/' + category + '/pos_dfs'
    graph_dir = 'Graph-Vizualisation-Rating-Metric-randoms/' + category + '/graph_objects'


    calculated_components_df = pd.DataFrame(columns=['posdf_id', 'score', 
                                                    'node_distribution_unreversed','node_distribution', 'node_distribution_raw', 
                                                    'distance_to_borderlines_unreversed','distance_to_borderlines', 'distance_to_borderlines_raw', 
                                                    'edge_length_sum', 'edge_length_sum_raw', 
                                                    'edge_node_distance_contribution_unreversed','edge_node_distance_contribution','edge_node_distance_contribution_raw', 
                                                    'count_edge_crossings',  'count_edge_crossings_raw',
                                                    'communities_closeness', 'sum_of_angles', 'symmetry'])

    # Iterate through files in the posdf_dir
    skip_layouts = ['lgl']

    for posdf_file in tqdm(os.listdir(posdf_dir)):

        if posdf_file.split('_')[1].split('.')[0] not in skip_layouts:
            # reading data

            if posdf_file.endswith('.csv'):
                posdf = pd.read_csv(os.path.join(posdf_dir, posdf_file))
                # print(f'Loaded {posdf_file}')

            graph_id = posdf_file.split('_')[0]
            layout = posdf_file.split('.')[0]
            if int(graph_id) < 10:
                layout = layout[2:]
            else:
                layout = layout[3:]
            score = annotations[annotations['graph_id'] == int(graph_id)][layout].values[0]


            graph_file = 'graph_' + graph_id + '.pkl' 
            with open(os.path.join(graph_dir, graph_file), 'rb') as f:
                G = pickle.load(f)
                
            components_dict = {
                # what to leave  -> conclusions from correlation analysis and vibes
                # node_distribtuion, edge_to_brodelines_raw, edge_length_sum, edge_node_distance_contribution_raw, count_edge crossings_raw
                'posdf_id': posdf_file,
                'score': score,
                'node_distribution_unreversed': node_distribution_unreversed(posdf),
                'node_distribution': node_distribution(posdf),
                'node_distribution_raw': node_distribution_raw(posdf),
                'distance_to_borderlines_unreversed': distance_to_borderlines_unreversed(posdf),
                'distance_to_borderlines': distance_to_borderlines(posdf),
                'distance_to_borderlines_raw': distance_to_borderlines_raw(posdf),
                'edge_length_sum': edge_length_sum(G, posdf),
                'edge_length_sum_raw': edge_length_sum_raw(G, posdf),
                'edge_node_distance_contribution_unreversed': edge_node_distance_contribution_unreversed(G, posdf)[0],
                'edge_node_distance_contribution': edge_node_distance_contribution(G, posdf)[0],
                'edge_node_distance_contribution_raw': edge_node_distance_contribution_raw(G, posdf)[0],

                'count_edge_crossings': count_edge_crossings(G, posdf),
                'count_edge_crossings_raw': count_edge_crossings_raw(G, posdf),

                'communities_closeness' : intra_cluster_distance(G, posdf)['overall_sum'],
                'sum_of_angles' : sum_of_angles(G, posdf),
                'symmetry' : measure_graph_symmetry(G, posdf)
            }
            new_row_df = pd.DataFrame([components_dict])
            calculated_components_df = pd.concat([calculated_components_df, new_row_df], ignore_index=True)
            calculated_components_df_numeric = calculated_components_df.drop(columns=['posdf_id'])
    
    calculated_components_df_numeric['category'] = category
    calculated_components_df_numeric.to_csv(rf'Graph-Vizualisation-Rating-Metric-randoms\calculated_components\calculated_components_{category}.csv', index=False)
    # Calculate the correlation matrix
    # correlation_matrix = calculated_components_df_numeric.corr()

    # Extract the correlation of all values to 'score'
    # score_correlation = correlation_matrix.loc[['score']]
    # score_correlation['category'] = category
  

    return calculated_components_df_numeric


def main():
    print(os.getcwd())
    #todo later change this paths
    df_ws =write_calculated_components_to_file(r'Graph-Vizualisation-Rating-Metric-randoms\WS\WS_annotations_scores.xlsx','WS')
    df_ba = write_calculated_components_to_file(r'Graph-Vizualisation-Rating-Metric-randoms\BA\BA_annotations_scores.xlsx','BA')
    df_er = write_calculated_components_to_file(r'Graph-Vizualisation-Rating-Metric-randoms\ER\ER_annotations_scores.xlsx','ER')
    df_rgg = write_calculated_components_to_file(r'Graph-Vizualisation-Rating-Metric-randoms\RGG\RGG_annotations_scores.xlsx','RGG')
    all_dfs = pd.concat([df_ws, df_ba, df_er, df_rgg], ignore_index=True) 
    all_dfs.to_csv(rf'Graph-Vizualisation-Rating-Metric-randoms\calculated_components\calculated_components_all.csv', index=False)



if __name__ == '__main__':
    main()
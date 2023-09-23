# com_city_clusters.py
import datetime

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from collections import defaultdict
import random
import matplotlib.font_manager as font_manager

task_path = ''
plt.rcParams.update({"font.family": "STIXGeneral",
                     "font.size": 16,
                     "mathtext.fontset": "cm"})


def generate_graph():
    with open('data_result_city_cluter.pkl', 'rb') as fid:
        _ = pickle.load(fid)
        _ = pickle.load(fid)
        _ = pickle.load(fid)
        df_network_weight = pickle.load(fid)
        df_network_2d = pickle.load(fid)

    df = df_network_2d[df_network_2d['huoliang'] != 0]
    # 创建图G
    G = nx.DiGraph()  # create a directed graph

    # Loop through each row in the dataframe and add edges
    for i, row in df.iterrows():
        # Add an edge from 'from_city' to 'to_city' with weight 'huoliang'
        G.add_edge(row['from_city'], row['to_city'], weight=row['huoliang'], weight_reversed=1 / row['huoliang'])

        # Add node attributes 'group_id' and 'group_id_to'
        G.nodes[row['from_city']]['group_id'] = row['group_id']
        G.nodes[row['to_city']]['group_id'] = row['group_id_to']
    return G, df


# intra_group_edges = {}
#
# # Loop over all nodes in the graph
# for node in G.nodes:
#     # Initialize the count for this node
#     intra_group_edges[node] = 0
#
#     # Get the group id for this node
#     group_id = G.nodes[node]['group_id']
#
#     # Loop over all connected nodes (neighbors)
#     for neighbor in G.neighbors(node):
#         # If the neighbor belongs to the same group, increment the count
#         if G.nodes[neighbor].get('group_id') == group_id:
#             intra_group_edges[node] += 1
#
# print(intra_group_edges)

# Prepare data for DataFrame
# data = []
# for node in G.nodes:
#     node_data = {
#         "node": node,
#         "group_id": G.nodes[node]['group_id'],
#         "intra_group_edges": intra_group_edges[node]
#     }
#     data.append(node_data)
#
# # Create DataFrame
# df_intra_group = pd.DataFrame(data)

def cal_intergroup_no_weight(G):
    result_dict = defaultdict(list)

    # Loop through each node in the graph
    # 计算连边数量
    for node in G.nodes():
        # Get group_id of the node
        node_group_id = G.nodes[node]['group_id']

        # Define a dictionary to count edges to each group
        edge_to_group_count = defaultdict(int)

        # Loop through each edge from the node
        for _, neighbor_node in G.edges(node):
            # Get the group_id of the neighbor node
            neighbor_group_id = G.nodes[neighbor_node]['group_id']

            # If the neighbor node's group_id is not the same as the node's group_id, increment the count
            # if neighbor_group_id != node_group_id:
            edge_to_group_count[neighbor_group_id] += 1

        # Store the results in result_dict
        for group_id_to, inter_group_edges in edge_to_group_count.items():
            result_dict['node'].append(node)
            result_dict['group_id'].append(node_group_id)
            result_dict['group_id_to'].append(group_id_to)
            result_dict['inter_group_edges'].append(inter_group_edges)

    # Convert result_dict to a DataFrame
    df_inter_group = pd.DataFrame(result_dict)
    df_inter_group.sort_values(['node', 'group_id', 'group_id_to'], inplace=True)
    return df_inter_group


# print(df_inter_group)
def cal_intergroup_weighted(G):
    # 计算连边权重
    data = []

    # Iterate over each edge in the graph
    for edge in G.edges(data=True):
        node = edge[0]
        node_to = edge[1]
        weight = edge[2][
            'weight']  # Extract weight information. Change this if your weight attribute is named differently.

        # Extract 'group_id' and 'group_id_to' attributes from the nodes. Change this if your group attributes are named differently.
        group_id = G.nodes[node]['group_id']
        group_id_to = G.nodes[node_to]['group_id']

        # Append the data to the list
        data.append([node, group_id, group_id_to, weight])

    # Create a DataFrame from the list
    df = pd.DataFrame(data, columns=['node', 'group_id', 'group_id_to', 'inter_group_edges_weight'])

    # Group by 'node', 'group_id', and 'group_id_to', and sum the 'inter_group_edges_weight'
    df_grouped = df.groupby(['node', 'group_id', 'group_id_to'], as_index=False).agg(
        {'inter_group_edges_weight': np.sum})
    df_grouped.sort_values(['node', 'group_id', 'group_id_to'], inplace=True)
    return df_grouped


def calculate_p_coeff(G, vi):
    # Get the total degree of node vi
    e_vi = G.degree(vi)

    # Calculate the sum of squared ratios for each module
    sum_squared_ratios = 0
    modules = nx.get_node_attributes(G, 'group_id')  # Assuming the modules are stored as node attributes

    for module in set(modules.values()):
        module_nodes = [n for n, m in modules.items() if m == module]
        e_j_vi = len([n for n in G.neighbors(vi) if n in module_nodes])
        ratio = e_j_vi / e_vi if e_vi != 0 else 0
        sum_squared_ratios += ratio ** 2

    # Calculate p^vi
    p_vi = 1 - sum_squared_ratios

    return p_vi


def calculate_p_coeff_w(G, node):
    e_vi = 0  # Total degree of node
    e_j_vi = defaultdict(int)  # Degrees within each module

    # Calculate the degree within each module and the total degree
    for neighbor in G.neighbors(node):
        weight = G[node][neighbor].get('weight', 1)  # Get weight of edge, default to 1 if not present
        e_vi += weight
        module = G.nodes[neighbor].get('group_id')
        e_j_vi[module] += weight

    # Calculate p^vi
    p_vi = 1 - sum((e_j / e_vi) ** 2 for e_j in e_j_vi.values())
    return p_vi


def calculate_z_e(G, df_intra_group):
    # Add a new column to df_intra_group for z^e_{v_i}
    df_intra_group['z_e'] = np.nan

    # Calculate the average and standard deviation of intra_group_edges for each group
    group_avg_std = df_intra_group.groupby('group_id')['inter_group_edges_weight'].agg(['max', 'min'])

    # Loop through each row of df_intra_group
    for index, row in df_intra_group.iterrows():
        group_id = row['group_id']
        # Calculate z^e_{v_i} using the formula
        df_intra_group.loc[index, 'z_e'] = (row['inter_group_edges_weight'] - group_avg_std.loc[group_id, 'min']) / \
                                           (group_avg_std.loc[group_id, 'max'] - group_avg_std.loc[group_id, 'min'])
        df_intra_group.loc[index, 'participant'] = calculate_p_coeff_w(G, df_intra_group.loc[index, 'node'])
    return df_intra_group


# 计算最短路
def cal_shortest_path_lengths(G):
    shortest_path_lengths = {}
    # Iterate through each pair of nodes
    for i in G.nodes:
        print(i)
        shortest_path_lengths[i] = {}
        for j in G.nodes:
            if i != j:
                try:
                    # Calculate the shortest path length
                    length = nx.shortest_path_length(G, source=i, target=j, weight='weight_reversed')
                    shortest_path_lengths[i][j] = length
                except nx.NetworkXNoPath:
                    # If there is no path, we can set the length as infinity or any large number
                    shortest_path_lengths[i][j] = float('inf')
    return shortest_path_lengths


def cal_audb(G):
    largest_cc = max(nx.strongly_connected_components(G.copy()), key=len)
    largest_cc_subgraph = G.subgraph(largest_cc)
    G = largest_cc_subgraph.to_undirected()

    betweenness = nx.current_flow_betweenness_centrality(G, weight='weight_reversed', normalized=False)

    # Calculate the degree for each node
    degree = G.degree(weight='weight_reversed')

    # Calculate the unit degree betweenness for each node
    unit_degree_betweenness = {node: betweenness[node] / degree[node] for node in G.nodes}

    # Calculate AUDB (Average Unit Degree Betweenness)
    audb = sum(unit_degree_betweenness.values()) / len(G.nodes)
    return audb


def remove_nodes_and_compute_audb(G, order_by, cal='AUDB'):
    # Percentage of nodes to be removed
    # percentages = [0.1 for i in range(1, 11)]
    percentage = 0.05
    steps = int(0.85 / percentage)
    # [percentage * (i + 1) for i in range(steps)]
    num_nodes = len(G.nodes)
    num_nodes_to_remove = int(num_nodes * percentage)

    audbs = []

    # Create a sorted list of nodes based on the ordering attribute
    nodes_sorted_by_attr = sorted(list(G.nodes()), key=lambda node: order_by[node], reverse=True)

    for _ in range(steps):

        # Calculate AUDB (Average Unit Degree Betweenness)
        if cal == 'AUDB':
            audb = cal_audb(G.copy())
        elif cal == 'Connected Component':
            audb = cal_cc(G.copy(), num_nodes)
        elif cal == 'Topological Efficiency':
            audb = cal_topo(G.copy(), num_nodes)
        else:
            largest_cc = max(nx.strongly_connected_components(G.copy()), key=len)
            largest_cc_subgraph = G.subgraph(largest_cc)
            audb = nx.diameter(largest_cc_subgraph, weight='weight_reversed')

        audbs.append(audb)

        nodes_to_remove = nodes_sorted_by_attr[:num_nodes_to_remove]

        # Remove nodes
        G.remove_nodes_from(nodes_to_remove)

        # Update the sorted list of nodes
        nodes_sorted_by_attr = [node for node in nodes_sorted_by_attr if node not in nodes_to_remove]

    return [percentage * i for i in range(steps)], audbs


def cal_topo(G, num_nodes):
    """
    Calculate the topological efficiency of a graph G

    Parameters:
    G (networkx.Graph): a networkx graph

    Returns:
    float: topological efficiency
    """
    N = len(G.nodes())
    # N = num_nodes
    d = nx.all_pairs_shortest_path_length(G)
    sum_of_reciprocal_d = sum(1 / v for _, paths in d for v in paths.values() if v > 0)
    E = 2 * sum_of_reciprocal_d / (N * (N - 1))
    return E


def cal_cc(G, total_graph_size):
    connected_components = sorted(nx.strongly_connected_components(G), key=len, reverse=True)

    # Size of the largest connected component
    largest_cc_size = len(connected_components[0])

    # Size of the entire graph
    total_graph_size = G.number_of_nodes()

    # Calculate the ratio
    ratio = largest_cc_size / total_graph_size
    return ratio


def plot_audb(G, df_intra_group, cal='AUDB'):
    # Create an order_by dictionary for each case
    # with open('data.pkl', 'rb') as f:
    #     audbs_ze, audbs_degree, G = pickle.load(f)
    order_by_random = {node: random.random() for node in G.nodes}
    order_by_ze = df_intra_group.set_index('node')['z_e'].to_dict()
    order_by_degree = dict(G.degree(weight='weight'))

    # Remove nodes and compute AUDbs for each case

    percentages, audbs_random = remove_nodes_and_compute_audb(G.copy(), order_by_random, cal)
    _, audbs_ze = remove_nodes_and_compute_audb(G.copy(), order_by_ze, cal)
    _, audbs_degree = remove_nodes_and_compute_audb(G.copy(), order_by_degree, cal)

    # Plot
    audbs_random_ = (audbs_random - np.min([audbs_random, audbs_ze, audbs_degree])) / (
            np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
    audbs_ze_ = (audbs_ze - np.min([audbs_random, audbs_ze, audbs_degree])) / (
            np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
    audbs_degree_ = (audbs_degree - np.min([audbs_random, audbs_ze, audbs_degree])) / (
            np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
    with open(f'plot_data_{cal}_zp.pkl', 'wb') as f:
        pickle.dump((audbs_ze_, audbs_degree_), f)
    plt.plot(percentages, audbs_random_, label='random', marker='o')
    plt.plot(percentages, audbs_ze_, label='z_e', marker='v')
    plt.plot(percentages, audbs_degree_, label='degree', marker='s')
    plt.xlabel('Percentage of Nodes Removed')
    plt.ylabel(cal)
    plt.legend()
    plt.savefig(f'Fig/Fig_resilience_{cal}_node.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def plot_group_audb():
    # get df and generate init group-wise graph
    audb_set = {}
    for i, order in enumerate(['random', 'z_e', 'degree', ]):
        G, graph_df = generate_graph()
        df_grouped = cal_intergroup_weighted(G.copy())
        # 提取其中社团内部部分
        df_intra_group = df_grouped.query('group_id == group_id_to')
        df_intra_group = calculate_z_e(G.copy(), df_intra_group)
        df_intra_group['participant_normalized'] = -(
                    df_intra_group['participant'] - df_intra_group['participant'].mean()) / \
                                                   df_intra_group['participant'].std()
        df_intra_group['z_e'] += df_intra_group['participant_normalized']
        df_intra_group = calculate_degree(G, df_intra_group)
        seed = np.random.randint(0, 100000)
        seed = 40882
        np.random.seed(seed)
        print(seed)
        df_intra_group['random'] = np.random.random(341, )
        # get new graph
        G = generate_graph_group_wise(graph_df.copy(deep=True))
        # cal audb for group-wise graph
        # audbs = cal_group_audb(G, cal='AUDB')
        # loop delete node in df
        percentage = 0.05
        steps = int(1 / percentage) - 2
        num_nodes = len(df_intra_group)
        num_nodes_to_remove = int(num_nodes * percentage)
        percentages = [percentage * (i + 1) for i in range(steps)]

        marker = ['s', 'o', 'v']
        audbs = [cal_group_audb(G, cal='AUDB')]
        for _ in range(steps - 1):
            # get new graph
            df_intra_group, graph_df = remove_nodes_in_df(df_intra_group, graph_df, order_by=order,
                                                          num=num_nodes_to_remove)
            G = generate_graph_group_wise(graph_df)
            audbs.append(cal_group_audb(G, cal='AUDB'))
        audb_set[order] = audbs

        # y = audbs
        # y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
        # plt.plot(percentages, y_normalized, label=order, marker=marker[i])
    audbs_random_ = (audb_set['random'] - np.min(list(audb_set.values()))) / (
            np.max(list(audb_set.values())) - np.min(list(audb_set.values())))
    audbs_ze_ = (audb_set['z_e'] - np.min(list(audb_set.values()))) / (
            np.max(list(audb_set.values())) - np.min(list(audb_set.values())))
    audbs_degree_ = (audb_set['degree'] - np.min(list(audb_set.values()))) / (
            np.max(list(audb_set.values())) - np.min(list(audb_set.values())))
    with open(f'plot_data_AUDBC_zp.pkl', 'wb') as f:
        pickle.dump((percentages, audbs_random_, audbs_ze_, audbs_degree_,), f)
    plt.plot(percentages, audbs_random_, label='random', marker='o')
    plt.plot(percentages, audbs_ze_, label='z_e', marker='v')
    plt.plot(percentages, audbs_degree_, label='degree', marker='s')
    plt.xlabel('Percentage of Nodes Removed')
    plt.ylabel('AUDBC')
    plt.legend()
    plt.savefig(f'Fig/Fig_resilience_AUDBC_Community_zp.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def generate_graph_group_wise(df):
    # turn df into group-wise df
    df = df.groupby(['group_id', 'group_id_to'])['huoliang'].sum().reset_index()
    G = nx.DiGraph()  # create a directed graph
    # Loop through each row in the dataframe and add edges
    for i, row in df.iterrows():
        # Add an edge from 'from_city' to 'to_city' with weight 'huoliang'
        G.add_edge(row['group_id'], row['group_id_to'], weight=row['huoliang'], weight_reversed=1 / row['huoliang'])
        # Add node attributes 'group_id' and 'group_id_to'
        # G.nodes[row['from_city']]['group_id'] = row['group_id']
        # G.nodes[row['to_city']]['group_id'] = row['group_id_to']
    return G


def calculate_degree(G, df):
    degree_df = pd.DataFrame(G.degree, columns=['node', 'degree'])
    return df.merge(degree_df, )


def gen_graph_from_df(df):
    G = nx.DiGraph()  # create a directed graph
    # Loop through each row in the dataframe and add edges
    for i, row in df.iterrows():
        # Add an edge from 'from_city' to 'to_city' with weight 'huoliang'
        G.add_edge(row['from_city'], row['to_city'], weight=row['huoliang'], weight_reversed=1 / row['huoliang'])

        # Add node attributes 'group_id' and 'group_id_to'
        G.nodes[row['from_city']]['group_id'] = row['group_id']
        G.nodes[row['to_city']]['group_id'] = row['group_id_to']
    return G


def remove_nodes_in_df(df, graph_df, order_by, num):
    # Create a sorted list of nodes based on the ordering attribute
    nodes_sorted_by_attr = df.sort_values([order_by], ascending=False)['node']
    nodes_to_remove = nodes_sorted_by_attr[:num]
    # Remove nodes
    graph_df = graph_df.loc[~graph_df['from_city'].isin(nodes_to_remove) & ~graph_df['to_city'].isin(nodes_to_remove),
               :]
    df = df.loc[~df['node'].isin(nodes_to_remove), :]
    return df, graph_df


def cal_group_audb(G, cal='AUDB'):
    # Calculate AUDB (Average Unit Degree Betweenness)
    if cal == 'AUDB':
        audb = cal_audb(G.copy())
    # elif cal == 'cc':
    #     audb = cal_cc(G.copy(), num_nodes)
    else:
        largest_cc = max(nx.strongly_connected_components(G.copy()), key=len)
        largest_cc_subgraph = G.subgraph(largest_cc)
        audb = nx.diameter(largest_cc_subgraph, weight='weight_reversed')
    return audb


def main(cal):
    G, _ = generate_graph()
    df_grouped = cal_intergroup_weighted(G.copy())
    # 提取其中社团内部部分
    df_intra_group = df_grouped.query('group_id == group_id_to')
    df_intra_group = calculate_z_e(G.copy(), df_intra_group)
    # df_intra_group['participant_normalized'] = (df_intra_group['participant'] - df_intra_group['participant'].mean()) / \
    #                                            df_intra_group['participant'].std()
    # df_intra_group['z_e'] += df_intra_group['participant_normalized']
    df_intra_group['z_e'] += df_intra_group['participant']

    # print(df_intra_group)
    df_intra_group.sort_values(['z_e'], ascending=False, inplace=True)
    plot_audb(G.copy(), df_intra_group, cal=cal)
    # nx.diameter(G, weight='weight_reversed')


def plot_22_audb():
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='normal',
                                       style='normal', size=16)
    # fig, axs = plt.subplots(2, 2, sharex='col', figsize=(10, 8))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    percentage = 0.05
    steps = int(0.85 / percentage)
    x = [percentage * i for i in range(steps)]
    fontsize = 16

    for cal in ['AUDB', 'Topological Efficiency', 'Connected Component']:
        # with open(f'plot_data_{cal}.pkl', 'rb') as f:
        #     # pickle.load(f)
        #     audbs_ze, audbs_degree, G = pickle.load(f)
        #
        # if cal == 'Topological Efficiency':
        #     seed = np.random.randint(0, 100000)
        #     print(cal, seed)
        #     random.seed(seed)
        #     order_by_random = {node: random.random() for node in G.nodes}
        #     percentages, audbs_random = remove_nodes_and_compute_audb(G.copy(), order_by_random, cal)
        #
        #     audbs_random_ = (audbs_random - np.min([audbs_random, audbs_ze, audbs_degree])) / (
        #             np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
        #     audbs_ze_ = (audbs_ze - np.min([audbs_random, audbs_ze, audbs_degree])) / (
        #             np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
        #     audbs_degree_ = (audbs_degree - np.min([audbs_random, audbs_ze, audbs_degree])) / (
        #             np.max([audbs_random, audbs_ze, audbs_degree]) - np.min([audbs_random, audbs_ze, audbs_degree]))
        #
        #     # Plotting the four subplots
        #     with open(f'plot_data_{cal}_final.pkl', 'wb') as f:
        #         pickle.dump((audbs_random_, audbs_ze_, audbs_degree_, G), f)
        with open(f'plot_data_{cal}_final_archive.pkl', 'rb') as f:
            audbs_random_, audbs_ze_, audbs_degree_, G = pickle.load(f)
        audbs_random_ = np.concatenate([np.array([1.0]), audbs_random_])
        audbs_ze_ = np.concatenate([np.array([1.0]), audbs_ze_])
        with open(f'plot_data_{cal}_zp_zmax_min.pkl', 'rb') as f:
            audbs_zp_, audbs_degree_ = pickle.load(f)
        if cal == "Connected Component":
            axs[0, 0].plot(x, audbs_random_, label='random', marker='o')
            # axs[0, 0].plot(x, audbs_ze_, label='z_e', marker='v')
            axs[0, 0].plot(x, audbs_degree_, label='degree', marker='s')
            axs[0, 0].plot(x, audbs_zp_, label='z_p', marker='^')
            axs[0, 0].set_xlabel('Percentage of removed nodes', fontproperties=font)
            axs[0, 0].set_ylabel('PNLCC', fontproperties=font)
            axs[0, 0].tick_params(axis='both', labelsize=fontsize - 4)

        elif cal == "Topological Efficiency":
            audbs_degree_ += 1-audbs_degree_[0]
            audbs_zp_ += 1-audbs_zp_[0]

            axs[0, 1].plot(x, audbs_random_, label='random', marker='o')
            # axs[0, 1].plot(x, audbs_ze_, label='z_e', marker='v')
            axs[0, 1].plot(x, audbs_degree_, label='degree', marker='s')
            axs[0, 1].plot(x, audbs_zp_, label='z_p', marker='^')
            axs[0, 1].set_xlabel('Percentage of removed nodes', fontproperties=font)
            axs[0, 1].set_ylabel('TE', fontproperties=font)
            axs[0, 1].tick_params(axis='both', labelsize=fontsize - 4)

        elif cal == "AUDB":
            axs[1, 0].plot(x, audbs_random_, label='random', marker='o')
            # axs[1, 0].plot(x, audbs_ze_, label='z_e', marker='v')
            axs[1, 0].plot(x, audbs_degree_, label='degree', marker='s')
            axs[1, 0].plot(x, audbs_zp_, label='z_p', marker='^')
            axs[1, 0].set_xlabel('Percentage of removed nodes', fontproperties=font)
            axs[1, 0].set_ylabel(cal, fontproperties=font)
            axs[1, 0].tick_params(axis='both', labelsize=fontsize - 4)

    # with open(f'plot_data_AUDBC.pkl', 'rb') as f:
    #     percentages, audbs_random_, audbs_ze_, audbs_degree_ = pickle.load(f)
    with open(f'plot_data_AUDBC_archive.pkl', 'rb') as f:
        percentages, audbs_random_, audbs_ze_, audbs_degree_ = pickle.load(f)
        # audbs_random_ = np.concatenate([np.array([1.0]), audbs_random_])
        # audbs_ze_ = np.concatenate([np.array([1.0]), audbs_ze_])
    with open(f'plot_data_AUDBC_zp_zmax_min.pkl', 'rb') as f:
        percentages, _, audbs_zp_, _ = pickle.load(f)
    axs[1, 1].plot(percentages, audbs_random_, label='random', marker='o')
    # axs[1, 1].plot(percentages, audbs_ze_, label='$z^e$', marker='v')
    axs[1, 1].plot(percentages, audbs_degree_, label='degree', marker='s')
    axs[1, 1].plot(percentages, audbs_zp_, label='$zp$', marker='^')

    axs[1, 1].set_xlabel('Percentage of removed nodes', fontproperties=font)
    axs[1, 1].set_ylabel('AUDBC', fontproperties=font)
    axs[1, 1].tick_params(axis='both', labelsize=fontsize - 4)
    axs[1, 1].legend()

    # Adding annotations
    axs[0, 0].annotate('(a)', xy=(0.05, 0.15), xycoords='axes fraction', fontproperties=font)
    axs[0, 1].annotate('(b)', xy=(0.05, 0.15), xycoords='axes fraction', fontproperties=font)
    axs[1, 0].annotate('(c)', xy=(0.05, 0.15), xycoords='axes fraction', fontproperties=font)
    axs[1, 1].annotate('(d)', xy=(0.05, 0.15), xycoords='axes fraction', fontproperties=font)

    # Adding a shared legend with larger font size
    # lines, labels = axs[0, 0].get_legend_handles_labels()
    # lines += axs[0, 1].get_legend_handles_labels()[0]
    # lines += axs[1, 0].get_legend_handles_labels()[0]
    # lines += axs[1, 1].get_legend_handles_labels()[0]
    #
    # fig.legend(lines, labels, loc='upper right', fontsize=fontsize)

    # Increasing font size of labels
    # for ax in axs.flat:
    #     ax.set_xlabel(fontsize=12)
    #     ax.set_ylabel(fontsize=12)
    plt.savefig(f'Fig/Fig_resilience_22_zp.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def main_ez(cal):
    plot_audb(None, None, cal=cal)


if __name__ == '__main__':
    # for cal in ['AUDB', 'Topological Efficiency']:
    #     main_ez(cal)
    # for cal in ['AUDB', 'Topological Efficiency', 'Connected Component']:
    #     main(cal)
    # plot_group_audb()
    # pass
    plot_22_audb()

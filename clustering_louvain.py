import snap
import numpy as np
import networkx as nx
import community
import matplotlib.pyplot as plt
import community_layout
import collections
import csv

G_nx = nx.Graph()
node_list = []
investor_list = []
node_size_list = []
weights_dict = {}
transactions_startup2vc = {}

def rescale(investor_size_list, new_min = 30, new_max = 500):
    old_max = max(investor_size_list)
    old_min = min(investor_size_list)
    return [int(1.0*(new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min) for x in investor_size_list]

def construct_graph(network, is_weighted = False):
    if network == "investor":
        node_list_file = "data/investor_list.txt"
        edge_list_file = "data/investor_network_undirected_unweighted.txt"
        edge_weights_file = "data/investor_network_undirected_weights.txt"
    elif network == "startup":
        node_list_file = "data/startup_list.txt"
        edge_list_file = "data/startup_network_undirected_unweighted.txt"
        edge_weights_file = "data/startup_network_undirected_weights.txt"

    f = open(node_list_file, "r")
    for line in f:
        name, w = line.strip().rsplit(' ', 1)
        node_list.append(name)
        node_size_list.append(int(w))

    if network == "startup":
        f = open("data/investor_list.txt", "r")
        for line in f:
            name, w = line.strip().rsplit(' ', 1)
            investor_list.append(name)

        f = open("data/transactions.csv", "r")
        csv_reader = csv.reader(f, delimiter=",", quotechar='"')
        for vc, startup, round, time in csv_reader:
            transactions_startup2vc[startup] = vc

    if is_weighted and edge_weights_file is not None:
        f = open(edge_weights_file, "r")
        for line in f:
            s, d, w = [int(i) for i in line.strip().split()]
            weights_dict[(s, d)] = float(w)

    f = open(edge_list_file, "r")
    f.readline()
    f.readline()
    f.readline()
    global G_nx
    for line in f:
        s, d = [int(i) for i in line.strip().split()]
        G_nx.add_node(s, name=node_list[s])
        G_nx.add_node(d, name=node_list[d])
        if is_weighted:
            G_nx.add_edge(s, d, weight=weights_dict[(s, d)])
        else:
            G_nx.add_edge(s, d)

    print("nodes", G_nx.number_of_nodes(), "edges", G_nx.number_of_edges())

    # configuration graph
    deg_seq = [d for n,d in G_nx.degree()]
    G_nx_conf = nx.configuration_model(deg_seq)
    G_nx = G_nx_conf
    print("config model nodes", G_nx.number_of_nodes(), "edges", G_nx.number_of_edges())

# def get_vc_for_startup_community(startup_communities):
#     f = open("data/transactions.csv", "r")
#     csv_reader = csv.reader(f, delimiter=",", quotechar='"')
#     transactions = {}
#     for vc, startup, round, time in csv_reader:
#         transactions[startup] = vc
#
#     vc_comm = {}
#     for com, startups in startup_communities.items():
#         vc_comm[com] =
#     return vc_comm

def louvain_partition_plot(network, is_weighted = False):
    construct_graph(network, is_weighted)

    ## degree distribution

    # degree_sequence = sorted([d for n, d in G_nx.degree()], reverse=True)
    # degreeCount = collections.Counter(degree_sequence)
    # print("max degree", max(degreeCount))
    # deg, cnt = zip(*degreeCount.items())
    # plt.plot(deg, cnt)
    # plt.xlabel("Node degree")
    # plt.ylabel("Count")
    # plt.title("Degree distribution")
    # plt.show()

    ## Louvain partitioning

    partition = community.best_partition(G_nx)

    size = float(len(set(partition.values())))
    print("# community", size)
    print("modularity", community.modularity(partition, G_nx))

    ## print communities

    # count = 0.
    communities = {}
    comm_size_list = []

    for com in set(partition.values()):
        # count += 1.
        list_nodes = [n for n in partition.keys() if partition[n] == com]
        communities[com] = list_nodes
        comm_size_list.append(len(list_nodes))
        print("#######community %i size %i" % (com, len(list_nodes)))

        if network == "startup":
            print("VCs who invest in this community:")
            for n in set([transactions_startup2vc[node_list[startup]] for startup in list_nodes]):
                print(n)
            print("\nStartups:")
        for n in list_nodes:
            print("%i %s" % (n, node_list[n]))
    # print("Communities", communities)

    # degree
    # for com in set(partition.values()):
    #     G_sub = G_nx.subgraph(communities[com])
    #     deg_in = 2*G_sub.number_of_edges()
    #     deg_total = 0
    #     for _,deg in G_nx.degree(communities[com]):
    #         deg_total += deg
    #     deg_out = deg_total - deg_in
    #     print
    #     # print(G_nx.degree(communities[com]))
    #     print("G_sub.number_of_edges", G_sub.number_of_edges())
    #     print("G_sub.degree in", deg_in)
    #     print("G_sub.degree total", deg_total)
    #     print("G_nx.degree out", deg_out)


    print("Community size", sorted(comm_size_list, reverse=True))
    # plt.plot(np.arange(size), sorted(comm_size_list, reverse=True))
    # plt.xlabel("Community")
    # plt.ylabel("Number of nodes in community")
    # plt.title("Community size distribution")
    # plt.show()

    ## plot

    # plot degree of nodes in communities
    list_nodes = []
    for com in set(partition.values()):
        # count += 1.
        list_nodes.extend([n for n in partition.keys() if partition[n] == com])
    deg_list = G_nx.degree(list_nodes)
    # print(deg_list)
    plt.plot(map(lambda x: x[1], deg_list))
    plt.xlabel("Nodes sorted by communities")
    plt.ylabel("Node degree")
    plt.title("Node degree by communities")
    plt.show()

    # draw graph
    # pos = nx.spring_layout(G_nx)
    pos = community_layout.community_layout(G_nx, partition)
    # pos = community_layout._position_communities(G_nx, partition)

    # nx.draw_networkx_nodes(G_nx, pos, list_nodes, node_size = 10, cmap=plt.cm.RdYlBu, node_color = np.array(partition.values())) #str(count / size)
    nx.draw_networkx_nodes(G_nx, pos, node_size=rescale(node_size_list), node_color=list(partition.values()))
    nx.draw_networkx_edges(G_nx, pos, alpha=0.1)
    # nx.draw_spring(G_nx, cmap = plt.get_cmap('jet'), node_color = partition.values(), node_size=50, with_labels=False)
    # nx.draw(G_nx, pos, node_size=100, node_color=partition.values())
    plt.show()

# louvain_partition_plot("investor", is_weighted=False)
# louvain_partition_plot("startup", is_weighted=False)




# unweighted
# (512, 3232)
# ('# community', 82.0)
# ('modularity', 0.23759403757903635)
# (439, 3232)
# ('# community', 9.0)
# ('modularity', 0.24210406953056077)

# weighted
# (439, 3232)
# ('# community', 9.0)
# ('modularity', 0.20910374606407073)


# G_snap = snap.LoadEdgeList(snap.PUNGraph, "data/investor_network_undirected_unweighted.txt", 0, 1)
# for n in G_snap.Nodes():
#     n.g


# G = nx.Graph()
# f = open("data/investor_list.txt", "r")
# i = 0
# for line in f:
#     G_nx.add_node(i, name=line)
#     i += 1
#
# print("# nodes", G.GetNodes(), "# edges", G.GetEdges())
#
# CmtyV = snap.TCnComV()
# modularity = snap.CommunityCNM(G, CmtyV)
# print("CommunityCNM modularity", modularity, "# community", CmtyV.Len())
#
# CmtyV = snap.TCnComV()
# modularity = snap.CommunityGirvanNewman(G, CmtyV)
# print("CommunityGirvanNewman modularity", modularity, "# community", CmtyV.Len())

# ('# nodes', 439, '# edges', 3232)
# ('CommunityCNM modularity', 0.22546502793843695, '# community', 9)
# ('CommunityGirvanNewman modularity', 0.02526693920939125, '# community', 302)

sizes = [434, 395, 344, 340, 323, 319, 306, 301, 300, 270, 251, 236, 173, 162, 158, 119, 118, 87, 17, 16, 15, 15, 12, 11, 7, 7, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
plt.plot(sizes)
plt.xlabel("Community")
plt.ylabel("Number of nodes in community")
plt.title("Community size distribution")
plt.show()
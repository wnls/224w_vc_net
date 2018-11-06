import snap
import numpy as np
import networkx as nx
import community
import matplotlib.pyplot as plt
import community_layout


def rescale(investor_size_list, new_min = 30, new_max = 500):
    old_max = max(investor_size_list)
    old_min = min(investor_size_list)
    return [int(1.0*(new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min) for x in investor_size_list]

def louvain_partition_plot(node_list_file, edge_list_file, edge_weights_file, is_weighted = False):
    G_nx = nx.Graph()
    f = open(node_list_file, "r")
    investor_list = []
    investor_size_list = []
    for line in f:
        name, w = line.strip().rsplit(' ', 1)
        investor_list.append(name)
        investor_size_list.append(int(w))

    if is_weighted and edge_weights_file is not None:
        weights_dict = {}
        f = open(edge_weights_file, "r")
        for line in f:
            s, d, w = [int(i) for i in line.strip().split()]
            weights_dict[(s, d)] = float(w)

    f = open(edge_list_file, "r")
    f.readline()
    f.readline()
    f.readline()
    for line in f:
        s, d = [int(i) for i in line.strip().split()]
        G_nx.add_node(s, name=investor_list[s])
        G_nx.add_node(d, name=investor_list[d])
        if is_weighted:
            G_nx.add_edge(s, d, weight=weights_dict[(s, d)])
        else:
            G_nx.add_edge(s, d)

    print(G_nx.number_of_nodes(), G_nx.number_of_edges())

    ## Louvain partitioning

    partition = community.best_partition(G_nx)

    size = float(len(set(partition.values())))
    print("# community", size)
    print("modularity", community.modularity(partition, G_nx))


    ## print communities

    # count = 0.
    communities = {}
    for com in set(partition.values()):
        # count += 1.
        list_nodes = [n for n in partition.keys() if partition[n] == com]
        communities[com] = list_nodes
        # print("#######community %i" % com)
        # for n in list_nodes:
        #     print("%i %s" % (n, investor_list[n]))
    print(communities)

    ## plot

    # plot degree of nodes in communities
    list_nodes = []
    for com in set(partition.values()):
        # count += 1.
        list_nodes.extend([n for n in partition.keys() if partition[n] == com])
    deg_list = G_nx.degree(list_nodes)
    print(deg_list)
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
    nx.draw_networkx_nodes(G_nx, pos, node_size=rescale(investor_size_list), node_color=list(partition.values()))
    nx.draw_networkx_edges(G_nx, pos, alpha=0.1)
    # nx.draw_spring(G_nx, cmap = plt.get_cmap('jet'), node_color = partition.values(), node_size=50, with_labels=False)
    # nx.draw(G_nx, pos, node_size=100, node_color=partition.values())
    plt.show()

louvain_partition_plot("data/investor_list.txt", "data/investor_network_undirected_unweighted.txt", "investor_network_undirected_weights.txt", is_weighted=False)
# louvain_partition_plot("data/startup_list.txt", "data/startup_network_undirected_unweighted.txt", "startup_network_undirected_weights.txt", is_weighted=False)




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
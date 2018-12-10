import csv
import snap
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from networkx.algorithms import approximation
from collections import defaultdict

def load_3_subgraphs():
    return [snap.LoadEdgeList(snap.PNGraph, "./subgraphs/{}.txt".format(i), 0, 1) for i in range(13)]

def match(G1, G2):
    if G1.GetEdges() > G2.GetEdges():
        G = G1
        H = G2
    else:
        G = G2
        H = G1
    # Only checks 6 permutations, since k = 3
    for p in permutations(range(3)):
        edge = G.BegEI()
        matches = True
        while edge < G.EndEI():
            if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
                matches = False
                break
            edge.Next()
        if matches:
            break
    return matches

def count_iso(G, sg, verbose=False):
    # if verbose:
    #     print(sg)
    nodes = snap.TIntV()
    for NId in sg:
        nodes.Add(NId)
    # This call requires latest version of snap (4.1.0)
    SG = snap.GetSubGraphRenumber(G, nodes)
    for i in range(len(directed_3)):
        if match(directed_3[i], SG):
            motif_counts[i] += 1
            # Count motif weight for motif clustering
            # if verbose and i == 12:
	           #  for u in sg:
	           #  	for v in sg:
	           #  		if G.IsEdge(u, v):
	           #  			if (u, v) not in motif_weights:
	           #  				motif_weights[(u, v)] = 0
	           #  			motif_weights[(u, v)] += 1

def enumerate_subgraph(G, k=3, verbose=False):
    global motif_counts
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)

    global motif_weights
    motif_weights = dict()

    for node in G.Nodes():
        node_id = node.GetId()
        sg = [node_id]
        v_ext = set()
        deg = node.GetDeg()
        for i in range(deg):
            nei = node.GetNbrNId(i)
            if nei > node_id: v_ext.add(nei)
        extend_subgraph(G, k, sg, list(v_ext), node_id, verbose)

def extend_subgraph(G, k, sg, v_ext, node_id, verbose=False):
    if len(sg) is k:
        count_iso(G, sg, verbose)
        return

    v_ext_set = set()
    for x in v_ext:
        v_ext_set.add(x)
    for x in sg:
        v_ext_set.add(x)

    while len(v_ext) > 0:
        w = v_ext.pop()
        w_node = G.GetNI(w)
        sg_copy = [x for x in sg]
        sg_copy.append(w)
        v_ext_copy = set([x for x in v_ext])
        deg = w_node.GetDeg()
        for i in range(deg):
            w_nei = w_node.GetNbrNId(i)
            if w_nei not in v_ext_set and w_nei > node_id:
                v_ext_copy.add(w_nei)
        extend_subgraph(G, k, sg_copy, list(v_ext_copy), node_id, verbose)

def gen_config_model_rewire(graph, iterations = 100):
    config_graph = snap.TNGraph.New()
    for node in graph.Nodes():
    	config_graph.AddNode(node.GetId())
    for edge in graph.Edges():
    	config_graph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
    
    edge_set = [(edge.GetSrcNId(), edge.GetDstNId()) for edge in config_graph.Edges()]
    m = len(edge_set)
    
    for ite in range(iterations):
        e1, e2 = np.random.randint(m), np.random.randint(m)
        if e1 == e2: 
        	continue
        
        # r1 = np.random.randint(2)
        # r2 = np.random.randint(2)
        # u, v = (edge_set[e1][0], edge_set[e1][1]) if r1 == 0 else (edge_set[e1][1], edge_set[e1][0])
        # w, x = (edge_set[e2][0], edge_set[e2][1]) if r2 == 0 else (edge_set[e2][1], edge_set[e2][0])
        
        # if u == w or v == x:
        # 	continue
        # if config_graph.IsEdge(u, w) or config_graph.IsEdge(v, x): 
        # 	continue
        
        # config_graph.DelEdge(u, v)
        # config_graph.DelEdge(w, x)
        # config_graph.AddEdge(u, w)
        # config_graph.AddEdge(v, x)

        # edge_set[e1] = (u, w)
        # edge_set[e2] = (v, x)

        u, v = (edge_set[e1][0], edge_set[e1][1])
        w, x = (edge_set[e2][0], edge_set[e2][1])

        if u == x or w == v:
        	continue
        if config_graph.IsEdge(u, x) or config_graph.IsEdge(w, v): 
        	continue

       	config_graph.DelEdge(u, v)
        config_graph.DelEdge(w, x)
        config_graph.AddEdge(u, x)
        config_graph.AddEdge(w, v)

        edge_set[e1] = (u, x)
        edge_set[e2] = (w, v)

    return config_graph




def read_vc_list():
	with open('vclist.csv') as file:
		reader = csv.reader(file, delimiter = ',')
		vc_list = []
		for row in reader:
			vc_list.append(row[0])
		vc_list = vc_list[1:]
	return vc_list


def read_transaction_data(vc_list):
	transactions = []
	for vc in vc_list:
		try:
			with open('VentureCapitals/' + vc + '.csv') as file:
				reader = csv.reader(file, delimiter = ',')
				line_count = 0
				for line in reader:
					line_count += 1
					if line_count == 1: continue
					record = [vc, line[2], line[4], line[8]]
					transactions.append(record)
				# print 'Collected records of vc firm: ' + vc
		except IOError as e:
			print 'No file for vc firm: ' + vc

	with open('transactions.csv', 'w') as file:
		writer = csv.writer(file)
		for item in transactions:
			writer.writerow(item)

	return transactions


def get_data_stats(transactions):
	'''
	Get information about the number of investments VCs have made and startups have received
	'''
	investor_investments, startup_investments = dict(), dict()
	investor_list, startup_list = [], []
	max_investor_investments, max_startup_investments = 0, 0

	startup_id, investor_id = dict(), dict()
	startup_num, investor_num = 0, 0

	for item in transactions:
		if item[0] not in investor_investments:
			investor_id[item[0]] = investor_num
			investor_num += 1
			investor_list.append(item[0])
			investor_investments[item[0]] = 0
		if item[1] not in startup_investments:
			startup_id[item[1]] = startup_num
			startup_num += 1
			startup_list.append(item[1])
			startup_investments[item[1]] = 0
		
		investor_investments[item[0]] += 1
		startup_investments[item[1]] += 1
		
		if investor_investments[item[0]] > max_investor_investments:
			max_investor_investments = investor_investments[item[0]]
		if startup_investments[item[1]] > max_startup_investments:
			max_startup_investments = startup_investments[item[1]]

	cnt_investor = np.zeros(max_investor_investments + 1)
	cnt_startup = np.zeros(max_startup_investments + 1)

	# print max_investor_investments, max_startup_investments

	for key in investor_investments:
		cnt_investor[investor_investments[key]] += 1
	for key in startup_investments:
		cnt_startup[startup_investments[key]] += 1

	# plt.loglog(np.array(range(max_investor_investments + 1)), cnt_investor)
	# plt.xlabel('Number of investments a vc has made')
	# plt.ylabel('Number of vc firms')
	# plt.savefig('vc_stats.png')

	# plt.clf()
	# plt.loglog(np.array(range(max_startup_investments + 1)), cnt_startup)
	# plt.xlabel('Number of investments a startup has received')
	# plt.ylabel('Number of startups')
	# plt.savefig('startup_stats.png')

	
	### get information about financing rounds
	round_names = set()
	round_counts = dict()

	for item in transactions:
		if item[2] not in round_names:
			round_names.add(item[2])
			round_counts[item[2]] = 0
		round_counts[item[2]] += 1

	name_list = list(round_names)
	count_list = np.array([round_counts[name] for name in name_list])
	indexes = np.argsort(-count_list)
	for idx in indexes:
		print name_list[idx], count_list[idx], np.around(float(count_list[idx]) / len(transactions) * 100.0, decimals = 2)


	### save investor and startup list
	# with open('investor_list.txt', 'w') as f:
	# 	for investor in investor_list:
	# 		f.write(investor + ' ' + str(investor_investments[investor]) + '\n')

	# with open('startup_list.txt', 'w') as f:
	# 	for startup in startup_list:
	# 		f.write(startup + ' ' + str(startup_investments[startup]) + '\n')

	return investor_id, startup_id, investor_list, startup_list


def build_graph(transactions, investor_id, startup_id):
	considered_rounds = ['Seed', 'Angel', 'Series A', 'Series B', 'Series C', 'Series D', 'Series E', 'Series F', 'Series G']

	startup_investors = dict()
	startup_round_investors = dict()
	investor_startups = dict()

	# startup_id, investor_id = dict(), dict()
	# startup_num, investor_num = 0, 0

	G1_weights, G1d_weights, G2_weights = dict(), dict(), dict()

	for item in transactions:
		investor = item[0]
		startup = item[1]
		invest_round = item[2]
		
		if startup not in startup_investors:
			# startup_id[startup] = startup_num
			# startup_num += 1
			startup_investors[startup] = set()
			startup_round_investors[startup] = dict()
		if invest_round not in startup_round_investors[startup]:
			startup_round_investors[startup][invest_round] = set()		
		if investor not in investor_startups:
			# investor_id[investor] = investor_num
			# investor_num += 1
			investor_startups[investor] = set()
		
		startup_investors[startup].add(investor)
		startup_round_investors[startup][invest_round].add(investor)
		investor_startups[investor].add(startup)

	

	### build investor graph
	G1 = snap.TUNGraph.New()
	for investor in investor_startups:
		G1.AddNode(investor_id[investor])
	for startup in startup_investors:
		# if len(startup_investors[startup]) > 1:
		# 	print startup, len(startup_investors[startup])
		for investor_a in startup_investors[startup]:
			for investor_b in startup_investors[startup]:
				if investor_a != investor_b:
					u = investor_id[investor_a]
					v = investor_id[investor_b]
					G1.AddEdge(u, v)
					if (u, v) not in G1_weights:
						G1_weights[(u, v)] = 0
					G1_weights[(u, v)] += 1
	print G1.GetNodes(), G1.GetEdges()

	### save investor graph
	# snap.SaveEdgeList(G1, 'investor_network_undirected_unweighted.txt')
	
	# FOut = snap.TFOut('investor_network_undirected_unweighted.graph')
	# G1.Save(FOut)
	# FOut.Flush()
	
	# with open('investor_network_undirected_weights.txt', 'w') as f:
	# 	for key in G1_weights:
	# 		f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G1_weights[key]) + '\n')
		
	


	### build directed investor graph
	G1d = snap.TNGraph.New()
	for investor in investor_startups:
		G1d.AddNode(investor_id[investor])
	for startup in startup_round_investors:
		rounds = []
		for item in considered_rounds:
			if item in startup_round_investors[startup]:
				rounds.append(item)
		for i in range(len(rounds) - 1):
			for investor_a in startup_round_investors[startup][rounds[i]]:
				for investor_b in startup_round_investors[startup][rounds[i + 1]]:
					if investor_a != investor_b:
						u = investor_id[investor_b]
						v = investor_id[investor_a]
						G1d.AddEdge(u, v)
						if (u, v) not in G1d_weights:
							G1d_weights[(u, v)] = 0
						G1d_weights[(u, v)] += 1
	print G1d.GetNodes(), G1d.GetEdges()

	### save directed investor graph
	# snap.SaveEdgeList(G1d, 'investor_network_directed_unweighted.txt')
	
	# FOut = snap.TFOut('investor_network_directed_unweighted.graph')
	# G1d.Save(FOut)
	# FOut.Flush()
	
	# with open('investor_network_directed_weights.txt', 'w') as f:
	# 	for key in G1d_weights:
	# 		f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G1d_weights[key]) + '\n')
	


	### build startup graph
	G2 = snap.TUNGraph.New()
	for startup in startup_investors:
		G2.AddNode(startup_id[startup])
	for investor in investor_startups:
		for startup_a in investor_startups[investor]:
			for startup_b in investor_startups[investor]:
				if startup_a != startup_b:
					u = startup_id[startup_a]
					v = startup_id[startup_b]
					G2.AddEdge(u, v)
					if (u, v) not in G2_weights:
						G2_weights[(u, v)] = 0
					G2_weights[(u, v)] += 1
	print G2.GetNodes(), G2.GetEdges()

	### save startup graph
	# snap.SaveEdgeList(G2, 'startup_network_undirected_unweighted.txt')
	
	# FOut = snap.TFOut('startup_network_undirected_unweighted.graph')
	# G2.Save(FOut)
	# FOut.Flush()
	
	# with open('startup_network_undirected_weights.txt', 'w') as f:
	# 	for key in G2_weights:
	# 		f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G2_weights[key]) + '\n')
	

	
	return G1, G1d, G2, G1_weights, G1d_weights, G2_weights


def rewire_undirected_graph(G):
	G_config = snap.TUNGraph.New()
	for node in G.Nodes():
		G_config.AddNode(node.GetId())
	for edge in G.Edges():
		G_config.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

	edge_set = [(edge.GetSrcNId(), edge.GetDstNId()) for edge in G_config.Edges()]
	m = len(edge_set)

	for ite in range(1000):
	    e1, e2 = np.random.randint(m), np.random.randint(m)
	    if e1 == e2: 
	    	continue

	    u, v = (edge_set[e1][0], edge_set[e1][1])
	    w, x = (edge_set[e2][0], edge_set[e2][1])

	    r = np.random.randint(2)
	    if r == 1:
	    	w, x = x, w

	    if u == x or w == v:
	    	continue
	    if G_config.IsEdge(u, x) or G_config.IsEdge(w, v): 
	    	continue

	   	G_config.DelEdge(u, v)
	    G_config.DelEdge(w, x)
	    G_config.AddEdge(u, x)
	    G_config.AddEdge(w, v)

	    edge_set[e1] = (u, x)
	    edge_set[e2] = (w, v)
		
	return G_config


def get_graph_overview(G, Gd = None):
	'''
	G here is an undirected graph
	'''

	# degree distribution
	CntV = snap.TIntPrV()
	snap.GetOutDegCnt(G, CntV)
	deg_x, deg_y = [], []
	max_deg = 0
	for item in CntV:
		max_deg = max(max_deg, item.GetVal1())
		deg_x.append(item.GetVal1())
		deg_y.append(item.GetVal2())
		# print item.GetVal1(), item.GetVal2()
	print 'max_deg = ', max_deg
	deg_cnt = np.zeros(max_deg + 1)
	for item in CntV:
		deg_cnt[item.GetVal1()] = item.GetVal2()
	print deg_cnt
	# plt.loglog(deg_x, deg_y)
	# plt.xlabel('Degree of nodes')
	# plt.ylabel('Number of nodes')
	# plt.savefig('Giu_deg_dist.png')
	# plt.clf()

	
	# clustering coefficient distribution
	cf = snap.GetClustCf(G)
	print 'average cf =', cf
	NIdCCfH = snap.TIntFltH()
	snap.GetNodeClustCf(G, NIdCCfH)
	ccf_sum = np.zeros(max_deg + 1)
	for item in NIdCCfH:
		ccf_sum[G.GetNI(item).GetDeg()] += NIdCCfH[item]
		# print item, NIdCCfH[item]
	ccf_x, ccf_y = [], []
	for i in range(max_deg + 1):
		if deg_cnt[i] != 0:
			ccf_sum[i] /= deg_cnt[i]
			ccf_x.append(i)
			ccf_y.append(ccf_sum[i])
	print ccf_y
	# plt.loglog(ccf_x, ccf_y)
	# plt.xlabel('Degree of nodes')
	# plt.ylabel('Average clustering coefficient of nodes with the degree')
	# plt.savefig('Giu_ccf_dist.png')
	# plt.clf()
	# snap.PlotClustCf(G, 'investor_network', 'Distribution of clustering coefficients')

	
	# diameter and shortest path distribution
	diam = snap.GetBfsFullDiam(G, 100)
	print diam
	# snap.PlotShortPathDistr(G, 'investor_network', 'Distribution of shortest path length')
	# rewired_diams = []
	# for i in range(100):
	# 	print 'rewire: ', i
	# 	G_config = rewire_undirected_graph(G)
	# 	rewired_diams.append(snap.GetBfsFullDiam(G_config, 400))
	# print rewired_diams
	# print 'null model diam mean: ', np.mean(rewired_diams)
	# print 'null model diam std: ', np.std(rewired_diams)

	
	# wcc and scc size distribution
	WccSzCnt = snap.TIntPrV()
	snap.GetWccSzCnt(G, WccSzCnt)
	print 'Distribution of wcc:'
	for item in WccSzCnt:
		print item.GetVal1(), item.GetVal2()

	if Gd != None:
		print 'Distribution of scc:'
		ComponentDist = snap.TIntPrV()
		snap.GetSccSzCnt(Gd, ComponentDist)
		for item in ComponentDist:
			print item.GetVal1(), item.GetVal2()


def do_motif_analysis(G):
	enumerate_subgraph(G, 3, True)
	motif_origin = np.array(motif_counts)
	print motif_counts

	### output edge weights for motif clustering
	# with open('Gid_motif_weitghts_13.txt', 'w') as f:
	# 	for edge in G.Edges():
	# 		u, v = edge.GetSrcNId(), edge.GetDstNId()
	# 		if (u, v) not in motif_weights:
	# 			motif_weights[(u, v)] = 0
	# 		f.write('{0} {1} {2}\n'.format(u, v, motif_weights[(u, v)]))

	motifs = np.zeros((10, 13))

	for i in range(10):
		print 'iteration ' + str(i)
		config_graph = gen_config_model_rewire(G, 500)
		enumerate_subgraph(config_graph, 3, False)
		motifs[i, :] = motif_counts
	print motifs

	motifs_mean = np.mean(motifs, axis = 0)
	motifs_std = np.std(motifs, axis = 0)
	z_scores = ((motif_origin - motifs_mean) / motifs_std)
	print z_scores


def build_networkx_graph(G, Gd = None, G_weights = None, Gd_weights = None):
	'''
	Building a graph of networkx using a graph of snap.py. If a directed graph is specified, also build a corresponding directed one.
	'''
	Gx = nx.Graph()
	for node in G.Nodes():
		Gx.add_node(node.GetId())
	for edge in G.Edges():
		Gx.add_edge(edge.GetSrcNId(), edge.GetDstNId())


	Gdx = None
	if Gd != None:
		Gdx = nx.DiGraph()
		for node in Gd.Nodes():
			Gdx.add_node(node.GetId())
		for edge in Gd.Edges():
			u, v = (edge.GetSrcNId(), edge.GetDstNId())
			if Gd_weights != None:
				Gdx.add_edge(u, v, weight = Gd_weights[(u, v)])
			else:
				Gdx.add_edge(u, v)

	return Gx, Gdx


def analyze_cliques(G, investor_list):
	### get maximal clique
	max_q = approximation.clique.max_clique(G)
	print 'clique size:', len(max_q)
	for node_id in max_q:
		print node_id, investor_list[node_id]


def analyze_node_centrality(Gd, investor_list):
	### calcualte pagerank
	pr = nx.pagerank(Gd, weight = 'weight')
	pr_scores = np.zeros(len(investor_list))
	for key in pr:
		pr_scores[key] = pr[key]
	idxs = np.argsort(-pr_scores)
	print 'pagerank --------------------------'
	for i in range(50):
		print i + 1, investor_list[idxs[i]], pr_scores[idxs[i]]
	

	### calcualte hub and authority values
	h, a = nx.hits(Gd)
	h_scores = np.zeros(len(investor_list))
	a_scores = np.zeros(len(investor_list))
	for key in h:
		h_scores[key] = h[key]
	for key in a:
		a_scores[key] = a[key]
	
	# idxs = np.argsort(-h_scores)
	# print 'hub score -------------------------'
	# for i in range(20):
	# 	print investor_list[idxs[i]], h_scores[idxs[i]]
	# idxs = np.argsort(-a_scores)
	# print 'authority score ------------------'
	# for i in range(20):
	# 	print investor_list[idxs[i]], a_scores[idxs[i]]



def divide_network_by_year(transactions):
	trans_by_years = dict()
	
	for item in transactions:
		year = int((item[3].split('-'))[0])
		if year < 2010:
			year = 2009
		if year not in trans_by_years:
			trans_by_years[year] = []
		trans_by_years[year].append(item)
	
	for year in trans_by_years:
		print year, len(trans_by_years[year])

	return trans_by_years



def analyze_graph_by_years(trans_by_years, investor_id, startup_id, investor_list, startup_list):
	for year in range(2009, 2019):
		print 'This is the graph of year ' + str(year) + '--------------------------------'
		
		G, _, _, _, _, _ = build_graph(trans_by_years[year], investor_id, startup_id)
		get_graph_overview(G)
		
		
		# FOut = snap.TFOut('Giu_{0}.graph'.format(year))
		# G.Save(FOut)
		# FOut.Flush()

		
		# with open('Giu_unweighted_by_year/Giu_{0}_louvain.txt'.format(year), 'r') as f:
		# 	lines = f.readlines()
			
		# 	communities = []
		# 	community = None

		# 	for line in lines:
		# 		if line[0] == '#':
		# 			if community != None:
		# 				communities.append(community)
		# 			community = []
		# 		else:
		# 			community.append(int(line.strip()))
		# 	communities.append(community)

		# 	with open('Giu_unweighted_by_year/Giu_{0}_common_startups.txt'.format(year), 'w') as fout:
		# 		for community in communities:
		# 			fout.write('###### Investors in this community:\n') 
		# 			for x in community:
		# 				fout.write('{0} {1}\n'.format(x, investor_list[x]))
					
		# 			commset = set(community)
					
		# 			startup_investors_list = [set([]) for _ in range(len(startup_list))]
		# 			for item in trans_by_years[year]:
		# 				if investor_id[item[0]] in commset:
		# 					startup_investors_list[startup_id[item[1]]].add(investor_id[item[0]])

		# 			startup_investors_count = np.array([len(startup_investors_list[i]) for i in range(len(startup_list))])
		# 			idx = np.argsort(-startup_investors_count)
					
		# 			fout.write('###### Representative startups in this community:\n')
		# 			for i in range(15):
		# 				if startup_investors_count[idx[i]] < 2:
		# 					break
		# 				fout.write('{0} {1} -- '.format(startup_list[idx[i]], startup_investors_count[idx[i]]))
		# 				fout.write(str(list(startup_investors_list[idx[i]])) + '\n')
					
		# 			fout.write('------------------------------------------\n\n')

def main():
	vc_list = read_vc_list()
	transactions = read_transaction_data(vc_list)

	investor_id, startup_id, investor_list, startup_list = get_data_stats(transactions)
	
	G1, G1d, G2, G1_weights, G1d_weights, G2_weights = build_graph(transactions, investor_id, startup_id)
	
	# get_graph_overview(G1, G1d)
	# do_motif_analysis(G1d)


	trans_by_years = divide_network_by_year(transactions)
	analyze_graph_by_years(trans_by_years, investor_id, startup_id, investor_list, startup_list)

	
	# G1x, G1dx = build_networkx_graph(G1, G1d, G1_weights, G1d_weights)
	# analyze_cliques(G1x, iv_list)
	# analyze_node_centrality(G1dx, iv_list)


if __name__ == '__main__':
	directed_3 = load_3_subgraphs()
	motif_counts = [0]*len(directed_3)

	main()

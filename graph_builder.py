import snap
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

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
    if verbose:
        print(sg)
    nodes = snap.TIntV()
    for NId in sg:
        nodes.Add(NId)
    # This call requires latest version of snap (4.1.0)
    SG = snap.GetSubGraphRenumber(G, nodes)
    for i in range(len(directed_3)):
        if match(directed_3[i], SG):
            motif_counts[i] += 1

def enumerate_subgraph(G, k=3, verbose=False):
    global motif_counts
    motif_counts = [0]*len(directed_3) # Reset the motif counts (Do not remove)

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
	### get information aboud the number of investments VCs have made and startups have received
	investor_investments, startup_investments = dict(), dict()
	max_investor_investments, max_startup_investments = 0, 0

	for item in transactions:
		if item[0] not in investor_investments:
			investor_investments[item[0]] = 0
		if item[1] not in startup_investments:
			startup_investments[item[1]] = 0
		investor_investments[item[0]] += 1
		startup_investments[item[1]] += 1
		if investor_investments[item[0]] > max_investor_investments:
			max_investor_investments = investor_investments[item[0]]
		if startup_investments[item[1]] > max_startup_investments:
			max_startup_investments = startup_investments[item[1]]

	cnt_investor = np.zeros(max_investor_investments + 1)
	cnt_startup = np.zeros(max_startup_investments + 1)

	print max_investor_investments, max_startup_investments

	for key in investor_investments:
		cnt_investor[investor_investments[key]] += 1
	for key in startup_investments:
		cnt_startup[startup_investments[key]] += 1

	plt.plot(np.array(range(max_investor_investments + 1)), cnt_investor)
	plt.xlabel('Number of investments a vc has made')
	plt.ylabel('Number of vc firms')
	plt.savefig('vc_stats.png')

	plt.clf()
	plt.plot(np.array(range(max_startup_investments + 1)), cnt_startup)
	plt.xlabel('Number of investments a startup has received')
	plt.ylabel('Number of startups')
	plt.savefig('startup_stats.png')

	
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

def build_graph(transactions):
	considered_rounds = ['Seed', 'Angel', 'Series A', 'Series B', 'Series C', 'Series D', 'Series E', 'Series F', 'Series G']

	startup_investors = dict()
	startup_round_investors = dict()
	investor_startups = dict()

	startup_id, investor_id = dict(), dict()
	startup_list, investor_list = [], []
	startup_num, investor_num = 0, 0

	G1_weights, G1d_weights, G2_weights = dict(), dict(), dict()

	for item in transactions:
		investor = item[0]
		startup = item[1]
		invest_round = item[2]
		
		if startup not in startup_id:
			startup_id[startup] = startup_num
			startup_list.append(startup)
			startup_num += 1
			startup_investors[startup] = set()
			startup_round_investors[startup] = dict()
		if invest_round not in startup_round_investors[startup]:
			startup_round_investors[startup][invest_round] = set()		
		if investor not in investor_id:
			investor_id[investor] = investor_num
			investor_list.append(investor)
			investor_num += 1
			investor_startups[investor] = set()
		
		startup_investors[startup].add(investor)
		startup_round_investors[startup][invest_round].add(investor)
		investor_startups[investor].add(startup)

	

	### build investor graph
	G1 = snap.TUNGraph.New()
	for investor in investor_id:
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
	FOut = snap.TFOut('investor_network_undirected_unweighted.graph')
	G1.Save(FOut)
	FOut.Flush()
	with open('investor_network_undirected_weights.txt', 'w') as f:
		for key in G1_weights:
			f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G1_weights[key]) + '\n')
		
	


	### build directed investor graph
	G1d = snap.TNGraph.New()
	for investor in investor_id:
		G1d.AddNode(investor_id[investor])
	for startup  in startup_round_investors:
		rounds = []
		for item in considered_rounds:
			if item in startup_round_investors[startup]:
				rounds.append(item)
		for i in range(len(rounds) - 1):
			for investor_a in startup_round_investors[startup][rounds[i]]:
				for investor_b in startup_round_investors[startup][rounds[i + 1]]:
					if investor_a != investor_b:
						u = investor_id[investor_a]
						v = investor_id[investor_b]
						G1d.AddEdge(u, v)
						if (u, v) not in G1d_weights:
							G1d_weights[(u, v)] = 0
						G1d_weights[(u, v)] += 1
	print G1d.GetNodes(), G1d.GetEdges()

	# ComponentDist = snap.TIntPrV()
	# snap.GetSccSzCnt(G1d, ComponentDist)
	# for item in ComponentDist:
	# 	print item.GetVal1(), item.GetVal2()

	### save directed investor graph
	# snap.SaveEdgeList(G1d, 'investor_network_directed_unweighted.txt')
	FOut = snap.TFOut('investor_network_directed_unweighted.graph')
	G1d.Save(FOut)
	FOut.Flush()
	with open('investor_network_directed_weights.txt', 'w') as f:
		for key in G1d_weights:
			f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G1d_weights[key]) + '\n')
	


	### build startup graph
	G2 = snap.TUNGraph.New()
	for startup in startup_id:
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
	FOut = snap.TFOut('startup_network_undirected_unweighted.graph')
	G2.Save(FOut)
	FOut.Flush()
	with open('startup_network_undirected_weights.txt', 'w') as f:
		for key in G2_weights:
			f.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(G2_weights[key]) + '\n')
	


	### save investor and startup list
	with open('investor_list.txt', 'w') as f:
		for investor in investor_list:
			f.write(investor + '\n')

	with open('startup_list.txt', 'w') as f:
		for startup in startup_list:
			f.write(startup + '\n')

	
	return G1, G1d, G2, investor_list, startup_list

def get_graph_overview(G):
	CntV = snap.TIntPrV()
	snap.GetOutDegCnt(G, CntV)
	# for item in CntV:
	# 	print item.GetVal1(), item.GetVal2()

	cf = snap.GetClustCf(G)
	# print cf
	# snap.PlotClustCf(G, "test1", "test2")

	diam = snap.GetBfsFullDiam(G, 100)
	# print diam

	WccSzCnt = snap.TIntPrV()
	snap.GetWccSzCnt(G, WccSzCnt)
	# for item in WccSzCnt:
	# 	print item.GetVal1(), item.GetVal2()

def do_motif_analysis(G):
	enumerate_subgraph(G, 3, False)
	motif_origin = np.array(motif_counts)
	print motif_counts

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
    
def main():
	vc_list = read_vc_list()
	transactions = read_transaction_data(vc_list)
	# get_data_stats(transactions)
	
	G1, G1d, G2, investor_list, startup_list = build_graph(transactions)
	# get_graph_overview(G1)
	# do_motif_analysis(G1d)

if __name__ == '__main__':
	directed_3 = load_3_subgraphs()
	motif_counts = [0]*len(directed_3)

	main()

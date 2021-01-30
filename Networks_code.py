# Learning Dynamics / Computational Game Theory Assignment 2
# 25 November 2020
# Horatiu Luci
# Year 1 - M-SECU-C @ ULB
#
# VUB Student ID: 0582214
# ULB Student ID: 000516512

import networkx as nx
import matplotlib.pyplot as plt
import random
import collections
from tqdm import tqdm # loading-bar
from scipy import stats
import numpy as np



def generate_erg(e, n):
    g = nx.Graph()
    for x in range (0, n):
        g.add_node(x)
    ne = 0
    while ne < e:
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        while y == x:
            y = random.randint(0, n - 1)
        if not(g.has_edge(x, y)):
            g.add_edge(x, y)
            ne += 1
        else: continue
    average_degree = 2*g.number_of_edges() / float(g.number_of_nodes())
    return g




def plot_deg_dist_erg(G):
    degree_sequence = [d for n, d in G.degree()]  # degree sequence
    mu, std = stats.norm.fit(degree_sequence)
    d = np.unique(degree_sequence, return_counts = True)[1]
    n, bins, patches = plt.hist(degree_sequence, bins=len(d), density = True, color = 'b', alpha = 1, histtype = 'barstacked', ec = 'white')
    xmin, xmax = plt.xlim()
    x = np.linspace(mu-6*std, mu+6*std, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=3)
    title = "Fit results: mean = {},  standard deviation = {}".format(mu, std)
    plt.title(title)
    plt.show()



def barabasi_add_nodes(g, n, m0):
    m = 4
    for i in tqdm(range(m0, n)):
        g.add_node(i)
        degrees = nx.degree(g)
        node_prob = {}
        s = 0
        for j in degrees:
            s += j[1]
        for each in g.nodes():
            node_prob[each] = (float)(degrees[each]) / s
        node_probabilities_cum = []
        prev = 0
        for n, p in node_prob.items():
            temp = [n, prev + p]
            node_probabilities_cum.append(temp)
            prev += p
        new_edges = []
        num_edges_added = 0
        target_nodes = []
        while (num_edges_added < m):
            prev_cum = 0
            r = random.random()
            k = 0
            while (not (r > prev_cum and r <= node_probabilities_cum[k][1])):
                prev_cum = node_probabilities_cum[k][1]
                k = k + 1
            target_node = node_probabilities_cum[k][0]
            if target_node in target_nodes:
                continue
            else:
                target_nodes.append(target_node)
            g.add_edge(i, target_node)
            num_edges_added += 1
            new_edges.append((i, target_node))
    return g



def plot_deg_dist_ba(g):
    all_degrees = []
    for i in nx.degree(g):
        all_degrees.append(i[1])
    N = g.number_of_nodes()
    ad = all_degrees
    unique_degrees = list(set(all_degrees))
    unique_degrees.sort()
    count_of_degrees = []
    for i in unique_degrees:
        c = all_degrees.count(i) / N
        count_of_degrees.append(c)
    mu, std = stats.expon.fit(ad)
    x = np.linspace(0, 65)
    p = stats.expon.pdf(x, scale=mu)
    plt.plot(x, p, 'k', linewidth=3, color='red')
    plt.plot(unique_degrees, count_of_degrees, 'o')
    plt.xlabel('Degrees')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution Plot')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



def plot_ccdf(g):
    all_degrees = []
    for i in nx.degree(g):
        all_degrees.append(i[1])
    N = g.number_of_nodes()
    unique_degrees = list(set(all_degrees))
    unique_degrees.sort()
    degree_dist = []
    for i in unique_degrees:
        c = all_degrees.count(i) / N
        degree_dist.append(c)
    c_dist = np.zeros(len(degree_dist), dtype=float)
    s = 0
    i = 0
    while s < 1 and i < len(degree_dist):
        s += degree_dist[i]
        c_dist[i] = 1 - s
        i += 1
    x = np.arange(len(c_dist))
    plt.title('Complementary cumulative distribution')
    plt.plot(x, c_dist)
    plt.show()



def rsim(G, rounds, payoff_matrix, max_diff):
    nd = list(G)
    N = G.number_of_nodes()
    nodes = []
    nodes_degrees = []
    for i in range (0, len(nd)):
        nodes.append([n for n in G.neighbors(i)])
    coop = []
    play_now = np.zeros(N)
    play_next = np.zeros(N)
    for i in range(0, N):     # round 1
        play_now[i] = np.random.randint(0, 2) # 1=COOP
    coop.append(play_now[play_now==0].size / play_now.size)
    for r in tqdm(range((rounds-1), 0, -1)):
        played = [False for i in range(0, N)]
        W = np.zeros(N)
        for i in range(0, N):
            for neighbor in nodes[i]:
                W[i] += payoff_matrix[int(play_now[i])][int(play_now[neighbor])]
        for i in range(0, N):
            if(len(nodes[i]) > 0):
                rn = np.random.choice(nodes[i])
                Pij = 0
                if(W[rn] > W[i]):
                    Kmax = max(len(nodes[i]), len(nodes[rn]))
                    Pij = (W[rn]-W[i])/(Kmax*max_diff)
                play_next[i] = np.random.choice([play_now[i], play_now[rn]], p=[1-Pij, Pij])
        play_now = play_next
        coop.append(play_now[play_now==0].size / play_now.size)
    return coop




def main():

    N = 10000
    rounds = 1000

    # Random net

    e = 20000
    G = generate_erg(e, N)         # generate the random network
    plot_deg_dist_erg(G)           # plot the random network
    R = 1
    P = 0
    S = -0.1
    T = [1.03, 1.07, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.95]
    coop = [[] for i in range(0, len(T))]
    for i in tqdm(range(0, len(T))):
        payoff_matrix = np.array([[1, -0.1], [T[i], 0]])
        max_diff = np.max(payoff_matrix[: , 0]) - np.min(payoff_matrix[: , 1])
        coop[i] = rsim(G, rounds, payoff_matrix, max_diff) #run simulation
    x = np.arange(1, rounds + 1)
    aa = np.linspace(0.1, 1, num=len(coop))
    for i in range(len(coop)):
        plt.plot(x, coop[i], color='b', alpha=aa[i])
    plt.xlabel('Rounds played')
    plt.ylabel('Cooperation (1 = Full cooperation)')
    plt.title('Prisoner dilemma simulation on a Random Network')
    plt.show()



    # Scale-free net

    m0 = 4
    G = nx.complete_graph(4)            # start from 4 interconnected nodes
    G = barabasi_add_nodes(G, N, m0)    # generate BA network starting from G and adding m0 links per new node until it has N nodes
    plot_deg_dist_ba(G)                 # plot degree distribution
    plot_ccdf(G)                        # plot complementary cumulative distribution

    R = 1
    P = 0
    S = -0.1
    T = [1.03, 1.07, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 1.95]
    coop = [[] for i in range(0, len(T))]
    for i in tqdm(range(0, len(T))):
        payoff_matrix = np.array([[1, -0.1], [T[i], 0]])
        max_diff = np.max(payoff_matrix[: , 0]) - np.min(payoff_matrix[: , 1])
        coop[i] = rsim(G, rounds, payoff_matrix, max_diff) #run simulation
    x = np.arange(1, rounds + 1)
    aa = np.linspace(0.1, 1, num=len(coop))
    for i in range(len(coop)):
        plt.plot(x, coop[i], color='b', alpha=aa[i])
    plt.xlabel('Rounds played')
    plt.ylabel('Cooperation (1 = Full cooperation)')
    plt.title('Prisoner dilemma simulation on a Scale-Free Network')
    plt.show()



if __name__ == '__main__':
    main()

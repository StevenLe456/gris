import networkx as nx
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind

tp = [nx.read_gml("graphs/" + f) for f in os.listdir("graphs") if f.startswith("tp")]
tn = [nx.read_gml("graphs/" + f) for f in os.listdir("graphs") if f.startswith("tn")]

results = []
cc = []
bc = []
idc = []
acc = []
gef = []

for g in tp:
    results.append(1)
    nums = np.array(list(nx.closeness_centrality(g).values()))
    cc.append(np.mean(nums))
    nums = np.array(list(nx.betweenness_centrality(g).values()))
    bc.append(np.mean(nums))
    nums = np.array(list(nx.in_degree_centrality(g).values()))
    idc.append(np.mean(nums))
    acc.append(nx.average_clustering(g))
    gef.append(nx.global_efficiency(g.to_undirected()))

for g in tn:
    results.append(0)
    nums = np.array(list(nx.closeness_centrality(g).values()))
    cc.append(np.mean(nums))
    nums = np.array(list(nx.betweenness_centrality(g).values()))
    bc.append(np.mean(nums))
    nums = np.array(list(nx.in_degree_centrality(g).values()))
    idc.append(np.mean(nums))
    acc.append(nx.average_clustering(g))
    gef.append(nx.global_efficiency(g.to_undirected()))

df = pd.DataFrame({"results": results, "closeness_centrality": cc, "betweenness_centrality": bc, 
    "in_degree_centrality": idc, "average_clustering_coefficient": acc, "global_efficiency": gef})

nonpsychotic = df[df["results"] == 0]
psychotic = df[df["results"] == 1]

with open("significance.txt", "w") as f:
    f.write("Significance of Closeness Centrality Measure\n")
    tup = ttest_ind(nonpsychotic["closeness_centrality"], psychotic["closeness_centrality"])
    f.write(f"T-test statistics: {tup[0]}, p-value: {tup[1]}\n")
    num1 = nonpsychotic["closeness_centrality"].mean()
    num2 = psychotic["closeness_centrality"].mean()
    f.write(f"Mean of non-psychotic sample's closeness centrality: {num1}\n")
    f.write(f"Mean of psychotic sample's closeness centrality: {num2}\n")
    f.write("Significance of Betweenness Centrality Measure\n")
    tup = ttest_ind(nonpsychotic["betweenness_centrality"], psychotic["betweenness_centrality"])
    f.write(f"T-test statistics: {tup[0]}, p-value: {tup[1]}\n")
    f.write("Significance of In-Degree Centrality Measure\n")
    tup = ttest_ind(nonpsychotic["in_degree_centrality"], psychotic["in_degree_centrality"])
    f.write(f"T-test statistics: {tup[0]}, p-value: {tup[1]}\n")
    num1 = nonpsychotic["in_degree_centrality"].mean()
    num2 = psychotic["in_degree_centrality"].mean()
    f.write(f"Mean of non-psychotic sample's in-degree centrality: {num1}\n")
    f.write(f"Mean of psychotic sample's in-degree centrality: {num2}\n")
    f.write("Significance of Average Clustering Coefficient Measure\n")
    tup = ttest_ind(nonpsychotic["average_clustering_coefficient"], psychotic["average_clustering_coefficient"])
    f.write(f"T-test statistics: {tup[0]}, p-value: {tup[1]}\n")
    num1 = nonpsychotic["average_clustering_coefficient"].mean()
    num2 = psychotic["average_clustering_coefficient"].mean()
    f.write(f"Mean of non-psychotic sample's average clustering coefficient: {num1}\n")
    f.write(f"Mean of psychotic sample's average clustering coefficient: {num2}\n")
    f.write("Significance of Global Efficiency Measure\n")
    tup = ttest_ind(nonpsychotic["global_efficiency"], psychotic["global_efficiency"])
    f.write(f"T-test statistics: {tup[0]}, p-value: {tup[1]}\n")
    num1 = nonpsychotic["global_efficiency"].mean()
    num2 = psychotic["global_efficiency"].mean()
    f.write(f"Mean of non-psychotic sample's global efficiency: {num1}\n")
    f.write(f"Mean of psychotic sample's global efficiency: {num2}\n")
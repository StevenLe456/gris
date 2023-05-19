import gensim.downloader
import networkx as nx
from nltk.tokenize import word_tokenize
import numpy as np
import os
import os.path as osp
import pickle
import torch
from torch_geometric.data import Data, Dataset, extract_zip

class PsychosisRedditDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["data/nonpsychotic_graphs/" + fil if fil.startswith("n") else "data/psychotic_graphs/" + fil for t in os.walk(self.raw_dir) for fil in t[2]] if osp.exists("tmp/raw") else []

    @property
    def processed_file_names(self):
        return["data_{}.pt".format(idx) for idx in range(9757)]

    def download(self):
        extract_zip("data.zip", self.raw_dir)

    def process(self):
        w2v = gensim.downloader.load("word2vec-google-news-300")
        
        with open("node_pca.pickle", "rb") as p:
            npca = pickle.load(p)

        with open("edge_pca.pickle", "rb") as p:
            epca = pickle.load(p)

        def graph_to_dict(g):
            nodes = g.nodes
            return {k: v for v, k in enumerate(nodes)}

        def embeddings(words):
            dummy = np.zeros((len(words), 1500))
            for k, w in enumerate(words):
                word_list = word_tokenize(w)
                temp = np.zeros((5, 300))
                for i in range(5):
                    try:
                        temp[i] = w2v[word_list[i]] if word_list[i] in w2v else np.zeroes((1, 300))
                    except:
                        break
                temp = temp.flatten()
                dummy[k] = temp
            return dummy
        
        def graph_to_x(g):
            nodes = g.nodes
            embd = embeddings(nodes)
            return torch.from_numpy(npca.transform(embd)).float()

        def graph_to_edge_index(g):
            d = graph_to_dict(g)
            l = [[d[i[0]], d[i[1]]] for i in g.edges]
            tensor = torch.LongTensor(l)
            return tensor.t().contiguous()

        def graph_to_edge_attr(g):
            edges = g.edges.data("relation")
            embd = embeddings([i[2] for i in edges])
            return torch.from_numpy(epca.transform(embd)).float()

        def graph_to_data(path):
            g = nx.read_gml(path)
            return Data(x=graph_to_x(g), edge_index=graph_to_edge_index(g), 
                edge_attr=graph_to_edge_attr(g), y=0 if "nonpsychotic" in path else 1)

        idx = 0
        for path in self.raw_paths:
            if not nx.is_empty(nx.read_gml(path)):
                data = graph_to_data(path)
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

from dataset import PsychosisRedditDataset
import gensim.downloader
from gnn import GNN
import networkx as nx
import pickle
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import to_networkx

with open("node_pca.pickle", "rb") as p:
    npca = pickle.load(p)

with open("edge_pca.pickle", "rb") as p:
    epca = pickle.load(p)

model = GNN()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

dataset = PsychosisRedditDataset("tmp")
dataset = dataset.shuffle()

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(),
    model_config=dict(
        mode="classification",
        task_level="graph",
        return_type="log_probs"
    ),
    explainer_config = dict(
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
    )
)

true_pos = []
true_neg = []

i = 0
while len(true_pos) <= 1000 or len(true_neg) <= 1000:
    y_hat = model(dataset[i].x, dataset[i].edge_index, dataset[i].edge_attr, dataset[i].batch)
    if torch.round(y_hat) == 0 and dataset[i].y == 0 and len(true_neg) <= 1000:
        true_neg.append("")
        explanation = explainer(dataset[i].x, dataset[i].edge_index, edge_attr=dataset[i].edge_attr, batch=dataset[i].batch)
        subgraph = explanation.get_explanation_subgraph()
        subgraph = to_networkx(subgraph)
        nx.write_gml(subgraph, f"graphs/tn{len(true_neg)}.gml")
    elif torch.round(y_hat) == 1 and dataset[i].y == 1 and len(true_pos) <= 1000:
        true_pos.append("")
        explanation = explainer(dataset[i].x, dataset[i].edge_index, edge_attr=dataset[i].edge_attr, batch=dataset[i].batch)
        subgraph = explanation.get_explanation_subgraph()
        subgraph = to_networkx(subgraph)
        nx.write_gml(subgraph, f"graphs/tp{len(true_pos)}.gml")
    i += 1
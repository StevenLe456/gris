from dataset import PsychosisRedditDataset
from gnn import GNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import torch.optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

writer = SummaryWriter()

with open("exp.txt", "r") as f:
    exp_num = int(f.read().strip())

dataset = PsychosisRedditDataset(root="tmp")
dataset = dataset.shuffle()
train = dataset[:int(len(dataset) * 0.8)]
test = dataset[int(len(dataset) * 0.8):]
train_loader = DataLoader(train, batch_size=1024, shuffle=True)
test_loader = DataLoader(test, batch_size=1024, shuffle=True)

model = GNN()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

for i in range(40):
    optimizer.zero_grad()
    j = 0
    k = 0
    train_acc = 0
    train_loss = 0
    test_acc = 0
    test_loss = 0
    print(f"Epoch:{i}")
    for data in train_loader:
        y_hat = model(data.x, data.edge_index, data.edge_attr, data.batch)
        train_acc += torch.sum(torch.round(y_hat.flatten()) == data.y.flatten()) / data.y.size(dim=0)
        loss = F.binary_cross_entropy(y_hat.flatten(), data.y.flatten().float())
        train_loss += loss
        loss.backward()
        optimizer.step()
        j += 1
    for data in test_loader:
        y_hat = model(data.x, data.edge_index, data.edge_attr, data.batch)
        test_acc += torch.sum(torch.round(y_hat.flatten()) == data.y.flatten()) / data.y.size(dim=0)
        test_loss += F.binary_cross_entropy(y_hat.flatten(), data.y.flatten().float())
        k += 1
    print(f"Training accuracy: {train_acc / j}, Training loss: {train_loss}")
    print(f"Test accuracy: {test_acc / k}, Test loss: {test_loss}")
    writer.add_scalar("Loss/train", train_loss, i)
    writer.add_scalar("Loss/test", test_loss, i)
    writer.add_scalar("Accuracy/train", train_acc / j, i)
    writer.add_scalar("Accuracy/test", test_acc / k, i)
    with open(f"training_{exp_num}.log", "w") as f:
        f.write(f"Final training accuracy: {train_acc / j}, Final training accuracy: {test_acc / k}")
    if train_acc / j > 0.95 and test_acc / k > 0.95:
        break

y_pred = []
y_true = []
cf_matrix = []

model.eval()
for data in test_loader:
    y_hat = model(data.x, data.edge_index, data.edge_attr, data.batch)
    output = (torch.round(y_hat.flatten())).data.cpu().numpy()
    y_pred.extend(output)
    labels = data.y.data.cpu().numpy()
    y_true.extend(labels)

cf_matrix = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cf_matrix, index = [i for i in range(2)],
                     columns = [i for i in range(2)])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)

plt.savefig(f'cf_{exp_num}.png')
torch.save(model.state_dict(), f"model_{exp_num}.pth")

writer.close()

with open("exp.txt", "w") as f:
    f.write(str(exp_num + 1))
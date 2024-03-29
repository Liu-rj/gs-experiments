import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
from model import PyGSAGEModel
import time
import pandas as pd
import numpy as np


dataset = PygNodePropPredDataset("ogbn-products", "/home/ubuntu/dataset")
split_idx = dataset.get_idx_split()
data = dataset[0].to("cuda")
fanouts = [25, 10]

train_idx, val_idx = split_idx["train"].to("cuda"), split_idx["valid"].to("cuda")
train_loader = NeighborSampler(
    data.edge_index,
    node_idx=train_idx,
    sizes=fanouts,
    batch_size=512,
    shuffle=True,
    num_workers=0,
)
val_loader = NeighborSampler(
    data.edge_index,
    node_idx=val_idx,
    sizes=fanouts,
    batch_size=512,
    shuffle=True,
    num_workers=0,
)


model = PyGSAGEModel(
    dataset.num_features, 256, dataset.num_classes, num_layers=len(fanouts)
)
model = model.to("cuda")

x = data.x.to("cuda")
y = data.y.squeeze().to("cuda")


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to("cuda") for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


def evaluate(epoch):
    model.eval()

    pbar = tqdm(total=val_idx.size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_correct = 0
    for batch_size, n_id, adjs in val_loader:
        adjs = [adj.to("cuda") for adj in adjs]
        out = model(x[n_id], adjs)

        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    approx_acc = total_correct / val_idx.size(0)

    return approx_acc


model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

epoch_time = []
cur_time = []
acc_list = []
start = time.time()
n_epochs = 100
for epoch in range(n_epochs):
    torch.cuda.synchronize()
    tic = time.time()
    loss, acc = train(epoch)
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")
    val_acc = evaluate(epoch)
    print(f"Epoch {epoch:02d}, Approx. Validation: {val_acc:.4f}")
    acc_list.append(val_acc)
    torch.cuda.synchronize()
    end = time.time()
    cur_time.append(end - start)
    epoch_time.append(end - tic)

torch.cuda.synchronize()
total_time = time.time() - start

print("Total Elapse Time:", total_time)
print("Average Epoch Time:", np.mean(epoch_time[3:]))
s5 = pd.Series(cur_time, name="cumulated time/s")
s1 = pd.Series(acc_list, name="acc")
s2 = pd.Series(epoch_time, name="time/s")
s3 = pd.Series([total_time], name="total time/s")
df = pd.concat([s5, s1, s2, s3], axis=1)
df.to_csv(
    "outputs/data/graphsage_pyg_products_{}.csv".format(time.ctime().replace(" ", "_")),
    index=False,
)

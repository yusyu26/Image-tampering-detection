import os, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from nets import DepthwiseSeparableCNN

# ---------- データ読み込み ----------
class GLCMDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        glcm = d["glcm"]
        if glcm.ndim == 4:           # (256,256,1,1) → (256,256)
            glcm = glcm[...,0,0]
        x = torch.from_numpy(glcm).unsqueeze(0).float()   # (1,H,W)
        y = torch.tensor(d["label"].item(), dtype=torch.float32)
        return x, y

files = np.array(glob.glob("glcm/*.npz"))
np.random.shuffle(files)
split = int(len(files)*0.8)
train_files, val_files = files[:split], files[split:]

# ---------- クラス不均衡対策：重み付きサンプラー ----------
labels = [np.load(f)["label"].item() for f in train_files]
class_counts = np.bincount(labels)        # [Au枚数, Tp枚数]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dl = DataLoader(GLCMDataset(train_files), batch_size=32,
                      sampler=sampler)
val_dl   = DataLoader(GLCMDataset(val_files),  batch_size=32)

# ---------- モデル & 損失 ----------
device  = torch.device("cpu")
model   = DepthwiseSeparableCNN().to(device)

# pos_weight = 負例/正例 ≒  1/3.6 ≈ 0.28
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.28]))
opt = optim.Adam(model.parameters(), lr=1e-3)

os.makedirs("checkpoints", exist_ok=True)   #  フォルダを作る
best = 0

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    for x, y in tqdm(train_dl, desc=f"epoch{epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device).unsqueeze(1)
        opt.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward(); opt.step()

    # ---------- validation ----------
    model.eval(); correct=total=0
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x.to(device)).cpu()
            preds  = (torch.sigmoid(logits) >= 0.5).int().squeeze(1)
            labels = y.int()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct/total
    print(f"val acc: {acc:.3f}")

    # ---------- checkpoint ----------
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "checkpoints/best.pt")
        print(f"saved checkpoints/best.pt  (val acc {best:.3f})")

import sys, numpy as np, torch
from nets import DepthwiseSeparableCNN

model = DepthwiseSeparableCNN()
model.load_state_dict(torch.load("checkpoints/casia2_cnn.pt", map_location="cpu"))
model.eval()

glcm = np.load(sys.argv[1])["glcm"][0]           # (H,W)
x = torch.from_numpy(glcm).unsqueeze(0).unsqueeze(0).float()
prob = model(x).item()
print(f"Tampered probability = {prob:.3f}")

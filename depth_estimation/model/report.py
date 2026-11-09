import os, re
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from peft import LoraConfig
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode
from Model import DepthAnythingPEFT
from Dataset import UVMultiDataset

MODEL = "LiheYoung/depth-anything-small-hf"
CKPT = r"C:\Users\yoame\Documents\ML\underwater\output\lora_10\depth-anything-small-hf_4.pth"
ROOT = r"C:\Users\yoame\Documents\ML\underwater\UVmulti\all_vid"
EVAL = r"C:\Users\yoame\Documents\ML\underwater\eval"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Loss curve (from train_log.txt) ----------
raw = open("train_log.txt", "rb").read()
txt = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", errors="ignore")
pairs = re.findall(r"LOSS train ([\d.]+) valid ([\d.]+)", txt)
train = [float(a) for a, _ in pairs]
valid = [float(b) for _, b in pairs]
epochs = range(1, len(pairs) + 1)
plt.figure(figsize=(6, 4))
plt.plot(epochs, train, marker="o", label="Train loss")
plt.plot(epochs, valid, marker="o", label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("L1 loss"); plt.title("Training curve"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/loss_curve.png", dpi=200); plt.close()

# ---------- 2. Before/after metrics table ----------
def mean_row(path):
    df = pd.read_csv(path, index_col=0).iloc[:-1]          # drop the 'Mean' row, recompute safely
    return df.replace([np.inf, -np.inf], np.nan).mean()

tbl = pd.DataFrame({"Without training": mean_row(f"{EVAL}/without_training.csv"),
                    "After training": mean_row(f"{EVAL}/after_training.csv")}).round(4)
tbl.to_csv(f"{OUT}/metrics_comparison.csv")
print(tbl)

# ---------- 3. Sample predictions ----------
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
proc = AutoImageProcessor.from_pretrained(MODEL)
tf = Compose([Resize((proc.size["height"], proc.size["width"]), interpolation=InterpolationMode.BICUBIC), ToTensor()])
ds = UVMultiDataset(root_dir=ROOT, transform=tf)

base = DepthAnythingPEFT(MODEL).model.eval().to(dev)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"],
                 lora_dropout=0.001, bias="lora_only", modules_to_save=["decode_head"])
tuned = DepthAnythingPEFT(MODEL).peft_model(cfg)
tuned.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state_dict"])
tuned.eval().to(dev)

def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

idxs = np.linspace(0, len(ds) - 1, 5, dtype=int)
fig, ax = plt.subplots(len(idxs), 4, figsize=(12, 3 * len(idxs)))
for r, i in enumerate(idxs):
    img, gt = ds[i]
    x = img.unsqueeze(0).to(dev)
    with torch.no_grad():
        p0 = base(x).predicted_depth[0].cpu().numpy()
        p1 = tuned(x).predicted_depth[0].cpu().numpy()
    panels = [img.permute(1, 2, 0).numpy(), norm(gt[0].numpy()), norm(p0), norm(p1)]
    titles = ["Input", "Ground truth", "Before training", "After training"]
    for c in range(4):
        ax[r, c].imshow(panels[c], cmap=None if c == 0 else "inferno")
        ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(titles[c])
plt.tight_layout(); plt.savefig(f"{OUT}/sample_predictions.png", dpi=150)
print("Saved everything in the 'results' folder")
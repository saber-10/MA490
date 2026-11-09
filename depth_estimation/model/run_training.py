from Training import PEFTTraining
from Model import DepthAnythingPEFT
from Dataset import UVMultiDataset
from peft import LoraConfig
import torch
from torch.utils.data import Subset
from torch import nn
import wandb
import os
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode

os.environ["WANDB_MODE"] = "offline"   # no wandb account needed

### Config
EXPERIMENT_NUM = 10
MODEL_CHECKPOINT = "LiheYoung/depth-anything-small-hf"
DATASET_ROOT_DIR = "C:/Users/yoame/Documents/ML/underwater/UVmulti/all_vid"
OUTPUT_DIR = rf"C:\Users\yoame\Documents\ML\underwater\output\lora_{EXPERIMENT_NUM}"
WANDB_USER = None
WANDB_PROJECT = "peft_training"
WANDB_DATASET = "UVMULTI"

### Hyperparameters
TRAIN_BATCH_SIZE = 4        # lower this if you get out-of-memory errors
VALID_BATCH_SIZE = 4
DATA_USE_PERCENTAGE = 100
TRAIN_SPLIT = 0.8
LOSS = nn.L1Loss()
OPTIM = "AdamW"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.001
BIAS = "lora_only"
GRAD_CLIP = 1.0
MIN_LR = 1e-7
LEARNING_RATE = 1e-3
WARMUP_PERIOD_PERCENTAGE = 30
EPOCH = 5

model = DepthAnythingPEFT(model_checkpoint=MODEL_CHECKPOINT)

data_transforms = Compose([
    Resize((model.image_processor.size["height"], model.image_processor.size["width"]),
           interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
])

dataset = UVMultiDataset(root_dir=DATASET_ROOT_DIR, transform=data_transforms)
useful_dataset_length = int(len(dataset) * DATA_USE_PERCENTAGE / 100)
print(f"Length of Dataset: {useful_dataset_length}")
train_size = int(TRAIN_SPLIT * useful_dataset_length)
valid_size = useful_dataset_length - train_size
useful_dataset = Subset(dataset, list(range(useful_dataset_length)))
train_dataset, valid_dataset = torch.utils.data.random_split(useful_dataset, [train_size, valid_size])

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "value"],
    lora_dropout=LORA_DROPOUT,
    bias=BIAS,
    modules_to_save=["decode_head"],
)

lora_model = model.peft_model(peft_config)
model.trainable_parameters(lora_model)
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)

config = {"lr": LEARNING_RATE, "batch_size": TRAIN_BATCH_SIZE, "epoch": EPOCH,
          "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA}
logger = wandb.init(project=WANDB_PROJECT, name=f"{WANDB_DATASET} Epoch: {EPOCH}", config=config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = PEFTTraining(MODEL_CHECKPOINT, OUTPUT_DIR, lora_model, train_dataset, valid_dataset,
                       TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, LOSS, optimizer, EPOCH, device,
                       WARMUP_PERIOD_PERCENTAGE, LEARNING_RATE, MIN_LR, GRAD_CLIP, True)

trainer.train(logger)
logger.finish()
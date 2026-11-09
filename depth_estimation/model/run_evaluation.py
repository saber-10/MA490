from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from Model import DepthAnythingPEFT
from Dataset import UVMultiDataset
import torch
import pandas as pd
from peft import LoraConfig
from Evaluation import EvaluationMetric
from torch.utils.data import Subset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    InterpolationMode,
)
import os


def model_evaluation(model,image_processor,dataset_root_dir,data_use_percentage,batch_size,output_file_name):

    print("Moddel Loaded")

    data_transforms = Compose(
        [
            # RandomResizedCrop(image_processor.size["height"]),
            # RandomHorizontalFlip(),
            Resize((image_processor.size["height"],image_processor.size["width"]),interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ]
    )

    print("Dataset Loaded")
    dataset = UVMultiDataset(root_dir= dataset_root_dir, transform=data_transforms)
    useful_dataset_length = int(len(dataset) * data_use_percentage /100)
    useful_dataset = Subset(dataset,list(range(useful_dataset_length)))
    dataloder = torch.utils.data.DataLoader(useful_dataset, batch_size= batch_size, shuffle= False)

    model.eval()
    evaluation = EvaluationMetric(False)
    all_metrics = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        print("evaluation started...")
        for i, (inputs, labels) in enumerate(dataloder):
            inputs, labels = inputs.to(device),labels.to(device)
            print('inputs',inputs.shape)
            print('labels',labels.shape)
            outputs = model(inputs).predicted_depth
            metrics = evaluation.compute_metrics(inputs,outputs,labels)
            print(metrics)
            all_metrics.append(metrics)
            print("evaluation num",i)

    print("evaluation ended; processing data...")
    # Create a DataFrame from the list of evaluation metrics
    df = pd.DataFrame(all_metrics, columns=['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10', 'rmse_log',
                                            'silog', 'sq_rel', 'psnr','ssim','pearson_corr'])

    # Add a row for the sum of all metrics
    mean_row = df.mean()
    mean_row.name = 'Mean'
    df = pd.concat([df, pd.DataFrame([mean_row])])

    # Save the DataFrame as a CSV file

    df.to_csv(output_file_name)

    print("Evaluation metrics saved as", output_file_name)


def main():
    MODEL_CHECKPOINT = "LiheYoung/depth-anything-small-hf"
    # the .pth saved by training, in your OUTPUT_DIR (use the highest epoch number, it's the best one)
    TRAINED_CHECKPOINT = r"C:\Users\yoame\Documents\ML\underwater\output\lora_10\depth-anything-small-hf_4.pth"
    IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    DATASET_ROOT_DIR = r"C:\Users\yoame\Documents\ML\underwater\UVmulti\all_vid"
    DATA_USE_PERCENTAGE = 100
    BATCH_SIZE = 4
    LORA_RANK = 16      # MUST match training
    LORA_ALPHA = 32     # MUST match training
    LORA_DROPOUT = 0.001
    BIAS = "lora_only"

    os.makedirs(r"C:\Users\yoame\Documents\ML\underwater\eval", exist_ok=True)
    depth_anything = DepthAnythingPEFT(model_checkpoint=MODEL_CHECKPOINT)

    # Before training
    model_evaluation(depth_anything.model, IMAGE_PROCESSOR, DATASET_ROOT_DIR, DATA_USE_PERCENTAGE,
                     BATCH_SIZE, r"C:\Users\yoame\Documents\ML\underwater\eval\without_training.csv")

    peft_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=["query", "value"],
                             lora_dropout=LORA_DROPOUT, bias=BIAS, modules_to_save=["decode_head"])
    lora_model = depth_anything.peft_model(peft_config)
    checkpoint = torch.load(TRAINED_CHECKPOINT, map_location="cpu")
    lora_model.load_state_dict(checkpoint["model_state_dict"])

    # After training
    model_evaluation(lora_model, IMAGE_PROCESSOR, DATASET_ROOT_DIR, DATA_USE_PERCENTAGE,
                     BATCH_SIZE, r"C:\Users\yoame\Documents\ML\underwater\eval\after_training.csv")

main()

    


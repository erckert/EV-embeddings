import torch

from config import App
from setup.configProcessor import select_model_from_config
from machine_learning.Predictions import predict
from machine_learning.Dataset import load_dataset
from machine_learning.Dataloader import get_dataloader
from output.OutputGenerator import write_prediction_fasta

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = select_model_from_config(device=device)

    dataset = load_dataset()
    dataloader = get_dataloader(dataset)

    predictions = predict(device, model, dataloader)

    write_prediction_fasta(predictions)
    print(f"Finished. Predicted structures for query sequences are available in {App.config()['GENERAL']['out_file']}")



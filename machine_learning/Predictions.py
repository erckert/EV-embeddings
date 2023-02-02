import itertools

import torch

from config import App


def predict(device, model, dataloader):
    embedding_type = App.config()["GENERAL"]["embedding_type"].lower()
    evolutionary_information = App.config()["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower()

    if embedding_type.lower() == "singleseq" and evolutionary_information.lower() == "msaconsensus":
        return predict_consensus_from_msa(device, model, dataloader)
    else:
        return predict_individual_sequences(device, model, dataloader)


def predict_individual_sequences(device, model, dataloader):
    predictions = {}
    with torch.no_grad():
        for i, (inputs, protein_names) in enumerate(dataloader):
            output = model(inputs.to(device))
            network_predictions = [prediction.argmax(dim=0) for prediction in output]

            for network_prediction, protein_name, network_input in zip(network_predictions, protein_names, inputs):
                network_prediction = [entry.item() for entry in network_prediction]
                mask = [all(embedding_dim != -100 for embedding_dim in entry) for entry in [residue_input for residue_input in network_input]]
                predictions[protein_name] = list(itertools.compress(network_prediction, mask))

    return predictions


def predict_consensus_from_msa(device, model, dataloader):
    predictions = {}
    with torch.no_grad():
        for i, (inputs, query_names, protein_names) in enumerate(dataloader):
            outputs = model(inputs.to(device))
            network_predictions = [prediction.argmax(dim=0) for prediction in outputs]

            for network_prediction, query_name, protein_name in zip(network_predictions, query_names, protein_names):
                if query_name not in predictions:
                    predictions[query_name] = {}
                # TODO: remove masked residue predictions, format of prediction
                predictions[query_name][protein_name] = network_prediction

    return predictions

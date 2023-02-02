import sys
import os
import torch
import colorama

from config import App
from colorama import Fore


embedding_models = ["SeqVec", "Bert", "ProtT5"]
embedding_types = ["SingleSeq", "MSA"]
evolutionary_information_types = ["None", "MSAConsensus", "PSSM"]
pssm_models = ["Concat", "Split"]
RED = "\033[31m"


def select_model_from_config(device):
    config = App.config()
    # we need this for color coded print statements
    colorama.init(autoreset=True)

    if not is_valid_config(config):
        sys.exit("Config is invalid. Please provide a valid config file.")
    print("Config file is valid")

    return select_model(config, device)


def is_valid_config(config):
    # No sections => File doesn't exist or is missing sections
    if len(config.sections()) == 0:
        print("Config file not found or empty")
        return False

    # Return False if any section in general is not filled out with a valid option
    general_section = config["GENERAL"]
    if general_section["embedding_model"].lower() not in map(lambda x: x.lower(), embedding_models):
        print(f"Config file does not specify a valid embedding_model. Please select one of the following: "
              f"{embedding_models}")
        return False
    if general_section.get("embedding_type").lower() not in map(lambda x: x.lower(), embedding_types):
        print(f"Config file does not specify a valid embedding_type. Please select one of the following: "
              f"{embedding_types}")
        return False
    if general_section.get("embedding_folder") == "" or not os.path.exists(general_section.get("embedding_folder")):
        print(f"File path to embedding_folder is missing or an invalid path. The path provided was: "
              f"{general_section.get('embedding_folder')}")
        return False
    if general_section.get("out_file") == "" or not os.path.exists(os.path.dirname(general_section.get("out_file"))):
        print(f"File path to out_file is missing or an invalid path. The path provided was: "
              f"{general_section.get('out_file')}")
        return False

    # Check if out_file already exists
    if os.path.exists(general_section.get("out_file")):
        print(Fore.RED + f"WARNING: File {general_section.get('out_file')} already exists and will be overwriten")

    # If MSA has been selected as embedding_type all further entries can be ignored,
    # else check remaining relevant entries
    if general_section.get("embedding_type").lower() == "msa":
        return True

    # Check if evolutionary information type is selected
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() not in \
            map(lambda x: x.lower(), evolutionary_information_types):
        print(f"Config file does not specify a valid evolutionary_information. Please select one of the following: "
              f"{evolutionary_information_types}")
        return False

    # If None has been selected as evolutionary_information all further entries can be ignored,
    # else check remaining relevant entries
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "none":
        return True
    # MSAConsensus needs MSA file in stockholm format and output folder path
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "msaconsensus":
        # check for MSA file
        if config["MSA CONSENSUS"]["msa_file"] == "" or not os.path.exists(config["MSA CONSENSUS"]["msa_file"]):
            print(f"File path to msa_file is missing or an invalid path. The path provided was: "
                  f"{config['MSA CONSENSUS']['msa_file']}")
            return False
        # TODO: Check if format is stockholm?
        # check for output folder path
        if config["MSA CONSENSUS"]["out_folder_msa_fastas"] == "" or \
                not os.path.exists(config["MSA CONSENSUS"]["out_folder_msa_fastas"]):
            print(f"File path to out_folder_msa_fastas is missing or an invalid path. The path provided was: "
                  f"{config['MSA CONSENSUS']['out_folder_msa_fastas']}")
            return False
        return True
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "pssm":
        if config["PSSM"]["model_selection"].lower() not in map(lambda x: x.lower(), pssm_models):
            print(f"Config file does not specify a valid model_selection. Please select one of the following: "
                  f"{pssm_models}")
            return False
        if config["PSSM"]["pssm_file"] == "" or not os.path.exists(config["PSSM"]["pssm_file"]):
            print(f"File path to pssm_file is missing or an invalid path. The path provided was: "
                  f"{config['PSSM']['pssm_file']}")
            return False
        # TODO: check PSSM format?

    return True


def select_model(config, device):
    config_general_section = config["GENERAL"]
    embedding_model = config_general_section["embedding_model"].lower()
    embedding_type = config_general_section["embedding_type"].lower()
    evolutionary_information = config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower()

    match embedding_model:
        case "seqvec":
            model_sub_folder = "SeqVec"
        case "bert":
            model_sub_folder = "Bert"
        case "prott5":
            model_sub_folder = "ProtT5"
        case _:
            sys.exit(f"Error: No sub folder found for model {embedding_model}. Please contact the development team.")

    model_folder_path = os.path.join("pretrained_models", model_sub_folder)

    match embedding_type:
        case "msa":
            model_name = "pretrained_msa_embedding.model"
        case "singleseq":
            match evolutionary_information:
                case "pssm":
                    model_name = f"pretrained_embedding_and_pssm_{config['PSSM']['model_selection'].lower()}.model"
                case "msaconsensus":
                    model_name = f"pretrained_embedding_and_msa_consensus.model"
                case "none":
                    model_name = "pretrained_embedding_only.model"
                case _:
                    sys.exit(
                        f'Error: No matching model "{embedding_model}" found for "{evolutionary_information}" '
                        f'as evolutionary information. Please contact the development team.')
        case _:
            sys.exit(f'Error: No matching model "{embedding_model}" found with "{embedding_type}" as embedding type. '
                     f'Please contact the development team.')

    model = torch.load(os.path.join(model_folder_path, model_name), map_location=device)
    model.eval()
    print(f'"{embedding_model}" model for "{embedding_type}" embeddings with evolutionary information '
          f'"{evolutionary_information if embedding_type != "msa" else "MSAEmbeddings"}" has been loaded')

    return model

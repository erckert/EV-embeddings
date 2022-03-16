import sys
import os

from config import App

embedding_models = ["SeqVec", "Bert", "ProtT5"]
embedding_types = ["SingleSeq", "MSA"]
evolutionary_informations = ["None", "MSAConsensus", "PSSM"]
pssm_models = [0, 1]


def select_model_from_config():
    config = App.config()
    if not is_valid_config(config):
        sys.exit("Config is invalid. Please provide a valid config file.")
    select_model(config)
    print("Config file is valid")


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
    if general_section.get("embedding_type").lower() not in map(lambda x:x.lower(), embedding_types):
        print(f"Config file does not specify a valid embedding_type. Please select one of the following: "
              f"{embedding_types}")
        return False
    if general_section.get("embedding_folder") == "" or not os.path.exists(general_section.get("embedding_folder")):
        print(f"File path to embedding_folder is missing or an invalid path. The path provided was: "
              f"{general_section.get('embedding_folder')}")
        return False

    # If MSA has been selected as embedding_type all further entries can be ignored,
    # else check remaining relevant entries
    if general_section.get("embedding_type").lower() == "msa":
        return True

    # Check if evolutionary information type is selected
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() not in \
            map(lambda x: x.lower(), evolutionary_informations):
        print(f"Config file does not specify a valid evolutionary_information. Please select one of the following: "
              f"{evolutionary_informations}")
        return False
    

    # If None has been selected as evolutionary_information all further entries can be ignored,
    # else check remaining relevant entries
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "none":
        return True
    # MSAConsensus needs MSA file in stockholm format
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "msaconsensus":
        if config["MSA CONSENSUS"]["msa_file"] == "" or not os.path.exists(config["MSA CONSENSUS"]["msa_file"]):
            print(f"File path to msa_file is missing or an invalid path. The path provided was: "
                  f"{config['MSA CONSENSUS']['msa_file']}")
            return False
        # TODO: Check if format is stockholm?
        return True
    if config["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower() == "pssm":
        if config["PSSM"]["model_selection"] not in map(lambda x: str(x), pssm_models):
            print(f"Config file does not specify a valid model_selection. Please select one of the following: "
                  f"{pssm_models}")
            return False
        if config["PSSM"]["pssm_file"] == "" or not os.path.exists(config["PSSM"]["pssm_file"]):
            print(f"File path to pssm_file is missing or an invalid path. The path provided was: "
                  f"{config['PSSM']['pssm_file']}")
            return False
        # TODO: check PSSM format?

    return True

def select_model(config):
    return ""
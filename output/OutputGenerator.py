from config import App


def map_structure_to_string(structure):
    match structure:
        case 0:
            return "C"
        case 1:
            return "H"
        case 2:
            return "E"
        case _:
            return "-"


def write_prediction_fasta(prediction):
    embedding_type = App.config()["GENERAL"]["embedding_type"].lower()
    evolutionary_information = App.config()["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower()

    if embedding_type.lower() == "singleseq" and evolutionary_information.lower() == "msaconsensus":
        return write_msa_consensus_fasta_files(prediction)
    else:
        return write_normal_fasta_file(prediction)


def write_msa_consensus_fasta_files(prediction):
    # TODO
    return None


def write_normal_fasta_file(prediction):
    out_file = App.config()['GENERAL']["out_file"]
    with open(out_file, 'w') as fh:
        for key, value in prediction.items():
            fh.write(f">{key}\n")
            fh.write("".join([map_structure_to_string(residue) for residue in value]))
            fh.write("\n")
    print(f"Predictions have been writen to {out_file}")
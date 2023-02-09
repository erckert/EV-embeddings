import io
import os

import numpy as np

from src.config import App
from Bio import AlignIO


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
    print("Writing output")
    embedding_type = App.config()["GENERAL"]["embedding_type"].lower()
    evolutionary_information = App.config()["SINGLE SEQUENCE EMBEDDINGS"]["evolutionary_information"].lower()

    if embedding_type.lower() == "singleseq" and evolutionary_information.lower() == "msaconsensus":
        return write_msa_consensus_fasta_files(prediction)
    else:
        return write_normal_fasta_file(prediction)


def write_msa_consensus_fasta_files(prediction):
    out_file = App.config()['GENERAL']["out_file"]
    out_folder = App.config()['MSA CONSENSUS']["out_folder_msa_fastas"]
    msa_file = App.config()['MSA CONSENSUS']["msa_file"]

    msa_consensus_writer = open(out_file, "w")

    # Write predictions for each sequence in MSA
    for query_id, prediction_dict in prediction.items():
        # write fasta with all sequences belonging to an individual query
        with open(os.path.join(out_folder, query_id) + ".fasta", "w") as fh:
            for key, value in prediction_dict.items():
                fh.write(f">{key}\n")
                fh.write("".join([map_structure_to_string(residue) for residue in value]))
                fh.write("\n")

    # Write MSAConsensus predictions
    temp_file = io.StringIO("")
    with open(msa_file, "r") as fh:
        for line in fh:
            temp_file.write(line)
            if line.startswith("//"):
                temp_file.seek(0, 0)
                msa = AlignIO.read(temp_file, "stockholm")
                temp_file.close()

                query_sequence_name = msa[0].id.replace(":", "_")
                prediction_average = np.zeros((len(msa[0].seq), 3))

                predictions_query_msa = prediction[query_sequence_name]
                for seq_id, seq_prediction in predictions_query_msa.items():
                    msa_entry = [entry for entry in msa if entry.id.replace(":", "_") == seq_id][0]
                    msa_seq = msa_entry.seq

                    # map predicted structures to corresponding position in MSA
                    index_prediction = 0
                    for i, character in enumerate(msa_seq):
                        if character != "-":
                            predicted_structure = seq_prediction[index_prediction]
                            prediction_average[i, predicted_structure] += 1
                            index_prediction += 1

                # initialize empty final prediction
                final_prediction = np.zeros(prediction_average.shape[0])
                for j, entry in enumerate(prediction_average):
                    final_prediction[j] = np.argmax(entry)
                out_string = ""
                for maximum in final_prediction:
                    out_string = out_string + map_structure_to_string(maximum)

                msa_consensus_writer.write(F">{query_sequence_name}\n")
                msa_consensus_writer.write(out_string + "\n")

                # reset temp_file to empty file for next MSA
                temp_file = io.StringIO("")

    msa_consensus_writer.close()


def write_normal_fasta_file(prediction):
    out_file = App.config()['GENERAL']["out_file"]
    with open(out_file, 'w') as fh:
        for key, value in prediction.items():
            fh.write(f">{key}\n")
            fh.write("".join([map_structure_to_string(residue) for residue in value]))
            fh.write("\n")
    print(f"Predictions have been writen to {out_file}")
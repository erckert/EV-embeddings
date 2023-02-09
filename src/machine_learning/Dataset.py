import os
import ntpath
import torch

import numpy as np
from src.setup.configProcessor import get_dataset_parameters, predict_msa_consensus


def map_pssms(use_pssm, pssm_file, lookup_file):
    numeric_id_to_pdb_id = {}
    id_to_pssm = {}

    if not use_pssm:
        return {}

    with open(lookup_file, 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            if line != '':
                parts = line.split('\t')
                numeric_id_to_pdb_id[parts[0].strip()]=parts[1].strip()

    with open(pssm_file, 'r') as fh:
        lines = fh.readlines()
        next_is_first_pssm_row = False
        pssm_nr = ""
        pssm = ""
        for line in lines:
            if next_is_first_pssm_row:
                pssm = ""
                next_is_first_pssm_row = False

            if line.startswith('Query profile of sequence'):
                next_is_first_pssm_row = True
                if pssm != "":
                    pssm_as_list = pssm.split('\n')[:-1]
                    pssm_as_list = [line.split('\t') for line in pssm_as_list]
                    id_to_pssm[numeric_id_to_pdb_id[pssm_nr].replace(':', '_')] = pssm_as_list
                pssm_nr = line.split('sequence')[1].strip()
            else:
                pssm = pssm + line
    pssm_as_list = pssm.split('\n')[:-1]
    pssm_as_list = [line.split('\t') for line in pssm_as_list]
    id_to_pssm[numeric_id_to_pdb_id[pssm_nr].replace(':', '_')] = pssm_as_list

    return id_to_pssm


def get_pssm_entries_as_matrix(pssm):
    return [entry[2:] for entry in pssm[1:]]


class ProteinBasedEmbeddingDataset:
    def __init__(self, embedding_folder, low_memory=True, use_pssm=False, pssm_file="", lookup_file=""):
        print('Loading dataset')
        self.low_memory = low_memory
        self.last_embedding_name = ''
        self.last_embedding = None

        self.embeddings = []
        self.protein_ids = []
        self.embedding_folder = embedding_folder

        self.pssms = map_pssms(use_pssm, pssm_file, lookup_file)

        embedding_paths = [os.path.join(self.embedding_folder, embedding_name)
                           for embedding_name in os.listdir(self.embedding_folder)]
        for embedding_path in embedding_paths:
            protein_id = ntpath.basename(embedding_path).replace(".npy", "")
            self.protein_ids.append(protein_id)
            if not low_memory:
                embedding = np.load(embedding_path)
                self.embeddings.append(embedding)

        print('Finished loading the dataset')

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, item):
        id_chain = self.protein_ids[item]

        if self.low_memory:
            if not self.last_embedding_name == id_chain:
                self.last_embedding_name = id_chain
                self.last_embedding = np.load(os.path.join(self.embedding_folder, id_chain.replace(':', '_') + '.npy'))

            data = torch.tensor(self.last_embedding).float()
        else:
            data = torch.tensor(self.embeddings[item]).float()

        if self.pssms:
            pssm = self.pssms[id_chain]
            pssm_entries = get_pssm_entries_as_matrix(pssm)
            pssm_entries = [[int(inner_entry) for inner_entry in entry] for entry in pssm_entries]
            pssm_entry_tensor = torch.tensor(np.array(pssm_entries)).float()
            if data.shape[0] != len(pssm_entries):
                print("Debug me!")
            data = torch.cat((data, pssm_entry_tensor), 1)

        return data, id_chain, id_chain


class ProteinBasedMSAConsensusEmbeddingDataset:
    def __init__(self, embedding_folder, low_memory=True):
        print('Loading dataset')
        self.low_memory = low_memory
        self.last_embedding_name = ''
        self.last_embedding = None

        self.embeddings = []
        self.protein_ids = []
        self.embedding_folder = embedding_folder

        embedding_sub_folders = [os.path.join(self.embedding_folder, embedding_folder) for embedding_folder
                                 in os.listdir(self.embedding_folder)]
        self.embedding_paths = []
        for folder in embedding_sub_folders:
            for entry in os.listdir(folder):
                self.embedding_paths.append(os.path.join(folder, entry))
        for embedding_path in self.embedding_paths:
            protein_id = ntpath.basename(embedding_path).replace(".npy", "")
            self.protein_ids.append(protein_id)
            if not low_memory:
                embedding = np.load(embedding_path)
                self.embeddings.append(embedding)

        print('Finished loading the dataset')

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, item):
        protein_id = self.protein_ids[item]
        embedding_path = self.embedding_paths[item]
        query_id = os.path.normpath(embedding_path).split(os.path.sep)[-2]

        if self.low_memory:
            if not self.last_embedding_name == protein_id:
                self.last_embedding_name = protein_id
                self.last_embedding = np.load(embedding_path)

            data = torch.tensor(self.last_embedding).float()
        else:
            data = torch.tensor(self.embeddings[item]).float()

        return data, protein_id, query_id


def load_dataset():
    embedding_folder, use_pssm, pssm_file, lookup_file, low_memory = get_dataset_parameters()
    if predict_msa_consensus():
        return ProteinBasedMSAConsensusEmbeddingDataset(embedding_folder)
    else:
        return ProteinBasedEmbeddingDataset(
            embedding_folder,
            low_memory=low_memory,
            use_pssm=use_pssm,
            pssm_file=pssm_file,
            lookup_file=lookup_file
            )

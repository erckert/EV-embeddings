from torch.nn.utils.rnn import pad_sequence


class ProteinCollate(object):
    def __call__(self, data, ignore_idx=-100):
        inputs, protein_info, query_name = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=ignore_idx)
        return inputs, protein_info, query_name

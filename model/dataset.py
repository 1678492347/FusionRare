import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class PatientDataset(Dataset):
    def __init__(self, data_split, json_data, hpo_vocab, disease_vocab):
        self.data = []

        for case in tqdm(json_data, desc=f"Loading {data_split}_data"):
            hpo_indices = [hpo_vocab[h] for h in case["term_id"] if h in hpo_vocab]

            self.data.append((
                torch.LongTensor(hpo_indices),
                torch.LongTensor([disease_vocab.get(case["diagnosis"])])
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    hpo_list, disease_list = zip(*batch)
    lengths = torch.tensor([len(h) for h in hpo_list])
    hpo_padded = pad_sequence(hpo_list, batch_first=True, padding_value=-1)
    mask = (hpo_padded != -1).float()
    return hpo_padded, torch.stack(disease_list), mask, lengths


class SparsePatientDataset(Dataset):
    def __init__(self, mixed_base, mixed_anc, labels):
        self.mixed_base = mixed_base.to_sparse().coalesce() if not mixed_base.is_sparse else mixed_base.coalesce()
        self.mixed_anc = mixed_anc.to_sparse().coalesce() if not mixed_anc.is_sparse else mixed_anc.coalesce()
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return idx


def sparse_collate_fn(batch_indices, dataset):
    indices = torch.as_tensor(batch_indices, dtype=torch.long, device='cpu')

    x1_batch = dataset.mixed_base.index_select(0, indices).to_dense()
    x2_batch = dataset.mixed_anc.index_select(0, indices).to_dense()
    y_batch = dataset.labels[indices]

    return x1_batch, x2_batch, y_batch

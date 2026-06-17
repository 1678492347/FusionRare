import gc

import numpy as np
import obonet
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from model.main import DiagDL, DynamicPatientSimulator
from model.dataset import PatientDataset, collate_fn, SparsePatientDataset, sparse_collate_fn
from utils.utils import *
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def load_pb_patients(model, dataset):
    pb_hpo, _, pd_disease, pb_mask, _ = list(DataLoader(
        dataset, batch_size=len(dataset),
        generator=torch.Generator(device=device), collate_fn=collate_fn))[0]
    pb_multi_hot_hpos = model.multi_hot_hpos(pb_hpo, pb_mask)
    pb_multi_hot_hpo_with_anc = model.propagate_vector(pb_multi_hot_hpos)
    pd_disease_label = torch.zeros(pd_disease.size(0), model.num_diseases, device=device)
    pd_disease_label.scatter_(1, pd_disease.to(device), 1.0)
    return pb_multi_hot_hpos.to_sparse().cpu(), pb_multi_hot_hpo_with_anc.to_sparse().cpu(), pd_disease_label.cpu()


def train_model(model, epochs=10):
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_cls = nn.CrossEntropyLoss()

    cumulative_recall = val(model)
    best_recall = sum(cumulative_recall)

    pb_base, pb_anc, pb_labels = load_pb_patients(model, train_dataset)

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()

        mixed_base, mixed_anc, mixed_labels = patient_simulator.generate_epoch_data(num_per_disease=10)
        # train_ds = SparsePatientDataset(mixed_base, mixed_anc, mixed_labels)
        # train_ds = SparsePatientDataset(pb_base, pb_anc, pb_labels)
        train_ds = SparsePatientDataset(
            torch.cat([mixed_base, pb_base], dim=0),
            torch.cat([mixed_anc, pb_anc], dim=0),
            torch.cat([mixed_labels, pb_labels], dim=0))
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator(device=device),
            collate_fn=lambda x: sparse_collate_fn(x, train_ds)
        )
        model.train()
        train_loss = 0
        for mixed_x_1, mixed_x_2, mixed_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            with autocast():
                x_1, x_2, y = mixed_x_1.to(device), mixed_x_2.to(device), mixed_y.to(device)
                disease_logits = model(x_1, x_2)
                loss_cls = criterion_cls(disease_logits, y)
                total_loss = loss_cls
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
        tqdm.write(f"Epoch {epoch + 1}: Loss={train_loss / len(train_loader):.4f}")

        cumulative_recall = val(model)
        scheduler.step()

        if sum(cumulative_recall) > best_recall:
            best_recall = sum(cumulative_recall)
            filtered_state_dict = {
                k: v.half() if v.is_floating_point() else v
                for k, v in model.state_dict().items()
                if 'specificity_prior.residual_net' in k or k == 'specificity_prior.alpha'
            }
            torch.save(filtered_state_dict, RNC_CHECKPOINT)


def val(model):
    cumulative_recall, _ = evaluate("Val: ", model, val_loader)
    return cumulative_recall


def evaluate(dataset_name, model, loader):
    model.eval()
    total_case = 0
    target_rank = []
    with torch.no_grad():
        for hpo, disease, mask, _ in loader:
            hpo, disease, mask = hpo.to(device), disease.to(device), mask.to(device)
            total_case += disease.size(0)
            disease_logits = model.test(hpo, mask)
            top_k_pred_spec = torch.topk(disease_logits, k=10, dim=-1)[1]

            cumulative_recall = []
            for k in range(1, 11):
                correct = (top_k_pred_spec[:, :k] == disease.view(-1, 1)).any(dim=1).sum().item()
                cumulative_recall.append(round(correct / total_case, 4))

            dise_matches = (top_k_pred_spec == disease)
            target_pos = dise_matches.int().argmax(dim=1, keepdim=True)
            target_pos[~dise_matches.any(dim=1, keepdim=True)] = 1000
            target_rank += (target_pos + 1).view(-1).tolist()
    tqdm.write(f"{dataset_name}Val Acc={cumulative_recall[0]:.4f}, t2_acc={cumulative_recall[1]:.4f}, t3_acc={cumulative_recall[2]:.4f}, t5_acc={cumulative_recall[4]:.4f}, t10_acc={cumulative_recall[-1]:.4f}")
    return cumulative_recall, target_rank


if __name__ == '__main__':
    """load data"""
    hpo_graph = obonet.read_obo(HP_OBO_PATH)
    dise_hpo_freq = load_disease_hpo_freq(ANNOTATION_FILE_PATH)
    propagated_freq = load_disease_hpo_freq(PROPAGATED_ANNO_PATH)

    """initialize model"""
    diag_dl = DiagDL(hpo_graph, dise_hpo_freq, propagated_freq)
    diag_dl.to(device)
    hpo_vocab = diag_dl.hpo_to_idx
    disease_vocab = diag_dl.omim_orpha_to_idx

    """load RD cases (sample)"""
    RPB_sample = load_json('data/RPB_sample.json')  # The complete RPB can be downloaded manually from data sources.
    train_data, val_data = train_test_split(
        RPB_sample,
        test_size=0.3,
        random_state=42
    )

    """analyze case quality"""
    phenotype_quality = np.column_stack(parse_pheno_quality(
        RPB_sample,
        dise_hpo_freq,
        hpo_graph))
    kde_model = KernelDensity(kernel='gaussian')
    kde_model.fit(phenotype_quality[:, 2:])

    """patient simulator"""
    patient_simulator = DynamicPatientSimulator(diag_dl, kde_model, dise_hpo_freq)

    """train"""
    train_dataset = PatientDataset('train', train_data, hpo_vocab, disease_vocab)
    val_dataset = PatientDataset('val', val_data, hpo_vocab, disease_vocab)
    val_loader = DataLoader(
        val_dataset, batch_size=len(val_dataset),
        generator=torch.Generator(device=device), collate_fn=collate_fn)
    train_model(diag_dl, epochs=10)

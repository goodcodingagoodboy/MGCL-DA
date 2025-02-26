from __future__ import print_function
import warnings
import math
import torch
from dataset_prep import Dataset_MMD
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from net.model import Model
from Loss import custom_loss_function
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os

# Define global hyperparameters
L2_DECAY = 1e-4
BATCH_SIZE = 15
TRAIN_EPOCHS = 200
LEARNING_RATE = 0.001
DROP = 0.5
EPOCHS_DROP = 20.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def step_decay(epoch, learning_rate, drop, epochs_drop):
    initial_lrate = learning_rate
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train_ddcnet(epoch, model, learning_rate, source_loader, drop, epochs_drop, loss_param, alpha=0.5, beta=0.3):
    """
    Training function
    Arguments:
        epoch: current training epoch
        model: model to train
        learning_rate: learning rate for the optimizer
        source_loader: training data loader
        drop: learning rate decay factor
        epochs_drop: number of epochs for learning rate decay
        loss_param: overall weight factor for the loss function
        alpha: weight factor for similarity between original and self_aware features
        beta: weight factor for similarity between cross_sample and self_aware features
    Returns:
        Training loss and various evaluation metrics
    """
    log_interval = 1

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=L2_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=TRAIN_EPOCHS // 3,
        T_mult=2,
        eta_min=1e-6
    )

    max_grad_norm = 1.0

    model.train()
    train_loss = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    tr_auc_y_gt, tr_auc_y_pred = [], []
    saved_group_outputs = None  # Used to save the output of the last batch

    len_dataloader = len(source_loader)
    for step, source_sample_batch in enumerate(source_loader):
        optimizer.zero_grad(set_to_none=True)

        source_data_batch = source_sample_batch['data'].to(device).float()
        source_label_batch = source_sample_batch['label'].to(device).float()

        with torch.cuda.amp.autocast(enabled=True):
            avg_output, group_outputs = model(source_data_batch)

            if step == len_dataloader - 1:  # Save the output of the last batch
                saved_group_outputs = [output.detach().clone() for output in group_outputs]

            loss = custom_loss_function(
                avg_output,
                group_outputs,
                source_label_batch,
                loss_param,
                alpha,
                beta
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch + step / len_dataloader)

        train_loss += loss.item()

        # Calculate accuracy
        source_pred_cpu = (torch.sigmoid(avg_output[:, 0]).detach().cpu().numpy() > 0.5).astype(int)
        source_labels_cpu = source_label_batch.cpu().numpy()

        TP += np.sum(np.logical_and(source_pred_cpu == 1, source_labels_cpu == 1))
        TN += np.sum(np.logical_and(source_pred_cpu == 0, source_labels_cpu == 0))
        FP += np.sum(np.logical_and(source_pred_cpu == 1, source_labels_cpu == 0))
        FN += np.sum(np.logical_and(source_pred_cpu == 0, source_labels_cpu == 1))

        tr_auc_y_gt.extend(source_labels_cpu)
        tr_auc_y_pred.extend(torch.sigmoid(avg_output[:, 0]).detach().cpu().numpy())

    # Compute the overall accuracy for the epoch
    train_acc = (TP + TN) / (TP + FP + TN + FN)
    train_AUC = roc_auc_score(tr_auc_y_gt, tr_auc_y_pred)
    train_precision = precision_score(tr_auc_y_gt, (np.array(tr_auc_y_pred) > 0.5).astype(int), zero_division=1)
    train_recall = recall_score(tr_auc_y_gt, (np.array(tr_auc_y_pred) > 0.5).astype(int), zero_division=1)
    train_f1 = f1_score(tr_auc_y_gt, (np.array(tr_auc_y_pred) > 0.5).astype(int), zero_division=1)

    # Update dynamically after each epoch
    if hasattr(model, 'dynamic_updater') and saved_group_outputs is not None:
        model.dynamic_updater.update_omega(
            accuracy=train_acc
        )
        # Update views and save them to the model
        with torch.no_grad():  # Ensure no computation graph is created
            updated_SA, updated_CA = model.dynamic_updater.update_views(
                saved_group_outputs[2],
                saved_group_outputs[0]
            )
            # Save the updated views to the model
            model.current_SA = updated_SA
            model.current_CA = updated_CA

    return train_loss / len_dataloader, train_acc, train_precision, train_recall, train_f1


def test_ddcnet(model, target_loader, loss_param, alpha=0.5, beta=0.3):
    model.eval()
    test_loss = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    te_auc_y_gt, te_auc_y_pred = [], []

    with torch.no_grad():
        for test_sample_batch in target_loader:
            test_data_batch = test_sample_batch['data'].to(device).float()
            test_label_batch = test_sample_batch['label'].to(device).float()

            avg_output, group_outputs = model(test_data_batch)

            loss = custom_loss_function(
                avg_output,
                group_outputs,
                test_label_batch,
                loss_param,
                alpha,
                beta
            )
            test_loss += loss.item()

            test_pred_cpu = (torch.sigmoid(avg_output[:, 0]).detach().cpu().numpy() > 0.5).astype(int)
            test_labels_cpu = test_label_batch.cpu().numpy()

            TP += np.sum(np.logical_and(test_pred_cpu == 1, test_labels_cpu == 1))
            TN += np.sum(np.logical_and(test_pred_cpu == 0, test_labels_cpu == 0))
            FP += np.sum(np.logical_and(test_pred_cpu == 1, test_labels_cpu == 0))
            FN += np.sum(np.logical_and(test_pred_cpu == 0, test_labels_cpu == 1))

            te_auc_y_gt.extend(test_labels_cpu)
            te_auc_y_pred.extend(torch.sigmoid(avg_output[:, 0]).detach().cpu().numpy())

    test_acc = (TP + TN) / (TP + FP + TN + FN)
    test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_pred)
    test_precision = precision_score(te_auc_y_gt, (np.array(te_auc_y_pred) > 0.5).astype(int), zero_division=1)
    test_recall = recall_score(te_auc_y_gt, (np.array(te_auc_y_pred) > 0.5).astype(int), zero_division=1)
    test_f1 = f1_score(te_auc_y_gt, (np.array(te_auc_y_pred) > 0.5).astype(int), zero_division=1)

    return {
        'test_loss': test_loss / len(target_loader),
        'test_acc': test_acc,
        'test_AUC': test_AUC,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }


def k_fold_cross_validation(k, data_path, loss_param, alpha=0.5, beta=0.3):
    results = []

    for fold in range(k):
        print(f'Fold {fold + 1}/{k} starts...')
        current_data_path = os.path.join(data_path, f'dataset_{fold + 1}')

        train_data = np.load(os.path.join(current_data_path, 'train_data.npy'))
        train_labels = np.load(os.path.join(current_data_path, 'train_lbl.npy'))
        test_data = np.load(os.path.join(current_data_path, 'test_data.npy'))
        test_labels = np.load(os.path.join(current_data_path, 'test_lbl.npy'))

        train_dataset = Dataset_MMD(train_data, train_labels)
        test_dataset = Dataset_MMD(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = Model(in_channels=1, num_class=1, edge_importance_weighting=True, root_path=current_data_path)
        model.to(device)

        best_acc = 0
        best_auc = 0

        for epoch in range(1, TRAIN_EPOCHS + 1):
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_ddcnet(
                epoch, model, LEARNING_RATE, train_loader, DROP, EPOCHS_DROP, loss_param, alpha, beta
            )
            test_results = test_ddcnet(model, test_loader, loss_param, alpha, beta)

            if (test_results['test_acc'] > best_acc) or (
                    test_results['test_acc'] == best_acc and test_results['test_AUC'] > best_auc):
                best_acc = test_results['test_acc']
                best_auc = test_results['test_AUC']
                torch.save(model.state_dict(), f'{current_data_path}/best_model_fold{fold + 1}.pth')
                print(f"Best model saved, accuracy: {best_acc}, AUC: {best_auc}")

            # Reset dynamic updater's state after each epoch (if present)
            if hasattr(model, 'dynamic_updater'):
                model.dynamic_updater.reset()

        results.append({
            'fold': fold + 1,
            'best_acc': best_acc,
            'best_auc': best_auc,
            'test_precision': test_results['test_precision'],
            'test_recall': test_results['test_recall'],
            'test_f1': test_results['test_f1']
        })

    df_results = pd.DataFrame(results)
    summary = df_results.describe().loc[['mean', 'std']].to_dict()

    return results, summary


def run_multiple_k_fold(k, times, data_path, save_path, excel_name, loss_param=0.1, alpha=0.5, beta=0.3):
    """
    Run multiple times of k-fold cross-validation
    Arguments:
        k: number of folds
        times: number of times to repeat the experiment
        data_path: path to the data
        save_path: path to save results
        excel_name: name of the Excel file
        loss_param: overall weight factor for the loss function
        alpha: weight factor for similarity between original and self_aware features
        beta: weight factor for similarity between cross_sample and self_aware features
    """
    all_results = []
    all_summaries = []

    for i in range(times):
        print(f"Run {i + 1} of {k}-fold cross-validation starts...")
        results, summary = k_fold_cross_validation(
            k=k,
            data_path=data_path,
            loss_param=loss_param,
            alpha=alpha,
            beta=beta
        )
        all_results.append(results)
        all_summaries.append(summary)

    os.makedirs(save_path, exist_ok=True)
    excel_path = os.path.join(save_path, excel_name)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        all_results_df = pd.concat([pd.DataFrame(res) for res in all_results])
        all_results_df.to_excel(writer, sheet_name='Sheet1', index=False)

        all_summaries_df = pd.concat([pd.DataFrame(sum_dict) for sum_dict in all_summaries])
        all_summaries_df.to_excel(writer, sheet_name='Sheet2', index=False)

    print(f"All experimental results and statistical data have been saved to the Excel file '{excel_name}'")


# Calling the main function:
if __name__ == '__main__':
    ROOT_PATH = '../mdd_data/'
    SAVE_PATH = '../save/'
    EXCEL_NAME = 'fold_results.xlsx'
    run_multiple_k_fold(k=5, times=5, data_path=ROOT_PATH, save_path=SAVE_PATH, excel_name=EXCEL_NAME)
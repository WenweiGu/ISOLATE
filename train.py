import pickle
import torch
from CVAE import loss_function, loss_function_positive
import torch.optim as optim
from model import RTAnomaly
from dataloader import load_dataset, get_dataloaders, get_positive_dataloaders
from data_preprocess import normalize, generate_windows, minmax_score
import logging
from tqdm import tqdm
from evaluate import get_anomaly_score
import numpy as np
from evaluate import compute_prediction, compute_binary_metrics

params = {
    'data_root': "./datasets/HW",
    'train_postfix': "train.pkl",
    'test_postfix': "test.pkl",
    'test_label_postfix': "test_label.pkl",
    'train_label_postfix': "train_label.pkl",
    'dim': 38,
    'entity': ['37f4ceba-f840-4c08-a488-676bce922fcf'],
    'valid_ratio': 0,
    'normalize': "minmax",
    'window_size': 20,
    'stride': 1,
    'batch_size': 32,
    'num_workers': 0,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'gnn_dim': 128,
    'pooling_ratio': 0.5,
    'threshold': 0.5,
    'dropout': 0.5,
    'filters': [256, 256, 256],
    'kernels': [8, 5, 3],
    'dilation': [1, 2, 4],
    'layers': [50, 10],
    'gru_dim': 128,
    'epoch': 1,
    'lr': 1e-4,
    'wd': 1e-3,
    'recon_filter': 5,
    'hidden_size': 100,
    'latent_size': 10,
    'cof': 0.5
}


def get_positive_label(model, item, threshold=0.9):
    model.train()

    data_dict_pu = load_dataset(
        data_root=params["data_root"],
        entities=params["entity"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
    )

    data_dict_pu = normalize(data_dict_pu, method=params["normalize"])

    windows_pu = generate_windows(
        data_dict_pu,
        window_size=params["window_size"],
        stride=1  # 确保每个点都有标签
    )

    train_window_pu = windows_pu[item]['train_windows']

    loader_train, _, loader_test = get_dataloaders(
        train_window_pu,
        train_window_pu,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]
    )

    for _ in range(params['epoch']):
        loss_pu = 0
        for n, x_pu in enumerate(tqdm(loader_train)):
            if x_pu.shape[0] == 1:
                continue

            x_pu = x_pu.to(params['device'])  # 先放GPU上
            x_pu = x_pu.permute(0, 2, 1)

            label = torch.zeros((x_pu.shape[0], 1)).to(params['device'])

            optimizer.zero_grad()
            x_recon_pu, recon_embed_pu, embed_pu, mu_pu, log_var_pu, _ = model(x_pu, label)

            # loss 部分可以加入别的部分, 有一定作用
            loss_train_pu = loss_function(x_pu, x_recon_pu, recon_embed_pu, embed_pu, mu_pu, log_var_pu,
                                          cof=params['cof'])
            loss_pu += loss_train_pu

            loss_train_pu.backward()
            optimizer.step()

    model.eval()
    score_pu, _, _ = get_anomaly_score(loader_test, encoder, params['device'], 1)
    score_pu = np.array(minmax_score(score_pu))

    train_label = np.zeros((score_pu.shape[0] + params['window_size'], 1))
    train_label[np.where(score_pu > threshold)] = 1

    pickle.dump(train_label, open(str(params['data_root']) + '/' + item + '_train_label.pkl', 'wb'))


for entity in params['entity']:
    logging.info("Fitting dataset: {}".format(entity))

    train_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entity"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"]
    )

    window = generate_windows(
        train_dict,
        window_size=params["window_size"],
        stride=params["stride"],
        positive_label=False
    )

    dim = window[entity]['train_windows'].shape[-1]

    encoder = RTAnomaly(
        ndim=dim,
        len_window=params['window_size'],
        gnn_dim=params['gnn_dim'],
        pooling_ratio=params['pooling_ratio'],
        threshold=params['threshold'],
        dropout=params['dropout'],
        filters=params['filters'],
        kernels=params['kernels'],
        dilation=params['dilation'],
        layers=params['layers'],
        gru_dim=params['gru_dim'],
        device=params['device'],
        recon_filter=params['recon_filter'],
        hidden_size=params['hidden_size'],
        latent_size=params['latent_size']
    )

    encoder.to(params['device'])

    optimizer = optim.Adam(encoder.parameters(),
                           lr=params['lr'], weight_decay=params['wd'])

    get_positive_label(encoder, entity)

    # reload dataset
    train_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entity"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        train_label_postfix=params["train_label_postfix"]
    )

    train_dict = normalize(train_dict, method=params["normalize"])

    window = generate_windows(
        train_dict,
        window_size=params["window_size"],
        stride=params["stride"],
        positive_label=True
    )

    train_windows = window[entity]['train_windows']
    test_windows = window[entity]['test_windows']
    test_labels = window[entity]['test_label'][:, -1].reshape(-1, 1)
    train_labels = window[entity]['train_label'][:, -1].reshape(-1, 1)

    train_loader, _, test_loader = get_positive_dataloaders(
        train_windows,
        train_labels,
        test_windows,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]
    )

    encoder.train()
    for epoch in range(params['epoch']):
        loss = 0
        for i, (x, y) in enumerate(tqdm(train_loader)):
            if x.shape[0] == 1:
                continue

            x = x.to(params['device'])  # 先放GPU上
            x = x.permute(0, 2, 1)
            y = y.to(params['device'])

            optimizer.zero_grad()
            x_recon, recon_embed, embed, mu, log_var, _ = encoder(x, y)

            # loss 部分可以加入别的部分, 有一定作用
            loss_train = loss_function_positive(x, x_recon, recon_embed, embed, mu, log_var, y,
                                                cof=params['cof'])
            loss += loss_train

            loss_train.backward()
            optimizer.step()

        loss /= train_loader.__len__()
        print(f'Training loss for epoch {epoch} is: {float(loss)}')
        torch.save(encoder.state_dict(), './save/checkpoint_' + entity + '.pth')

    logging.info("Finish dataset: {}".format(entity))
    encoder.load_state_dict(torch.load('./save/checkpoint_' + entity + '.pth'))
    encoder.eval()

    score, _, score_metrics = get_anomaly_score(test_loader, encoder, params['device'], 1)
    score = minmax_score(score)

    test_labels = test_labels.flatten()

    pred, pred_adjust, _ = compute_prediction(score, test_labels).values()
    f1, pre, re = compute_binary_metrics(pred_adjust, test_labels).values()

    print(f'Results for {entity}:' + str(compute_binary_metrics(pred_adjust, test_labels)))

import pickle
import torch
from CVAE import loss_function, loss_function_positive
import torch.optim as optim
from model import RTAnomaly
from dataloader import load_dataset, get_dataloaders, get_positive_dataloaders
from data_preprocess import normalize, generate_windows, minmax_score
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from evaluate import get_anomaly_score
import numpy as np

params = {
    'data_root': "./datasets/HW",
    'train_postfix': "train.pkl",
    'test_postfix': "test.pkl",
    'test_label_postfix': "test_label.pkl",
    'train_label_postfix': "train_label.pkl",
    'positive': True,
    'dim': 38,
    'entity': ['37f4ceba-f840-4c08-a488-676bce922fcf'],
    'valid_ratio': 0,
    'normalize': "minmax",
    'window_size': 20,
    'stride': 10,
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
    'epoch': 50,
    'lr': 1e-4,
    'wd': 1e-3,
    'recon_filter': 5,
    'hidden_size': 100,
    'latent_size': 10,
    'cof': 0.5
}


def get_positive_label(model, item, threshold=0.9):
    model.train()

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entity"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
    )

    data_dict = normalize(data_dict, method=params["normalize"])

    windows = generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=1  # 确保每个点都有标签
    )

    train_window = windows[item]['train_windows']

    loader_train, _, loader_test = get_dataloaders(
        train_window,
        train_window,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]
    )

    for epoch in range(params['epoch']):
        loss = 0
        for n, x in enumerate(tqdm(loader_train)):
            if x.shape[0] == 1:
                continue

            x = x.to(params['device'])  # 先放GPU上
            x = x.permute(0, 2, 1)

            label = torch.zeros((x.shape[0], 1)).to(params['device'])

            optimizer.zero_grad()
            x_recon, recon_embed, embed, mu, log_var, _ = model(x, label)

            # loss 部分可以加入别的部分, 有一定作用
            loss_train = loss_function(x, x_recon, recon_embed, embed, mu, log_var, cof=params['cof'])
            loss += loss_train

            loss_train.backward()
            optimizer.step()

    model.eval()
    score, _, _ = get_anomaly_score(loader_test, encoder, params['device'], 1)
    score = np.array(minmax_score(score))

    plt.plot(score)
    plt.axhline(threshold, color='r')

    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'./PU_{entity}.jpg', bbox_inches='tight', dpi=600)
    plt.close()

    train_label = np.zeros((score.shape[0] + params['window_size'], 1))
    train_label[np.where(score > threshold)] = 1

    pickle.dump(train_label, open(str(params['data_root']) + '/' + item + '_train_label.pkl', 'wb'))


def anomaly_segment(label):
    # find the change point
    tag = [0 for j in range(len(label))]
    for i in range(1, len(label)):
        if label[i] > label[i - 1]:
            tag[i] = 1
        elif label[i] < label[i - 1]:
            tag[i] = -1

    # flag for change
    flag = 0
    start = []
    end = []

    for i in range(0, len(label)):
        if flag == 0:
            if tag[i] == 1:
                start.append(i)
                flag = 1  # go into anomaly pattern
        if flag == 1:
            if tag[i] == -1:
                end.append(i)
                flag = 0  # go out anomaly pattern

    if len(start) != len(end):
        end.append(len(label))
    anomaly_segment = [(start[i], end[i]) for i in range(len(start))]

    return anomaly_segment


for entity in params['entity']:
    logging.info("Fitting dataset: {}".format(entity))

    avg1 = None
    avg2 = None

    if params['positive']:
        train = True
        test = True
        get_positive = True

        t_dict = load_dataset(
            data_root=params["data_root"],
            entities=params["entity"],
            dim=params["dim"],
            valid_ratio=params["valid_ratio"],
            test_label_postfix=params["test_label_postfix"],
            test_postfix=params["test_postfix"],
            train_postfix=params["train_postfix"]
        )

        w = generate_windows(
            t_dict,
            window_size=params["window_size"],
            stride=params["stride"],
            positive_label=False
        )

        dim = w[entity]['train_windows'].shape[-1]

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

        if get_positive:
            get_positive_label(encoder, entity, threshold=0.9)

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

        if train:
            A_n = []
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
                    x_recon, recon_embed, embed, mu, log_var, a_n = encoder(x, y)

                    A_n.extend(a_n)

                    # loss 部分可以加入别的部分, 有一定作用
                    loss_train = loss_function_positive(x, x_recon, recon_embed, embed, mu, log_var, y,
                                                        cof=params['cof'])
                    loss += loss_train

                    loss_train.backward()
                    optimizer.step()

                loss /= train_loader.__len__()
                print(f'Training loss for epoch {epoch} is: {float(loss)}')

                torch.save(encoder.state_dict(), './save/checkpoint_' + entity + '.pth')

            A_n = np.array([n.cpu().detach().numpy() for n in A_n])

            avg1 = np.mean(np.mean(np.array(A_n), axis=0), axis=0)

        if test:
            logging.info("Finish dataset: {}".format(entity))
            encoder.load_state_dict(torch.load('./save/checkpoint_' + entity + '.pth'))
            encoder.eval()

            score, A, score_metrics = get_anomaly_score(test_loader, encoder, params['device'], 1)
            score = minmax_score(score)
            np.save(f'./score/score_{entity}.npy', score)

            A = np.array(A)

            eval_score = np.load(f'./score/score_{entity}.npy')
            test_labels = test_labels.flatten()

            segments = anomaly_segment(test_labels)

            average = []

            avg_anomaly = np.mean(np.array(score_metrics), axis=0)

            for seg in segments:
                average = np.mean(np.mean(np.array(A[seg[0]:seg[1], :, :]), axis=0), axis=0)

            avg2 = np.array(average)

            dif = np.abs(avg1 - avg2)

            idx_anomaly = np.argsort(-avg_anomaly)
            rank_anomaly = np.argsort(idx_anomaly)

            np.save(f'./RCA_{entity}.txt', rank_anomaly)
            print(f'Metric localization ranking is: {rank_anomaly}')

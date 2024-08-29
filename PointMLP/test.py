"""
python test.py --model pointMLP --msg 20220209053148-404
"""
import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import progress_bar, IOStream
from data import NifosTree
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, linewidths=.5, square=True, cmap=cmap,
                xticklabels=classes, yticklabels=classes, cbar=False)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--model', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=4, type=int, help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    assert args.checkpoint is not None, 'checkpoint directory is not given'

    print('==> Preparing data..')
    test_loader = DataLoader(NifosTree(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    test_out = validate(net, test_loader, criterion, device, args.num_classes)
    print(f"Vanilla out: {test_out}")


def validate(net, testloader, criterion, device, num_classes):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    np.savetxt("test_data_pred.csv", np.vstack(test_pred), delimiter=',')
    np.savetxt("test_data_label.csv", np.vstack(test_true), delimiter=',')
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    # After the loop where test_pred and test_label are gathered
    # Compute confusion matrix
    print("=====================================")
    print(test_pred)
    # save as csv
    np.savetxt("test_pred.csv", test_pred, delimiter=",")
    print("=====================================")
    print(test_true)
    # save as csv
    np.savetxt("test_label.csv", test_true, delimiter=",")

    # Confusion matrix를 계산합니다.
    cm = confusion_matrix(test_true, test_pred)
    class_names = ['Densi', 'Koraiensis', 'Larix', 'Obtusa']
    # class_names = [f'class_{i}' for i in range(num_classes)]  # 클래스 이름 목록을 생성합니다.
    plot_confusion_matrix(cm, class_names, title='Confusion Matrix')

    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()

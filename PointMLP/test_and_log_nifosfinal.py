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
from data_nifosfinal import NifosTreeFinal
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import wandb

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.savefig(f'{title}.png')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--model', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    return parser.parse_args()

def test_net(args):
    wandb.init(project="LiDCLS", config=args)
    wandb.run.name = f"test-{args.model}-{args.num_points}-{args.batch_size}"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")

    print('==> Preparing data..')
    valid_loader = DataLoader(NifosTreeFinal(partition='valid', num_points=args.num_points), num_workers=4,
                              batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(NifosTreeFinal(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    print('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    valid_out = validate(net, valid_loader, criterion, device, args.num_classes, 'Validation')
    print(f"Validation results: {valid_out}")

    test_out = validate(net, test_loader, criterion, device, args.num_classes, 'Test')
    print(f"Test results: {test_out}")

    # Log results to wandb
    for split, results in [('valid', valid_out), ('test', test_out)]:
        wandb.log({
            f"{split}_loss": results['loss'],
            f"{split}_accuracy": results['acc'],
            f"{split}_balanced_accuracy": results['acc_avg'],
            f"{split}_f1_score": results['f1'],
            f"{split}_precision": results['precision'],
            f"{split}_recall": results['recall'],
            f"{split}_time": results['time'],
            "num_points": args.num_points,
            "batch_size": args.batch_size,
            f"{split}_predictions": results['pred'],
            f"{split}_true_labels": results['true'],
        })

    wandb.finish()

def validate(net, dataloader, criterion, device, num_classes, split):
    net.eval()
    loss = 0
    true = []
    pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss += criterion(logits, label).item()
            preds = logits.max(dim=1)[1]
            true.append(label.cpu().numpy())
            pred.append(preds.detach().cpu().numpy())
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f'
                         % (loss / (batch_idx + 1)))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    true = np.concatenate(true)
    pred = np.concatenate(pred)

    # Compute metrics
    acc = 100. * metrics.accuracy_score(true, pred)
    acc_avg = 100. * metrics.balanced_accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')
    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')

    # Compute and plot confusion matrix
    cm = confusion_matrix(true, pred)
    class_names = ['Densi', 'Koraiensis', 'Larix', 'obtusa']
    plot_confusion_matrix(cm, class_names, title=f'{split} Confusion Matrix')

    return {
        "loss": float("%.6f" % (loss / len(dataloader))),
        "acc": float("%.6f" % acc),
        "acc_avg": float("%.6f" % acc_avg),
        "f1": float("%.6f" % f1),
        "precision": float("%.6f" % precision),
        "recall": float("%.6f" % recall),
        "time": time_cost,
        "pred": pred.tolist(),
        "true": true.tolist()
    }

if __name__ == '__main__':
    args = parse_args()
    test_net(args)

# python test_and_log_nifosfinal.py --model pointMLP --num_points 1024 --batch_size 32 --checkpoint checkpoints/pointmlp-20240919132617-1
# python test_and_log_nifosfinal.py --model pointMLP --num_points 2048 --batch_size 32 --checkpoint checkpoints/pointmlp-20240919151605-1
# python test_and_log_nifosfinal.py --model pointMLP --num_points 2048 --batch_size 16 --checkpoint checkpoints/pointmlp-20240919105057-1
# python test_and_log_nifosfinal.py --model pointMLP --num_points 1024 --batch_size 16 --checkpoint checkpoints/pointmlp-20240919075324-1
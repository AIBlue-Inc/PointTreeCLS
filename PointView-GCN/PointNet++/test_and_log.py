import argparse
import numpy as np
import os
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from dataloader import SinglePoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser('PointNet2 Testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--model', default='POINTNET_MSG_CLS', help='model name')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--worker', type=int, default=4, help='number of workers for dataloader')
    return parser.parse_args()


def test(model, loader, criterion, num_class=4, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    all_preds = []
    all_targets = []
    total_loss = 0.0

    for points, target in tqdm(loader, total=len(loader), desc="Testing"):
        points = points.transpose(2, 1)
        points, target = points.float().to(device), target.to(device)
        classifier = model.eval()
        vote_pool = torch.zeros(target.shape[0], num_class).to(device)

        for _ in range(vote_num):
            pred, trans_feat = classifier(points)
            vote_pool += pred
            loss = criterion(pred, target, trans_feat)
            total_loss += loss.item()

        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        all_preds.extend(pred_choice.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        for cat in torch.unique(target):
            classacc = pred_choice[target == cat].eq(target[target == cat]).cpu().sum()
            class_acc[cat, 0] += classacc.item() / points[target == cat].size()[0]
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = np.nan_to_num(class_acc[:, 0] / class_acc[:, 1])
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    avg_loss = total_loss / len(loader)

    return instance_acc, class_acc, all_preds, all_targets, avg_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # Set up logging
    experiment_dir = Path(args.log_dir)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    log_dir = experiment_dir.joinpath('logs/')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)

    # Initialize W&B
    wandb.init(project='LiDCLS', config=vars(args))
    wandb.run.name = f"test-{args.model}-{args.num_point}-{args.batch_size}"

    # Load dataset
    log_string('Load dataset ...')
    valid_path = f'../../data/NIFOS_TREE_240827_{args.num_point}_xyz/*/valid'
    test_path = f'../../data/NIFOS_TREE_240827_{args.num_point}_xyz/*/test'

    VALID_DATASET = SinglePoint(valid_path, npoint=args.num_point)
    TEST_DATASET = SinglePoint(test_path, npoint=args.num_point)

    validDataLoader = DataLoader(VALID_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Load model
    num_class = 4
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_class).to(device)
    criterion = MODEL.get_loss().to(device)

    # Load the best model
    checkpoint = torch.load(str(checkpoints_dir / 'best_model.pth'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Test the model
    with torch.no_grad():
        # Validation set
        valid_instance_acc, valid_class_acc, valid_preds, valid_targets, valid_loss = test(classifier.eval(), validDataLoader, criterion,
                                                                               num_class=num_class,
                                                                               vote_num=args.num_votes)

        # Test set
        test_instance_acc, test_class_acc, test_preds, test_targets, test_loss = test(classifier.eval(), testDataLoader, criterion,
                                                                           num_class=num_class, vote_num=args.num_votes)

    # Calculate additional metrics
    valid_f1 = f1_score(valid_targets, valid_preds, average='weighted')
    valid_precision = precision_score(valid_targets, valid_preds, average='weighted')
    valid_recall = recall_score(valid_targets, valid_preds, average='weighted')
    valid_balanced_acc = balanced_accuracy_score(valid_targets, valid_preds)

    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    test_precision = precision_score(test_targets, test_preds, average='weighted')
    test_recall = recall_score(test_targets, test_preds, average='weighted')
    test_balanced_acc = balanced_accuracy_score(test_targets, test_preds)

    # Log results
    log_string('Valid Instance Accuracy: %f, Class Accuracy: %f' % (valid_instance_acc, valid_class_acc))
    log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (test_instance_acc, test_class_acc))

    # Log to W&B
    wandb.log({
        "valid_acc": valid_instance_acc,
        "valid_loss": valid_loss,
        "valid_class_acc": valid_class_acc,
        "valid_f1": valid_f1,
        "valid_precision": valid_precision,
        "valid_recall": valid_recall,
        "valid_balanced_acc": valid_balanced_acc,
        "test_acc": test_instance_acc,
        "test_loss": test_loss,
        "test_class_acc": test_class_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_balanced_acc": test_balanced_acc,
    })

    # Save predictions and true labels
    wandb.log({
        "valid_predictions": valid_preds,
        "valid_true_labels": valid_targets,
        "test_predictions": test_preds,
        "test_true_labels": test_targets,
    })

    wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)

# python test_and_log.py --model POINTNET_MSG_CLS --num_point 1024 --batch_size 32 --log_dir log/classification/2024-09-03_11-39 --num_votes 3
# python test_and_log.py --model POINTNET_MSG_CLS --num_point 2048 --batch_size 32 --log_dir log/classification/2024-09-03_11-41 --num_votes 3
# python test_and_log.py --model POINTNET_MSG_CLS --num_point 2048 --batch_size 16 --log_dir log/classification/2024-09-04_01-39 --num_votes 3
# python test_and_log.py --model POINTNET_MSG_CLS --num_point 1024 --batch_size 16 --log_dir log/classification/2024-09-04_05-45 --num_votes 3
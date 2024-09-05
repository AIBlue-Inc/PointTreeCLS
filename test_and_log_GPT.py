import argparse
import torch
import wandb
from tools import builder
from utils import misc, dist_utils
import numpy as np
from torchvision import transforms
from pointnet2_ops import pointnet2_utils
from datasets import data_transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from utils.config import get_config


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--ckpts', type=str, help='path to checkpoint file')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--local_rank', default=None)
    return parser.parse_args()


def test_transforms(points):
    return data_transforms.PointcloudScaleAndTranslate()(points)


def evaluate(base_model, dataloader, criterion, args, split):
    base_model.eval()
    test_pred = []
    test_label = []
    npoints = config.npoints
    total_loss = 0.0

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits, loss1 = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        loss = base_model.get_loss_acc(logits, label)[0]

    preds = np.array(test_pred.cpu())
    labels = np.array(test_label.cpu())

    # Calculate metrics
    acc = (preds == labels).sum() / float(labels.size) * 100.
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    print(f'{split} Accuracy: {acc:.6f}%')
    print(f'{split} Loss: {avg_loss:.6f}')
    print(f'{split} F1 Score: {f1:.6f}')
    print(f'{split} Precision: {precision:.6f}')
    print(f'{split} Recall: {recall:.6f}')

    # Log results to wandb
    wandb.log({
        f"{split}_accuracy": acc,
        f"{split}_loss": avg_loss,
        f"{split}_f1_score": f1,
        f"{split}_precision": precision,
        f"{split}_recall": recall,
        f"{split}_predictions": preds.tolist(),
        f"{split}_true_labels": labels.tolist(),
    })

    return acc, avg_loss, f1, precision, recall


def test_net(args, config):
    model_name = "Point"
    wandb.init(project="LiDCLS", config=config)
    wandb.run.name = f"test-{model_name}-{args.num_points}-{args.batch_size}"

    _, valid_dataloader = builder.dataset_builder(args, config.dataset.val)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)

    if args.gpu:
        base_model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating on validation set:")
    evaluate(base_model, valid_dataloader, criterion, args, "valid")

    print("\nEvaluating on test set:")
    evaluate(base_model, test_dataloader, criterion, args, "test")

    wandb.log({
        "num_points": args.num_points,
        "batch_size": args.batch_size,
    })

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    args.distributed = False
    args.num_workers = 4
    config = get_config(args)
    test_net(args, config)

# cd tools
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_1k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_1k_bs16/ModelNet_models/PointBERT-1024-16-1/ckpt-best.pth --num_points=1024 --batch_size=16
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_1k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_1k_bs32/ModelNet_models/PointBERT-1024-32-1/ckpt-best.pth --num_points=1024 --batch_size=32
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_2k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_2k_bs16/ModelNet_models/PointBERT-2048-16-1/ckpt-best.pth --num_points=2048 --batch_size=16
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_2k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_2k_bs32/ModelNet_models/PointBERT-2048-32-1/ckpt-best.pth --num_points=2048 --batch_size=32

# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_1k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_1k_bs16/PointGPT-S/PointGPT-1024-16-1/ckpt-best.pth --num_points=1024 --batch_size=16
# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_1k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_1k_bs32/PointGPT-S/PointGPT-1024-32-1/ckpt-best.pth --num_points=1024 --batch_size=32
# python test_and_log_GPT.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_2k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_2k_bs16/PointGPT-S/PointGPT-2048-16-1/ckpt-best.pth --num_points=2048 --batch_size=16
# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_2k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_2k_bs32/PointGPT-S/PointGPT-2048-32-1/ckpt-best.pth --num_points=2048 --batch_size=32
import argparse
import torch
import wandb
from tools import builder
from utils import misc
import numpy as np
from torchvision import transforms
from pointnet2_ops import pointnet2_utils
from datasets import data_transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--ckpts', type=str, help='path to checkpoint file')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    return parser.parse_args()


def test_transforms(points):
    return data_transforms.PointcloudScaleAndTranslate()(points)


def evaluate(base_model, dataloader, criterion, args, split):
    base_model.eval()
    preds = []
    labels = []
    total_loss = 0.0

    with torch.no_grad():
        for idx, (_, _, data) in enumerate(dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, args.num_points)
            points = test_transforms(points)

            logits, _ = base_model(points)
            loss = criterion(logits, label)
            pred = logits.argmax(-1).view(-1)

            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
            total_loss += loss.item()

    preds = np.array(preds)
    labels = np.array(labels)

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

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

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
    wandb.init(project="LiDCLS", config=config)
    wandb.run.name = f"test-{args.num_points}-{args.batch_size}-{args.ckpts}"

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
    config = misc.Config.fromfile(args.config)
    test_net(args, config)

# cd tools
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_1k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_1k_bs16/ModelNet_models/PointBERT-1024-16-1/ckpt-best.pth --num_points=1024 --batch_size=16
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_1k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_1k_bs32/ModelNet_models/PointBERT-1024-32-1/ckpt-best.pth --num_points=1024 --batch_size=32
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_2k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_2k_bs16/ModelNet_models/PointBERT-2048-16-1/ckpt-best.pth --num_points=2048 --batch_size=16
# python test_and_log.py --config=cfgs/ModelNet_models/PointTransformer_NifosTree4_2k_test.yaml --ckpts=experiments/PointTransformer_NifosTree4_2k_bs32/ModelNet_models/PointBERT-2048-32-1/ckpt-best.pth --num_points=2048 --batch_size=32

# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_1k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_1k_bs16/PointGPT-S/PointGPT-1024-16-1/ckpt-best.pth --num_points=1024 --batch_size=16
# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_1k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_1k_bs32/PointGPT-S/PointGPT-1024-32-1/ckpt-best.pth --num_points=1024 --batch_size=32
# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_2k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_2k_bs16/PointGPT-S/PointGPT-2048-16-1/ckpt-best.pth --num_points=2048 --batch_size=16
# python test_and_log.py --config=cfgs/PointGPT-S/finetune_modelnet_NifosTree4_2k_test.yaml --ckpts=experiments/finetune_modelnet_NifosTree4_2k_bs32/PointGPT-S/PointGPT-2048-32-1/ckpt-best.pth --num_points=2048 --batch_size=32

# python test_and_log.py --cfgs/PointGPT-S/finetune_modelnet_NifosTree4_2k_test.yaml \
# --ckpts=experiments/finetune_modelnet_NifosTree4_2k_bs16/PointGPT-S/PointGPT-2048-16-1/ckpt-best.pth --num_points=2048 --batch_size=16
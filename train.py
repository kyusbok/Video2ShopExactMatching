import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from easydict import EasyDict as edict

from models import AsymNet
from data.deepfashion2 import DeepFashion2Dataset
from torchvision import transforms as T

def get_transform():
    transforms = []
    transforms.append(T.Resize([256, 256]))
    transforms.append(T.RandomCrop(227))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    cfg.weights = [1, cfg.n_shops // 2]
    model = AsymNet(cfg)

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

    optim_sd = None
    ep = 0
    if cfg.pretrained is not None:
        print("Loading checkpoint...")
        ckpt = torch.load(cfg.pretrained, map_location=device)
        model.load_state_dict(ckpt['model'])
        ep = ckpt['e']
        # optim_sd = ckpt['optimizer']

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    save_tensor = True
    name = f'{cfg.dataset}_TOT-E_{cfg.num_epoch}_LR_{cfg.lr}_SHOP_{cfg.n_shops}_FRAMES_{cfg.n_frames}_WEIGHTS_{cfg.weights}_TR_PHASE_{cfg.training_phase}'
    if save_tensor:
        writer = SummaryWriter(
            log_dir=os.path.join(cfg.log_dir, name))

    # Define Dataset
    if cfg.dataset == 'deepfashion2':
        from data import dataloader

        train_dataset = DeepFashion2Dataset('./dataset/train/annots.json',
                                            './dataset/train/image', get_transform())

        data_loader = dataloader.get_dataloader(train_dataset, batch_size=(cfg.n_frames + 1) * cfg.n_shops,
                                                n_frames=cfg.n_frames,
                                                n_shops=cfg.n_shops,
                                                is_parallel=torch.cuda.device_count() > 1)

    optimizer = optim.SGD(
        params=model.parameters(),
        lr=cfg.lr, momentum=0.9, weight_decay=0.0005)


    if optim_sd is not None:
        optimizer.load_state_dict(optim_sd)

    losses_match = []
    losses_snn = []
    model.train()
    for epoch in range(ep, cfg.num_epoch):
        with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{cfg.num_epoch}', unit='iter', dynamic_ncols=True) as pbar:
            for i, batch in enumerate(data_loader):
                imgs = batch[0]
                imgs = imgs.to(device)

                scores, snn_score, l_match, l_snn = model(imgs, cfg.training_phase == 1)

                loss = l_match + l_snn

                losses_match.append(l_match.detach().cpu())
                losses_snn.append(l_snn.detach().cpu())

              
                if i % cfg.log_step == 0 and save_tensor:
                    global_step = (epoch * len(data_loader)) + i
                    log_loss_match = torch.stack(losses_match).mean()
                    log_loss_snn = torch.stack(losses_snn).mean()
                    writer.add_scalar('train/Loss_match', log_loss_match, global_step)
                    writer.add_scalar('train/Loss_snn', log_loss_snn, global_step)
                    writer.add_scalar('train/Loss_total', log_loss_match + log_loss_snn, global_step)
                    pbar.write(
                        'Train Epoch_{} step_{}: loss : {},loss match : {},loss ssn : {}, max_snn : {}, min_snn : {}, max_score : {}, min_score : {}'
                            .format(epoch, i, log_loss_match + log_loss_snn, log_loss_match,
                                    log_loss_snn, snn_score.max(), snn_score.min(), scores.max(), scores.min()))
                    losses_match = []
                    losses_snn = []

                if cfg.training_phase == 0:
                    l_snn.backward()
                elif cfg.training_phase == 1:
                    l_match.backward()
                else:
                    loss.backward()

                optimizer.step()
                #
                model.zero_grad()
                pbar.update()

            os.makedirs(os.path.join(cfg.save_folder, name), exist_ok=True)
            if epoch % cfg.save_step == 0:

                save_path = os.path.join(cfg.save_folder, name, str(epoch) + '_model.pth')
                if torch.cuda.device_count() > 1:
                    torch.save({'e': epoch,
                                'model': model.module.state_dict(),
                                'optimizer': optimizer.state_dict()}, save_path)
                else:
                    torch.save({'e': epoch,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, save_path)

            save_path = os.path.join(cfg.save_folder, name, 'last.pth')
            if torch.cuda.device_count() > 1:
                torch.save({'e': epoch,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()}, save_path)
            else:
                torch.save({'e': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Video2Shop models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default="deepfashion2",
                        help="Dataset used to train the model")
    parser.add_argument('-g', '--gpus', dest='gpus', type=str, default="3", help="GPUs IDs (0,1)")
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('-fr', '--frame', dest='n_frames', type=int, default=8,
                        help="Frames for each video (Power of 2)")
    parser.add_argument('-sh', '--shops', dest='n_shops', type=int, default=16,
                        help="Number of matching ")
    parser.add_argument('-lr_st', '--learning_rate_step', dest='learning_rate_step', type=int, default=7,
                        help="Epoch save step")
    parser.add_argument('-e', '--epochs', dest='num_epoch', type=int, default=160, help="Epochs")

    parser.add_argument('-p', '--pretrained', dest='pretrained', type=str, default=None, help="Pretrained Model")
    parser.add_argument('-tph', '--training_phase', dest='training_phase', type=int, default=2,
                        help="0: SNN, 1: FN, 2: BOTH")
    parser.add_argument('-log', '--log_dir', dest='log_dir', type=str, default='log', help="Log directory")
    parser.add_argument('-step', '--log_step', dest='log_step', type=int, default=100, help="Log step")
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=10, help="Epoch save step")
    parser.add_argument('-f', '--save_folder', dest='save_folder', type=str,
                        default="checkpoints",
                        help="Checkpoints saving folder")

    args = vars(parser.parse_args())
    args = edict(args)
    train(args)

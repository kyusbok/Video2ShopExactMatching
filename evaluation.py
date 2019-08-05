import os
import argparse
import torch

from tqdm import tqdm
from easydict import EasyDict as edict

from models import AsymNet
from data.deepfashion2 import DeepFashion2Dataset
from torchvision import transforms as T
from data import dataloader

import json

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def get_transform():
    transforms = []
    transforms.append(T.Resize([227, 227]))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def compute_scores(scores):
    top1 = 0
    top5 = 0
    top10 = 0
    top15 = 0
    top20 = 0
    for i, sc in enumerate(scores):
        indx = list(torch.argsort(sc, descending=True))
        if i in indx[:20]:
            top20 += 1
            if i in indx[:15]:
                top15 += 1
                if i in indx[:10]:
                    top10 += 1
                    if i in indx[:5]:
                        top5 += 1
                        if i == indx[0]:
                            top1 += 1

    top1 = top1 / len(scores)
    top5 = top5 / len(scores)
    top10 = top10 / len(scores)
    top15 = top15 / len(scores)
    top20 = top20 / len(scores)
    return top1,top5,top10,top15,top20

def eval(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    cfg.weights = [1, 1]
    model = AsymNet(cfg)

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

    print("Loading checkpoint...")
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Define Dataset
    train_dataset = DeepFashion2Dataset('./dataset/deepfashion2/validation/annots.json',
                                        './dataset/validation/image', get_transform(), False)

    data_loader = dataloader.get_dataloader(train_dataset, batch_size=(cfg.n_frames + 1),
                                            n_frames=cfg.n_frames,
                                            n_shops=1,
                                            is_parallel=torch.cuda.device_count() > 1)

    model.eval()
    cfg.save_folder = os.path.join(cfg.save_folder, cfg.checkpoint.split('/')[-2])
    if not os.path.isfile(os.path.join(cfg.save_folder, 'shops.pth')) \
            or not os.path.isfile(os.path.join(cfg.save_folder, 'videos.pth')):

        os.makedirs(cfg.save_folder, exist_ok=True)

        dict_shops = {}
        dict_videos = {}

        with tqdm(total=len(data_loader), desc='Extracting features', unit='iter', dynamic_ncols=True) as pbar:
            for i, batch in enumerate(data_loader):
                imgs = batch[0]
                imgs = imgs.to(device)
                pair_key = batch[1][0]
                pair_key = str(pair_key[0].detach().numpy()) + '_' + str(pair_key[1].detach().numpy())
                with torch.no_grad():
                    shop_features = model.IFN(imgs[0, ...].unsqueeze(0))
                    video_features = torch.stack(model.VFN(imgs[1:, ...]))

                dict_shops.update({pair_key: shop_features.detach().cpu()})
                dict_videos.update({pair_key: video_features.detach().cpu()})

                pbar.update()
        pbar.write(f"{len(dict_shops.keys())} shops and {len(dict_videos.keys())} videos extracted")
        pbar.close()
        #
        torch.save(dict_shops, os.path.join(cfg.save_folder, 'shops.pth'))
        torch.save(dict_videos, os.path.join(cfg.save_folder, 'videos.pth'))
    else:
        dict_shops = torch.load(os.path.join(cfg.save_folder, 'shops.pth'))
        dict_videos = torch.load(os.path.join(cfg.save_folder, 'videos.pth'))

    if not os.path.isfile(os.path.join(cfg.save_folder, 'scores.pth')):
        keys = dict_shops.keys()
        scores = torch.zeros((len(keys), len(keys)))
        scores_mean = torch.zeros((len(keys), len(keys)))
        scores_max = torch.zeros((len(keys), len(keys)))
        step = cfg.n_shops
        with tqdm(total=len(keys), desc='Extracting matching score', unit='iter', dynamic_ncols=True) as pbar:
            for i, sh in enumerate(keys):
                sh_f = dict_shops[sh].repeat(step, 1)
                for j in range(0, len(keys), step):
                    vi = [list(keys)[jj] for jj in range(j, min([j + step, len(keys)]))]
                    vi_f = torch.cat([dict_videos[vii] for vii in vi], dim=1)
                    if vi_f.size(1) != cfg.n_frames * step:
                        vi_f = torch.cat([vi_f, torch.zeros((1, cfg.n_frames * step - vi_f.size(1), 1024))], dim=1)
                    with torch.no_grad():
                        tmp, tmp_y = model.SN(sh_f.squeeze().to(device), vi_f.squeeze().to(device))
                        scores_mean[i, j:min([j + step, len(keys)])] = torch.diag(
                            torch.mean(tmp_y.view(cfg.n_shops ** 2, cfg.n_frames, -1), dim=1).view(step, step))[
                                                                       :min([j + step, len(keys)]) - j].detach().cpu()
                        scores_max[i, j:min([j + step, len(keys)])] = torch.diag(
                            torch.max(tmp_y.view(cfg.n_shops ** 2, cfg.n_frames, -1), dim=1)[0].view(step, step))[
                                                                      :min([j + step, len(keys)]) - j].detach().cpu()
                        scores[i, j:min([j + step, len(keys)])] = torch.diag(tmp.view(step, step))[
                                                                  :min([j + step, len(keys)]) - j].detach().cpu()
                pbar.update()
        torch.save({'scores': scores, 'scores_avg': scores_mean, 'scores_max': scores_max},
                    os.path.join(cfg.save_folder, 'scores.pth'))
    else:
        scores = torch.load(os.path.join(cfg.save_folder, 'scores.pth'))['scores']
        scores_mean = torch.load(os.path.join(cfg.save_folder, 'scores.pth'))['scores_avg']
        scores_max = torch.load(os.path.join(cfg.save_folder, 'scores.pth'))['scores_max']

    top1, top5, top10, top15, top20 = compute_scores(scores_mean)
    print(f"Final score [AVG]:\nTop1Acc:{top1}\nTop5Acc:{top5}\nTop10Acc:{top10}\nTop15Acc:{top15}\nTop20Acc:{top20}\n")

    top1, top5, top10, top15, top20 = compute_scores(scores_max)
    print(f"Final score [MAX]:\nTop1Acc:{top1}\nTop5Acc:{top5}\nTop10Acc:{top10}\nTop15Acc:{top15}\nTop20Acc:{top20}\n")

    top1, top5, top10, top15, top20 = compute_scores(scores)
    print(f"Final score:\nTop1Acc:{top1}\nTop5Acc:{top5}\nTop10Acc:{top10}\nTop15Acc:{top15}\nTop20Acc:{top20}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Video2Shop models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpus', dest='gpus', type=str, default="1", help="GPUs IDs (0,1)")

parser.add_argument('-fr', '--frame', dest='n_frames', type=int, default=8,
                    help="Frames for each video (Power of 2)")
parser.add_argument('-sh', '--shops', dest='n_shops', type=int, default=128,
                    help="Number of matching ")

parser.add_argument('-ckpt', '--checkpoint', dest='checkpoint', type=str,
                    default="./checkpoints/last.pth",
                    help="Pretrained Model")
parser.add_argument('-tph', '--training_phase', dest='training_phase', type=int, default=1,
                    help="0: SNN, 1: FN, 2: BOTH")
parser.add_argument('-f', '--save_folder', dest='save_folder', type=str,
                    default="results",
                    help="Checkpoints saving folder")

args = vars(parser.parse_args())
args = edict(args)
eval(args)

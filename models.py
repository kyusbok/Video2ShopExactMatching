import torch
import torch.nn as nn
import torchvision
from math import sqrt
from math import log2

d = 1024


class AsymNet(nn.Module):
    def __init__(self, args):
        super(AsymNet, self).__init__()
        self.IFN = ImageFeatureNetwork()
        self.VFN = VideoFeatureNetwork(n_frame=args.n_frames)
        self.SN = SimilarityNetwork(args.n_frames, args.n_shops)

        self.loss_weight = torch.FloatTensor(args.weights)
        self.criterionMatch = torch.nn.BCELoss(reduce=False)
        self.criterionSNN = torch.nn.BCELoss(reduce=False)

        self.n_frames = args.n_frames
        self.n_shops = args.n_shops

    def forward(self, input, lock_first=False):
        inds = torch.arange(0, input.size(0))
        inds = inds % (self.n_frames + 1) == 0
        if lock_first:
            self.IFN.eval()
            self.VFN.eval()
            with torch.no_grad():
                M = self.IFN(input[inds, ...])
                H = self.VFN(input[inds == 0, ...])

        else:
            M = self.IFN(input[inds, ...])
            H = self.VFN(input[inds == 0, ...])

        scores, out_snn = self.SN(M, torch.cat(H, dim=0), lock_first)
        targets = torch.eye(self.n_shops).to(input.device).flatten()
        loss_match = torch.mean(
            self.criterionMatch(scores, targets) * self.loss_weight[targets.long()].to(input.device))

        targets_snn = targets.unsqueeze(1).repeat(1, self.n_frames).view(-1)
        loss_snn = torch.mean(
            self.criterionSNN(out_snn, targets_snn) * self.loss_weight[targets_snn.long()].to(input.device))

        return scores, out_snn, loss_match, loss_snn


class FrameLSTM(nn.Module):
    def __init__(self):
        super(FrameLSTM, self).__init__()
        self.lstm1 = nn.LSTM(d, d, batch_first=True)
        # self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(d, d, batch_first=True)

    def forward(self, input):
        out1, hidden1 = self.lstm1(input.view(1, input.size(0), input.size(1)))

        out, _ = self.lstm2(out1)

        return out.squeeze()


class VideoFeatureNetwork(nn.Module):
    def __init__(self, n_frame):
        super(VideoFeatureNetwork, self).__init__()

        self.IFN = ImageFeatureNetwork()

        self.lstm_frame = FrameLSTM()
        self.n_frame = n_frame

    def forward(self, input):
        x = self.IFN(input)
        x_list = torch.split(x, self.n_frame)
        H = [self.lstm_frame(x_i) for x_i in x_list]

        return H


class ImageFeatureNetwork(nn.Module):
    def __init__(self):
        super(ImageFeatureNetwork, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(vgg.features.children()))
        self.fcs = nn.Sequential(*list(vgg.classifier.children())[:-1],
                                 nn.Linear(4096, d))

        self.bn = nn.BatchNorm1d(d)

    def forward(self, input):
        x = self.backbone(input)
        return self.bn(self.fcs(x.view(x.size(0), -1)))


class SingleSimilarityBlock(nn.Module):
    def __init__(self, n_frames, n_shops):
        super(SingleSimilarityBlock, self).__init__()

        self.fc1 = nn.Linear(d * 2, d)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, 2)

        self.n_frames = n_frames
        self.n_shops = n_shops

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1).repeat(1, self.n_frames, 1).view(-1, d).unsqueeze(1).repeat(1, self.n_shops, 1).view(-1,
                                                                                                                  d)
        x2 = x2.repeat(self.n_shops, 1).view(-1, d)
        x = torch.cat([x1, x2], dim=-1)
        x = self.bn(self.relu(self.fc1(x)))
        z = self.fc2(x)
        return torch.nn.functional.sigmoid(z)[:, 1], x


class FusionNode(nn.Module):
    def __init__(self, n_shops=0, n_frames=0):
        super(FusionNode, self).__init__()

        self.vij = nn.Linear(d, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.n_shops = n_shops
        self.n_frames = n_frames
        self.n_eps = n_frames

    def forward(self, X):
        eps = self.vij(X)

        g = self.softmax(eps)[..., 1]
        if g.size(-1) > 1:
            g = g.view(g.size(0), -1, 2)
            g = self.softmax(g).view(g.size(0), -1)
        r = self.n_eps // g.size(1)
        g = g.view(g.size(0), g.size(1), 1).repeat(1, 1, r).view(g.size(0), self.n_eps)
        #
        # if X.size(0) == 1:
        return g
        # gs = self.forward(X)
        # # g = torch.cat([g, gs])
        # return gs


class SimilarityNetwork(nn.Module):
    def __init__(self, n_frames, n_shops):
        super(SimilarityNetwork, self).__init__()

        self.sim_block = SingleSimilarityBlock(n_frames, n_shops)
        self.n_frames = n_frames
        self.n_shops = n_shops
        self.n_layer = int(log2(n_frames)) + 1
        self.fusionNode = nn.ModuleList([FusionNode(self.n_shops, self.n_frames) for i in range(self.n_layer)])

    def forward(self, im, frames, lock=False):
        if lock:
            self.sim_block.eval()
            with torch.no_grad():
                (ys, X) = self.sim_block(im, frames)
        else:
            (ys, X) = self.sim_block(im, frames)
        X = X.view(self.n_shops ** 2, -1, d)
        ys = ys.view(self.n_shops ** 2, 1, -1)
        G = ys
        for i in range(self.n_layer):
            gs = self.fusionNode[i](X).unsqueeze(1)
            X = torch.split(X, 2, dim=1)
            X = torch.stack([torch.mean(x_i, dim=1) for x_i in X]).permute((1, 0, 2))
            G = torch.cat([G, gs], dim=1)

        l = torch.sum((torch.prod(G, dim=1).view(self.n_shops ** 2, G.size(-1))), dim=1)

        return l, ys.view(-1)

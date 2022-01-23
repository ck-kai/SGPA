import torch
import torch.nn as nn

from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG
from lib.adaptor import PriorAdaptor

class SPGANet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024, num_structure_points=128):
        super(SPGANet, self).__init__()
        self.n_cat = n_cat
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.instance_geometry = Pointnet2MSG(0)
        self.num_structure_points = num_structure_points

        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=256, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

        self.lowrank_projection = None
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.category_local = Pointnet2MSG(0)

        self.prior_enricher = PriorAdaptor(emb_dims=64, n_heads=4)

        self.category_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def get_prior_enricher_lowrank_projection(self):
        return self.prior_enricher.get_lowrank_projection()
    
    def forward(self, points, img, choose, cat_id, prior):
        
        input_points = points.clone()
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        points = self.instance_geometry(points)
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat

        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb)

        inst_local = torch.cat((points, emb), dim=1)     # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1

        self.lowrank_projection = self.conv1d_stpts_prob(inst_local)
        weighted_xyz = torch.sum(self.lowrank_projection[:, :, :, None] * input_points[:, None, :, :], dim=2)

        weighted_points_features = torch.sum(self.lowrank_projection[:, None, :, :] * points[:, :, None, :], dim=3)
        weighted_img_features = torch.sum(self.lowrank_projection[:, None, :, :] * emb[:, :, None, :], dim=3)

        # category-specific features
        cat_points = self.category_local(prior)    # bs x 64 x n_pts
        cat_color = self.prior_enricher(cat_points, weighted_points_features, weighted_img_features)
        cat_local = torch.cat((cat_points, cat_color), dim=1)
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1

        # assignemnt matrix
        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts

        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv

        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return weighted_xyz, assign_mat, deltas


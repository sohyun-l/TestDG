from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

import PIL
from my_transforms import get_tta_transforms
from inject_amplifier_return import inject_trainable_amplifier_return
from proj_layer import ProjLayer
from collections import OrderedDict

from memory import Memory
from rbf_kernel import rbf_kernel
from mmd_critic import select_prototypes
from time import time
import logging
import time
import resource 

batch = 40

def update_ema_variables(ema_model, model, alpha_teacher, alpha_amplifier):
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        if "amplifier_" in name:
            ema_param.data[:] = alpha_amplifier * ema_param[:].data[:] + (1 - alpha_amplifier) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class testdg_cifar10c(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, ema=0.99, ema_amplifier = 0.99, rst_m=0.1, anchor_thr=0.9, unc_thr = 0.2, mse_weight = 0.1, proj_lr=0.001, cls_weight = 1.0, cls_lr=0.001):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.proj_lr = proj_lr
        self.cls_lr = cls_lr
        self.model_state, self.optimizer_state, self.model_ema = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.alpha_teacher = ema
        self.alpha_amplifier = ema_amplifier
        self.rst = rst_m
        self.thr = unc_thr
        self.mse_weight = mse_weight
        self.ProjLayer = ProjLayer(10)
        self.ProjLayer.cuda()
        self.classifier = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        ).cuda()

        self.optimizer_proj = torch.optim.Adam(self.ProjLayer.parameters(), lr=self.proj_lr)
        self.optimizer_cls = torch.optim.Adam(self.classifier.parameters(), lr=self.cls_lr)
        self.cls_weight = cls_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, x, c_iter, pre_conf, domain_change, i_c, conf_diff, domain_label, domain_memory, prototypes, random_vector, clean):
        if clean == True:
            with torch.no_grad():
                self.model_ema.eval()
                self.model.eval()
                outputs = self.model(x)
                conf = outputs.softmax(1).max(1).values.mean()
                random_vector_new = random_vector
        else:
            if self.episodic:
                self.reset()

            for i in range(self.steps):
                if domain_change == True:
                    domain_label += 1

                outputs, conf, domain_memory, prototypes, random_vector_new = self.forward_and_adapt(x, self.model, self.optimizer, c_iter, domain_label, domain_change, domain_memory, prototypes, random_vector)
                
        return outputs, conf, domain_label, domain_memory, prototypes, random_vector_new

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)             
        self.model_state, self.optimizer_state, self.model_ema = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def set_scale(self, update_model, high):
        for name, module in update_model.named_modules():
            if hasattr(module, 'scale2'):
                module.scale2 = high.item()
    @torch.enable_grad()  
    def forward_and_adapt(self, x, model, optimizer, i_c, domain_label, domain_change, domain_memory, prototypes, random_vector):
        self.model_ema.eval()
        self.domain_memory = domain_memory
        N = 4
        outputs_uncs = []


        if i_c % 50 < 20:
            for name, param in self.model.named_parameters():
                if 'amplifier_up' in name:
                    param.requires_grad = True
                elif 'amplifier_down' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for param in self.ProjLayer.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
            outputs_  = self.model_ema(self.transform(x)).detach()
            
            pair_aug = self.ProjLayer(outputs_)
            
            lambda_high = torch.tensor(1.0).cuda()

            self.set_scale(update_model = model, high = lambda_high)
            self.set_scale(update_model = self.model_ema, high = lambda_high)
            standard_ema = self.model_ema(x)
            outputs = self.model(x)
            conf = standard_ema.softmax(1).max(1).values.mean()

            
            loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0) 
            pair_ori = self.ProjLayer(outputs)

            if domain_label == 0:
                self.domain_memory.enqueue(pair_ori.detach())
                cls_ori = self.classifier(pair_ori)
                cls_aug = self.classifier(pair_aug)
                labels_ori = torch.ones(batch, dtype=torch.float).cuda()
                labels_aug = torch.zeros(batch, dtype=torch.float).cuda()

                loss_ori = torch.nn.functional.binary_cross_entropy_with_logits(cls_ori, labels_ori.unsqueeze(1))
                loss_aug = torch.nn.functional.binary_cross_entropy_with_logits(cls_aug, labels_aug.unsqueeze(1))
                domain_loss = self.cls_weight * (loss_ori + loss_aug) / 2
            else:
                if domain_change == True:
                    selected = self.generate_representation(self.domain_memory.get_mem(), 
                                        rbf_gamma=1.0,
                                        sample_num=4, 
                                        std_mul=0.25)
                    prototypes = self.domain_memory.get_mem()[selected]
                    self.domain_memory.clear()
                    print(domain_change)
                    print(prototypes[0])
                self.domain_memory.enqueue(pair_ori.detach())
                cls_aug = self.classifier(prototypes)
                cls_ori = self.classifier(pair_ori)
                labels_ori = torch.ones(batch, dtype=torch.float).cuda()
                labels_aug = torch.zeros(batch, dtype=torch.float).cuda()
                loss_ori = torch.nn.functional.binary_cross_entropy_with_logits(cls_ori, labels_ori.unsqueeze(1))
                loss_aug = torch.nn.functional.binary_cross_entropy_with_logits(cls_aug, labels_aug.unsqueeze(1))
                domain_loss = self.cls_weight * (loss_ori + loss_aug) / 2


            loss += domain_loss
            loss.backward()
            self.optimizer_proj.step()
            self.optimizer_cls.step()
            self.optimizer_proj.zero_grad() 
            self.optimizer_cls.zero_grad()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            self.ProjLayer.eval()
            self.classifier.eval()
            loss_chamfer = 0
            if domain_label != 0:
                chamfer_dist_pre = chamfer_dist(pair_ori.detach().unsqueeze(0), prototypes.detach().unsqueeze(0))
                outputs = self.model(x).detach()
                new_pair_ori = self.ProjLayer(outputs).detach()
                prototypes_new = nn.Parameter(torch.randn(batch, 4, requires_grad=True, device="cuda"))
                optimizer_proto = torch.optim.Adam([prototypes_new], lr=0.01)  
                chamfer_dist_curr = chamfer_dist(new_pair_ori.unsqueeze(0), prototypes_new.unsqueeze(0))
                loss_chamfer = torch.abs(chamfer_dist_pre - chamfer_dist_curr)
                loss_chamfer.backward()
                optimizer_proto.step()
                optimizer_proto.zero_grad()
                prototypes = prototypes_new.detach()
                loss_chamfer = 0


            for name, param in self.model.named_parameters():
                if 'amplifier_up' in name:
                    param.requires_grad = False
                elif 'amplifier_down' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for param in self.ProjLayer.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

            lambda_high = torch.tensor(1.0).cuda()
            self.set_scale(update_model = model, high = lambda_high)
            self.set_scale(update_model = self.model_ema, high = lambda_high)
            standard_ema = self.model_ema(x)
            outputs = self.model(x)

            
            loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0) 
        else:
            for name, param in self.model.named_parameters():
                if 'amplifier_up' in name:
                    param.requires_grad = True
                elif 'amplifier_down' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for param in self.ProjLayer.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
            outputs_  = self.model_ema(self.transform(x)).detach()
            
            pair_aug = self.ProjLayer(outputs_)
            
            lambda_high = torch.tensor(1.0).cuda()
            
            self.set_scale(update_model = model, high = lambda_high)
            self.set_scale(update_model = self.model_ema, high = lambda_high)
            standard_ema = self.model_ema(x)
            outputs = self.model(x)
            conf = standard_ema.softmax(1).max(1).values.mean()
            loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0)

            pair_ori = self.ProjLayer(outputs)
            

            if domain_label == 0:

                self.domain_memory.enqueue(pair_ori.detach())
                cls_ori = self.classifier(pair_ori)
                cls_aug = self.classifier(pair_aug)
                labels_ori = torch.ones(batch, dtype=torch.float).cuda()
                labels_aug = torch.zeros(batch, dtype=torch.float).cuda()

                loss_ori = torch.nn.functional.binary_cross_entropy_with_logits(cls_ori, labels_ori.unsqueeze(1))
                loss_aug = torch.nn.functional.binary_cross_entropy_with_logits(cls_aug, labels_aug.unsqueeze(1))
                domain_loss = self.cls_weight * (loss_ori + loss_aug) / 2
            else:
                if domain_change == True:
                    selected = self.generate_representation(self.domain_memory.get_mem(), 
                                        rbf_gamma=1.0,
                                        sample_num=4, 
                                        std_mul=0.25)
                    prototypes = self.domain_memory.get_mem()[selected]
                    self.domain_memory.clear()
                
                self.domain_memory.enqueue(pair_ori.detach())
                cls_aug = self.classifier(prototypes)
                pair_aug = prototypes
                cls_ori = self.classifier(pair_ori)
                labels_ori = torch.ones(batch, dtype=torch.float).cuda()
                labels_aug = torch.zeros(batch, dtype=torch.float).cuda()
                loss_ori = torch.nn.functional.binary_cross_entropy_with_logits(cls_ori, labels_ori.unsqueeze(1))
                loss_aug = torch.nn.functional.binary_cross_entropy_with_logits(cls_aug, labels_aug.unsqueeze(1))
                domain_loss = self.cls_weight * (loss_ori + loss_aug) / 2

            loss += domain_loss
            loss.backward()
            self.optimizer_proj.step()
            self.optimizer_cls.step()
            self.optimizer_proj.zero_grad() 
            self.optimizer_cls.zero_grad()
            optimizer.step()
            optimizer.zero_grad()


            model.eval()
            self.ProjLayer.eval()
            self.classifier.eval()
            loss_chamfer = 0
            if domain_label != 0:
                chamfer_dist_pre = chamfer_dist(pair_ori.detach().unsqueeze(0), prototypes.detach().unsqueeze(0))
                outputs = self.model(x).detach()
                new_pair_ori = self.ProjLayer(outputs).detach()
                prototypes_new = nn.Parameter(torch.randn(batch, 4, requires_grad=True, device="cuda"))
                optimizer_proto = torch.optim.Adam([prototypes_new], lr=0.01)  
                chamfer_dist_curr = chamfer_dist(new_pair_ori.unsqueeze(0), prototypes_new.unsqueeze(0))
                loss_chamfer = torch.abs(chamfer_dist_pre - chamfer_dist_curr)
                loss_chamfer.backward()
                optimizer_proto.step()
                optimizer_proto.zero_grad()
                prototypes = prototypes_new.detach()
                loss_chamfer = 0


            for name, param in self.model.named_parameters():
                if 'amplifier_up' in name:
                    param.requires_grad = False
                elif 'amplifier_down' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for param in self.ProjLayer.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

            lambda_high = torch.tensor(1.0).cuda()
            self.set_scale(update_model = model, high = lambda_high)
            self.set_scale(update_model = self.model_ema, high = lambda_high)
            standard_ema = self.model_ema(x)
            outputs = self.model(x)
            pair_ori = self.ProjLayer(outputs)

            loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0)

            pair_aug = prototypes
            if domain_label == 0:
                pair_aug = self.ProjLayer(outputs_)
            l1_loss = self.mse_weight * torch.nn.L1Loss(reduction='mean')(pair_ori, pair_aug)   
            loss += l1_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        model.eval()
        self.ProjLayer.eval()
        self.classifier.eval()
        loss_chamfer = 0
        if domain_label != 0:
            chamfer_dist_pre = chamfer_dist(pair_ori.detach().unsqueeze(0), prototypes.detach().unsqueeze(0))
            outputs = self.model(x).detach()
            new_pair_ori = self.ProjLayer(outputs).detach()
            prototypes_new = nn.Parameter(torch.randn(batch, 4, requires_grad=True, device="cuda"))
            optimizer_proto = torch.optim.Adam([prototypes_new], lr=0.01)  
            chamfer_dist_curr = chamfer_dist(new_pair_ori.unsqueeze(0), prototypes_new.unsqueeze(0))
            loss_chamfer = torch.abs(chamfer_dist_pre - chamfer_dist_curr)
            loss_chamfer.backward()
            optimizer_proto.step()
            optimizer_proto.zero_grad()
            prototypes = prototypes_new.detach()
            loss_chamfer = 0

        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher= self.alpha_teacher, alpha_amplifier = self.alpha_amplifier)
        if True:
            for npp, p in model.named_parameters():
                if p.requires_grad:
                    mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                    with torch.no_grad():
                        p.data = self.model_state[npp] * mask + p * (1.-mask)

        return standard_ema, conf, self.domain_memory, prototypes, random_vector

    def generate_representation(self, feat, rbf_gamma=None, sample_num=1, std_mul=0.0):
        feat = feat.detach()
        kernel_matrix = rbf_kernel(feat, gamma=rbf_gamma)
        selected = select_prototypes(kernel_matrix, num_prototypes=batch)
        
        return selected


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    
def chamfer_dist(a,b):
    batch_size, num_points, dim = a.shape
    _, num_points, dim = b.shape
    a = a.unsqueeze(2).expand(batch_size, num_points, num_points, dim)
    b = b.unsqueeze(1).expand(batch_size, num_points, num_points, dim)
    dist = torch.norm(a - b, dim=3)
    min_dist_a_to_b = torch.min(dist, dim=2)[0]
    min_dist_b_to_a = torch.min(dist, dim=1)[0]
    chamfer_dist = torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)

    return chamfer_dist


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    amplifier_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'amplifier_' in name:
            amplifier_params_list.append(param)
        else:
            model_params_lst.append(param)     
    return model_params_lst, amplifier_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg):
    """Configure model for use with tent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
    amplifier_params, amplifier_names = inject_trainable_amplifier_return(model = model, target_replace_module = ["CrossAttention", "Attention"], \
            r = cfg.TEST.amplifier_rank1, r2 = cfg.TEST.amplifier_rank2)
    new_state_dict = OrderedDict()
    if cfg.TEST.ckpt!=None:
        checkpoint = torch.load(cfg.TEST.ckpt)
        for k,v in checkpoint['model'].items():
            name = k.replace('model.module.','module.') # remove `module.`
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

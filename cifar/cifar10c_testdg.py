import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import accuracy_cifar10c
from collections import OrderedDict
from memory import Memory
import numpy as np

import testdg_cifar10c
import torch.nn as nn
from datetime import datetime
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def evaluate(description):
    args = load_cfg_fom_args(description)
    
    model = cfg.MODEL.ADAPTATION
    arc = cfg.MODEL.ARCH
    mse = cfg.OPTIM.MSE_WEIGHT
    projlr = cfg.OPTIM.PROJ_LR
    clslr = cfg.OPTIM.CLS_LR
    lr = cfg.OPTIM.LR
    amplr = cfg.OPTIM.AMPLIFIERLR
    clsweight = cfg.OPTIM.CLS_WEIGHT
    adt = cfg.MODEL.ADAPTATION
    batch = cfg.TEST.BATCH_SIZE
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{model}-{arc}-{adt}-test-cifar10c-{now}'
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    

    if cfg.TEST.ckpt is not None:
        base_model = torch.nn.DataParallel(base_model) 
        checkpoint = torch.load(cfg.TEST.ckpt)
        base_model.load_state_dict(checkpoint, strict=False)
    else:
        base_model = torch.nn.DataParallel(base_model) 
    base_model.cuda()

    if cfg.MODEL.ADAPTATION == "testdg_cifar10c":
        logger.info("test-time adaptation: Test-DG")
        model = setup_testdg_cifar10c(args, base_model)
    All_error = []
    All_error_clean = []
    acc_list = []
    pre_conf = [0.0, 0.0]
    pre_iter = 0
    domain_label = 0
    conf_diff = 0        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    domain_memory = Memory(queue_size=256, channel=4, gpu_idx=device)
    prototypes = []

    single_row = torch.rand(1, 4) * 20 - 10
    random_vector = single_row.repeat(batch, 1)
    random_vector = random_vector.to(device) 

    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                        severity, cfg.DATA_DIR, False,
                                        [corruption_type])
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            acc, num, pre_conf, domain_label, conf_diff, domain_memory, prototypes, random_vector, acc_list = accuracy_cifar10c(model, x_test, y_test, acc_list, random_vector, cfg.TEST.BATCH_SIZE, pre_iter, pre_conf, i_c, domain_label, conf_diff, domain_memory, prototypes, device = 'cuda')
            
            pre_iter += num
            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

def setup_testdg_cifar10c(args, model):
    model = testdg_cifar10c.configure_model(model, cfg)
    model_param, amplifier_param = testdg_cifar10c.collect_params(model)
    optimizer = setup_optimizer_testdg(model_param, amplifier_param, cfg.OPTIM.LR, cfg.OPTIM.AMPLIFIERLR)
    ours_model = testdg_cifar10c.testdg_cifar10c(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_amplifier = cfg.OPTIM.MT_AMPLIFIER,
                           mse_weight = cfg.OPTIM.MSE_WEIGHT,
                           proj_lr = cfg.OPTIM.PROJ_LR,
                           cls_weight = cfg.OPTIM.CLS_WEIGHT,
                           cls_lr = cfg.OPTIM.CLS_LR,
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return ours_model



def setup_optimizer_testdg(params, params_testdg, model_lr, testdg_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
                                  {"params": params_testdg, "lr": testdg_lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                                  {"params": params_testdg, "lr": testdg_lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError
if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')

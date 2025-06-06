import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import accuracy_cifar100c
from memory import Memory

import testdg_cifar100c
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
    amp_lr = cfg.OPTIM.AMPLIFIERLR
    clsweight = cfg.OPTIM.CLS_WEIGHT
    adt = cfg.MODEL.ADAPTATION
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{model}-{arc}-{adt}-test-cifar100c-{now}'
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    if cfg.TEST.ckpt is not None:
        base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(cfg.TEST.ckpt)
        base_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        base_model = torch.nn.DataParallel(base_model)
    base_model.cuda()

    if cfg.MODEL.ADAPTATION == "testdg_cifar100c":
        logger.info("test-time adaptation: Test-DG")
        model = setup_testdg_cifar100c(args, base_model)
    All_error = []
    pre_conf = [0.0, 0.0]
    pre_iter = 0
    domain_label = 0
    conf_diff = 0        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    domain_memory = Memory(queue_size=512, channel=4, gpu_idx=device)
    prototypes = []

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
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            acc, num, pre_conf, domain_label, conf_diff, domain_memory, prototypes = accuracy_cifar100c(model, x_test, y_test, cfg.TEST.BATCH_SIZE, pre_iter, pre_conf, i_c, domain_label, conf_diff, domain_memory, prototypes, device = 'cuda')
            err = 1. - acc
            All_error.append(err)
            pre_iter += num
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_testdg_cifar100c(args, model):
    model = testdg_cifar100c.configure_model(model, cfg)
    model_param, amplifier_param = testdg_cifar100c.collect_params(model)
    optimizer = setup_optimizer_testdg(model_param, amplifier_param, cfg.OPTIM.LR, cfg.OPTIM.AMPLIFIERLR)
    testdg_model = testdg_cifar100c.testdg_cifar100c(model, optimizer,
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
    return testdg_model


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

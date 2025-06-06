import logging

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import accuracy as accuracy
from memory import Memory
import numpy as np

import testdg
from datetime import datetime

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)



def evaluate(description):
    args = load_cfg_fom_args(description)
    
    model = cfg.MODEL.ADAPTATION
    arc = cfg.MODEL.ARCH
    mse = cfg.OPTIM.MSE_WEIGHT
    lr = cfg.OPTIM.LR
    adapt = cfg.MODEL.ADAPTATION
    wd = cfg.OPTIM.WD
    momentum = cfg.OPTIM.MOMENTUM
    mt = cfg.OPTIM.MT
    mt_amplifier = cfg.OPTIM.MT_AMPLIFIER
    beta = cfg.OPTIM.BETA
    now = datetime.now().strftime('%m-%d-%H-%M')
    batch_size = cfg.TEST.BATCH_SIZE
    run_name = f'imagenetc-{model}-{arc}-{adapt}-{now}'
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    if cfg.MODEL.ADAPTATION == "testdg":
        logger.info("test-time adaptation: Test-DG")
        model = setup_testdg(args, base_model)
    prev_ct = "x0"
    All_error = []
    acc_list = []
    pre_conf = [0.0, 0.0]
    pre_iter = 0
    domain_label = 0
    conf_diff = 0        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    domain_memory = Memory(queue_size=512, channel=4, gpu_idx=device)
    prototypes = []

    single_row = torch.rand(1, 4) * 20 - 10
    random_vector = single_row.repeat(batch_size, 1)  
    random_vector = random_vector.to(device)

    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            try:
                if i_c == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc, num, pre_conf, domain_label, conf_diff, domain_memory, prototypes, random_vector, acc_list = accuracy(model, x_test, y_test, acc_list, random_vector, cfg.TEST.BATCH_SIZE, pre_iter, pre_conf, i_c, domain_label, conf_diff, domain_memory, prototypes, device = 'cuda')
            
            pre_iter += num
            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    

def setup_testdg(args, model):
    model = testdg.configure_model(model, cfg)
    model_param, amplifier_param = testdg.collect_params(model)
    optimizer = setup_optimizer_amplifier(model_param, amplifier_param, cfg.OPTIM.LR, cfg.OPTIM.AMPLIFIERLR)
    testdg_model = testdg.testdg(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           unc_thr = args.unc_thr,
                           ema = cfg.OPTIM.MT,
                           ema_amplifier = cfg.OPTIM.MT_AMPLIFIER,
                           mse_weight = cfg.OPTIM.MSE_WEIGHT,
                           proj_lr = cfg.OPTIM.PROJ_LR,
                           cls_weight = cfg.OPTIM.CLS_WEIGHT,
                           cls_lr = cfg.OPTIM.CLS_LR
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return testdg_model

def setup_optimizer_amplifier(params, params_testdg, model_lr, testdg_lr):
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
    evaluate('"Imagenet-C evaluation.')

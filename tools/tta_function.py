import numpy as np
import os
import torch
from tqdm import tqdm
import ttach as tta

def tta_score_ours(model, data_loader, device, predict_folder, eval_score_iou=False, eval_depth=False, eval_trunc_recall=False):
    lr = 0.5 * 1e-3
    transforms = tta.aliases.d4_transform()
    for idx, batch in enumerate(tqdm(data_loader)):
        model.train()
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # simple test-time augmentations
        aug_scores = []
        aug_total_scores = []
        aug_probs = []

        for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform()
            tmp_imgs = images.tensors
            aug_imgs = transformer.augment_image(tmp_imgs)
            scores, score_dict, prob_dict = model(aug_imgs, targets, tta=True, temp=1.0)

            aug_scores.append(scores)
            aug_total_scores.append(score_dict)
            aug_probs.append(prob_dict)

        # high-confidence detection optimization
        debug_mean_score = torch.tensor(0.).cuda()
        tmp_len = 0
        for each_score in aug_scores:
            debug_mask = each_score >= 0.2  # remove the background part, following the orignal setting
            if debug_mask.sum() > 0:
                debug_mean_score += torch.mean(each_score[debug_mask])
                tmp_len += len(each_score[debug_mask])
        debug_mean_score /= len(aug_scores)
        tmp_len /= len(aug_scores)

        if idx == 0:
            cur_threshold = 0.2
            pre_threshold = 0.2
            alpha = 0.1
        elif tmp_len > 0:
            cur_threshold = alpha * debug_mean_score + (1 - alpha) * pre_threshold
            pre_threshold = cur_threshold
        print('iteration {}, cur_threshold {}, pre_threshold {}'.format(idx, cur_threshold, pre_threshold))



def run_tta(cfg, model, datasets, vis, eval_score_iou, eval_all_depths=True, clean_model=None):
    data_loaders_val = build_data_loader(cfg, datasets)
    for data_loader_val in data_loaders_val:
        tta_ckpt = tta_score_ours(model, data_loader_val, cfg, vis, eval_score_iou, eval_all_depths, clean_model)
    # comm.synchronize()
    return tta_ckpt

def build_data_loader(cfg, datasets):
    data_loaders = []
    for dataset in datasets:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.samples_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=True,
            pin_memory=False,
            drop_last=True)
        data_loaders.append(data_loader)
    # add an attribute for visualization convenience 
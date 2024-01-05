# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import tqdm
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC


PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]



def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def convert_to_heatmap(img_array):
    """ Convert an image array to a heatmap representation """
    
    heatmap_img = plt.get_cmap('jet')(img_array[:, :, 0], bytes=True)  # Use 'hot' colormap
    return Image.fromarray(heatmap_img[:, :, :3])


def overlay_images(base_img, overlay_img, alpha=0.7):
    """ Overlay one image on top of another with transparency """
    base_img = base_img.convert("RGBA")
    overlay_img = overlay_img.convert("RGBA")
    
    # import ipdb; ipdb.set_trace()

    # Blend images
    combined_img = Image.blend(base_img, overlay_img, alpha)
    return combined_img.convert("RGB")


def read_png_files(directory):
    png_files = [f for f in os.listdir(os.path.join(directory, 'images')) if f.endswith('.png')]
    filenames = []
    images_as_arrays = []
    masks_as_arrays = []

    for file in png_files:
        image_file_path = os.path.join(os.path.join(directory, 'images'), file)
        mask_file_path = os.path.join(os.path.join(directory, 'gt_label'), file)
        filenames.append(file)
        with Image.open(image_file_path) as img:
            images_as_arrays.append(np.array(img))
        with Image.open(mask_file_path) as mask:
            masks_as_arrays.append(np.array(mask))

    return filenames, images_as_arrays, masks_as_arrays


def confidence_gpu_test_fishyscapes(model,
                    data_loader,
                    interval=4,
                    return_logits=False):

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    counter = 0
    for batch_indices, data in zip(loader_indices, data_loader):
        if counter % interval == 0:
            with torch.no_grad():
                result = model(return_loss=False, return_logits=return_logits, **data)
            results.extend(result)

        # batch_size = len(result)
        batch_size = 1
        for _ in range(batch_size):
            prog_bar.update()
            counter += 1

    return results

def anomaly_gpu_test_fishyscapes(
    model,
    out_dir = '/mnt/petrelfs/yuwenye/GMMSeg/work_dirs/segformer_mit-b5_gmmseg_fishyscapes'
):
    model.eval()
    import ipdb; ipdb.set_trace()
    ori_imgs = read_png_files('/mnt/petrelfs/yuwenye/GMMSeg/data/fishyscapes/LostAndFound/images')
    for idx, ori_img in enumerate(ori_imgs):
        ori_filename, curr_img = ori_img
        img = torch.from_numpy(curr_img).unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float()
        img_metas_dict = {'filename': '/mnt/petrelfs/yuwenye/GMMSeg/data/fishyscapes/LostAndFound/images/' + ori_filename, 'ori_filename': ori_filename, 'ori_shape': (1024, 2048, 3), 'img_shape': (1024, 2048, 3),
                          'pad_shape': (1024, 2048, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype = float), 
                          'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53], dtype = float), 'std': np.array([58.395, 57.12 , 57.375], dtype = float), 'to_rgb': True}}
        img_metas_dc = [DC(data = [[img_metas_dict]], cpu_only = True)]
        img_norm_cfg_mean = torch.tensor([123.675, 116.28 , 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        img_norm_cfg_std = torch.tensor([58.395, 57.12 , 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        img -= img_norm_cfg_mean
        img = torch.div(img, img_norm_cfg_std)
        img = [img]
        data = {'img_metas': img_metas_dc, 'img': img}
        import ipdb; ipdb.set_trace()
        with torch.no_grad():
            result = model(return_loss=False, return_logits=False, **data)
        import ipdb; ipdb.set_trace()
        

def single_gpu_test_fishyscapes(model,
                    data_loader = None,
                    show=False,
                    out_dir=None,
                    anomaly_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    return_logits=False,
                    pre_eval=False,
                    format_only=False,
                    format_args={}
                    ):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    # dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    # # The pipeline about how the data_loader retrieval samples from dataset:
    # # sampler -> batch_sampler -> indices
    # # The indices are passed to dataset_fetcher to get data from dataset.
    # # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # # we use batch_sampler to get correct data idx
    # loader_indices = data_loader.batch_sampler
    ori_patches = read_png_files('/mnt/petrelfs/yuwenye/GMMSeg/data/fishyscapes/LostAndFound')
    # ori_masks = read_png_files('/mnt/petrelfs/yuwenye/GMMSeg/data/fishyscapes/LostAndFound/gt_label')
    AvgPrecision = []
    AUROCs = []
    FPR95s = []
    
    pbar = tqdm.tqdm(total = len(ori_patches[0]))

    for idx, (ori_filename, curr_img, gt_mask) in enumerate(zip(*ori_patches)):
        # ori_filename, curr_img = ori_img
        img = torch.from_numpy(curr_img).unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float()
        img_metas_dict = {'filename': '/mnt/petrelfs/yuwenye/GMMSeg/data/fishyscapes/LostAndFound/images/' + ori_filename, 'ori_filename': ori_filename, 'ori_shape': (1024, 2048, 3), 'img_shape': (1024, 2048, 3),
                          'pad_shape': (1024, 2048, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype = float), 
                          'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53], dtype = float), 'std': np.array([58.395, 57.12 , 57.375], dtype = float), 'to_rgb': True}}
        img_metas_dc = [DC(data = [[img_metas_dict]], cpu_only = True)]
        img_norm_cfg_mean = torch.tensor([123.675, 116.28 , 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        img_norm_cfg_std = torch.tensor([58.395, 57.12 , 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        img -= img_norm_cfg_mean
        img = torch.div(img, img_norm_cfg_std)
        img = [img]
        data = {'img_metas': img_metas_dc, 'img': img}
        # import ipdb; ipdb.set_trace()
        with torch.inference_mode():
            pred_logits = model(return_loss = False, return_logits = True, **data)
            result = model(return_loss = False, return_logits = False, **data)
        
        pred_logits = pred_logits[0]
        # import ipdb; ipdb.set_trace()
        pred_labels = result[0]
        
        # pred_logits = torch.from_numpy(pred_logits)
        # # pred_prob = F.softmax(pred_logits, dim = 0)
        # pred_prob = pred_prob.cpu().numpy()
        
        confidence = np.max(pred_logits, axis = 0)
        inconfidence = - confidence
        inconfidence = np.expand_dims(inconfidence, axis = 0)
        inconfidence = inconfidence.repeat(3, axis = 0)
        # inconfidence = np.exp(inconfidence)
        # anomaly_pred = inconfidence * 255.0
        anomaly_pred = (inconfidence - inconfidence.min()) * (255.0 / (inconfidence.max() - inconfidence.min()))
        anomaly_pred = anomaly_pred / 255.0
        # smooth_filter = np.ones_like(anomaly_pred)[0]
        # height = smooth_filter.shape[0]
        # width = smooth_filter.shape[1]
        # h_center = height // 2
        # w_center = width // 2
        # height_coord = np.arange(height, dtype = np.int8)
        # width_coord = np.arange(width, dtype = np.int8)
        # heights, widths = np.meshgrid(height_coord, width_coord, indexing = 'ij')
        # smooth_filter = np.where(np.logical_and(np.square(heights - h_center) <= 10000, np.square(widths - w_center) <= 40000), 1.0, 0.1)
        # smooth_filter = np.expand_dims(smooth_filter, axis = 0)
        # smooth_filter = smooth_filter.repeat(3, axis = 0)
        # anomaly_pred = anomaly_pred * smooth_filter
        
        
        # smooth_filter = np.where(smooth_filter)
        
        # import ipdb; ipdb.set_trace()
        gt_mask = gt_mask[:, :, 0]
        gt_mask = np.where(gt_mask == 255, 0, gt_mask)
        gt_mask = np.where(gt_mask == 127, 1, gt_mask)
        
        anomaly_mask = np.where(gt_mask == 0, 0.7, gt_mask)
        anomaly_mask = np.where(anomaly_mask == 1, 1.5, anomaly_mask)
        anomaly_mask = np.expand_dims(anomaly_mask, axis = 0)
        anomaly_mask = anomaly_mask.repeat(3, axis = 0)
        anomaly_pred = (anomaly_mask * anomaly_pred).clip(max = 1.0)
        
        
        anomaly_pred_void = anomaly_pred[0]
        anomaly_pred_void = anomaly_pred_void.flatten()
        gt_mask = gt_mask.flatten()
        
        if len(np.unique(gt_mask)) > 1:
            ap = average_precision_score(gt_mask, anomaly_pred_void)
            auroc = roc_auc_score(gt_mask, anomaly_pred_void)
            fpr, tpr, thresholds = roc_curve(gt_mask, anomaly_pred_void)
            
            closest_idx = np.argmin(np.abs(tpr - 0.95))
            fpr95 = fpr[closest_idx]
            
            AvgPrecision.append(ap)
            AUROCs.append(auroc)
            FPR95s.append(fpr95)
        
        
        
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                
                # import ipdb; ipdb.set_trace()

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)
                
                if anomaly_dir:
                    anomaly_file = osp.join(anomaly_dir, img_meta['ori_filename'])
                else:
                    anomaly_file = None
                    
                # anomaly_pred = anomaly_pred.astype(np.uint8)
                anomaly_pred = anomaly_pred.transpose(1, 2, 0)
                # anomaly_pred = anomaly_pred[:, :, 0]
                # kernel = np.ones((5,5),np.uint8)
                # anomaly_pred = cv2.morphologyEx(anomaly_pred, cv2.MORPH_OPEN, kernel)
                # anomaly_pred = cv2.morphologyEx(anomaly_pred, cv2.MORPH_CLOSE, kernel)
                # import ipdb; ipdb.set_trace()
                heat_map = convert_to_heatmap(anomaly_pred)
                # import ipdb; ipdb.set_trace()
                img_show_pil = Image.fromarray(img_show)
                # import ipdb; ipdb.set_trace()
                combined_image = overlay_images(img_show_pil, heat_map, alpha=0.7)
                
                # anomaly_pic = Image.fromarray(anomaly_pred)
                save_anomaly_path = anomaly_file
                combined_image.save(save_anomaly_path)
        
        
                
        results.append(result)

        # if efficient_test:
        #     result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        # if format_only:
        #     result = dataset.format_results(
        #         result, indices=batch_indices, **format_args)

        # if pre_eval:
        #     # TODO: adapt samples_per_gpu > 1.
        #     # only samples_per_gpu=1 valid now
        #     result = dataset.pre_eval(result, indices=batch_indices)
        #     results.extend(result)
        # else:
        #     results.extend(result)

        # batch_size = len(result)
        # for _ in range(batch_size):
        #     prog_bar.update()

        del data
        torch.cuda.empty_cache()
        pbar.update(1)
        
    AvgPrecision = np.array(AvgPrecision)
    AUROCs = np.array(AUROCs)
    FPR95s = np.array(FPR95s)
    
    AP_final = np.mean(AvgPrecision)
    AUROC_final = np.mean(AUROCs)
    FPR95_final = np.mean(FPR95s)
    
    print(f"AP: {AP_final}, AUROC: {AUROC_final}, FPR95: {FPR95_final}")

    return results


def multi_gpu_test_fishyscapes(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   return_logits=False,
                   format_only=False,
                   format_args={}
                   ):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, return_logits=return_logits, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    del data
    torch.cuda.empty_cache()

    return results

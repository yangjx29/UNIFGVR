from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment # LOOK I DON'T HAVE THIS VERSION
import scipy.io
from tqdm import tqdm
import random
import os
import argparse


import torch
import numpy as np
from difflib import SequenceMatcher

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # Samples

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]):  # Classes
            r.append((1 / 1) ** (i / (Q.shape[0] - 1.0)))
        r = np.array(r)
        r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
        r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum()  # Scaling to make it prob

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        # import pdb; pdb.set_trace()
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)

#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_arr, jnd_arr = linear_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CentroidTracker(object):
    def __init__(self, model, labeled_train_loader, num_labeled_classes, device,
                 dataset_name, train_stage, save_root, mode='dynamic'):
        self.model = model
        self.loader = labeled_train_loader
        self.num_labeled_classes = num_labeled_classes
        self.device = device
        self.dataset_name = dataset_name
        self.train_stage = train_stage
        self.root = save_root+'/'
        self.stats_dir = os.path.join(self.root, (self.dataset_name+'_'+self.train_stage+'_stats'))
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        self.file_prefix = self.stats_dir+'/'+'epoch'
        self.mode = mode
        self.flying_mean = None
        self.flying_sig = None
        self.flying_cov = None

    def initialize_stats(self, init_mean, init_sig, init_cov):
        self.flying_mean = init_mean
        self.flying_sig = init_sig
        self.flying_cov = init_cov
        print("...... Initialized centroids from the old feature space")

    def generate(self, current_epoch, save_featmap=True):
        # create individual containers for extracted feature, labels, class mean, class sig and class cov
        all_feat = []
        all_labels = []
        class_mean = torch.zeros(self.num_labeled_classes, 512)
        class_sig = torch.zeros(self.num_labeled_classes, 512)

        # extract the feat and label using the current learned model
        self.model.eval()
        print("Start to calculate the statistics of the labeled features for epoch: [{}]".format(current_epoch))
        print("Extract labeled features")
        for batch_idx, (x, label, idx) in enumerate(tqdm(self.loader)):
            # print("---extracting from batch [{}]".format(batch_idx))
            x, label = x.to(self.device), label.to(self.device)
            _, _, feat = self.model(x)
            all_feat.append(feat.detach().clone())
            all_labels.append(label.detach().clone())

        # organize it a bit
        # print(len(all_feat))
        # print(all_feat[0].shape)
        all_feat = torch.cat(all_feat, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        # check the shapes
        print("all_feats shape: {}".format(all_feat.shape))
        print("all_labels shape: {}".format(all_labels.shape))

        print("Calculate labeled Mean-Var-Cov")
        for i in range(self.num_labeled_classes):
            this_feat = all_feat[all_labels==i]
            this_mean = this_feat.mean(dim=0)
            this_var = this_feat.var(dim=0)
            class_mean[i,:] = this_mean
            class_sig[i,:] = (this_var + 1e-5).sqrt()

        ### Calculate Class-Wise Cov
        N = all_feat.size(0)
        C = self.num_labeled_classes
        A = all_feat.size(1)

        NxCxFeatures = all_feat.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C)  # .cuda()
        onehot.scatter_(1, all_labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        class_cov = torch.bmm(var_temp.permute(1, 2, 0),
                              var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        if self.mode == 'dynamic' and current_epoch != 0:
            self.flying_mean = class_mean.cuda()
            self.flying_sig = class_sig.cuda()
            self.flying_cov = class_cov.cuda()

        this_epoch = {
                        'class_mean': class_mean.cpu().numpy(),
                        'class_sig': class_sig.cpu().numpy(),
                        'class_cov': class_cov.cpu().numpy(),
                        'all_feats': all_feat.numpy(),
                        'all_labels': all_labels.numpy()
                       }
        if save_featmap:
            scipy.io.savemat(self.file_prefix+'{}.mat'.format(current_epoch), this_epoch)
        print("Class mean, sig, cov, feats, labels saved at epoch [{}]".format(current_epoch))
        # return this_epoch
        # comment above to save memory
        return True

    def sample_labeled_features(self, dataset_name):
        if self.mode == 'static':
            print("This centroid tracker in in static mode. It can't sample feantures.")
            return False
        if self.flying_mean is None or self.flying_sig is None or self.flying_cov is None:
            print("Centroid tracker does not have correct statistics, pls check")
            return False

        feats = []
        labels = []

        if dataset_name == 'cifar10':
            num_per_class = 20
        elif dataset_name == 'cifar100':
            num_per_class = 2
        else:
            num_per_class = 3

        for i in range(self.num_labeled_classes):
            dist = torch.distributions.Normal(self.flying_mean[i], self.flying_sig.mean(dim=0))
            this_feat = dist.sample((num_per_class,)).cuda()  # new API
            this_label = torch.ones(this_feat.size(0)).cuda() * i

            feats.append(this_feat)
            labels.append(this_label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0).long()
        # print("..... sampled dynamic centroids in the current feature space.")

        return feats.cuda(), labels.cuda()

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter
from skimage.measure import block_reduce
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs

def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att

def map_attention_to_inputs(att_map, ori_inputs, processor):
    """
    Maps attention weights to the corresponding image tokens in the original inputs.
    
    This function takes an attention map (2D array) and maps it to the image token positions
    in the original inputs, creating a weight vector that can be used to identify important
    image tokens.
    
    Args:
        att_map: 2D numpy array representing attention weights (shape: [h, w])
        ori_inputs: Original inputs from prepare_qwen2_5_input containing input_ids and image_grid_thw
        processor: The processor used to prepare the inputs
        
    Returns:
        tuple: (image_token_weights, image_token_positions)
            - image_token_weights: 1D tensor with attention weights for each image token
            - image_token_positions: tuple of (start_pos, end_pos) indicating image token range
    """
    
    # Get the shape of the attention map
    att_h, att_w = att_map.shape
    
    # Get vision token IDs
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    # Find the position range of image tokens in input_ids
    input_ids = ori_inputs['input_ids'].tolist()[0]
    start_pos = input_ids.index(vision_start_token_id) + 1
    end_pos = input_ids.index(vision_end_token_id)
    
    # Flatten the attention map to match the image token sequence
    # The attention map is reshaped to match the spatial arrangement of image tokens
    att_flat = att_map.flatten()  # Shape: [h*w]
    
    # Convert numpy array to torch tensor and create image token weights
    att_tensor = torch.from_numpy(att_flat).float()
    image_token_weights = torch.zeros(end_pos - start_pos, dtype=torch.float32)
    
    print(f'att_flat:{len(att_flat)}\nimage_token_weights:{len(image_token_weights)}')
    
    # Map attention weights to image tokens
    # Note: The mapping depends on how the image tokens are arranged spatially
    # For Qwen2.5VL, image tokens are typically arranged in row-major order
    min_length = min(len(att_tensor), len(image_token_weights))
    image_token_weights[:min_length] = att_tensor[:min_length]
    
    return image_token_weights, (start_pos, end_pos)

def get_important_image_tokens(att_map, ori_inputs, processor, threshold=None, top_k=None):
    """
    Identifies the most important image tokens based on attention weights.
    
    Args:
        att_map: 2D numpy array representing attention weights
        ori_inputs: Original inputs from prepare_qwen2_5_input
        processor: The processor used to prepare the inputs
        threshold: Optional threshold for filtering tokens (tokens with weight > threshold)
        top_k: Optional number of top tokens to select (if specified, threshold is ignored)
        
    Returns:
        dict: Contains information about important tokens
            - 'weights': tensor of attention weights for image tokens
            - 'positions': tuple of (start_pos, end_pos) for image token range
            - 'important_indices': indices of important tokens (relative to image token start)
            - 'important_weights': weights of important tokens
    """
    
    # 将att_map的值映射到image_token_weights
    image_token_weights, image_positions = map_attention_to_inputs(att_map, ori_inputs, processor)
    
    if top_k is not None:
        # Select top-k tokens
        top_k = min(top_k, len(image_token_weights))
        important_indices = torch.topk(image_token_weights, top_k).indices
        important_weights = image_token_weights[important_indices]
    elif threshold is not None:
        # Select tokens above threshold
        print(f'选取值大于1的token')
        important_mask = image_token_weights > threshold # bool
        # 返回所有满足 important_mask 为 True 的元素的索引
        important_indices = torch.where(important_mask)[0]
        important_weights = image_token_weights[important_indices]
    else:
        # Return all tokens
        important_indices = torch.arange(len(image_token_weights))
        important_weights = image_token_weights
    
    return {
        'weights': image_token_weights,
        'positions': image_positions,
        'important_indices': important_indices,
        'important_weights': important_weights
    }

def create_attention_mask(att_map, ori_inputs, important_tokens_info, processor, threshold=None, top_k=None):
    """
    Creates a binary mask for important image tokens based on attention weights.
    
    Args:
        att_map: 2D numpy array representing attention weights
        ori_inputs: Original inputs from prepare_qwen2_5_input
        processor: The processor used to prepare the inputs
        threshold: Optional threshold for filtering tokens
        top_k: Optional number of top tokens to select
        
    Returns:
        torch.Tensor: Binary mask with 1 for important tokens, 0 for others
    """
    
    # important_tokens_info = get_important_image_tokens(att_map, ori_inputs, processor, threshold, top_k)
    
    # Create a mask for the entire input sequence
    input_length = ori_inputs['input_ids'].shape[1]
    print(f'input_length:{input_length}')
    mask = torch.zeros(input_length, dtype=torch.bool)
    
    start_pos, end_pos = important_tokens_info['positions']
    important_indices = important_tokens_info['important_indices']
    
    # Set mask to True for important image tokens
    for idx in important_indices:
        if start_pos + idx < end_pos:
            mask[start_pos + idx] = True
    
    return mask

def visualize_attention_mapping(att_map, ori_inputs, processor, important_tokens_info,  top_k=None):
    """
    Visualizes the mapping between attention map and image tokens.
    
    Args:
        att_map: 2D numpy array representing attention weights
        ori_inputs: Original inputs from prepare_qwen2_5_input
        processor: The processor used to prepare the inputs
        top_k: Number of top tokens to highlight
        
    Returns:
        dict: Visualization information
    """
    
    # important_tokens_info = get_important_image_tokens(att_map, ori_inputs, processor, top_k=top_k)
    
    # Get input tokens for visualization
    input_ids = ori_inputs['input_ids'].tolist()[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    start_pos, end_pos = important_tokens_info['positions']
    
    # Create visualization data
    visualization = {
        'attention_map_shape': att_map.shape,
        'image_token_range': (start_pos, end_pos),
        'total_image_tokens': end_pos - start_pos,
        'top_k_tokens': top_k,
        'top_attention_weights': important_tokens_info['important_weights'].tolist(),
        'top_token_positions': (important_tokens_info['important_indices'] + start_pos).tolist(),
        'sample_tokens': tokens[start_pos:start_pos+10] if start_pos + 10 <= end_pos else tokens[start_pos:end_pos]
    }
    
    return visualization

def is_similar(str1, str2, threshold=0.7):
        """判断两个字符串是否语义相似"""
        
        similarity = SequenceMatcher(None, str1, str2).ratio()
        return similarity >= threshold
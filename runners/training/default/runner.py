import math
import torch
from runners.common.branch_utils import get_branch_specific_objects
from core.run.metric_logger.context import get_logger
from runners.interface import BaseRunner
import numpy as np


def make_gaussian(size, fwhm = 5, center=None, norm=False):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    if norm:
        max_num = np.max(gaussian)
        gaussian = gaussian/max_num

    return gaussian



class DefaultTrainer(BaseRunner):
    def __init__(self, criteria, optimizer,
                 lr_scheduler_per_iteration,
                 lr_scheduler_per_epoch, loss_composer,
                 grad_max_norm=None, iteration_step=1):
        self.criteria = criteria
        self.loss_composer = loss_composer

        self.optimizer = optimizer
        self.lr_scheduler_per_iteration = lr_scheduler_per_iteration
        self.lr_scheduler_per_epoch = lr_scheduler_per_epoch

        self.grad_max_norm = grad_max_norm

        self.data_pipeline_on_host = None
        self.branch_name = None
        self.is_train = True
        self.iteration_index = 0
        self.iteration_step = iteration_step

        self.gaussian_map = torch.zeros((196, 196)).cuda()
        for ctr in range(196):
            ctr_y = ctr // 14
            ctr_x = ctr - ctr_y * 14
            self.gaussian_map[:, ctr] = torch.from_numpy(make_gaussian(14, center=(ctr_x, ctr_y))).flatten().cuda()

        self.gaussian_map_large = torch.zeros((576, 576)).cuda()
        for ctr in range(576):
            ctr_y = ctr // 24
            ctr_x = ctr - ctr_y * 24
            self.gaussian_map_large[:, ctr] = torch.from_numpy(make_gaussian(24, center=(ctr_x, ctr_y))).flatten().cuda()

    def get_iteration_index(self):
        return self.iteration_index

    def register_data_pipelines(self, branch_name, data_pipelines):
        if 'data_pipeline' not in data_pipelines:
            return
        if self.data_pipeline_on_host is None:
            self.data_pipeline_on_host = {}
        if branch_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[branch_name] = []
        for data_pipeline in data_pipelines['data_pipeline']:
            self.data_pipeline_on_host[branch_name].append(data_pipeline)

    def get_metric_definitions(self):
        if self.is_train:
            runner_metric_definitions = {
                'local': [
                    {'name': 'lr', 'window_size': 1, 'fmt': '{value:.6f}'},
                    {'name': 'loss'}
                ]}
        else:
            runner_metric_definitions = {
                'local': [
                    {'name': 'loss'}
                ]}
        metric_definitions = [runner_metric_definitions]
        data_pipelines = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        if data_pipelines is not None:
            for data_pipeline in data_pipelines:
                if hasattr(data_pipeline, 'get_metric_definitions'):
                    metric_definitions.append(data_pipeline.get_metric_definitions())
        return metric_definitions

    def switch_branch(self, branch_name):
        self.branch_name = branch_name

    def train(self, is_train):
        self.is_train = is_train

    def class_score_apply_gaussian(self, class_label, positive_batch, positive_dim):
        if class_label.shape[-1] == 196:
            size = [14, 14]
            gaussian_map = self.gaussian_map
        elif class_label.shape[-1] == 576:
            size = [24, 24]
            gaussian_map = self.gaussian_map_large

        positive_dim_y = torch.div(positive_dim, size[0], rounding_mode = 'floor')
        positive_dim_x = positive_dim - positive_dim_y * size[0]

        batch, _ = class_label.shape
        for b in range(batch):

            batch_slice = (positive_batch == b)
            x_min, _ = torch.min(positive_dim_x[batch_slice], dim=-1)
            x_max, _ = torch.max(positive_dim_x[batch_slice], dim=-1)
            y_min, _ = torch.min(positive_dim_y[batch_slice], dim=-1)
            y_max, _ = torch.max(positive_dim_y[batch_slice], dim=-1)

            xc = int((x_min + x_max)/2)
            yc = int((y_min + y_max)/2)

            class_label[b] = class_label[b] * gaussian_map[:, xc + yc * size[0]]

        return class_label

    def get_gt_bbox(self, samples, targets):
        # return B，196, 5
        if samples is not None and targets is not None:
            if isinstance(samples, (tuple, list)):
                b, c, h, w = samples[0].shape
            elif isinstance(samples, dict):
                b, c, h, w = samples['z'].shape
            else:
                b, c, h, w = samples.shape

            gt_bbox = torch.zeros((b, int(w*h/64), 4)).cuda().float()  # b, 196, 4
            # class_label = self.class_score_apply_gaussian(targets['class_label'], targets['positive_sample_batch_dim_index'], targets['positive_sample_feature_map_dim_index'])
            gt_bbox[targets['positive_sample_batch_dim_index'], targets['positive_sample_feature_map_dim_index']] = targets['bounding_box_label']
            class_label = targets['class_label']
            gt_bbox = torch.concat([gt_bbox, class_label.unsqueeze(-1)], dim=-1)

        else:
            gt_bbox = None
        return gt_bbox

    def run_iteration(self, model, data):
        samples, targets, miscellanies_on_host, miscellanies_on_device = data
        data_pipeline_on_host = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        if data_pipeline_on_host is not None:
            for data_pipeline in data_pipeline_on_host:
                if hasattr(data_pipeline, 'pre_processing'):
                    samples, targets, miscellanies_on_host, miscellanies_on_device = data_pipeline.pre_processing(samples, targets, miscellanies_on_host, miscellanies_on_device)

        gt_bbox = self.get_gt_bbox(samples, targets)
        outputs = None
        loss = None
        if samples is not None:
            if isinstance(samples, (tuple, list)):
                outputs = model(*samples, gt_bbox=gt_bbox)
            elif isinstance(samples, dict):
                outputs = model(**samples, gt_bbox=gt_bbox)
            else:
                outputs = model(samples, gt_bbox=gt_bbox)
            if targets is not None:
                loss, loss_value, loss_stats = self.loss_composer(self.criteria(outputs, targets))
                get_logger().log({'loss': loss_value, **loss_stats})

                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Loss is {loss_value}, stopping training\n{loss_stats}")

        if data_pipeline_on_host is not None:
            for data_pipeline in reversed(data_pipeline_on_host):
                if hasattr(data_pipeline, 'post_processing'):
                    outputs = data_pipeline.post_processing(outputs)

        if loss is not None and self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_max_norm)
            self.optimizer.step()
            if self.lr_scheduler_per_iteration is not None:
                self.lr_scheduler_per_iteration.step()

            get_logger().log({'lr': self.optimizer.param_groups[0]["lr"]})

        if self.is_train:
            self.iteration_index += self.iteration_step

    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict(), 'iter': self.iteration_index}
        if self.lr_scheduler_per_iteration is not None:
            state_dict['lr_scheduler_per_iteration'] = self.lr_scheduler_per_iteration.state_dict()
        if self.lr_scheduler_per_epoch is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler_per_epoch.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler_per_iteration is not None:
            self.lr_scheduler_per_iteration.load_state_dict(state_dict['lr_scheduler_per_iteration'])
        if self.lr_scheduler_per_epoch is not None:
            self.lr_scheduler_per_epoch.load_state_dict(state_dict['lr_scheduler'])
        self.iteration_index = state_dict['iter']

    def on_device_changed(self, device):
        self.criteria.to(device)

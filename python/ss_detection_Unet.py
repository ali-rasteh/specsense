import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import cv2




class NumpyDataset(Dataset):

    def __init__(self, dataset_path, mode='segmentation', norm_mode_data='std', norm_mode_mask='none', norm_mode_bbox='len', mask_mode='binary'):
        dataset = np.load(dataset_path)

        self.data = dataset['data']
        self.masks = dataset['masks']
        self.bboxes = dataset['bboxes']
        self.objectnesses = dataset['objectnesses']
        self.classes = dataset['classes']

        self.mode = mode
        self.norm_mode_data = norm_mode_data
        self.norm_mode_mask = norm_mode_mask
        self.norm_mode_bbox = norm_mode_bbox
        self.mask_mode = mask_mode

        # self.data_transform = data_transform
        # self.mask_transform = mask_transform

        self.min_data = self.data.min()
        self.max_data = self.data.max()
        self.mean_data = self.data.mean()
        self.std_data = self.data.std()
        self.min_masks = self.masks.min()
        self.max_masks = self.masks.max()
        self.mean_masks = self.masks.mean()
        self.std_masks = self.masks.std()
        self.min_bboxes = self.bboxes.min()
        self.max_bboxes = self.bboxes.max()
        self.mean_bboxes = self.bboxes.mean()
        self.std_bboxes = self.bboxes.std()
        
        if self.norm_mode_data=='max':
            self.data = (self.data - self.min_data) / (self.max_data - self.min_data)
        elif self.norm_mode_data=='std':
            self.data = (self.data - self.mean_data) / self.std_data
        elif self.norm_mode_data=='max&std':
            self.data = (self.data - self.min_data) / (self.max_data - self.min_data)
            self.data = (self.data - self.data.mean()) / self.data.std()
        elif self.norm_mode_data=='none':
            pass

        if self.norm_mode_mask=='max':
            self.masks = (self.masks - self.min_masks) / (self.max_masks - self.min_masks)
        elif self.norm_mode_mask=='std':
            self.masks = (self.masks - self.mean_masks) / self.std_masks
        elif self.norm_mode_mask=='max&std':
            self.masks = (self.masks - self.min_masks) / (self.max_masks - self.min_masks)
            self.masks = (self.masks - self.masks.mean()) / self.masks.std()
        elif self.norm_mode_mask=='none':
            pass
        
        if self.norm_mode_bbox=='max':
            self.bboxes = (self.bboxes - self.min_bboxes) / (self.max_bboxes - self.min_bboxes)
        elif self.norm_mode_bbox=='std':
            self.bboxes = (self.bboxes - self.mean_bboxes) / self.std_bboxes
        elif self.norm_mode_bbox=='max&std':
            self.bboxes = (self.bboxes - self.min_bboxes) / (self.max_bboxes - self.min_bboxes)
            self.bboxes = (self.bboxes - self.bboxes.mean()) / self.bboxes.std()
        elif self.norm_mode_bbox=='len':
            self.bboxes = self.bboxes/max(self.data.shape[1:])
        elif self.norm_mode_bbox=='none':
            pass


    def __len__(self):
        return len(self.data)
    

    def _get_masks_zeroone_ratio(self):
        n_zeros = (self.masks == 0).sum()
        n_ones = (self.masks == 1).sum()
        return n_zeros / n_ones
    

    def _transform(self, data, min=0, max=1, mean=0, std=1, norm_mode='std'):
        if self.mask_mode=='channels':
            data_t = torch.tensor(data, dtype=torch.float32)
        else:
            data_t = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        # if self.data_transform:
        #     data = self.data_transform(data)
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)

        return data_t


    def __getitem__(self, idx):
        data = self.data[idx]
        mask = self.masks[idx]
        data = self._transform(data)
        mask = self._transform(mask)
        
        if self.mode=='segmentation':
            return data, (mask,)
        elif self.mode=='detection':
            bbox = self.bboxes[idx]
            objectness = self.objectnesses[idx]
            classes = self.classes[idx]
            bbox = self._transform(bbox)
            objectness = self._transform(objectness)
            classes = self._transform(classes)
            return data, (mask, bbox, objectness, classes)




# Custom binarization function using Straight-Through Estimator
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thr=0.0):
        ctx.save_for_backward(input)
        return (input > thr).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0.0] = 0.0
        grad_input[input > 1.0] = 0.0
        return grad_input, None
    


class UNet(nn.Module):
    def __init__(self, shape=(1024,), n_layers=10, dim=1, n_out_channels=1, n_classes=1, mode='segmentation'):
        super(UNet, self).__init__()

        self.shape = shape
        self.n_layers = n_layers
        self.dim = dim
        self.n_out_channels = n_out_channels
        self.mode = mode
        self.n_classes = n_classes

        def Conv(in_channels=2, out_channels=1, kernel_size=1, padding=0):
            if self.dim == 1:
                return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            elif self.dim == 2:
                return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            elif self.dim == 3:
                return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        def ConvTranspose(in_channels=1, out_channels=1, kernel_size=2, stride=2):
            if self.dim == 1:
                return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            elif self.dim == 2:
                return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            elif self.dim == 3:
                return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        def MaxPool(kernel_size=2):
            if self.dim == 1:
                return nn.MaxPool1d(kernel_size=kernel_size)
            elif self.dim == 2:
                return nn.MaxPool2d(kernel_size=kernel_size)
            elif self.dim == 3:
                return nn.MaxPool3d(kernel_size=kernel_size)

        def BatchNorm(num_features=1):
            if self.dim == 1:
                return nn.BatchNorm1d(num_features=num_features)
            elif self.dim == 2:
                return nn.BatchNorm2d(num_features=num_features)
            elif self.dim == 3:
                return nn.BatchNorm3d(num_features=num_features)
            
        def conv_block(in_channels=1, out_channels=1, kernel_size=3):
            return nn.Sequential(
                Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.ModuleList([])
        for i in range(self.n_layers):
            self.encoder.append(conv_block(in_channels=2**i, out_channels=2**(i+1)))

        self.pool = MaxPool(kernel_size=2)

        self.middle = conv_block(in_channels=2**(self.n_layers), out_channels=2**(self.n_layers+1))

        self.up = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(self.n_layers):
            self.up.append(ConvTranspose(in_channels=2**(self.n_layers-i+1), out_channels=2**(self.n_layers-i), kernel_size=2, stride=2))
            self.decoder.append(conv_block(in_channels=2**(self.n_layers-i+1), out_channels=2**(self.n_layers-i)))

        self.out_conv = Conv(in_channels=2, out_channels=self.n_out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # self.fc_bbox = nn.Linear(int(np.prod(self.shape)), 2*len(self.shape) * self.n_classes)
        # self.fc_obj = nn.Linear(int(np.prod(self.shape)), self.n_classes)
        # self.fc_class = nn.Linear(int(np.prod(self.shape)), self.n_classes)


    def forward(self, x):
        enc = []
        dec = []
        enc.append(self.encoder[0](x))
        for i in range(1,self.n_layers):
            enc.append(self.encoder[i](self.pool(enc[-1])))

        middle = self.middle(self.pool(enc[-1]))

        dec.append(self.up[0](middle))
        dec[-1] = torch.cat((dec[-1], enc[-1]), dim=1)
        dec[-1] = self.decoder[0](dec[-1])
        for i in range(1,self.n_layers):
            dec.append(self.up[i](dec[-1]))
            dec[-1] = torch.cat((dec[-1], enc[self.n_layers-i-1]), dim=1)
            dec[-1] = self.decoder[i](dec[-1])

        output = self.out_conv(dec[-1])
        # output = self.sigmoid(self.out_conv(dec[-1]))
        return (output,)
    
        # if self.mode=='segmentation':
        #     return (output,)
        
        # elif self.mode=='detection':
        #     output_flat = torch.flatten(output, start_dim=2)
        #     bboxes  = self.fc_bbox(output_flat)
        #     objectnesses = self.fc_obj(output_flat)
        #     # class_probs = self.fc_class(output_flat)
        #     class_probs = None
        
        #     return (bboxes, objectnesses, class_probs)


class UNetDetectionHead(nn.Module):
    def __init__(self, shape=(1024,), dim=1, n_classes=1):
        super(UNetDetectionHead, self).__init__()

        self.shape = shape
        self.dim = dim
        self.n_classes = n_classes

        # self.fc_bbox = nn.Linear(int(np.prod(self.shape)), 2*len(self.shape) * self.n_classes)
        self.fc_bbox = nn.Sequential(
                nn.Linear(int(np.prod(self.shape)), int(np.prod(self.shape))//100),
                nn.ReLU(),
                nn.Linear(int(np.prod(self.shape))//100, 2*len(self.shape) * self.n_classes))
        # self.fc_obj = nn.Linear(int(np.prod(self.shape)), self.n_classes)
        self.fc_obj = nn.Sequential(
                nn.Linear(int(np.prod(self.shape)), int(np.prod(self.shape))//100),
                nn.ReLU(),
                nn.Linear(int(np.prod(self.shape))//100, self.n_classes))
        # self.fc_class = nn.Linear(int(np.prod(self.shape)), self.n_classes)

    def forward(self, x):
        x_flat = torch.flatten(x, start_dim=2)
        bboxes  = self.fc_bbox(x_flat)
        objectnesses = self.fc_obj(x_flat)
        # class_probs = self.fc_class(x_flat)
        class_probs = None
    
        return (bboxes, objectnesses, class_probs)
    


class UNetWithDetectionHead(nn.Module):
    def __init__(self, unet, dethead, problem_mode='detection', train_mode='end2end', mask_thr=0.0):
        super(UNetWithDetectionHead, self).__init__()
        self.unet = unet
        self.dethead = dethead
        self.problem_mode = problem_mode
        self.train_mode = train_mode
        self.mask_thr = mask_thr

    def forward(self, x):
        if self.train_mode=='end2end':
            mask = self.unet(x)[0]
            # mask = torch.sigmoid(mask)
            mask = (mask > self.mask_thr).float()
        elif self.train_mode=='separate':
            with torch.no_grad():
                mask = self.unet(x)[0]
                # mask = torch.sigmoid(mask)
                mask = (mask > self.mask_thr).float()
        det_output = self.dethead(mask)
    
        return det_output
        
    
class ObjectDetectionLoss(nn.Module):
    def __init__(self, shape=(1024,), lambda_start=1.0, lambda_length=1.0, lambda_obj=1.0, lambda_class=1.0, n_sigs_max=1, eval_smooth=1e-6, mask_thr=0.0, gt_mask_thr=0.5):
        super(ObjectDetectionLoss, self).__init__()
        self.shape = shape
        self.n_sigs_max = n_sigs_max
        self.eval_smooth = eval_smooth
        self.mask_thr = mask_thr
        self.gt_mask_thr = gt_mask_thr
        self.lambda_start = lambda_start  # Weighting factor for the start of the bounding box
        self.lambda_length = lambda_length  # Weighting factor for the length of the bounding box
        self.lambda_obj = lambda_obj  # Weighting factor for the no object loss
        self.lambda_class = lambda_class  # Weighting factor for the no object loss
        
        self.bbox_loss = nn.SmoothL1Loss()  # Smooth L1 loss for bounding box regression
        self.objectness_loss = nn.BCEWithLogitsLoss()  # BCE loss for objectness
        self.class_loss = nn.CrossEntropyLoss()  # Cross-entropy loss for class probabilities

    def forward(self, pred, gt):
        (pred_bbox, pred_objectness, pred_classes) = pred
        (gt_bbox, gt_objectness, gt_classes) = gt

        # Objectness loss
        objectness_loss = self.objectness_loss(pred_objectness, gt_objectness)

        # Class probability loss
        # class_loss = self.class_loss(pred_classes, gt_classes)
        class_loss = 0


        pred_objectness = (pred_objectness > self.mask_thr).float()
        # pred_objectness = torch.sigmoid(pred_objectness)
        
        batch_size = pred_bbox.shape[0]
        pred_bbox = pred_bbox.reshape((batch_size, self.n_sigs_max, -1))
        gt_bbox = gt_bbox.reshape((batch_size, self.n_sigs_max, -1))
        pred_objectness = pred_objectness.reshape((batch_size, self.n_sigs_max))
        gt_objectness = gt_objectness.reshape((batch_size, self.n_sigs_max))

        pred_bbox *= pred_objectness[:,:,None]
        gt_bbox *= gt_objectness[:,:,None]

        pred_starts = pred_bbox[:,:,:len(self.shape)]
        pred_lengths = pred_bbox[:,:,len(self.shape):]
        gt_starts = gt_bbox[:,:,:len(self.shape)]
        gt_lengths = gt_bbox[:,:,len(self.shape):]
        pred_stops = pred_starts + pred_lengths
        gt_stops = gt_starts + gt_lengths

        # intersection_start = torch.max(pred_starts, gt_starts)
        # intersection_stop = torch.min(pred_stops, gt_stops)
        # intersection = intersection_stop-intersection_start
        # # intersection = torch.clamp(intersection, min=0.0, max=1.0)
        # intersection_size = torch.prod(intersection, dim=-1)

        # union_start = torch.min(pred_starts, gt_starts)
        # union_stop = torch.max(pred_stops, gt_stops)
        # union = union_stop-union_start
        # # union = torch.clamp(union, min=0.0, max=1.0)
        # union_size = torch.prod(union, dim=-1)

        # iou = ((intersection_size+self.eval_smooth)/(union_size+self.eval_smooth)).mean()
        # # Use lambda_start=1.0 and lambda_obj=1.0 for this loss
        # bbox_start_loss = -iou
        # bbox_length_loss = 0.0

        # Use lambda_start=10.0, lambda_length=1.0 and lambda_obj=1.0 for this loss
        # bbox_loss = self.bbox_loss(pred_bbox, gt_bbox)
        bbox_start_loss = self.bbox_loss(pred_starts, gt_starts)
        bbox_length_loss = self.bbox_loss(pred_lengths, gt_lengths)
        
        # Total loss
        total_loss = self.lambda_start * bbox_start_loss + self.lambda_length * bbox_length_loss + self.lambda_obj * objectness_loss + self.lambda_class * class_loss
        
        return total_loss
    



class ss_detection_Unet(object):
    
    def __init__(self, params=None):

        self.dataset_path = params.data_dir+params.dataset_name
        self.shape = params.shape
        self.n_sigs_max = params.n_sigs_max
        self.mask_mode = params.mask_mode
        self.eval_smooth = params.eval_smooth
        self.train_ratio = params.train_ratio
        self.val_ratio = params.val_ratio
        self.test_ratio = params.test_ratio
        self.seed = params.seed
        self.batch_size = params.batch_size
        self.n_layers = params.n_layers
        self.problem_mode = params.problem_mode
        self.lr = params.lr
        self.apply_pos_weight = params.apply_pos_weight
        self.mask_thr = params.mask_thr
        self.gt_mask_thr = params.gt_mask_thr
        self.draw_histogram = params.draw_histogram
        self.hist_thr = params.hist_thr
        self.hist_bins = params.hist_bins
        self.sched_gamma = params.sched_gamma
        self.sched_step_size = params.sched_step_size
        self.n_epochs_tot = params.n_epochs_tot
        self.n_epochs_unet = params.n_epochs_unet
        self.n_epochs_dethead = params.n_epochs_dethead
        self.nepoch_save = params.nepoch_save
        self.nbatch_log = params.nbatch_log
        self.norm_mode_data = params.norm_mode_data
        self.norm_mode_mask = params.norm_mode_mask
        self.norm_mode_bbox = params.norm_mode_bbox
        self.lambda_start=params.lambda_start
        self.lambda_length=params.lambda_length
        self.lambda_obj=params.lambda_obj
        self.lambda_class=params.lambda_class
        self.contours_min_area=params.contours_min_area
        self.contours_max_gap=params.contours_max_gap
        self.train = params.train
        self.det_mode = params.det_mode
        self.train_mode = params.train_mode
        self.test = params.test
        self.load_model_params = params.load_model_params
        self.random_str = params.random_str
        self.model_dir = params.model_dir
        self.model_name = params.model_name
        self.model_unet_name = params.model_unet_name
        self.figs_dir = params.figs_dir

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("Initialized U-net Instance.")


    def generate_model(self):
        if self.mask_mode=='channels':
            n_out_channels = self.n_sigs_max
        else:
            n_out_channels = 1

        self.model_unet = UNet(shape=self.shape, n_layers=self.n_layers, dim=len(self.shape), n_out_channels=n_out_channels, n_classes=self.n_sigs_max, mode=self.problem_mode)
        if self.problem_mode == 'segmentation':
            self.model_dethead = None
            self.model = self.model_unet
        elif self.problem_mode == 'detection' and self.det_mode=='contours':
            self.model_dethead = None
            self.model = self.model_unet
        elif self.problem_mode == 'detection' and self.det_mode=='nn':
            self.model_dethead = UNetDetectionHead(shape=self.shape, dim=len(self.shape), n_classes=self.n_sigs_max)
            self.model = UNetWithDetectionHead(unet=self.model_unet, dethead=self.model_dethead, problem_mode=self.problem_mode, train_mode=self.train_mode, mask_thr=self.mask_thr)
        print('Total Number of parameters in the model: {}'.format(self.count_parameters(self.model)))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Unet device: {}'.format(self.device))
        
        self.model_unet.to(self.device)
        if self.model_dethead is not None:
            self.model_dethead.to(self.device)
        self.model.to(self.device)

        print("Generated Neural network's model.")


    def load_model(self):
        if 'model' in self.load_model_params:
            self.model.load_state_dict(torch.load(self.model_dir+self.model_name))
            # self.model = torch.load(self.model_dir+self.model_name)
            print("Loaded Neural network model for the whole network.")
        if 'unet' in self.load_model_params:
            self.model_unet.load_state_dict(torch.load(self.model_dir+self.model_unet_name))
            print("Loaded Neural network model for the U-net.")
        

    def count_parameters(self, model=None):
        return sum(p.numel() for p in model.parameters())


    def load_optimizer(self):
        if self.mask_mode=='binary' or self.mask_mode=='channels':
            if self.apply_pos_weight:
                pos_weight=torch.tensor([self.dataset_zeroone_ratio]).to(self.device)
                print("Applied pos_weight to the loss function: {}".format(pos_weight))
            else:
                pos_weight=None
            if self.problem_mode=='segmentation':
                self.criterion_unet = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                # self.criterion_unet = nn.BCELoss()
                self.criterion_dethead = None
                self.criterion = self.criterion_unet
            elif self.problem_mode=='detection' and self.det_mode=='contours':
                self.criterion_unet = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.criterion_dethead = None
                self.criterion = self.criterion_unet
            elif self.problem_mode=='detection' and self.det_mode=='nn':
                self.criterion_unet = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.criterion_dethead = ObjectDetectionLoss(shape=self.shape, lambda_start=self.lambda_start, lambda_length=self.lambda_length, lambda_obj=self.lambda_obj, lambda_class=self.lambda_class, n_sigs_max=self.n_sigs_max, eval_smooth=self.eval_smooth, mask_thr=self.mask_thr, gt_mask_thr=self.gt_mask_thr)
                self.criterion = self.criterion_dethead
        elif self.mask_mode=='snr':
            self.criterion_unet = nn.MSELoss()
            self.criterion_dethead = None
            self.criterion = self.criterion_unet
        
        self.optimizer_unet = optim.Adam(self.model_unet.parameters(), lr=self.lr)
        self.scheduler_unet = optim.lr_scheduler.StepLR(self.optimizer_unet, step_size=self.sched_step_size, gamma=self.sched_gamma)
        if self.model_dethead is not None:
            self.optimizer_dethead = optim.Adam(self.model_dethead.parameters(), lr=self.lr)
            self.scheduler_dethead = optim.lr_scheduler.StepLR(self.optimizer_dethead, step_size=self.sched_step_size, gamma=self.sched_gamma)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma)

        print("Loaded optimizer.")


    def generate_data_loaders(self):
        self.dataset = NumpyDataset(dataset_path=self.dataset_path, mode=self.problem_mode, norm_mode_data=self.norm_mode_data, norm_mode_mask=self.norm_mode_mask, norm_mode_bbox=self.norm_mode_bbox, mask_mode=self.mask_mode)
        if self.apply_pos_weight:
            self.dataset_zeroone_ratio = self.dataset._get_masks_zeroone_ratio()
        else:
            self.dataset_zeroone_ratio = None

        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        torch.manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        print("Generated Dataloaders.")


    def extract_bbox(self, mask, min_area=1, max_gap=1):
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()
        # mask = mask.astype(np.uint8)

        ndims = len(mask.shape)
        bboxes = []
        objectnesses = 0.0
        max_area = 0

        if ndims == 1:
            bbox = [0,0]
            length = mask.shape[0]
            for x in range(length):
                if mask[x] == 1.0:
                    # Initialize bounds
                    min_x = x
                    max_x = x
                    component_area = 0

                    # Expand the bounds to include the entire object
                    stack = [x]
                    while stack:
                        cx = stack.pop()
                        if mask[cx] == 1.0:
                            mask[cx] = 0.0  # Mark as visited
                            component_area += 1

                            # Update bounding box coordinates
                            min_x = min(min_x, cx)
                            max_x = max(max_x, cx)

                            # Add neighboring pixels to the stack
                            # if cx > 0:
                            #     stack.append(cx - 1)
                            # if cx < length - 1:
                            #     stack.append(cx + 1)

                            # Add neighboring pixels to the stack including those within the gap size
                            for dx in range(-max_gap, max_gap + 1):
                                nx = cx + dx
                                if 0 <= nx < length and mask[nx] == 1.0:
                                    stack.append(nx)

                    if component_area >= min_area:
                        w = max_x - min_x + 1
                        bboxes.append([min_x, w])
                        area = w
                        if area > max_area:
                            max_area = area
                            bbox = bboxes[-1]
                            objectnesses = 1.0
        elif ndims == 2:
            height, width = mask.shape
            # Iterate over the mask to find bounding boxes
            for y in range(height):
                for x in range(width):
                    # If we find a pixel belonging to an object
                    if mask[y, x] == 1.0:
                        # Initialize bounds
                        min_x, min_y = x, y
                        max_x, max_y = x, y
                        component_area = 0

                        # Expand the bounds to include the entire object
                        stack = [[x, y]]
                        while stack:
                            cx, cy = stack.pop()
                            if mask[cy, cx] == 1.0:
                                mask[cy, cx] = 0.0  # Mark as visited
                                component_area += 1

                                # Update bounding box coordinates
                                min_x = min(min_x, cx)
                                min_y = min(min_y, cy)
                                max_x = max(max_x, cx)
                                max_y = max(max_y, cy)

                                # Add neighboring pixels to the stack
                                # if cx > 0:
                                #     stack.append([cx - 1, cy])
                                # if cx < width - 1:
                                #     stack.append([cx + 1, cy])
                                # if cy > 0:
                                #     stack.append([cx, cy - 1])
                                # if cy < height - 1:
                                #     stack.append([cx, cy + 1])

                                # Add neighboring pixels to the stack including those within the gap size
                                for dx in range(-max_gap, max_gap + 1):
                                    for dy in range(-max_gap, max_gap + 1):
                                        nx, ny = cx + dx, cy + dy
                                        if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] == 1.0:
                                            stack.append([nx, ny])

                        if component_area >= min_area:
                            w = max_x - min_x + 1
                            h = max_y - min_y + 1
                            bboxes.append([min_x, min_y, w, h])
                            area = w * h
                            if area > max_area:
                                max_area = area
                                bbox = bboxes[-1]
                                objectnesses = 1.0

        return (bbox, objectnesses)


    def extract_bbox_efficient(self, masks, min_area=1, max_gap=1):
        if not isinstance(masks, np.ndarray):
            masks = masks.cpu().numpy()

        bboxes_out = np.zeros((masks.shape[0], masks.shape[1], 2*len(self.shape)*self.n_sigs_max), dtype=float)
        objectnesses_out = np.zeros((masks.shape[0], masks.shape[1], self.n_sigs_max), dtype=float)
        classes_out = np.zeros((masks.shape[0], masks.shape[1], self.n_sigs_max), dtype=int)

        for batch_id, mask in enumerate(masks):
            for ch, mask_c in enumerate(mask):
                ndims = len(mask_c.shape)
                bboxes = []
                objectnesses = 0.0

                if ndims == 1:
                    # Convert to a 2D array for contour finding
                    mask_2d = mask_c.reshape(1, -1).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # if w >= min_area:  # Only consider boxes larger than min_area
                        bboxes.append([x, w])
                    
                    # Sort bounding boxes by their X coordinate
                    bboxes.sort()
                    # Merge boxes with gaps less than or equal to max_gap
                    merged_boxes = []
                    current_box = bboxes[0] if bboxes else None

                    for i in range(1, len(bboxes)):
                        next_box = bboxes[i]
                        # Check if the gap between current and next box is <= max_gap
                        if next_box[0] - (current_box[0] + current_box[1]) <= max_gap:
                            # Merge the boxes
                            new_x = current_box[0]
                            new_w = (next_box[0] + next_box[1]) - current_box[0]
                            current_box = (new_x, new_w)
                        else:
                            # Save the current box and move to the next
                            merged_boxes.append(current_box)
                            current_box = next_box
                    
                    # Add the last box
                    if current_box:
                        merged_boxes.append(current_box)
                    
                    merged_boxes = [item for item in merged_boxes if item[1] >= min_area]

                    # Find the largest bounding box
                    if merged_boxes:
                        # bboxes = max(merged_boxes, key=lambda b: b[1])
                        bboxes = sorted(merged_boxes, key=lambda b: b[1])
                    else:
                        bboxes = []
                    n_sigs = min(len(bboxes), self.n_sigs_max)
                    bboxes = bboxes[:n_sigs]
                    bboxes = bboxes + [[0,0]] * (self.n_sigs_max - n_sigs)
                    objectnesses = [1.0] * n_sigs + [0.0] * (self.n_sigs_max - n_sigs)
                    classes = [0] * n_sigs + [-1] * (self.n_sigs_max - n_sigs)
                

                # return (bbox, objectness)
                bboxes_out[batch_id, ch] = np.array(bboxes).flatten()
                objectnesses_out[batch_id, ch] = np.array(objectnesses).flatten()
                classes_out[batch_id, ch] = np.array(classes).flatten()

        bboxes_out /= max(self.shape)
        bboxes_out = torch.tensor(bboxes_out, dtype=torch.float32).to(self.device)
        objectnesses_out = torch.tensor(objectnesses_out, dtype=torch.float32).to(self.device)
        classes_out = torch.tensor(classes_out, dtype=torch.int32).to(self.device)
        return (bboxes_out, objectnesses_out, classes_out)


    def intersection_over_union(self, pred, target, mode='segmentation'):
        if mode=='segmentation' or mode=='detection-unet':
            pred_c = pred > self.mask_thr
            target_c = target > self.gt_mask_thr
            # target_c = target
            
            intersection = (pred_c & target_c).float().sum()
            union = (pred_c | target_c).float().sum()
            
            iou = (intersection + self.eval_smooth) / (union + self.eval_smooth)
            iou = iou.mean().item()

        elif 'detection' in mode:
            (pred_bbox, pred_objectness, pred_classes) = pred
            (gt_bbox, gt_objectness, gt_classes) = target
            # print(pred_bbox[:10])
            # print(gt_bbox[:10])

            pred_objectness = (pred_objectness > self.mask_thr).float()
            # gt_objectness = (gt_objectness > self.gt_mask_thr).float()

            batch_size = pred_bbox.shape[0]
            pred_bbox = pred_bbox.reshape((batch_size, self.n_sigs_max, -1))
            gt_bbox = gt_bbox.reshape((batch_size, self.n_sigs_max, -1))
            pred_objectness = pred_objectness.reshape((batch_size, self.n_sigs_max))
            gt_objectness = gt_objectness.reshape((batch_size, self.n_sigs_max))

            pred_bbox *= pred_objectness[:,:,None]
            gt_bbox *= gt_objectness[:,:,None]
            pred_bbox *= max(self.shape)
            gt_bbox *= max(self.shape)
            pred_bbox = pred_bbox.round().int()
            gt_bbox = gt_bbox.round().int()

            pred_starts = pred_bbox[:,:,:len(self.shape)]
            pred_lengths = pred_bbox[:,:,len(self.shape):]
            gt_starts = gt_bbox[:,:,:len(self.shape)]
            gt_lengths = gt_bbox[:,:,len(self.shape):]
            pred_stops = pred_starts + pred_lengths
            gt_stops = gt_starts + gt_lengths

            intersection_start = torch.max(pred_starts, gt_starts)
            intersection_stop = torch.min(pred_stops, gt_stops)
            intersection = intersection_stop-intersection_start
            intersection = torch.clamp(intersection, min=0, max=max(self.shape))
            intersection = torch.prod(intersection, dim=-1)

            union_start = torch.min(pred_starts, gt_starts)
            union_stop = torch.max(pred_stops, gt_stops)
            union = union_stop-union_start
            # union = pred_lengths + gt_lengths - intersection
            union = torch.clamp(union, min=0, max=max(self.shape))
            union = torch.prod(union, dim=-1)

            iou = ((intersection+self.eval_smooth)/(union+self.eval_smooth))
            iou = iou.mean().item()
            # raise InterruptedError("Interrupted manually.")

        return iou


    def dice_coefficient(self, pred, target):
        pred_c = pred > self.mask_thr
        target_c = target > self.gt_mask_thr
        
        intersection = (pred_c & target_c).float().sum()
        dice = (2. * intersection + self.eval_smooth) / (pred_c.float().sum() + target_c.float().sum() + self.eval_smooth)
        
        return dice.mean().item()


    def evaluate_model(self, model, data_loader, mode='segmentation'):
        model.eval()
        score = 0
        mask_energy = 0
        print("Dataloader length: {}".format(len(data_loader)))
        with torch.no_grad():
            for data, gt in data_loader:
                data = data.to(self.device)
                gt = tuple(gt[i].to(self.device) for i in range(len(gt)))
                if len(gt)==1:
                    gt = gt[0]
                else:
                    if mode=='detection-unet':
                        gt = gt[0]
                    else:
                        gt = gt[1:]
                output = model(data)
                if len(output)==1:
                    output = output[0]
                if mode=='detection-contours':
                    output = (output>self.mask_thr).float()
                    output = self.extract_bbox_efficient(output, min_area=self.contours_min_area, max_gap=self.contours_max_gap)
                if (mode=='segmentation' and (self.mask_mode=='binary' or self.mask_mode=='channels')) or mode=='detection-unet':
                    score += self.intersection_over_union(output, gt, mode=mode)
                elif mode=='segmentation' and self.mask_mode=='snr':
                    score += F.mse_loss(output, gt).item()
                    # score += ((gt.cpu().numpy()-output.cpu().numpy())**2).mean()
                    mask_energy += torch.mean(gt**2).item()
                elif 'detection' in mode:
                    score += self.intersection_over_union(output, gt, mode=mode)
                if self.draw_histogram:
                    plt.figure(figsize=(10, 6))
                    output_h=output.clone()
                    output_h=output_h[output_h>-1*self.hist_thr]
                    output_h[output_h<-1*self.hist_thr]=-1*self.hist_thr
                    output_h[output_h>self.hist_thr]=self.hist_thr
                    plt.hist(output_h.flatten().cpu().numpy(), bins=self.hist_bins, edgecolor='black')
                    plt.title('Histogram of Neural Network Outputs')
                    plt.xlabel('Output Values')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.savefig(self.figs_dir + 'NN_hist.pdf', format='pdf')
                    plt.show()
                    raise InterruptedError("Interrupted manually after plot.")
        if len(data_loader)==0:
            return 0.0
        score /= len(data_loader)
        mask_energy /= len(data_loader)
        if self.mask_mode=='snr':
            score = score/mask_energy

        return score


    def train_model(self):
        if self.train:
            print("Beginning to train the whole Neural Network...")
            if self.problem_mode=='segmentation':
                self.train_model_one(mode='segmentation')
            elif self.problem_mode=='detection' and self.det_mode=='contours':
                self.train_model_one(mode='detection-contours')
            elif self.problem_mode=='detection' and self.det_mode=='nn':
                if self.train_mode=='end2end':
                    self.train_model_one(mode='detection-end2end')
                elif self.train_mode=='separate':
                    self.train_model_one(mode='detection-unet')
                    for param in self.model_unet.parameters():
                        param.requires_grad = False
                    for param in self.model.unet.parameters():
                        param.requires_grad = False
                    self.train_model_one(mode='detection-dethead')


    def train_model_one(self, mode='segmentation'):
        if self.train:
            print("Beginning to train the Neural Network in mode: {}...".format(mode))
            if mode=='segmentation' or mode=='detection-end2end' or mode=='detection-contours':
                model = self.model
                name = ''
                criterion = self.criterion
                optimizer = self.optimizer
                scheduler = self.scheduler
                n_epochs = self.n_epochs_tot
            elif mode=='detection-unet':
                model = self.model_unet
                name = '_unet'
                criterion = self.criterion_unet
                optimizer = self.optimizer_unet
                scheduler = self.scheduler_unet
                n_epochs = self.n_epochs_unet
            elif mode=='detection-dethead':
                model = self.model
                name = ''
                criterion = self.criterion_dethead
                optimizer = self.optimizer_dethead
                scheduler = self.scheduler_dethead
                n_epochs = self.n_epochs_dethead


            for epoch in range(n_epochs):
                batch_id = 0
                model.train()
                epoch_loss = 0
                for data, gt in self.train_loader:
                    data = data.to(self.device)
                    gt = tuple(gt[i].to(self.device) for i in range(len(gt)))
                    if len(gt)==1:
                        gt = gt[0]
                    else:
                        if mode=='detection-unet' or mode=='detection-contours':
                            gt = gt[0]
                        else:
                            gt = gt[1:]
                    optimizer.zero_grad()
                    output = model(data)
                    if len(output)==1:
                        output = output[0]
                    # output = BinarizeSTE.apply(output, self.mask_thr)
                    loss = criterion(output, gt)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    if (batch_id+1) % self.nbatch_log == 0 and self.nbatch_log != -1:
                        print(f"Batch {batch_id + 1}/{len(self.train_loader)}, Loss: {loss.item()}, lr: {scheduler.get_last_lr()}")
                    batch_id += 1

                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(self.train_loader)}, lr: {scheduler.get_last_lr()}\n")
                # self.test_model(mode='test')
                
                if ((epoch+1) % self.nepoch_save == 0 and self.nepoch_save != -1) or (epoch+1 == n_epochs):
                    torch.save(model.state_dict(), self.model_dir+self.random_str+name+'_weights_{}.pth'.format(epoch + 1))
                    torch.save(model, self.model_dir+self.random_str+name+'_model_{}.pth'.format(epoch + 1))
                    print("Saved the Neural Network's model")
                    self.test_acc = self.evaluate_model(model, self.test_loader, mode=mode)
                    print('Accuracy on test data: {}\n'.format(self.test_acc))

                scheduler.step()
    

    def test_model(self, mode='both', eval_mode=None):
        if self.test:
            print("Starting to test the Neural Network...")
            if eval_mode is None:
                if self.problem_mode=='segmentation':
                    eval_mode='segmentation'
                elif self.problem_mode=='detection' and self.det_mode=='contours':
                    eval_mode='detection-contours'
                elif self.problem_mode=='detection' and self.det_mode=='nn':
                    eval_mode='detection-end2end'
                else:
                    eval_mode=eval_mode
            else:
                eval_mode=eval_mode

            if mode=='both':
                self.train_acc = self.evaluate_model(self.model, self.train_loader, mode=eval_mode)
                print('Accuracy on train data: {}'.format(self.train_acc))
            self.test_acc = self.evaluate_model(self.model, self.test_loader, mode=eval_mode)
            print('Accuracy on test data: {}\n'.format(self.test_acc))




if __name__ == '__main__':
    from ss_detection_test import params_class
    params = params_class()
    ss_det_unet = ss_detection_Unet(params)
    # mask = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # mask = mask.reshape(1, 1, -1)
    mask = np.random.randint(0, 2, (40000,1,1024))
    bounding_boxes = ss_det_unet.extract_bbox_efficient(mask, min_area=2, max_gap=2)
    print("Bounding Boxes:", bounding_boxes)
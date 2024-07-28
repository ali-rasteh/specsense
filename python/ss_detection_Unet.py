import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional as F




class NumpyDataset(Dataset):

    def __init__(self, dataset_path, data_transform=None, mask_transform=None, norm_mode='max'):
        dataset = np.load(dataset_path)

        self.data = dataset['data']
        self.masks = dataset['masks']
        self.norm_mode = norm_mode

        self.data_transform = data_transform
        self.mask_transform = mask_transform

        self.min_data = self.data.min()
        self.max_data = self.data.max()
        self.mean_data = self.data.mean()
        self.std_data = self.data.std()
        self.min_masks = self.masks.min()
        self.max_masks = self.masks.max()
        self.mean_masks = self.masks.mean()
        self.std_masks = self.masks.std()


    def __len__(self):
        return len(self.data)
    

    def _transform(self, data, min=0, max=1, mean=0, std=1, norm_mode='max'):
        if norm_mode=='max':
            data_t = (data - min) / (max - min)
        elif norm_mode=='std':
            data_t = (data - mean) / std
        elif norm_mode=='none':
            data_t = data.copy()
        data_t = torch.tensor(data_t, dtype=torch.float32).unsqueeze(0)

        # if self.data_transform:
        #     data = self.data_transform(data)
        
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)

        return data_t


    def __getitem__(self, idx):
        data = self.data[idx]
        mask = self.masks[idx]
        data = self._transform(data, min=self.min_data, max=self.max_data, mean=self.mean_data, std=self.std_data, norm_mode=self.norm_mode)
        mask = self._transform(mask, min=self.min_masks, max=self.max_masks, mean=self.mean_masks, std=self.std_masks, norm_mode='none')

        return data, mask




class UNet1D(nn.Module):
    def __init__(self, n_layers = 10):
        super(UNet1D, self).__init__()

        self.n_layers = n_layers

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
                # nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            )

        self.encoder = nn.ModuleList([])
        for i in range(self.n_layers):
            self.encoder.append(conv_block(2**i, 2**(i+1)))

        self.pool = nn.MaxPool1d(2)

        self.middle = conv_block(2**(i+1), 2**(i+2))

        self.up = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(self.n_layers):
            self.up.append(nn.ConvTranspose1d(2**(self.n_layers-i+1), 2**(self.n_layers-i), kernel_size=2, stride=2))
            self.decoder.append(conv_block(2**(self.n_layers-i+1), 2**(self.n_layers-i)))

        self.out_conv = nn.Conv1d(2, 1, kernel_size=1)


    def forward(self, x):
        enc = []
        dec = []
        enc.append(self.encoder[0](x))
        # print(enc[-1].shape)
        for i in range(1,self.n_layers):
            enc.append(self.encoder[i](self.pool(enc[-1])))
            # print(enc[-1].shape)

        middle = self.middle(self.pool(enc[-1]))
        # print(1,middle.shape)

        dec.append(self.up[0](middle))
        # print(dec[-1].shape)
        dec[-1] = torch.cat((dec[-1], enc[-1]), dim=1)
        # print(dec[-1].shape)
        dec[-1] = self.decoder[0](dec[-1])
        # print(dec[-1].shape)
        for i in range(1,self.n_layers):
            dec.append(self.up[i](dec[-1]))
            # print(dec[-1].shape)
            dec[-1] = torch.cat((dec[-1], enc[self.n_layers-i-1]), dim=1)
            # print(dec[-1].shape)
            dec[-1] = self.decoder[i](dec[-1])
            # print(dec[-1].shape)

        return self.out_conv(dec[-1])





class ss_detection_Unet(object):
    
    def __init__(self, params=None):

        self.dataset_path = params.dataset_path
        self.eval_smooth = params.eval_smooth
        self.train_ratio = params.train_ratio
        self.val_ratio = params.val_ratio
        self.test_ratio = params.test_ratio
        self.seed = params.seed
        self.batch_size = params.batch_size
        self.n_layers = params.n_layers
        self.lr = params.lr
        self.sched_gamma = params.sched_gamma
        self.sched_step_size = params.sched_step_size
        self.n_epochs = params.n_epochs
        self.nepoch_save = params.nepoch_save
        self.nbatch_log = params.nbatch_log
        self.norm_mode = params.norm_mode
        self.train = params.train
        self.test = params.test
        self.load_model_params = params.load_model_params
        self.model_save_path = params.model_save_path
        self.model_load_path = params.model_load_path


        self.data_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("Initialized U-net Instance.")


    def generate_model(self):
        self.model = UNet1D(n_layers=self.n_layers)
        print('Number of parameters in the model: {}'.format(self.count_parameters()))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(self.device))
        
        self.model.to(self.device)

        print("Generated Neural network's model.")


    def load_model(self):
        if self.load_model_params:
            self.model.load_state_dict(torch.load(self.model_load_path))
            # self.model = torch.load(self.model_load_path)
            print("Loaded Neural network's model.")


    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


    def load_optimizer(self):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma)

        print("Loaded optimizer.")


    def generate_data_loaders(self):
        self.dataset = NumpyDataset(dataset_path=self.dataset_path, data_transform=self.data_transform, mask_transform=self.mask_transform, norm_mode=self.norm_mode)

        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        torch.manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        print("Generated Dataloaders.")


    def intersection_over_union(self, pred, target):
        pred = pred > 0.5
        target = target > 0.5
        
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
        
        iou = (intersection + self.eval_smooth) / (union + self.eval_smooth)
        return iou.mean().item()


    def dice_coefficient(self, pred, target):
        pred = pred > 0.5
        target = target > 0.5
        
        intersection = (pred & target).float().sum()
        dice = (2. * intersection + self.eval_smooth) / (pred.float().sum() + target.float().sum() + self.eval_smooth)
        
        return dice.mean().item()


    def evaluate_model(self, model, device, data_loader):
        model.eval()
        score = 0
        with torch.no_grad():
            for data, mask in data_loader:
                data = data.to(device)
                mask = mask.to(device)
                output = model(data)
                score += self.intersection_over_union(output, mask)
        score = score/len(data_loader)

        return(score)


    def train_model(self):
        print("Starting to train the Neural Network...")
        if self.train:
            for epoch in range(self.n_epochs):
                batch_id = 0
                self.model.train()
                epoch_loss = 0
                for data, mask in self.train_loader:
                    data = data.to(self.device)
                    mask = mask.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, mask)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    if (batch_id+1) % self.nbatch_log == 0 and self.nbatch_log != -1:
                        print(f"Batch {batch_id + 1}/{len(self.train_loader)}, Loss: {loss.item()}, lr: {self.scheduler.get_last_lr()}")
                    batch_id += 1

                self.scheduler.step()

                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss / len(self.train_loader)}, lr: {self.scheduler.get_last_lr()}\n")
                
                if (epoch+1) % self.nepoch_save == 0 and self.nepoch_save != -1:
                    torch.save(self.model.state_dict(), self.model_save_path+'model_weights_{}.pth'.format(epoch + 1))
                    torch.save(self.model, self.model_save_path+'model_{}.pth'.format(epoch + 1))
                    print("Saved the Neural Network's model")
                    test_acc = self.evaluate_model(self.model, self.device, self.test_loader)
                    print('Accuracy on test data: {}\n'.format(test_acc))

    
    def test_model(self):
        print("Starting to test the Neural Network...")
        if self.test:
            train_acc = self.evaluate_model(self.model, self.device, self.train_loader)
            print('Accuracy on train data: {}'.format(train_acc))

            test_acc = self.evaluate_model(self.model, self.device, self.test_loader)
            print('Accuracy on test data: {}\n'.format(test_acc))




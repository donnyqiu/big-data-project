import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from log import setup_logging
from model import get_model
import os

class Trainer:
    def __init__(self, 
                 distributed,
                 rank,
                 world_size,
                 epochs, 
                 batch_size, 
                 encoder_lr, 
                 head_lr, 
                 data_root,
                 save_model_path,
                 log_file, 
                 save_loss_path,
                 save_accuracy_path):
        
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr

        self.data_root = data_root
        self.save_model_path = save_model_path
        self.logger = setup_logging(log_file)
        self.save_loss_path = save_loss_path
        self.save_accuracy_path = save_accuracy_path

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        if self.distributed == "data_parallel":
            self.setup()

        self.device = torch.device(f'cuda:{self.rank}' if self.distributed == "data_parallel" and self.rank >= 0 else 'cuda:0')
        self.get_model_to_device()

        self.train_dataloader, self.test_dataloader = self.get_dataloaders()

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW([
            {'params': self.model.module.encoder.parameters() if self.distributed == "data_parallel" else self.model.encoder.parameters(), 'lr': self.encoder_lr},
            {'params': self.model.module.heads.parameters() if self.distributed == "data_parallel" else self.model.heads.parameters(), 'lr': self.head_lr}
        ])
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.2)

    def setup(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    def get_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.CIFAR100(root=self.data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root=self.data_root, train=False, download=True, transform=test_transform)

        if self.distributed == "data_parallel":
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            test_sampler = None

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

        return train_dataloader, test_dataloader

    def get_model_to_device(self):
        self.model = get_model()
        if self.distributed == "local":
            self.model.to(self.device)
        elif self.distributed == "data_parallel":
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        elif self.distributed == "model_parallel":
            device_ids = list(range(self.world_size))
            self.model.conv_proj = self.model.conv_proj.to(f'cuda:0')
            self.model.class_token.data = self.model.class_token.to(f'cuda:0')
            self.model.encoder.dropout = self.model.encoder.dropout.to(f'cuda:0')
            self.model.encoder.pos_embedding.data = self.model.encoder.pos_embedding.to(f'cuda:0')
            
            self.encoder_layer_num_per_device = [len(self.model.encoder.layers) // (len(device_ids) - 2)] * (len(device_ids) - 2)
            remain = len(self.model.encoder.layers) % (len(device_ids) - 2)
            for i in range(remain):
                self.encoder_layer_num_per_device[i] += 1
            self.encoder_layer_num_per_device = [0] + self.encoder_layer_num_per_device + [0]

            current_layer_index = 0
            for device_index, num_layers in enumerate(self.encoder_layer_num_per_device):
                for i in range(num_layers):
                    self.model.encoder.layers[current_layer_index + i] = self.model.encoder.layers[current_layer_index + i].to(f'cuda:{device_ids[device_index]}')
                current_layer_index += num_layers
            
            self.model.encoder.ln = self.model.encoder.ln.to(f'cuda:{device_ids[-1]}')
            self.model.heads = self.model.heads.to(f'cuda:{device_ids[-1]}')

            def forward_with_model_parallel(model_self, x):
                x = x.to(f'cuda:0')
                x = model_self._process_input(x)
                n = x.shape[0]

                batch_class_token = model_self.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                
                # encoder
                x += model_self.encoder.pos_embedding
                x = model_self.encoder.dropout(x)

                current_layer_index = 0
                for device_index, num_layers in enumerate(self.encoder_layer_num_per_device):
                    if num_layers == 0:
                        continue
                    x = x.to(f'cuda:{device_ids[device_index]}')
                    for i in range(num_layers):
                        x = model_self.encoder.layers[i + current_layer_index](x)
                    current_layer_index += num_layers

                
                x = x.to(f'cuda:{device_ids[-1]}')
                x = model_self.encoder.ln(x)

                # heads
                x = x[:, 0]
                output = model_self.heads(x)
                output = output.to(f'cuda:0')
                return output
            
            self.model.forward = forward_with_model_parallel.__get__(self.model, type(self.model))

    def cleanup(self):
        dist.destroy_process_group()

    def train(self):
        for epoch in range(self.epochs):
            if self.distributed == "data_parallel":
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data in self.train_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.scheduler.step()

            train_loss, train_accuracy = running_loss / len(self.train_dataloader), correct / total
            test_loss, test_accuracy = self.evaluate(self.test_dataloader)

            if not self.distributed == "data_parallel" or self.rank == 0:
                self.logger.info(f'[Epoch {epoch+1}] Train Loss: {train_loss:.4f}')
                self.logger.info(f'[Epoch {epoch+1}] Train Accuracy: {train_accuracy:.4f}')
                self.logger.info(f'[Epoch {epoch+1}] Test Loss: {test_loss:.4f}')
                self.logger.info(f'[Epoch {epoch+1}] Test Accuracy: {test_accuracy:.4f}')

                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_accuracy)
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)

                if (epoch + 1) % 5 == 0:
                    if self.distributed == "data_parallel":
                        torch.save(self.model.module.state_dict(), self.save_model_path + "/epoch_" + str(epoch+1) + "_model.pth")
                    else:
                        torch.save(self.model.state_dict(), self.save_model_path + "/epoch_" + str(epoch+1) + "_model.pth")
        
        if self.distributed == "data_parallel":
            self.cleanup()

        if not self.distributed == "data_parallel" or self.rank == 0:
            self.plot_loss_accuracy()

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(dataloader), correct / total

    def plot_loss_accuracy(self):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.train_losses, 'r', label='Training loss')
        plt.plot(epochs, self.test_losses, 'b', label='Test loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_loss_path)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.train_accuracies, 'r', label='Training accuracy')
        plt.plot(epochs, self.test_accuracies, 'b', label='Test accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.save_accuracy_path)
        plt.close()

import argparse
import os
import threading
import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.distributed.rpc as rpc
from torchvision import transforms
import torchvision
from model import get_model
from log import setup_logging

class ParameterServer(object):
    def __init__(self, num_workers, encoder_lr, heads_lr):
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.num_workers = num_workers
        self.model = get_model()
        self.encoder_lr = encoder_lr
        self.heads_lr = heads_lr
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.encoder_lr},
            {'params': self.model.heads.parameters(), 'lr': self.heads_lr}
        ])
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.2)
        self.epoch_barrier = threading.Barrier(self.num_workers)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        with self.lock:
            for p, g in zip(self.model.parameters(), grads):
                p.grad = g
            self.optimizer.step()
            self.optimizer.zero_grad()

            fut = self.future_model

            fut.set_result(self.model)
            self.future_model = torch.futures.Future()

        return fut


def run_worker(ps_rref, num_workers, rank, data_dir, batch_size, num_epochs):
    logger = setup_logging("logs/parameter_server/rank_" + str(rank) + "_train_log")

    # prepare dataset
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

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=num_workers, rank=rank-1, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=num_workers, rank=rank-1, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

    device_id = rank - 1
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # get initial model from the PS
    m = ps_rref.rpc_sync().get_model().to(device)

    for i in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_id, (data, target) in enumerate(train_loader):
            m.train()
            data, target = data.to(device), target.to(device)
            output = m(data)
            loss = criterion(output, target)
            loss.backward()

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            m = rpc.rpc_sync(to=ps_rref.owner(),
                            func=ParameterServer.update_and_fetch_model,
                            args=(ps_rref, [p.grad for p in m.cpu().parameters()])
                            ).to(device)
            

        m.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = m(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        logger.info(f'[Epoch {i+1}] Train Loss: {train_loss/len(train_loader):.4f}')
        logger.info(f'[Epoch {i+1}] Train Accuracy: {train_correct/train_total:.4f}')
        logger.info(f'[Epoch {i+1}] Test Loss: {test_loss/len(test_loader):.4f}')
        logger.info(f'[Epoch {i+1}] Test Accuracy: {train_correct/train_total:.4f}')
        logger.info(f'finish epoch {i+1}')

def main():
    parser = argparse.ArgumentParser(description="Train models under ASGD")
    parser.add_argument("--rank", type=int, default=0, help="Global rank of this process.")
    parser.add_argument("--world_size", type=int, default=2, help="Total number of workers.")
    parser.add_argument("--data_dir", type=str, default="./data", help="The location of dataset.")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="12345", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of each worker during training.")
    parser.add_argument("--encoder_lr", type=float, default=5e-7, help="Encoder learning rate.")
    parser.add_argument("--heads_lr", type=float, default=5e-5, help="Heads learning rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs.")

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    options = rpc.TensorPipeRpcBackendOptions(rpc_timeout=1e5)

    if args.rank == 0:
        rpc.init_rpc(f"PS{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options)

        ps_rref = rpc.RRef(ParameterServer(args.world_size-1, args.encoder_lr, args.heads_lr))

        futs = []
        for r in range(1, args.world_size):
            worker = f'worker{r}'
            futs.append(rpc.rpc_async(to=worker,
                                      func=run_worker,
                                      args=(ps_rref, args.world_size-1, r, args.data_dir, args.batch_size, args.num_epochs)))

        torch.futures.wait_all(futs)

    else:
        rpc.init_rpc(f"worker{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options)

    rpc.shutdown()


if __name__ == "__main__":
    main()
from tqdm.auto import tqdm
import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from data import load_datasets
from net import Net

# see https://github.com/pytorch/opacus/issues/612
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

LR = 0.001
EPOCHS = 1
EPSILON = 50.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.
BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128

privacy_engine = PrivacyEngine(
    accountant="rdp",
)

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

def test(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)


def main():
    print(f"Using device: {device}")
    trainloader, valloader, testloader = load_datasets('./dataset-cache', [1.0])
    trainloader, valloader = trainloader[0], valloader[0]

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    model = model.to(device)
    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device)

    top1_acc = test(model, testloader, device)

if __name__ == "__main__":
    main()

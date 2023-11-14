import torch
import sys
import network 

def train(model, device, train_loader, optimizer, epoch, switch, weight=0.5):
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, decoded, classification_output = model(data)

        encoder_loss = F.mse_loss(decoded, data)
        classifier_loss = F.nll_loss(classification_output, target)

        if switch == 'encoder_loss':
            alpha = 1
        elif switch == 'classifier_loss':
            alpha = 0
        elif switch == 'joint':
            alpha = weight
        else:
            print("Error, expected switch to be one of encoder_loss, classifier_loss or joint.")
            sys.exit()

        loss = (alpha * encoder_loss) + ((1-alpha) * classifier_loss)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, decoded, output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


def main(model, switch):
    torch.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss, accuracy = [], []

    for epoch in range(1, 10 + 1):
        train_losses = train(model, device, train_loader, optimizer, epoch, switch)
        test_loss, acc = test(model, device, test_loader)
        loss.append(train_losses)
        accuracy.append(acc)

    return loss, accuracy



random_seed = 456
learning_rate = 0.005
num_epochs = 10
batch_size = 128
torch.manual_seed(random_seed)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = network.ConvolutionalAutoencoder()

classifier_loss, classifier_accuracy = main(model, 'classifier_loss')
encoder_loss, encoder_accuracy = main(model, 'encoder_loss')
joint_loss, joint_accuracy = main(model, 'joint')
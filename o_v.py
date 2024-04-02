import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(num_epochs = 50, batch=16):
    model = CNN()

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()])

    train_data = datasets.ImageFolder('data/img', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
    torch.save(model.state_dict(), 'cnn_model.pth')

def test(test_loader, path= "cnn_model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0
    clases = ['Орел', 'Всадник']

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            print(clases[predicted[0]], '_', clases[outputs[0]])
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()

        print('Accuracy of the model based on the test set of',
              'inputs is: %d %%' % (100 * running_accuracy / total))

def classif(img_loader, path="cnn_model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(path))
    clases = ['Орел', 'Всадник']

    with torch.no_grad():
        for data in img_loader:
            inputs, outputs = data
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            print(clases[predicted[0]])

data_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
data = datasets.ImageFolder('data/test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
# test(test_loader)

data = datasets.ImageFolder('data/detec', transform=data_transform)
img_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
classif(img_loader)
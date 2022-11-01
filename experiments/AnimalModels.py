import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
import time

BATCH_SIZE = 32

# move the input and model to GPU for speed if available
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

AlexNet_Model = torch.hub.load("pytorch/vision:v0.6.0", "alexnet", pretrained=True)

new_features = torch.nn.Sequential(
    # conv1
    nn.Conv2d(
        1, 16, kernel_size=3, stride=2, padding=2
    ),  # 3表示输入的图像是3通道的。64表示输出通道数，也就是卷积核的数量。
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # conv2
    nn.Conv2d(16, 64, kernel_size=3, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # conv3
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    # nn.MaxPool2d(kernel_size=2, stride=2),
)
# AlexNet_Model.features[0] = nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=2)
# AlexNet_Model.features[3] = nn.Conv2d(64, 192, kernel_size=2, stride=2, padding=2)
# AlexNet_Model.features[6] = nn.Conv2d(192, 384, kernel_size=2, stride=2, padding=2)
# AlexNet_Model.features[8] = nn.Conv2d(384, 256, kernel_size=2, stride=2, padding=2)
# AlexNet_Model.features[10] = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=2)
AlexNet_Model.features = new_features

AlexNet_Model.classifier[1] = nn.Linear(64 * 6 * 6, 1024)
AlexNet_Model.classifier[4] = nn.Linear(1024, 256)
AlexNet_Model.classifier[6] = nn.Linear(256, 10)
AlexNet_Model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.001, momentum=0.9)


# inputs = torch.randn(32, 1, 32, 32).to(device)
# inputs = AlexNet_Model(inputs)
# print(inputs.shape)
# print()


def train(x, y):
    AlexNet_Model.eval()
    with torch.enable_grad():
        # x y tensor
        train_dataset = Data.TensorDataset(x, y)

        trainloader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )

        for epoch in range(5):  # loop over the dataset multiple times
            running_loss = 0.0
            start_time = time.time()

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.type(torch.LongTensor).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                print(inputs.shape)
                # forward + backward + optimize
                output = AlexNet_Model(inputs)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Time
                end_time = time.time()
                time_taken = end_time - start_time

                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                print("Time:", time_taken)
                running_loss = 0.0


def test(x, y):
    test_dataset = Data.TensorDataset(x, y)

    testloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # Testing Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = AlexNet_Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

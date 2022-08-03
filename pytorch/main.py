import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html

#데이터셋 다운로드
training_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download =True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform = ToTensor(),  #[0,1]값으로 변경
)

#print(test_data[0]) 한 행의 type는 튜플

batch_size = 64

#데이터셋을 데이터 로더 - 64개의 feature, lable를 묶은것
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W] : {X.shape}")
    print(f"Shape of y:{y.shape} {y.dtype}")
    break
    #데이터 불러오는법 https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html

#모델 만들기
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#모델 매개변수 최적화 - 손실 함수 & 옵티마이저
loss_fn = nn.CrossEntropyLoss()
#SGD 확률적 경사 하강법 /  lr = Learning Rate 미분값 이동 정도
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        #역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch - len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0;

    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)


            test_loss += loss_fn(pred ,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /=size
    print(f"Test Error :\n Accuracy: /{(100-correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def print_epochs():
    # 에폭 출력
    epochs = 5

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done")


#print_epochs()
#모델 학습법 https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html

#모델 저장
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

#모델 불러오기
classes = [
"T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x,y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

#저장하고 불러오는 법 https://tutorials.pytorch.kr/beginner/basics/saveloadrun_tutorial.html
















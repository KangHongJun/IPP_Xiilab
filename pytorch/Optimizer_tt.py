#모델 매개변수 최적화

#모델을 학습하는 과정은 반복적인 과정을 거친다. - epoch
#각 반복단계에서 모델은 출력을 추측, 추측과 정답 사이의 오류(loss)를 계산
#매개변수에 대한 오류의 도함수(derivative) 수집 후 경사하강법을 사용하여 파라미터들 optimizer

#pre-requisite

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import  datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

#Hyperparameter는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수
#서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율(covergence rate)에 영향을 미칠 수 있다.
#epoch, batch size, leaning rate정의
learning_rate = 1e-3
batch_size = 64
epochs = 5

#최적화 단계 - 모델 최적화 반복(iternation) : 에폭
#에폭은 학습단계(train loop), 검증/테스트 단계(validation, test loop) - 성능개선 테스트를 위해 반복

#손실 함수
#얻은 결과와 실제 값 사이의 틀린정도(degree of dissmilarity) 측정하고 이 값을 학습 중에 최소화
#회귀문제(regression task) -MSELoss(평균 제곱 오차:Mean Square Error)
#분류(classification) - NLLLoss(음의 로그 우도 : Negative Log Likelihood)
#LogSoftmax와 NLLLoss합친 CrossEnrotpyLoss

# 손실함수 초기화
loss_fn = nn.CrossEntropyLoss()

#옵티마이저(Optimizer)
#모든 최적화 절차(logic)는 optimizer 객체에 캡슐화(encapsulate)

#모델의 매개변수와 학습률 하이퍼파라매터를 등록하여 초기화
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#학습 단계
#zero_grad()를 호출하여 매개변수의 변화도 재설정(기본적으로 변화도는 더해지기(add up)때문에 중복 계산을 막기 위해 반복마다 0으로 설정)
#loss.backward를 호출하여 예측 손실(prdiction loss)을 역전파, 각 매개변수에 대한 손실의 변화도 저장
#변화도 계산 후 optimizer.step() 호출하여 역전파 단계에서 수집된 변화도로 매개변수 조정

def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    for bacth, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if bacth % 100 ==0:
            loss,current = loss.item(), bacth*len(X)
            print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

#
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------")
    train_loop(training_dataloader, model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
print("Done")






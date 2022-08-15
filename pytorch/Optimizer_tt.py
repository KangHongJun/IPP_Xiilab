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
#얻은 결과와 실제 값 사이의 틀린정도(degree of dissmilarity)를 측정하며
#


















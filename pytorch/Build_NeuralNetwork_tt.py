#신경망은 데이터에 대한 연산을 수행하는 layer와 module로 구성되어 있다.


#FashionMNIST 분류 신경망 구성
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using{device} device")

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

#구조 출력
model = NeuralNetwork().to(device)
print(model)

#모델을 사용하기 위한 데이터 전달
#모델에 입력을 호출하면 각 class에 대한 원시(raw) 예측값이 있느 10-차원 텐서 반환 후
#원시 예측값을 softmax모듈의 인스턴스에 통과시켜 예측확률 얻음
X = torch.rand(1,28,28,device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predited class: {y_pred}")

#모델 계층
input_image = torch.rand(3,28,28) #28*28 크기, 이미지 3개로 구성된 미니배치
print(input_image.size())

#Flatten계층을 초기화하여 28*28 2D이미지를 784팍셀값을 가지는 배열로 반환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#선형 계층은 저장된 가중치(weight)와 편향(bias)을 입력에 선형변환을 적용
layer1 = nn.Linear(in_features=28*28,out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#비선형 활성화(activaton)는 모델의 입력과 출력사이에 복잡한 관계를 만든다
#비선형 활성화는 선형 변환 후에 적용되어 비선형성(nonliearity)을 도입하고, 학습할 수 있도록 한다.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")








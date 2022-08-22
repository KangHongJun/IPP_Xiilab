#모든 torchvisio 데이터셋은 변형로직을 가지고,
#호출 가능한 객체를 받는 매개변수 두개를 변경하기 위한 transform과
#정답을 면경하기 위한 target_transform을 가진다

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


#데이터 다운로드
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    #람다식 정수를 원-핫으로 부호화된 텐서로 변형, 데이터셋 정답의 개수크기의 zero tensor를 만들고
    #scatter을 호출하여 정답 y에 해당하는 index에 value=1을 할당
    #나중에 다시 보기
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)








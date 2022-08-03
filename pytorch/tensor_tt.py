#https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html

#numpy배열과 매우 유사하지만 tensor는 GPU & 하드웨어 가속기에서 사용가능
#ndarray와 내부 메모리를 공유 가능, 자동미분에 최적화

import torch
import numpy as np

#텐서 초기화

#데이터로부터 직접 생성
data = [[1,2],[3,4]]

x_data = torch.tensor(data)
print(x_data, type(x_data))

#numpy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_data, type(x_np))

#다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}\n")

x_rand = torch.rand_like(x_data,dtype=torch.float)
print((f"Random Tensor: \n {x_rand} \n"))

#복사 tensor와 torch를 사용하는 방법도 있지만 아래의 방법을 추천한다고 한다.
x_test = x_data.clone().detach()
print(f"Ones Tensor: \n {x_test}\n")

#무작위 또는 상수 값 사용하기
shape = (2,3,) #차원을 나타내는 튜플
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_torch = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_torch}\n")

#텐서의 속성

tensor = torch.rand(3,4)

#모양 자료형 저장된 장치
print(f"Shape af torch: \n {tensor.shape}\n")
print(f"Datat type of tensor: \n {tensor.dtype}\n") 
print(f"Device tensor is stored on: \n {tensor.device}\n")

#텐서 연산 https://pytorch.org/docs/stable/torch.html
#텐서는 기본적으로 CPU에 생성되기 때문에 .to메소드를 사용하여 GPU가용성을 확인하고 명시적으로 텐서를 이동할 수 있다.

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#numpy식 표준 인덱싱 & 슬라이싱
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First colum: {tensor[:,0]}")
print(f"Last colum: {tensor[...,-1]}")

tensor[:,1] = 0 #2열을 0ㅇ,로
print(tensor)

#텐서 합치기 - torch.cat/stack

t1 = torch.cat([tensor,tensor,tensor],dim=1)#옆으로 붙는 형식
print(t1)

t2 = torch.stack((tensor, tensor), dim=1,out=None)#아래로 붙는 형식
print(t2,type(t2))

#산술 연산 - 두 텐서의 행렬곱 y1 y2 y3는 같은 값을 가진다.

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(f"{y1,y2,y3}\n")

#요소별 곱
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor, out=z3)

print(z1, z2, z3)

#단일 요소 텐서 - 텐서의 모든 값을 하나로 집계하여 요소가 하나인 경우 ,item을 사용하여 python 숫자 값으로 변환
#type torch.Tensor -> float
agg = tensor.sum()
agg_item = agg.item()

print(agg, type(agg))
print(agg_item, type(agg_item))

#바꿔치기(in place) 연산 - x.copy_(y)나 x.t_()는 x를 변경
#메모리를 일부 절약하지만, 기록이 삭제되어 비추
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

#numpy 변환(Bridge) - Cpu상의 텐서의 numpy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됨

#텐서 -> 넘파이
t = torch.ones(5)
print(f"t:{t}",type(t))
n = t.numpy()
print(f"n : {n}",type(n))

t.add_(1) #텐서에 연산한 내용이 numpy n에도 반영된다. - 매우 신기한 부분이다.
print(f"t:{t}")
print(f"n : {n}")

#넘파이 -> 텐서 마찬가지로 같이 반영된다.
np.add(n,1,out=n)
print(f"t: {t}")
print(f"n: {n}")
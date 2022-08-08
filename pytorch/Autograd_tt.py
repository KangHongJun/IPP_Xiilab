#신경망을 학습할때 가장 자주 사용되는 알고리즘은 역전파
#이 알고리즘에서 매개변수는 주어진 매개변수에 대한 손실 함수의 변화도(gradient)에 따라 조정
#변화도를 계산하기 위해 torch.autogard 자동 미분 엔진으로 계산 그래프 계산

#입력x, 매개변수x&b, 일부 손실함수가 있는 간단한 단일계층 신경망
import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3,requires_grad=True)
b = torch.rand(3,requires_grad=True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#변화도 계산하기
#신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수를 계산해야 한다
#즉 x와y의 일부 고정값에서 ∂loss/∂w와 ∂loss/∂b가 필요하다
#이러한 도함수를 계산하기 위해 loss.backward()이후 w.grad, b.grad값 가져옴
loss.backward()
print(w.grad)
print(b.grad)







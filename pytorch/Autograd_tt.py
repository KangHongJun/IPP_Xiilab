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

#requires_grad=Ture로 설정한 노드들의 grad값만 구할수 있다.
#backward를 사용한 변화도 계산은 한번만 수행가능, 여러번 해야한다면 retrain_graph=true전달 해야한다.

#변화도 추적 멈추기
#기본적으로 requires_grad=True인 모든 텐서는 기록을 추적하고 변화도 계산을 지원한다.
#그러나 모델을 학습한 뒤 데이터를 단순히 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우엔
#추적이 필요없을 수 있다
z = torch.matmul(x,w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w)+b
print(z.requires_grad)

#동일한 결과를 결과를 얻는 다른 방법은 텐서에 detach메소드 사용
z = torch.matmul(x,w)+b
z_det = z.detach()
print(z_det.requires_grad)
#추적을 멈추는 이유 : 사전 학습된 신경망을 미세조정 할때 필요(추적하면 고정된 매개변수로 표시하나봄)
#변화도를 추적하지 않는 텐서의 연산이 더 효율적

#연산 그래프에
#개념적으로 autogard는 텐서 및 실행된 모든 연산들의 기록을 Function객첼 구성된
#방향성 비순한 그래프(DAG : Directed Acyilc Graph)에 저장
#이 때 그래프의 잎은 입력텐서, 뿌리는 결과 텐서이다.
#뿌리부터 잎까지 추적하면 연쇄 법칙에 따라 변화도를 자동으로 계산할 수 있다.
#순전파 단계에서 autogard는 요청된 연산을 수행하여 결과텐서에 계산하고, DAG에 연산의 변화도 기능을 유지
#역전파 단계는 DAG에서 bacward가 호출될 때 시작된다
#autogard는 이 때 각 .grad_fn으로부터 변화도를 계산하고,
#각텐서의 .grad 속성에 계산 결과를 쌓고(accumulate), 연쇄 법칙을 사용하여, 모든 잎 텐서들까지 전파

#파이토치에서 DAG동적이고, 주목해야하는 부분은 그래프가 처음부터 다시 생성된다는것
#매번 bacward가 호출되면 새로운 그래프를 채운다(populate)
#이 때문에 모델에서 흐름제어(control flow)구문들을 사용할 수 있다
#매번 반복(iteration) 할때마다 필요하면 모양,크기, 연산을 바꿀 수 있다.






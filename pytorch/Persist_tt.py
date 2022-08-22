#모델을 저장하고 불러와서 모델의 상태 유지(persist), 모델의 예측 실행

import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
#state_dict(내부 상태 사전)에 저장 및 save로 저장
torch.save(model.state_dict(), 'model_weight.pth')

#모델 가중치를 불러오기 이해 instance를 생성한 후 load_State_dict사용하여 매개변수 불라옴
model = models.vgg16()
model.load_state_dict(torch.load('model_weight.pth'))
model.eval()
#dropout과 배치 정규화(batch normalization)를 평가모드(evaluation mode)로 설정해야 일관성 있는 결고 ㅏ생성

#모델의 형태 포함하여 저장 및 불러오기
torch.save(model, 'model.pth')

model = torch.load('model.pth')
#pickle 모듈을 사용하여 모델을 직렬화(serialize)하므로
#모델을 불러올 때 실제 클래스 정의(definition)를 적용(rely on)






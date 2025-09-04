# 2_model.py

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """트랜스포머 기반의 Q-Value 예측 모델"""
    def __init__(self, state_dim, num_boxes, num_orientations, container_dims, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.container_dims = container_dims
        
        # 입력 상태를 모델이 이해할 수 있는 벡터로 변환
        self.embedding = nn.Linear(state_dim, d_model)
        
        # 인코더: 상태 정보를 분석하고 맥락(context)을 추출
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 디코더 입력 (학습 가능한 파라미터로 생성)
        self.decoder_input = nn.Parameter(torch.rand(1, 3, d_model)) # (상자, 방향, 위치) 3가지 요소를 디코딩하기 위함

        # 디코더: 인코더의 출력을 바탕으로 각 행동 요소의 Q-value를 계산
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 최종 Q-value를 위한 출력 레이어
        self.fc_out_box = nn.Linear(d_model, num_boxes)
        self.fc_out_orientation = nn.Linear(d_model, num_orientations)
        # 위치는 3D 공간의 각 좌표에 대한 Q-value를 예측
        self.fc_out_position = nn.Linear(d_model, container_dims[0] * container_dims[1] * container_dims[2])

    def forward(self, state):
        # state: (batch_size, state_dim)
        
        # 1. 임베딩 및 차원 확장
        # (batch_size, state_dim) -> (batch_size, d_model) -> (batch_size, 1, d_model)
        src = self.embedding(state).unsqueeze(1) 
        
        # 2. 인코더
        # memory: (batch_size, 1, d_model)
        memory = self.transformer_encoder(src)
        
        # 3. 디코더
        # 디코더 입력을 배치 크기에 맞게 복제
        batch_size = state.shape[0]
        tgt = self.decoder_input.repeat(batch_size, 1, 1) # (batch_size, 3, d_model)
        
        # output: (batch_size, 3, d_model)
        output = self.transformer_decoder(tgt, memory)
        
        # 4. 각 행동 요소에 대한 Q-Value 계산
        # output의 각 슬라이스는 (상자, 방향, 위치)에 해당
        q_box = self.fc_out_box(output[:, 0, :])           # (batch_size, num_boxes)
        q_orientation = self.fc_out_orientation(output[:, 1, :]) # (batch_size, num_orientations)
        q_position = self.fc_out_position(output[:, 2, :])   # (batch_size, W*L*H)
        
        return q_box, q_orientation, q_position
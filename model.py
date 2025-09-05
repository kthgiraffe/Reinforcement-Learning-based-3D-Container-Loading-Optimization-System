# 2_model.py
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """유연한 상태 입력을 처리하는 트랜스포머 기반 Q-value 예측 모델"""
    def __init__(self, model_params, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # 1. 각 상태 요소를 위한 개별 임베딩 레이어
        #    - 서로 다른 종류의 정보를 모델이 이해할 수 있는 동일한 차원의 벡터로 변환
        self.height_map_embedding = nn.Linear(model_params['height_map_dim'], d_model)
        self.boxes_embedding = nn.Linear(model_params['boxes_obs_dim'], d_model)
        self.container_embedding = nn.Linear(model_params['container_dims_dim'], d_model)
        
        # 2. 트랜스포머 인코더
        #    - 임베딩된 상태 정보들 간의 관계(어텐션)를 분석하여 맥락 정보를 추출
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. 트랜스포머 디코더
        #    - 인코더가 추출한 맥락 정보를 바탕으로 각 행동(상자, 방향, 위치)에 대한 출력을 생성
        self.decoder_input = nn.Parameter(torch.rand(1, 3, d_model)) # 디코더의 초기 입력
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 4. 최종 출력 헤드 (각 행동에 대한 Q-value를 계산)
        self.fc_out_box = nn.Linear(d_model, model_params['num_box_actions'])
        self.fc_out_orientation = nn.Linear(d_model, model_params['num_orientations'])
        self.fc_out_position = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Sigmoid() # 0~1 사이의 상대 좌표(x,y,z)를 출력
        )

    def forward(self, state):
        """모델의 순전파 로직"""
        # state 딕셔너리에서 각 요소 추출
        height_map = state['height_map']
        boxes_obs = state['boxes_obs']
        container_dims = state['container_dims']
        
        # 각 요소를 개별적으로 임베딩
        hm_emb = self.height_map_embedding(height_map)
        box_emb = self.boxes_embedding(boxes_obs)
        cont_emb = self.container_embedding(container_dims)
        
        # 임베딩된 벡터들을 하나의 시퀀스로 결합 (batch_size, 3, d_model)
        # -> [높이맵 정보, 상자 정보, 컨테이너 정보]가 인코더의 입력이 됨
        src = torch.stack([hm_emb, box_emb, cont_emb], dim=1)
        
        # 인코더와 디코더를 통과하여 최종 출력 생성
        memory = self.transformer_encoder(src)
        batch_size = src.shape[0]
        tgt = self.decoder_input.repeat(batch_size, 1, 1)
        output = self.transformer_decoder(tgt, memory)
        
        # 각 행동 요소에 대한 Q-Value 계산
        q_box = self.fc_out_box(output[:, 0, :])
        q_orientation = self.fc_out_orientation(output[:, 1, :])
        q_position_relative = self.fc_out_position(output[:, 2, :])
        
        return q_box, q_orientation, q_position_relative
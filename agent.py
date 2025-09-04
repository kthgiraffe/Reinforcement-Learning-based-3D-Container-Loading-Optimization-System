# 3_agent.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from model import TransformerModel

class DQNAgent:
    def __init__(self, state_dim, num_boxes, container_dims, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.num_boxes = num_boxes
        self.num_orientations = 6
        self.container_dims = container_dims
        self.pos_dim = container_dims[0] * container_dims[1] * container_dims[2]
        self.gamma = gamma
        self.batch_size = batch_size
        
        # 기본 모델과 목표 모델(Target Network) 생성
        self.policy_net = TransformerModel(state_dim, num_boxes, self.num_orientations, container_dims)
        self.target_net = TransformerModel(state_dim, num_boxes, self.num_orientations, container_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 평가 모드

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state, epsilon):
        """Epsilon-Greedy 방식으로 행동 선택"""
        if random.random() > epsilon: # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_box, q_orientation, q_position = self.policy_net(state_tensor)
                
                box_idx = q_box.argmax(1).item()
                orientation_idx = q_orientation.argmax(1).item()
                pos_flat_idx = q_position.argmax(1).item()
                position = np.unravel_index(pos_flat_idx, self.container_dims)
        else: # Random action
            box_idx = random.randrange(self.num_boxes)
            orientation_idx = random.randrange(self.num_orientations)
            position = (random.randrange(self.container_dims[0]),
                        random.randrange(self.container_dims[1]),
                        random.randrange(self.container_dims[2]))

        return box_idx, orientation_idx, position

    def store_transition(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """리플레이 버퍼에서 샘플링하여 모델 학습"""
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        # 데이터를 텐서로 변환
        state_batch = torch.FloatTensor(np.array(batch_state))
        reward_batch = torch.FloatTensor(batch_reward)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state))
        done_batch = torch.FloatTensor(batch_done)

        # 행동 인덱스 분리
        box_actions = torch.LongTensor([a[0] for a in batch_action])
        orient_actions = torch.LongTensor([a[1] for a in batch_action])
        pos_flat_actions = torch.LongTensor([np.ravel_multi_index(a[2], self.container_dims) for a in batch_action])

        # 1. 현재 Q-Value 계산
        q_box, q_orientation, q_position = self.policy_net(state_batch)
        
        q_box_current = q_box.gather(1, box_actions.unsqueeze(1))
        q_orient_current = q_orientation.gather(1, orient_actions.unsqueeze(1))
        q_pos_current = q_position.gather(1, pos_flat_actions.unsqueeze(1))
        
        # 각 행동 요소의 Q-value를 합산하여 최종 Q-value 계산 (단순화된 방식)
        current_q = (q_box_current + q_orient_current + q_pos_current) / 3

        # 2. 목표 Q-Value 계산
        with torch.no_grad():
            next_q_box, next_q_orient, next_q_pos = self.target_net(next_state_batch)
            
            # 다음 상태에서 가장 큰 Q-value 선택
            max_next_q_box = next_q_box.max(1)[0]
            max_next_q_orient = next_q_orient.max(1)[0]
            max_next_q_pos = next_q_pos.max(1)[0]
            
            # 각 요소를 평균내어 다음 상태의 Q-value 계산
            max_next_q = (max_next_q_box + max_next_q_orient + max_next_q_pos) / 3
            
            # 벨만 방정식 적용
            target_q = reward_batch + (1 - done_batch) * self.gamma * max_next_q

        # 3. 손실 함수 계산 및 모델 업데이트
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """목표 모델의 가중치를 현재 모델의 가중치로 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
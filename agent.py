# 3_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from model import TransformerModel

class DQNAgent:
    """DQN 알고리즘을 구현한 에이전트 클래스"""
    def __init__(self, model_params, lr=1e-4, gamma=0.99, buffer_size=20000, batch_size=128):
        self.gamma = gamma  # 미래 보상에 대한 감가율
        self.batch_size = batch_size  # 한 번에 학습할 경험의 수
        self.num_box_actions = model_params['num_box_actions']
        
        # 실제 학습에 사용될 정책망(policy_net)과 목표값 계산에 사용될 목표망(target_net)
        self.policy_net = TransformerModel(model_params)
        self.target_net = TransformerModel(model_params)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 목표망은 학습하지 않으므로 평가 모드로 설정

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size) # 경험을 저장할 리플레이 버퍼

    def _state_to_tensor(self, state):
        """상태 딕셔너리를 PyTorch 텐서로 변환합니다."""
        return {k: torch.FloatTensor(v).unsqueeze(0) for k, v in state.items()}

    def select_action(self, state, env, epsilon):
        """Epsilon-Greedy 정책에 따라 행동을 선택합니다."""
        # Epsilon 확률로 무작위 행동(탐험), 1-Epsilon 확률로 최적 행동(활용)
        if random.random() > epsilon:
            with torch.no_grad(): # 평가 시에는 그래디언트 계산 불필요
                state_tensor = self._state_to_tensor(state)
                q_box, q_orientation, q_position = self.policy_net(state_tensor)
                
                # 마스킹: 현재 놓을 수 없는 상자 선택지를 제외
                mask = torch.zeros_like(q_box)
                num_remaining = len(env.remaining_boxes)
                if num_remaining < self.num_box_actions:
                    mask[:, num_remaining:] = -1e9 # Q-value를 매우 낮춰 선택되지 않게 함
                
                box_idx = (q_box + mask).argmax(1).item()
                orientation_idx = q_orientation.argmax(1).item()
                position_relative = q_position.squeeze(0).cpu().numpy()
        else: # 무작위 행동 (탐험)
            box_idx = random.randrange(min(len(env.remaining_boxes), self.num_box_actions))
            orientation_idx = random.randrange(6)
            position_relative = np.random.rand(3)

        return box_idx, orientation_idx, position_relative

    def store_transition(self, state, action, reward, next_state, done):
        """경험(S, A, R, S')을 리플레이 버퍼에 저장합니다."""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """리플레이 버퍼에서 경험을 샘플링하여 모델을 학습합니다."""
        if len(self.memory) < self.batch_size:
            return # 버퍼에 데이터가 충분히 쌓일 때까지 기다림

        # 버퍼에서 배치 크기만큼의 경험을 무작위로 샘플링
        transitions = random.sample(self.memory, self.batch_size)
        
        # 상태 딕셔너리 재구성 (배치 처리를 위해)
        states = {k: torch.FloatTensor(np.array([t[0][k] for t in transitions])) for k in transitions[0][0]}
        next_states = {k: torch.FloatTensor(np.array([t[3][k] for t in transitions])) for k in transitions[0][3]}
        
        actions = [t[1] for t in transitions]
        rewards = torch.FloatTensor([t[2] for t in transitions])
        dones = torch.FloatTensor([t[4] for t in transitions])

        box_actions = torch.LongTensor([a[0] for a in actions]).unsqueeze(1)
        orient_actions = torch.LongTensor([a[1] for a in actions]).unsqueeze(1)
        
        # 1. 현재 Q-Value 계산: Q(s, a)
        q_box, q_orient, _ = self.policy_net(states)
        current_q_box = q_box.gather(1, box_actions)
        current_q_orient = q_orient.gather(1, orient_actions)
        current_q = (current_q_box + current_q_orient) / 2 # Q-value 단순 평균

        # 2. 목표 Q-Value 계산: R + γ * max_a' Q(s', a')
        with torch.no_grad():
            next_q_box, next_q_orient, _ = self.target_net(next_states)
            max_next_q = (next_q_box.max(1)[0] + next_q_orient.max(1)[0]) / 2
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 3. 손실(Loss) 계산 및 모델 업데이트
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward() # 역전파
        self.optimizer.step() # 정책망의 가중치 업데이트

    def soft_update(self, tau=1e-3):
        """목표망을 정책망 쪽으로 서서히 업데이트하여 학습 안정성 향상"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
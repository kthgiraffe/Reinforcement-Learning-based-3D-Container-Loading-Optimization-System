# 4_main.py

from environment import PackingEnv
from agent import DQNAgent
import numpy as np
import torch

# --- 하이퍼파라미터 설정 ---
CONTAINER_DIMS = (100, 100, 100)  # 컨테이너 크기 (W, L, H)
# 적재할 상자 목록 (W, L, H)
BOXES_TO_PLACE = [
    (14, 3, 12), (7, 14, 0), (10, 12, 1), (14, 11, 2), (7, 14, 14), (12, 6, 2), (0, 11, 10), (10, 1, 10), (9, 11, 8), (12, 11, 6), (4, 6, 5), (11, 0, 14), (5, 5, 1), (8, 9, 5), (1, 3, 0), (0, 15, 10), (12, 15, 8), (9, 2, 14), (12, 11, 9), (1, 4, 15), (13, 14, 8), (2, 13, 5), (0, 3, 1), (14, 5, 3), (13, 2, 4)
]
NUM_BOXES = len(BOXES_TO_PLACE)

# 학습 관련 파라미터
NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # 몇 에피소드마다 Target Net 업데이트할지

def main():
    # 1. 환경 및 에이전트 초기화
    env = PackingEnv(CONTAINER_DIMS, BOXES_TO_PLACE)
    state_dim = np.prod(CONTAINER_DIMS) # 상태 크기
    
    agent = DQNAgent(state_dim=state_dim, 
                     num_boxes=NUM_BOXES, 
                     container_dims=CONTAINER_DIMS)

    epsilon = EPSILON_START
    scores = []
    
    # 2. 학습 루프
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 행동 선택
            action = agent.select_action(state, epsilon)
            
            # 환경에서 행동 수행
            # 이 부분은 환경의 남은 상자 수에 맞게 action의 box_idx를 조정해야 함
            current_num_boxes = len(env.remaining_boxes)
            if current_num_boxes > 0:
                action = (action[0] % current_num_boxes, action[1], action[2])
            
            next_state, reward, done = env.step(action)
            
            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)
            
            # 모델 학습
            agent.learn()
            
            state = next_state
            total_reward += reward

        scores.append(total_reward)
        
        # Epsilon 값 감소 (탐험 -> 활용)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # 목표 신경망 업데이트
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()

        print(f"Episode {episode+1}/{NUM_EPISODES}, Score: {total_reward:.2f}, Epsilon: {epsilon:.2f}, UR: {env.container.get_utilization_rate():.2%}")
    
    print("Training finished. Saving model...")
    torch.save(agent.policy_net.state_dict(), "container_packer_model.pth")
    print("Model saved to container_packer_model.pth")

if __name__ == '__main__':
    main()
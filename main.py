# 4_main.py
import numpy as np
import random
import torch
from tqdm import tqdm # 학습 진행률 시각화를 위한 라이브러리
from environment import PackingEnv
from agent import DQNAgent

# --- 하이퍼파라미터 ---
# 문제 생성 관련
MAX_CONTAINER_DIM = 50
MAX_BOX_DIM = 25
# 모델 구조 관련
MAX_BOXES_OBS = 5 # 에이전트가 한 번에 관찰할 상자 수
HEIGHT_MAP_SIZE = (MAX_CONTAINER_DIM, MAX_CONTAINER_DIM) # 고정된 높이 맵 크기
# 학습 관련
NUM_EPISODES = 5000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

# --- 동적 문제 생성 함수 ---
def generate_random_container():
    """무작위 크기의 컨테이너 규격을 생성합니다."""
    return (random.randint(20, MAX_CONTAINER_DIM), random.randint(20, MAX_CONTAINER_DIM), random.randint(20, MAX_CONTAINER_DIM))

def generate_random_boxes(container_dims):
    """컨테이너에 들어갈 만한 무작위 상자 목록을 생성합니다."""
    num_boxes = random.randint(5, 20)
    boxes = []
    for _ in range(num_boxes):
        box_dim = (
            random.randint(3, min(container_dims[0] - 2, MAX_BOX_DIM)),
            random.randint(3, min(container_dims[1] - 2, MAX_BOX_DIM)),
            random.randint(3, min(container_dims[2] - 2, MAX_BOX_DIM))
        )
        boxes.append(box_dim)
    return boxes

def main():
    """메인 학습 루프를 실행합니다."""
    
    # 모델 파라미터를 딕셔너리로 정의하여 에이전트에 전달
    model_params = {
        'height_map_dim': np.prod(HEIGHT_MAP_SIZE),
        'boxes_obs_dim': MAX_BOXES_OBS * 3,
        'container_dims_dim': 3,
        'num_box_actions': MAX_BOXES_OBS, # 모델은 다음 N개 상자 중 하나를 선택
        'num_orientations': 6
    }
    agent = DQNAgent(model_params)
    epsilon = EPSILON_START

    # 지정된 에피소드 수만큼 훈련 반복
    for episode in tqdm(range(NUM_EPISODES), desc="Training Progress"):
        # 1. 매 에피소드마다 새로운 무작위 문제 생성
        container_dims = generate_random_container()
        boxes_to_place = generate_random_boxes(container_dims)
        
        # 2. 생성된 문제로 환경 초기화
        env = PackingEnv(container_dims, boxes_to_place, MAX_BOXES_OBS, HEIGHT_MAP_SIZE)
        state = env.reset()
        done = False
        total_reward = 0

        step_count = 0
        max_step = 2500
        
        # 3. 한 에피소드가 끝날 때까지 반복
        while not done:
            action = agent.select_action(state, env, epsilon)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            agent.soft_update() # 목표망 소프트 업데이트
            state = next_state
            total_reward += reward

            if step_count >= max_step and not done:
                done = True
        
        # Epsilon 값을 점차 감소시켜 탐험의 비중을 줄임
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # 50번 에피소드마다 학습 결과 출력
        if (episode + 1) % 50 == 0:
            tqdm.write(f"Episode {episode+1}, Score: {total_reward:.2f}, UR: {env.container.get_utilization_rate():.2%}, Epsilon: {epsilon:.3f}")

    # 4. 학습 완료 후 모델 저장
    print("Training finished. Saving model...")
    torch.save(agent.policy_net.state_dict(), "container_packer_model.pth")
    print("Model saved to container_packer_model.pth")

if __name__ == '__main__':
    main()
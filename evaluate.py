# 5_evaluate.py
import torch
import numpy as np
from environment import PackingEnv
from agent import DQNAgent
from main import HEIGHT_MAP_SIZE, MAX_BOXES_OBS

def evaluate():
    """훈련된 모델의 성능을 처음 보는 문제로 평가합니다."""

    # 1. 훈련 때 사용되지 않은 새로운 테스트 문제 정의
    test_container_dims = (45, 35, 30)
    test_boxes = [(12, 18, 22), (10, 10, 10), (15, 12, 8), (20, 10, 5), (7, 7, 7), (18, 15, 12)]
    model_path = "container_packer_model.pth"

    # 2. 에이전트 및 모델 구조 초기화 (훈련 때와 동일한 파라미터)
    model_params = {
        'height_map_dim': np.prod(HEIGHT_MAP_SIZE),
        'boxes_obs_dim': MAX_BOXES_OBS * 3,
        'container_dims_dim': 3,
        'num_box_actions': MAX_BOXES_OBS,
        'num_orientations': 6
    }
    agent = DQNAgent(model_params)

    # 3. 저장된 모델의 가중치(기억) 불러오기
    try:
        agent.policy_net.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("먼저 4_main.py를 실행하여 모델을 훈련하고 저장해주세요.")
        return
        
    agent.policy_net.eval() # 모델을 평가 모드로 설정
    print("Transformer Model loaded successfully.")

    # 4. 평가 실행
    env = PackingEnv(test_container_dims, test_boxes, MAX_BOXES_OBS, HEIGHT_MAP_SIZE)
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon=0으로 설정하여 오직 학습된 전략만을 사용
        action = agent.select_action(state, env, epsilon=0) 
        state, _, done = env.step(action)
    
    # 5. 최종 결과 출력
    print("\n--- Evaluation Result on Unseen Problem ---")
    print(f"Container Dims: {test_container_dims}")
    print(f"Placed Boxes: {len(env.container.placed_boxes)} / {len(test_boxes)}")
    print(f"Final Utilization Rate: {env.container.get_utilization_rate():.2%}")
    print("-------------------------------------------")

if __name__ == '__main__':
    evaluate()
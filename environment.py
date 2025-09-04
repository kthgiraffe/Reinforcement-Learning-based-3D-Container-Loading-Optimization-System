# 1_environment.py

import numpy as np

class Box:
    """상자의 정보를 다루는 클래스"""
    def __init__(self, id, width, length, height):
        self.id = id
        self.dims = np.array([width, length, height])
        # 상자를 회전시켜 6가지 방향의 크기 정보를 생성
        self.orientations = self._generate_orientations()

    def _generate_orientations(self):
        w, l, h = self.dims
        return [
            np.array([w, l, h]), np.array([w, h, l]), np.array([l, w, h]),
            np.array([l, h, w]), np.array([h, w, l]), np.array([h, l, w])
        ]

    def __repr__(self):
        return f"Box(id={self.id}, dims={self.dims})"

class Container:
    """컨테이너의 상태를 관리하는 클래스"""
    def __init__(self, width, length, height):
        self.dims = np.array([width, length, height])
        # 컨테이너 내부 공간을 3D 배열로 표현 (0: 비어있음, 1: 채워짐)
        self.space = np.zeros(self.dims, dtype=np.int8)
        self.placed_boxes = []
        self.total_volume = width * length * height

    def reset(self):
        """환경 초기화"""
        self.space.fill(0)
        self.placed_boxes = []

    def check_placement(self, box_dims, position):
        """특정 위치에 상자를 놓을 수 있는지 확인"""
        pos = np.array(position)
        
        # 컨테이너 경계를 벗어나는지 확인
        if np.any(pos < 0) or np.any(pos + box_dims > self.dims):
            return False
            
        # 다른 상자와 겹치는지 확인
        target_space = self.space[pos[0]:pos[0]+box_dims[0], 
                                  pos[1]:pos[1]+box_dims[1], 
                                  pos[2]:pos[2]+box_dims[2]]
        if np.sum(target_space) > 0:
            return False
            
        return True

    def place_box(self, box, box_dims, position):
        """상자를 컨테이너에 배치"""
        pos = np.array(position)
        self.space[pos[0]:pos[0]+box_dims[0], 
                   pos[1]:pos[1]+box_dims[1], 
                   pos[2]:pos[2]+box_dims[2]] = 1 # 상자 ID 대신 1로 채움
        
        self.placed_boxes.append({
            'box_id': box.id,
            'dims': box_dims,
            'position': position
        })

    def get_utilization_rate(self):
        """현재 공간 활용률(UR) 계산"""
        filled_volume = np.sum(self.space)
        return filled_volume / self.total_volume if self.total_volume > 0 else 0


class PackingEnv:
    """강화학습을 위한 컨테이너 적재 환경"""
    def __init__(self, container_dims, boxes_to_place):
        self.container = Container(*container_dims)
        self.initial_boxes = [Box(i, *dims) for i, dims in enumerate(boxes_to_place)]
        self.remaining_boxes = []

    def reset(self):
        """환경을 초기 상태로 리셋하고 첫 상태를 반환"""
        self.container.reset()
        self.remaining_boxes = self.initial_boxes.copy()
        return self._get_state()

    def _get_state(self):
        """현재 상태를 신경망 입력 형태로 반환"""
        # 상태: 컨테이너의 3D 공간 맵 + 남은 상자들의 정보
        # 실제 구현에서는 이 부분을 정교하게 벡터화해야 함
        # 여기서는 단순화를 위해 컨테이너 공간만 상태로 사용
        return self.container.space.flatten()

    def step(self, action):
        """행동을 수행하고 다음 상태, 보상, 종료 여부를 반환"""
        box_idx, orientation_idx, position = action
        
        if box_idx >= len(self.remaining_boxes):
            # 유효하지 않은 상자 선택 시 페널티
            return self._get_state(), -10.0, True 

        selected_box = self.remaining_boxes[box_idx]
        selected_dims = selected_box.orientations[orientation_idx]

        if self.container.check_placement(selected_dims, position):
            # 행동 수행: 상자 배치
            self.container.place_box(selected_box, selected_dims, position)
            self.remaining_boxes.pop(box_idx)
            
            # 보상 획득: 공간 활용률 증가분
            reward = self.container.get_utilization_rate() * 100
            done = len(self.remaining_boxes) == 0
        else:
            # 잘못된 위치에 놓으려고 할 때 페널티
            reward = -1.0
            done = False # 계속 시도

        # 모든 상자를 다 놓았거나 더 이상 놓을 공간이 없을 때 종료
        if not self._can_any_box_be_placed():
            done = True
            # 최종 보상으로 전체 공간 활용률 제공
            reward = self.container.get_utilization_rate() * 100
            
        return self._get_state(), reward, done

    def _can_any_box_be_placed(self):
        """남은 상자 중 하나라도 배치 가능한지 확인 (단순화된 버전)"""
        if not self.remaining_boxes:
            return False
        # 실제로는 모든 남은 상자와 모든 위치를 탐색해야 하지만,
        # 여기서는 남은 상자가 있는지 여부만으로 간단히 판단
        return True
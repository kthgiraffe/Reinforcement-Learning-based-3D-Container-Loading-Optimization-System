# 1_environment.py
import numpy as np

class Box:
    """상자의 정보를 다루는 클래스"""
    def __init__(self, id, width, length, height):
        self.id = id  # 상자의 고유 ID
        self.dims = np.array([width, length, height])  # 상자의 크기 (W, L, H)
        # 상자를 회전시켜 만들 수 있는 6가지 방향의 크기 정보를 미리 생성
        self.orientations = self._generate_orientations()

    def _generate_orientations(self):
        """상자의 6가지 회전 상태에 대한 크기 목록을 생성합니다."""
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
        self.dims = np.array([width, length, height])  # 컨테이너 크기 (W, L, H)
        # 컨테이너 내부 공간을 3D 배열로 표현 (0: 비어있음, 1: 채워짐)
        self.space = np.zeros(self.dims, dtype=np.int8)
        self.placed_boxes = []  # 배치된 상자들의 정보를 저장하는 리스트
        self.total_volume = width * length * height  # 컨테이너의 총 부피

    def reset(self):
        """컨테이너 상태를 초기화합니다 (모두 비움)."""
        self.space.fill(0)
        self.placed_boxes = []

    def check_placement(self, box_dims, position, theta=0.7):
        """특정 위치에 상자를 놓을 수 있는지 물리 법칙에 따라 확인합니다."""
        pos = np.array(position)
        
        # 1. 경계 검사: 상자가 컨테이너 밖으로 나가는지 확인
        if np.any(pos < 0) or np.any(pos + box_dims > self.dims):
            return False
        
        # 상자를 놓을 공간이 컨테이너 크기를 벗어나는 경우 예외 처리
        try:
            target_space = self.space[pos[0]:pos[0]+box_dims[0], pos[1]:pos[1]+box_dims[1], pos[2]:pos[2]+box_dims[2]]
        except IndexError:
            return False

        # 2. 겹침 검사: 해당 공간에 다른 상자가 이미 있는지 확인
        if np.sum(target_space) > 0:
            return False
        
        # 3. 안정성 검사 (중력): 상자가 공중에 뜨지 않는지 확인
        if pos[2] == 0:  # 컨테이너 바닥에 놓는 경우, 무조건 안정적
            return True
    
        # 상자 바로 아래층(z-1)의 공간을 확인
        support_area_slice = self.space[pos[0] : pos[0] + box_dims[0], pos[1] : pos[1] + box_dims[1], pos[2] - 1]
        contact_area = np.sum(support_area_slice)  # 아래층에서 받쳐주는 면적
        footprint_area = box_dims[0] * box_dims[1]  # 상자의 바닥 면적

        # 받쳐주는 면적이 상자 바닥 면적의 theta 비율 미만이면 불안정
        if footprint_area > 0 and contact_area / footprint_area < theta:
            return False
            
        return True # 모든 검사를 통과하면 배치 가능

    def place_box(self, box, box_dims, position):
        """검사를 통과한 상자를 컨테이너에 배치합니다."""
        pos = np.array(position)
        self.space[pos[0]:pos[0]+box_dims[0], pos[1]:pos[1]+box_dims[1], pos[2]:pos[2]+box_dims[2]] = 1
        self.placed_boxes.append({'box_id': box.id, 'dims': box_dims, 'position': position})

    def get_utilization_rate(self):
        """현재 컨테이너의 공간 활용률(UR)을 계산합니다."""
        placed_volume = sum(np.prod(item['dims']) for item in self.placed_boxes)
        return placed_volume / self.total_volume if self.total_volume > 0 else 0

    def get_height_map(self):
        """3D 공간 정보를 2D 높이 맵으로 변환하여 반환합니다."""
        height_map = np.zeros(self.dims[:2], dtype=np.float32)
        for x in range(self.dims[0]):
            for y in range(self.dims[1]):
                # 해당 (x, y) 좌표에 쌓인 상자의 가장 높은 z좌표를 찾음
                filled_zs = np.where(self.space[x, y, :] == 1)[0]
                if len(filled_zs) > 0:
                    height_map[x, y] = filled_zs.max() + 1
        return height_map

class PackingEnv:
    """강화학습 에이전트와 상호작용하는 환경 클래스"""
    def __init__(self, container_dims, boxes_to_place, max_boxes_obs=5, height_map_size=(50,50)):
        self.container = Container(*container_dims)
        self.initial_boxes = [Box(i, *dims) for i, dims in enumerate(boxes_to_place)]
        self.remaining_boxes = []
        self.max_boxes_obs = max_boxes_obs  # 에이전트가 한 번에 관찰할 수 있는 최대 상자 수
        self.height_map_size = height_map_size  # 모델 입력을 위한 고정된 높이 맵 크기

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.container.reset()
        # 부피가 큰 상자부터 놓도록 정렬하여 문제 난이도를 약간 낮춤
        self.remaining_boxes = sorted(self.initial_boxes, key=lambda b: np.prod(b.dims), reverse=True)
        return self._get_state()

    def _get_state(self):
        """현재 환경 상태를 모델이 이해할 수 있는 형태로 변환합니다."""
        # 1. Height Map: 컨테이너의 현재 채워진 상태 (고정 크기로 만들고 정규화)
        height_map = self.container.get_height_map()
        padded_height_map = np.zeros(self.height_map_size, dtype=np.float32)
        h, w = height_map.shape
        h_max, w_max = self.height_map_size
        padded_height_map[:min(h, h_max), :min(w, w_max)] = height_map[:min(h, h_max), :min(w, w_max)]
        normalized_height_map = padded_height_map / self.container.dims[2]

        # 2. 남은 상자 정보: 다음에 놓아야 할 상자들의 크기 정보 (정규화)
        box_obs = np.zeros((self.max_boxes_obs, 3), dtype=np.float32)
        for i in range(min(len(self.remaining_boxes), self.max_boxes_obs)):
            box_obs[i, :] = self.remaining_boxes[i].dims / self.container.dims
            
        # 3. 컨테이너 규격 정보: 현재 문제의 컨테이너 크기 정보 (정규화)
        normalized_container_dims = self.container.dims / 100.0

        # 세 가지 정보를 딕셔너리 형태로 묶어 반환
        return {
            "height_map": normalized_height_map.flatten(),
            "boxes_obs": box_obs.flatten(),
            "container_dims": normalized_container_dims
        }

    def step(self, action):
        """에이전트의 행동을 실행하고 결과를 반환합니다."""
        box_idx_in_obs, orientation_idx, relative_position = action

        if box_idx_in_obs >= len(self.remaining_boxes):
            return self._get_state(), -1.0, False # 잘못된 행동이지만 에피소드는 계속

        selected_box = self.remaining_boxes[box_idx_in_obs]
        selected_dims = selected_box.orientations[orientation_idx]
        
        # 모델이 예측한 상대 좌표(0~1)를 실제 컨테이너의 절대 좌표로 변환
        position = np.floor(relative_position * (np.array(self.container.dims) - selected_dims)).astype(int)

        # 물리 법칙 검사 후 행동 수행
        if self.container.check_placement(selected_dims, position):
            self.container.place_box(selected_box, selected_dims, position)
            self.remaining_boxes.pop(box_idx_in_obs)
            # 보상: 배치한 상자의 부피 비율만큼 긍정적 보상
            reward = np.prod(selected_dims) / self.container.total_volume * 10
        else:
            # 잘못된 위치에 놓으려 하면 부정적 보상 (패널티)
            reward = -1.0
        
        done = len(self.remaining_boxes) == 0
        if done:
            # 모든 상자를 다 놓으면 최종 공간 활용률을 큰 보상으로 제공
            reward = self.container.get_utilization_rate() * 100
            
        return self._get_state(), reward, done
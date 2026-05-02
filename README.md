# Classical Mechanics Visualizer (CMV)
**고전역학 시각화 + ML 역학습 도구**

## 설치

```bash
pip install numpy scipy matplotlib torch pandas
```

## 실행

```bash
cd "C:\Users\leosc\OneDrive\바탕 화면\ai와 머신러닝\leo"
python -m cmv.main
```

## 지원 운동 유형 (7가지)

| 운동 | 모델 | 특징 |
|------|------|------|
| 단진자 | `PendulumODE` | 감쇠 계수 조절, 에너지 보존 확인 |
| 이중진자 | `DoublePendulumODE` | 혼돈(Chaos) 특성, DOP853 적분 |
| 포물체 운동 | `ProjectileODE` | 공기저항 포함, 착지 이벤트 |
| SHM / 강제진동 | `SHMODE` | 감쇠·공명 주파수 탐색 |
| 원운동 | `CircularODE` | 일정 반경 검증 |
| 케플러 궤도 | `KeplerODE` | 이심률 0~0.95, 각운동량 보존 |

## UI 사용법

1. **운동 유형 선택** (왼쪽 RadioButton)
2. **슬라이더**로 파라미터 실시간 조정 (200ms 디바운싱)
3. **좌표계 전환**: Cartesian / Cylindrical / Spherical
4. **저장 버튼**: 시뮬레이션 데이터 → `data/<motion>/<timestamp>/`
5. **ML 학습**: 저장된 데이터로 MLP 학습 (별도 스레드, UI 비블로킹)
6. **결과 비교**: 예측 궤적 vs 실제 궤적 오버레이

## ML 역학습 흐름

```
시뮬레이션 실행 → 저장 → ML 학습 (100 epoch) → 결과 비교
```

- 입력: `[x, y, vx, vy, t]` → 출력: `[ax, ay]`
- 모델: MLP (128-128-64, Tanh 활성화)
- 손실: MSELoss on accelerations
- 학습 중 손실 곡선 실시간 표시

## 디렉터리 구조

```
cmv/
├── physics/        # 물리 엔진 (ODE + 수치적분)
├── coords/         # 좌표계 변환
├── viz/            # Matplotlib 시각화 + 위젯
├── data/           # 데이터 저장/로드
├── ml/             # PyTorch MLP 모델 및 학습
└── main.py         # 진입점
tests/              # 22개 자동화 테스트
data/               # 시뮬레이션 데이터 (gitignore)
models/             # 학습된 모델 (gitignore)
```

## 테스트 실행

```bash
python -m pytest tests/ -v
```

# Scheduler (간호사 근무표 CP-SAT)

이 프로젝트는 **OR-Tools CP-SAT(제약 충족)**으로 간호사 근무표를 생성합니다.  
각 간호사/각 날짜에 대해 하나의 상태를 선택합니다:
- **근무**: `D`(Day), `E`(Evening), `N`(Night)
- **비근무**: `O`(휴무), `WR`(주휴) *(그 외 `VAC`, `BIRTH_REST`, `SLEEP`, `LEGAL_HOLIDAY`, `OFFICIAL_LEAVE`도 enum에 포함)*

---

## 현재 적용된 제약조건(요약)

### 하드 제약 (반드시 만족)
- **H-01 일별 정원/커버리지**: 매일 `D/E/N` 각각 `required_*_per_day`명 배치 (현재 코드상 `==`로 강제)
- **H-02 N 블록(2~3일)**: `N`은 최소 2일 연속, 최대 3일 연속만 허용 (1일 N 금지, 4일 연속 N 금지)
- **H-03 N 블록 종료 후 2일 휴식**: `N` 블록이 끝난 다음 **연속 2일은 근무(D/E/N) 금지** → `O` 또는 `WR`만 가능
- **H-04 연속 근무 제한**: **6일 이상 연속 근무 금지** (어떤 6일 구간에서도 근무(D/E/N) 합이 6이 되면 안 됨)
- **H-05 최소 휴식(전환 금지)**: `N→D`, `E→D` 전환 금지
- **주휴 고정(`set_weekly_rest_days`)**:
  - 입력한 요일을 기준으로 주휴(`WR`)를 **고정**
  - **7일마다 주휴 요일이 -1씩 당김** (예: 월→일→토→…)
  - `set_weekly_rest_days`로 고정된 날이 아닌데 비근무가 필요하면 **`WR`이 아니라 `O`로 표현되도록 강제**
- **주 단위 최소 비근무**: 각 간호사에 대해 **매 7일마다 `WR` 최소 1개 & `O` 최소 1개** 보장
- **지정 근무 불가(H-06)**: `add_forbidden_assignments()`로 지정한 (간호사, 날짜, 근무타입) 조합 금지

### 소프트 제약 (가능하면 만족, 가중치로 조절)
목적함수는 “점수 최소화(Minimize)” 형태이며, 하드 제약을 만족하는 해들 중 점수가 가장 좋은 해를 선택합니다.

- **S-01 D/E/N 균형**: 개인별 D/E/N 횟수 편차 최소화  
  - 가중치: `ScheduleConfig.balance_shift` (기본 5)
- **S-02 비근무(O+WR) 형평성**: 개인별 비근무일 수 편차 최소화  
  - 가중치: `ScheduleConfig.balance_off` (기본 5)
- **O/WR 연속성 보너스(낮은 가중치)**: `O→O`, `WR→WR` 연속이 많을수록 선호  
  - 가중치: `ScheduleConfig.consecutive_rest_bonus` (기본 2)

---

## 사용 방법 (test.ipynb 기준)

### 1) 설정 및 모델 생성
`test.ipynb`의 흐름은 아래 순서입니다.

1. `ScheduleConfig(cycle_days=28)` 생성
2. `ScheduleModel(num_nurses, config)` 생성
3. `model.create_variables()` 호출
4. **주휴를 최우선으로 고정**: `model.set_weekly_rest_days({...})`
5. `model.add_hard_constraints()` 호출
6. `model.solve(time_limit_seconds=...)` 호출 후 결과를 DataFrame으로 출력

> 주의: **`set_weekly_rest_days()`는 반드시 `add_hard_constraints()`보다 먼저 호출**해야 합니다.

### 2) 주휴 설정 예시
```python
model.set_weekly_rest_days(
    {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5}
)
```
- `{nurse_id: day_of_week}` 형식
- `day_of_week`: `0=일, 1=월, ..., 6=토`

### 3) 소프트 제약 가중치 조정
```python
config = ScheduleConfig(cycle_days=28)
config.balance_shift = 5
config.balance_off = 5
config.consecutive_rest_bonus = 2  # 낮게 유지하면 "있으면 좋고 없어도 됨" 수준
```

---

## 주요 파일
- `constraint.py`: 제약조건(하드/소프트) 및 솔버 모델
- `test.ipynb`: 실행 예시(입력/출력/집계)


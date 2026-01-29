# 근무표 스케줄링 모델링 문서

## 모델 개요

제약조건 프로그래밍(Constraint Programming)을 사용하여 근무표를 생성하는 모델입니다.
OR-Tools의 CP-SAT 솔버를 사용합니다.

## 변수 정의

### 의사결정 변수
- `assignments[nurse][day]`: 각 간호사(nurse)의 각 날짜(day)에 할당된 근무 유형
  - 값 범위: 0~9 (ShiftType enum)
  - D(0), E(1), N(2), OFF(3), 주휴(4), VAC(5), 생휴(6), 수면(7), 법휴(8), 공가(9)

## 하드 제약조건 구현

### H-01: 일별 D/E/N 정원 준수
- 각 날짜별로 D, E, N 각각 최대 2명까지 배정 가능
- 구현: 각 날짜별로 D/E/N 개수를 세어 제한

### H-02: Night는 반드시 2~3일 연속 블록으로만 배정
- 1일 N 금지: N이면 이전 또는 다음 날도 N이어야 함
- 4일 이상 연속 N 금지: 4일 연속 N이 불가능하도록 제약

### H-03: N 블록 종료 후 연속 2일은 근무 금지
- N 블록이 끝난 후 2일간은 OFF 또는 주휴만 가능
- D/E/N 근무 불가

### H-04: 5일 초과 연속 근무 금지
- D/E/N을 근무로 간주
- OFF/주휴는 연속 근무를 끊음
- 6일 이상 연속 근무 불가

### H-05: 근무 간 최소 16시간 휴식
- 금지 전환: N→D, E→D
- N 다음날 D 불가, E 다음날 D 불가

### H-06: 지정 근무 불가 준수
- `add_forbidden_assignments()` 메서드로 설정
- 특정 간호사의 특정 날짜에 특정 근무 유형 배정 금지

### H-07: 주휴 먼저 고정
- `set_weekly_rest_days()` 메서드로 간호사별 주휴 요일 설정
- 주기 규칙: 1주기 고정, 2주기부터 하루씩 당김 (구현 예정)

## 비근무일 관련 제약

### 일주일 안에 주휴 1개, OFF 1개 필수
- 일주일(일요일~토요일) 단위로 적용
- 각 주마다 주휴 1개, OFF 1개 필수 배정

## 모델 구조

### ScheduleModel 클래스
- `create_variables()`: 의사결정 변수 생성
- `add_hard_constraints()`: 모든 하드 제약조건 추가
- `set_weekly_rest_days()`: 주휴 설정
- `add_forbidden_assignments()`: 지정 근무 불가 설정
- `solve()`: 모델 풀이

### ScheduleConfig 클래스
- 스케줄 설정 파라미터 관리
- cycle_days, 정원, 최대 연속 근무일 등

## 사용 예시

```python
from main import ScheduleModel, ScheduleConfig, ShiftType

# 모델 생성
num_nurses = 3
config = ScheduleConfig()
model = ScheduleModel(num_nurses, config)

# 변수 생성
model.create_variables()

# 하드 제약 추가
model.add_hard_constraints()

# 주휴 설정 (간호사1: 월요일, 간호사2: 화요일, 간호사3: 수요일)
model.set_weekly_rest_days({
    0: 1,  # 간호사1: 월요일
    1: 2,  # 간호사2: 화요일
    2: 3,  # 간호사3: 수요일
})

# 지정 근무 불가 설정 (예시)
model.add_forbidden_assignments([
    (0, 5, ShiftType.N),  # 간호사1의 6일차에 N 배정 불가
])

# 모델 풀이
solution = model.solve(time_limit_seconds=60)
```

## 향후 추가 예정

1. 소프트 제약조건 (S-01, S-02, S-03)
2. 비근무일 세부 제약 (법휴, 공가, 생휴, 수면 등)
3. 주기별 주휴 당김 규칙 구현
4. 해 출력 및 시각화 기능

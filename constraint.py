"""
근무표 스케줄링을 위한 제약조건 프로그래밍 모델
OR-Tools CP-SAT 솔버 사용
"""

# ============================================================
# 이 파일은 "간호사 근무표"를 자동으로 만드는 규칙(제약조건)을 정의합니다.
#
# 핵심 아이디어(비전공자용 요약)
# - 우리는 각 간호사/각 날짜에 대해 "D/E/N/O/WR/..." 중 하나를 고릅니다.
# - 그리고 "반드시 지켜야 하는 규칙(하드 제약)"을 모두 만족시키는 배치를 찾습니다.
# - 하드 제약을 만족하는 해가 여러 개이면,
#   "되도록 지키면 좋은 규칙(소프트 제약)"을 점수화해서 가장 점수가 좋은 해를 선택합니다.
#
# 용어
# - D/E/N: Day/Evening/Night(근무)
# - O: 휴무(OFF)
# - WR: 주휴(Weekly Rest)  ← set_weekly_rest_days로 '고정'되는 날
# - VAC/생휴/수면/법휴/공가 등: 비근무(확장 가능)
#
# 이 모델에서의 "가중치(weight)"
# - 하드 제약: 가중치 개념 없음(무조건 만족해야 함)
# - 소프트 제약: 아래 ScheduleConfig의 balance_* 값이 "얼마나 중요하게" 볼지 결정
# ============================================================

from ortools.sat.python import cp_model
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

__all__ = ["ScheduleConfig", "ScheduleModel"]


class ShiftType(IntEnum):
    """근무 유형 정의"""

    # 근무
    D = 0  # 데이: 6am-2pm
    E = 1  # 이브닝: 2pm-10pm
    N = 2  # 나이트: 10pm-6am(+1)

    # 비근무
    O = 3  # 휴무(OFF)
    WR = 4  # 주휴(Weekly Rest) - set_weekly_rest_days로 '고정'되는 날
    VAC = 5  # 휴가
    BIRTH_REST = 6  # 생휴
    SLEEP = 7  # 수면
    LEGAL_HOLIDAY = 8  # 법휴
    OFFICIAL_LEAVE = 9  # 공가


@dataclass
class ScheduleConfig:
    """스케줄 설정"""

    cycle_days: int  # 주기 일수
    max_d_per_day: int = 2  # 일별 D 정원
    max_e_per_day: int = 2  # 일별 E 정원
    max_n_per_day: int = 2  # 일별 N 정원
    # 실제로 “근무 배치가 되도록” 일별 필수 배치 인원(커버리지)을 둔다.
    # 기본값은 정원과 동일하게 설정(=매일 D/E/N 각각 2명씩 배치).
    required_d_per_day: int = 2
    required_e_per_day: int = 2
    required_n_per_day: int = 2
    max_consecutive_work: int = 5  # 최대 연속 근무일
    min_n_block: int = 2  # N 최소 연속일
    max_n_block: int = 3  # N 최대 연속일
    rest_after_n_block: int = 2  # N 블록 후 필수 휴무일
    # 소프트 제약 가중치 (CP_constraint.md)
    balance_shift: int = 5  # S-01: D/E/N 개수 균형
    balance_off: int = 5  # S-02: 비근무일 개수 형평성
    consecutive_rest_bonus: int = 2  # O와 WR 연속성 보너스 (낮은 가중치)

    def __init__(self, cycle_days: int):
        # dataclass에 기본값이 있어도, __init__을 직접 정의하면
        # 여기서 최소한 cycle_days는 반드시 세팅해줘야 합니다.
        # 나머지 값은 "클래스 기본값"을 사용하게 됩니다.
        # (필요하면 사용자가 config.balance_shift = 10 처럼 덮어쓸 수 있음)
        self.cycle_days = cycle_days


class ScheduleModel:
    """근무표 스케줄링 모델"""

    def __init__(self, num_nurses: int, config: ScheduleConfig = None):
        """
        Args:
            num_nurses: 간호사 수
            config: 스케줄 설정
        """
        self.num_nurses = num_nurses
        self.config = config or ScheduleConfig()
        self.model = cp_model.CpModel()

        # 변수: assignments[nurse][day] = shift_type
        # 각 간호사의 각 날짜에 할당된 근무 유형
        self.assignments = {}

        # 주휴 정보: nurse -> 주휴 요일 (0=일요일, 1=월요일, ..., 6=토요일)
        self.weekly_rest_days = {}

        # 지정 근무 불가 정보: (nurse, day, shift_type) -> bool
        self.forbidden_assignments = {}

    def create_variables(self):
        """의사결정 변수 생성"""
        # assignments[nurse][day]는 정수(IntVar)입니다.
        # 예: 0이면 D, 1이면 E, 2이면 N, 3이면 O, 4이면 WR ...
        # 즉, "그날 그 간호사가 어떤 상태인지"를 하나의 값으로 표현합니다.
        # 각 간호사와 각 날짜에 대해 근무 유형 할당 변수 생성
        for nurse in range(self.num_nurses):
            self.assignments[nurse] = {}
            for day in range(self.config.cycle_days):
                # 각 날짜에 할당 가능한 근무 유형 중 하나를 선택
                self.assignments[nurse][day] = self.model.NewIntVar(
                    0, len(ShiftType) - 1, f"nurse_{nurse}_day_{day}"
                )

    def add_hard_constraints(self):
        """하드 제약조건 추가

        주의: 주휴 설정(set_weekly_rest_days)은 이 메서드 호출 전에 먼저 수행되어야 합니다.
        주휴는 최우선 조건으로 고정되며, 다른 제약보다 우선 적용됩니다.
        """
        # 하드 제약(H-xx)은 "반드시" 지켜야 합니다.
        # 하드 제약을 하나라도 어기면 해(근무표)가 존재하지 않습니다.
        # H-01: 일별 D/E/N 정원 준수
        self._add_daily_quota_constraint()

        # H-02: Night는 최대 3번 연속 가능
        self._add_night_block_constraint()

        # H-03: N 블록 종료 후 연속 2일은 근무 금지(OFF/주휴만 가능)
        self._add_rest_after_night_constraint()

        # H-04: 연속 5일 초과 연속근무 금지
        self._add_max_consecutive_work_constraint()

        # # H-05: 근무 간 최소 16시간 휴식 (금지 전환: N→D, E→D)
        self._add_minimum_rest_constraint()

        # (요청에 따라) H-07 관련(주휴 고정/주기 당김)은 여기서 제외
        # 주휴 고정은 set_weekly_rest_days()를 사용자가 별도로 호출

        # # 비근무일 관련 기준: 일주일(일~토) 안에 “주휴” 1개, “OFF” 1개 필수
        # 7일마다 최소 O 1개, WR 1개 필수
        self._add_weekly_rest_and_off_constraint()

        # 소프트 제약(S-xx)은 "가능하면" 만족하도록 점수화해서 최적화합니다.
        # 이 코드는 목적함수(점수)를 설정하는 부분입니다.
        # 목적함수
        self._add_objective_maximize_work()

    def _add_daily_quota_constraint(self):
        """H-01: 일별 D/E/N 정원 준수"""
        # 매일 D/E/N이 "몇 명씩" 배치되어야 하는지(커버리지)를 강제합니다.
        # required_*_per_day는 '정확히 몇 명'을 의미하도록 (==)로 넣어두었습니다.
        for day in range(self.config.cycle_days):
            # D 정원
            d_count = []
            for nurse in range(self.num_nurses):
                d_count.append(self.model.NewBoolVar(f"d_nurse_{nurse}_day_{day}"))
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.D
                ).OnlyEnforceIf(d_count[nurse])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.D
                ).OnlyEnforceIf(d_count[nurse].Not())

            # “정원 준수”는 상한(<=)이지만, 근무표가 전부 OFF로 가는 것을 막기 위해
            # 기본적으로는 required_* 만큼 “필수 배치(==)”를 강제한다.
            self.model.Add(sum(d_count) <= self.config.max_d_per_day)
            self.model.Add(sum(d_count) == self.config.required_d_per_day)

            # E 정원
            e_count = []
            for nurse in range(self.num_nurses):
                e_count.append(self.model.NewBoolVar(f"e_nurse_{nurse}_day_{day}"))
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.E
                ).OnlyEnforceIf(e_count[nurse])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.E
                ).OnlyEnforceIf(e_count[nurse].Not())

            self.model.Add(sum(e_count) <= self.config.max_e_per_day)
            self.model.Add(sum(e_count) == self.config.required_e_per_day)

            # N 정원
            n_count = []
            for nurse in range(self.num_nurses):
                n_count.append(self.model.NewBoolVar(f"n_nurse_{nurse}_day_{day}"))
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(n_count[nurse])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(n_count[nurse].Not())

            self.model.Add(sum(n_count) <= self.config.max_n_per_day)
            self.model.Add(sum(n_count) == self.config.required_n_per_day)

    def _add_night_block_constraint(self):
        """H-02: N은 최대 3번 연속 가능, 최소 2번 연속"""
        # 핵심: 야간(N)은 2~3일 연속 블록으로만 존재해야 합니다.
        # - 1일짜리 N은 금지
        # - 4일 이상 연속 N도 금지
        #
        # 이를 위해 is_night[day]라는 보조 Bool 변수를 만들어
        # "그날이 N인가?"를 0/1로 표현합니다.
        self.is_night = {}  # nurse -> [day별 N 여부] (N 블록 균형 제약에서 재사용)
        for nurse in range(self.num_nurses):
            # 각 날짜가 N인지 여부
            is_night = [
                self.model.NewBoolVar(f"is_night_nurse_{nurse}_day_{day}")
                for day in range(self.config.cycle_days)
            ]
            self.is_night[nurse] = is_night

            for day in range(self.config.cycle_days):
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(is_night[day])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(is_night[day].Not())

            # 최대 3번 연속 N: 4일 이상 연속 N 금지
            # 각 4일 연속 구간에서 최대 3개까지만 N이 될 수 있음
            for start_day in range(self.config.cycle_days - 3):
                self.model.Add(sum(is_night[start_day + i] for i in range(4)) <= 3)

            # 최소 2번 연속 N: N이 하루만 있는 경우 금지
            # N이 배정되면 이전날 또는 다음날 중 최소 하나는 N이어야 함
            for day in range(self.config.cycle_days):
                if day == 0:
                    # 첫날: 다음날이 N이어야 함
                    self.model.AddImplication(is_night[day], is_night[day + 1])
                elif day == self.config.cycle_days - 1:
                    # 마지막날: 이전날이 N이어야 함
                    self.model.AddImplication(is_night[day], is_night[day - 1])
                else:
                    # 중간날: 이전날 또는 다음날 중 최소 하나는 N이어야 함
                    # 즉, is_night[day]가 True이면 is_night[day-1] OR is_night[day+1]이 True
                    self.model.AddBoolOr(
                        [is_night[day - 1], is_night[day + 1]]
                    ).OnlyEnforceIf(is_night[day])

    def _add_rest_after_night_constraint(self):
        """H-03: N 블록 종료 후 연속 2일은 근무 금지 (OFF/주휴만 가능)"""
        # 야간(N) 블록이 끝난 직후에는 피로 회복을 위해 연속 2일은 "근무(D/E/N)" 금지
        # 즉, (O 또는 WR 같은 비근무)만 허용합니다.
        for nurse in range(self.num_nurses):
            # 각 날짜가 N인지 여부
            is_night = [
                self.model.NewBoolVar(f"is_night_nurse_{nurse}_day_{day}")
                for day in range(self.config.cycle_days)
            ]

            for day in range(self.config.cycle_days):
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(is_night[day])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(is_night[day].Not())

            # N 블록 종료 지점 찾기: N이었다가 다음날 N이 아닌 경우
            for day in range(self.config.cycle_days - self.config.rest_after_n_block):
                # day가 N이고, day+1이 N이 아니면 day는 블록 종료일
                block_end = self.model.NewBoolVar(
                    f"n_block_end_nurse_{nurse}_day_{day}"
                )

                # block_end = is_night[day] AND NOT is_night[day+1]
                self.model.AddBoolAnd(
                    [is_night[day], is_night[day + 1].Not()]
                ).OnlyEnforceIf(block_end)
                self.model.AddBoolOr(
                    [is_night[day].Not(), is_night[day + 1]]
                ).OnlyEnforceIf(block_end.Not())

                # 블록 종료 후 2일은 OFF 또는 주휴만 가능 (근무 금지)
                for rest_day in range(1, self.config.rest_after_n_block + 1):
                    if day + rest_day < self.config.cycle_days:
                        rest_day_idx = day + rest_day
                        # 해당 날짜가 OFF 또는 주휴인지 확인
                        is_off = self.model.NewBoolVar(
                            f"is_off_nurse_{nurse}_day_{rest_day_idx}"
                        )
                        is_weekly_rest = self.model.NewBoolVar(
                            f"is_weekly_rest_nurse_{nurse}_day_{rest_day_idx}"
                        )

                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] == ShiftType.O
                        ).OnlyEnforceIf(is_off)
                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] != ShiftType.O
                        ).OnlyEnforceIf(is_off.Not())

                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] == ShiftType.WR
                        ).OnlyEnforceIf(is_weekly_rest)
                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] != ShiftType.WR
                        ).OnlyEnforceIf(is_weekly_rest.Not())

                        # block_end가 True이면 is_off OR is_weekly_rest이어야 함
                        is_off_or_rest = self.model.NewBoolVar(
                            f"off_or_rest_nurse_{nurse}_day_{rest_day_idx}"
                        )
                        self.model.AddBoolOr([is_off, is_weekly_rest]).OnlyEnforceIf(
                            is_off_or_rest
                        )
                        self.model.AddBoolAnd(
                            [is_off.Not(), is_weekly_rest.Not()]
                        ).OnlyEnforceIf(is_off_or_rest.Not())
                        self.model.AddImplication(block_end, is_off_or_rest)

    def _add_max_consecutive_work_constraint(self):
        """H-04: 6일 이상 연속 근무 금지 (최대 5일 연속 근무까지 허용)"""
        # 어떤 6일 연속 구간에서도 "근무(D/E/N)"가 6번 모두 나오면 안 됩니다.
        # 즉, 6일 중 최소 1일은 비근무(O/WR/...)여야 합니다.
        # 구현: 6일 구간에서 근무일 합 <= 5
        for nurse in range(self.num_nurses):
            # 각 날짜가 근무일인지 여부 (D/E/N은 근무, OFF/주휴는 비근무)
            is_work = [
                self.model.NewBoolVar(f"is_work_nurse_{nurse}_day_{day}")
                for day in range(self.config.cycle_days)
            ]

            for day in range(self.config.cycle_days):
                # D, E, N 중 하나면 근무
                is_d = self.model.NewBoolVar(f"is_d_nurse_{nurse}_day_{day}")
                is_e = self.model.NewBoolVar(f"is_e_nurse_{nurse}_day_{day}")
                is_n = self.model.NewBoolVar(f"is_n_nurse_{nurse}_day_{day}")

                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.D
                ).OnlyEnforceIf(is_d)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.D
                ).OnlyEnforceIf(is_d.Not())

                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.E
                ).OnlyEnforceIf(is_e)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.E
                ).OnlyEnforceIf(is_e.Not())

                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(is_n)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(is_n.Not())

                # is_work = is_d OR is_e OR is_n
                self.model.AddBoolOr([is_d, is_e, is_n]).OnlyEnforceIf(is_work[day])
                self.model.AddBoolAnd(
                    [is_d.Not(), is_e.Not(), is_n.Not()]
                ).OnlyEnforceIf(is_work[day].Not())

            # 6일 이상 연속 근무 금지: 어떤 6일 연속 구간에서도 최대 5일까지만 근무 가능
            for start_day in range(self.config.cycle_days - 5):
                self.model.Add(sum(is_work[start_day + i] for i in range(6)) <= 5)

    def _add_objective_maximize_work(self):
        """목적함수: S-01/S-02 소프트 제약(점수 최소화) 반영.
        S-01: 개인별 D/E/N 개수 균형(평균 대비 편차 최소화)
        S-02: 개인별 비근무일 개수 형평성(편차 최소화)
        추가: O와 WR 연속성 보너스 (낮은 가중치)
        """
        # ============================================================
        # 소프트 제약은 "점수"로 변환하여 Minimize(점수)합니다.
        #
        # (1) S-01 D/E/N 균형:
        # - 각 간호사별 D/E/N 횟수가 평균에 가깝도록(편차 최소)
        #
        # (2) S-02 비근무(O+WR) 형평성:
        # - 각 간호사별 비근무일 수가 평균에 가깝도록(편차 최소)
        #
        # (3) O/WR 연속성(낮은 가중치):
        # - O가 연속(O→O), WR이 연속(WR→WR)으로 이어지면 "보너스"
        # - 목적함수는 Minimize이므로, 보너스는 점수에서 '빼기(-)'로 반영
        #
        # 가중치 의미:
        # - balance_shift(기본 5): D/E/N 균형을 얼마나 중요하게 볼지
        # - balance_off(기본 5): 비근무 형평성을 얼마나 중요하게 볼지
        # - consecutive_rest_bonus(기본 2): O/WR 연속성을 얼마나 중요하게 볼지 (낮게 설정)
        # ============================================================
        cycle = self.config.cycle_days
        n = self.num_nurses
        rd = self.config.required_d_per_day
        re = self.config.required_e_per_day
        rn = self.config.required_n_per_day
        w_shift = getattr(self.config, "balance_shift", 5)
        w_off = getattr(self.config, "balance_off", 5)
        w_consecutive = getattr(self.config, "consecutive_rest_bonus", 2)

        # 목표 개수 (전체 일정이 정원으로 채워지므로 평균 = 목표)
        target_d = (rd * cycle) // n
        target_e = (re * cycle) // n
        target_n = (rn * cycle) // n
        total_work = (rd + re + rn) * cycle
        target_off = (n * cycle - total_work) // n

        penalty_terms = []

        for nurse in range(self.num_nurses):
            is_d_list = []
            is_e_list = []
            is_n_list = []
            is_non_work_list = []
            is_o_list = []
            is_wr_list = []

            for day in range(cycle):
                is_d = self.model.NewBoolVar(f"obj_d_n{nurse}_d{day}")
                is_e = self.model.NewBoolVar(f"obj_e_n{nurse}_d{day}")
                is_n = self.model.NewBoolVar(f"obj_n_n{nurse}_d{day}")
                is_o = self.model.NewBoolVar(f"obj_o_n{nurse}_d{day}")
                is_wr = self.model.NewBoolVar(f"obj_wr_n{nurse}_d{day}")

                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.D
                ).OnlyEnforceIf(is_d)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.D
                ).OnlyEnforceIf(is_d.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.E
                ).OnlyEnforceIf(is_e)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.E
                ).OnlyEnforceIf(is_e.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(is_n)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(is_n.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.O
                ).OnlyEnforceIf(is_o)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.O
                ).OnlyEnforceIf(is_o.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.WR
                ).OnlyEnforceIf(is_wr)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.WR
                ).OnlyEnforceIf(is_wr.Not())

                is_non_work = self.model.NewBoolVar(f"obj_nonwork_n{nurse}_d{day}")
                self.model.AddBoolOr([is_o, is_wr]).OnlyEnforceIf(is_non_work)
                self.model.AddBoolAnd([is_o.Not(), is_wr.Not()]).OnlyEnforceIf(
                    is_non_work.Not()
                )

                is_d_list.append(is_d)
                is_e_list.append(is_e)
                is_n_list.append(is_n)
                is_non_work_list.append(is_non_work)
                is_o_list.append(is_o)
                is_wr_list.append(is_wr)

            count_d = self.model.NewIntVar(0, cycle, f"count_d_n{nurse}")
            count_e = self.model.NewIntVar(0, cycle, f"count_e_n{nurse}")
            count_n = self.model.NewIntVar(0, cycle, f"count_n_n{nurse}")
            count_off = self.model.NewIntVar(0, cycle, f"count_off_n{nurse}")
            self.model.Add(count_d == sum(is_d_list))
            self.model.Add(count_e == sum(is_e_list))
            self.model.Add(count_n == sum(is_n_list))
            self.model.Add(count_off == sum(is_non_work_list))

            # S-01: 평균 대비 편차 (절대값) -> dev >= |count - target|
            dev_d = self.model.NewIntVar(0, cycle, f"dev_d_n{nurse}")
            dev_e = self.model.NewIntVar(0, cycle, f"dev_e_n{nurse}")
            dev_n = self.model.NewIntVar(0, cycle, f"dev_n_n{nurse}")
            self.model.Add(dev_d >= count_d - target_d)
            self.model.Add(dev_d >= target_d - count_d)
            self.model.Add(dev_e >= count_e - target_e)
            self.model.Add(dev_e >= target_e - count_e)
            self.model.Add(dev_n >= count_n - target_n)
            self.model.Add(dev_n >= target_n - count_n)

            # S-02: 비근무일 개수 편차
            dev_off = self.model.NewIntVar(0, cycle, f"dev_off_n{nurse}")
            self.model.Add(dev_off >= count_off - target_off)
            self.model.Add(dev_off >= target_off - count_off)

            # O와 WR 연속성 보너스: O→O, WR→WR 연속 전환에 보너스
            # 연속 전환이 많을수록 더 긴 연속 블록 = 보너스 (페널티에서 빼기)
            consecutive_bonus = []
            for day in range(cycle - 1):
                # O→O 연속
                o_to_o = self.model.NewBoolVar(f"o_to_o_n{nurse}_d{day}")
                self.model.AddBoolAnd(
                    [is_o_list[day], is_o_list[day + 1]]
                ).OnlyEnforceIf(o_to_o)
                self.model.AddBoolOr(
                    [is_o_list[day].Not(), is_o_list[day + 1].Not()]
                ).OnlyEnforceIf(o_to_o.Not())
                consecutive_bonus.append(o_to_o)

                # WR→WR 연속
                wr_to_wr = self.model.NewBoolVar(f"wr_to_wr_n{nurse}_d{day}")
                self.model.AddBoolAnd(
                    [is_wr_list[day], is_wr_list[day + 1]]
                ).OnlyEnforceIf(wr_to_wr)
                self.model.AddBoolOr(
                    [is_wr_list[day].Not(), is_wr_list[day + 1].Not()]
                ).OnlyEnforceIf(wr_to_wr.Not())
                consecutive_bonus.append(wr_to_wr)

            # 연속성 점수 (최대화하려면 페널티에서 빼기 = 보너스)
            consecutive_score = sum(consecutive_bonus)
            # YLW Modified : end

            penalty_terms.append(
                w_shift * (dev_d + dev_e + dev_n)
                + w_off * dev_off
                - w_consecutive * consecutive_score
            )

        self.model.Minimize(sum(penalty_terms))

    def _add_minimum_rest_constraint(self):
        """H-05: 근무 간 최소 16시간 휴식 (N→D, E→D 금지)"""
        # 인력 피로/수면시간을 고려해서 다음 전환을 금지합니다.
        # - N 다음날 D 금지
        # - E 다음날 D 금지
        for nurse in range(self.num_nurses):
            for day in range(self.config.cycle_days - 1):
                # N 다음날 D 금지
                is_n = self.model.NewBoolVar(f"is_n_nurse_{nurse}_day_{day}")
                is_d_next = self.model.NewBoolVar(f"is_d_next_nurse_{nurse}_day_{day}")

                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.N
                ).OnlyEnforceIf(is_n)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.N
                ).OnlyEnforceIf(is_n.Not())

                self.model.Add(
                    self.assignments[nurse][day + 1] == ShiftType.D
                ).OnlyEnforceIf(is_d_next)
                self.model.Add(
                    self.assignments[nurse][day + 1] != ShiftType.D
                ).OnlyEnforceIf(is_d_next.Not())

                # N AND D_next는 동시에 True가 될 수 없음
                self.model.AddBoolOr([is_n.Not(), is_d_next.Not()])

                # E 다음날 D 금지
                is_e = self.model.NewBoolVar(f"is_e_nurse_{nurse}_day_{day}")
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.E
                ).OnlyEnforceIf(is_e)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.E
                ).OnlyEnforceIf(is_e.Not())

                # E AND D_next는 동시에 True가 될 수 없음
                self.model.AddBoolOr([is_e.Not(), is_d_next.Not()])

    def _add_weekly_rest_constraint(self):
        """H-07: 주휴 먼저 고정"""
        # 주휴는 나중에 설정할 수 있도록 메서드만 정의
        # 실제 주휴 설정은 set_weekly_rest_days() 메서드로 수행
        pass

    def _add_weekly_rest_and_off_constraint(self):
        """7일마다 최소 O(휴무) 1개, WR(주휴) 1개 필수"""
        # 각 간호사에 대해 "매 7일(1주)" 안에
        # - WR(주휴) 최소 1개
        # - O(휴무) 최소 1개
        # 가 반드시 존재하도록 강제합니다.
        #
        # 주의: 마지막 주(주기가 7의 배수가 아닐 때)는 남은 날짜만 검사합니다.
        num_weeks = (self.config.cycle_days + 6) // 7
        for nurse in range(self.num_nurses):
            for week in range(num_weeks):
                week_start = week * 7
                week_end = min(week_start + 7, self.config.cycle_days)

                # 해당 주에 주휴(WR) 개수
                weekly_rest_count = []
                for day in range(week_start, week_end):
                    is_weekly_rest = self.model.NewBoolVar(
                        f"is_weekly_rest_nurse_{nurse}_week_{week}_day_{day}"
                    )
                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.WR
                    ).OnlyEnforceIf(is_weekly_rest)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.WR
                    ).OnlyEnforceIf(is_weekly_rest.Not())
                    weekly_rest_count.append(is_weekly_rest)

                # 해당 주에 휴무(O) 개수
                off_count = []
                for day in range(week_start, week_end):
                    is_off = self.model.NewBoolVar(
                        f"is_off_nurse_{nurse}_week_{week}_day_{day}"
                    )
                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.O
                    ).OnlyEnforceIf(is_off)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.O
                    ).OnlyEnforceIf(is_off.Not())
                    off_count.append(is_off)

                # 주휴 최소 1개 필수
                self.model.Add(sum(weekly_rest_count) >= 1)

                # 휴무 최소 1개 필수
                self.model.Add(sum(off_count) >= 1)

    def set_weekly_rest_days(self, weekly_rest_days: Dict[int, int]):
        """
        간호사별 주휴 요일 설정 (최우선 조건 - 반드시 먼저 호출되어야 함)

        주휴는 고정되며 다른 제약보다 우선 적용됩니다.
        주기 규칙: 1주기는 지정된 요일, 2주기부터는 7일마다 -1씩 당김

        Args:
            weekly_rest_days: {nurse_id: day_of_week}
                             day_of_week: 0=일요일, 1=월요일, ..., 6=토요일
        """
        self.weekly_rest_days = weekly_rest_days

        # ============================================================
        # 이 함수는 "주휴(WR)"를 미리 고정합니다.
        # - weekly_rest_days는 {간호사ID: 요일(0~6)} 형태입니다.
        # - 주기 규칙: 7일마다 주휴 요일이 하루씩 앞당겨집니다. (예: 월→일→토→…)
        #
        # 그리고 중요한 추가 규칙:
        # - set_weekly_rest_days로 고정되지 않은 날에 "비근무"가 발생한다면,
        #   그것은 WR이 아니라 O(휴무)로 표현되도록 강제합니다.
        #   (즉, WR은 이 함수가 고정해 준 날짜에만 등장)
        # ============================================================
        # 주기 규칙: 1주기 고정, 2주기부터 7일마다 -1씩 당김 (월→일→토→…)
        # week 0: day_of_week, week 1: day_of_week-1, week 2: day_of_week-2, ...
        num_weeks = (self.config.cycle_days + 6) // 7
        weekly_rest_dates_set = set()  # 주휴로 설정된 (nurse, day) 집합

        for nurse, day_of_week in weekly_rest_days.items():
            for week in range(num_weeks):
                # 해당 주의 주휴 요일 = (day_of_week - week) mod 7
                rest_dow = (day_of_week - week) % 7
                day_in_cycle = week * 7 + rest_dow
                if day_in_cycle < self.config.cycle_days:
                    # 해당 날짜는 주휴로 고정 (최우선 조건)
                    self.model.Add(
                        self.assignments[nurse][day_in_cycle] == ShiftType.WR
                    )
                    weekly_rest_dates_set.add((nurse, day_in_cycle))

        # 주휴로 설정되지 않은 날 중 비근무일(D/E/N이 아닌 날)은 모두 O로 설정
        for nurse in range(self.num_nurses):
            for day in range(self.config.cycle_days):
                if (nurse, day) not in weekly_rest_dates_set:
                    # 핵심:
                    # - 이 날은 WR(주휴)이 될 수 없습니다. (주휴는 고정된 날에만)
                    # - 그럼에도 근무가 아닌 상태가 필요하다면 O(휴무)로 표현해야 합니다.
                    #
                    # 따라서 가능한 값은 D/E/N/O 중 하나이며,
                    # D/E/N이 아니면 자동으로 O가 되도록(implication) 만듭니다.
                    # 주휴가 아닌 날: D/E/N 중 하나이거나 O여야 함 (WR 불가)
                    # D/E/N이 아닌 경우 O로 강제
                    is_d = self.model.NewBoolVar(f"non_wr_d_n{nurse}_d{day}")
                    is_e = self.model.NewBoolVar(f"non_wr_e_n{nurse}_d{day}")
                    is_n = self.model.NewBoolVar(f"non_wr_n_n{nurse}_d{day}")
                    is_o = self.model.NewBoolVar(f"non_wr_o_n{nurse}_d{day}")

                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.D
                    ).OnlyEnforceIf(is_d)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.D
                    ).OnlyEnforceIf(is_d.Not())

                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.E
                    ).OnlyEnforceIf(is_e)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.E
                    ).OnlyEnforceIf(is_e.Not())

                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.N
                    ).OnlyEnforceIf(is_n)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.N
                    ).OnlyEnforceIf(is_n.Not())

                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.O
                    ).OnlyEnforceIf(is_o)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.O
                    ).OnlyEnforceIf(is_o.Not())

                    # WR은 불가 (주휴가 아닌 날이므로)
                    self.model.Add(self.assignments[nurse][day] != ShiftType.WR)

                    # D OR E OR N OR O 중 하나여야 함 (ExactlyOne)
                    self.model.AddBoolOr([is_d, is_e, is_n, is_o])

                    # D/E/N이 모두 False이면 O가 True여야 함
                    # 즉: NOT(D OR E OR N) => O
                    is_work = self.model.NewBoolVar(f"is_work_n{nurse}_d{day}")
                    self.model.AddBoolOr([is_d, is_e, is_n]).OnlyEnforceIf(is_work)
                    self.model.AddBoolAnd(
                        [is_d.Not(), is_e.Not(), is_n.Not()]
                    ).OnlyEnforceIf(is_work.Not())
                    self.model.AddImplication(is_work.Not(), is_o)

    def add_forbidden_assignments(self, forbidden: List[Tuple[int, int, ShiftType]]):
        """
        지정 근무 불가 제약 추가 (H-06)

        Args:
            forbidden: [(nurse_id, day, shift_type), ...] 리스트
        """
        for nurse, day, shift_type in forbidden:
            self.model.Add(self.assignments[nurse][day] != shift_type)

    def solve(
        self, time_limit_seconds: int = 30
    ) -> Optional[Dict[int, Dict[int, ShiftType]]]:
        """
        모델 풀이

        Returns:
            해가 있으면 {nurse: {day: shift_type}} 딕셔너리, 없으면 None
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds

        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for nurse in range(self.num_nurses):
                solution[nurse] = {}
                for day in range(self.config.cycle_days):
                    solution[nurse][day] = ShiftType(
                        solver.Value(self.assignments[nurse][day])
                    )
            return solution
        else:
            return None

"""
간호사 근무표 자동 작성 모델 (OR-Tools CP-SAT 사용).

- 각 날짜·간호사마다 D/E/N/OFF/주휴 등 하나의 근무유형을 고른다.
- 반드시 지켜야 할 규칙(하드 제약)을 모두 만족하는 배치만 허용한다.
- 그중에서 "되도록 지키면 좋은 규칙"(소프트 제약)을 점수로 쳐서, 점수가 가장 좋은 해를 고른다.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime, timedelta

from ortools.sat.python import cp_model

from kr_holidays import is_holiday

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["ScheduleConfig", "ScheduleModel"]


class ShiftType(IntEnum):
    """그날 그 사람이 무엇인지 나타내는 유형 (한 날에 하나만 선택)."""

    D = 0  # 데이 (06~14시)
    E = 1  # 이브닝 (14~22시)
    N = 2  # 나이트 (22시~익일 06시)
    OFF = 3  # 휴무
    WR = 4  # 주휴 (요일 고정)
    VAC = 5  # 휴가
    BR = 6  # 생휴
    SL = 7  # 수면
    LH = 8  # 법휴
    OL = 9  # 공가 (사전 지정 비근무)
    D1 = 10  # 사전 지정 근무 (정원에는 안 들어감)


@dataclass
class ScheduleConfig:
    """스케줄 설정"""

    cycle_days: int
    max_d_per_day: int = 2
    max_e_per_day: int = 2
    max_n_per_day: int = 2
    required_d_per_day: int = 2  # 매일 D 몇 명 꼭
    required_e_per_day: int = 2
    required_n_per_day: int = 2
    max_consecutive_work: int = 5  # 연속 근무 최대 5일
    min_n_block: int = 2  # N은 2일 이상 연속
    max_n_block: int = 3  # N은 3일 이하 연속
    rest_after_n_block: int = 2  # N 블록 끝나고 휴무 2일
    off_max_per_cycle: int = (
        8  # 한 주기 동안 휴무(OFF) 상한. 하한 없음(주당 WR 1·OFF 1이 최소).
    )
    balance_shift: int = 5  # 소프트: D/E/N 골고루 (클수록 비중↑)
    balance_off: int = 5  # 소프트: 휴무일 개수 형평
    consecutive_rest_bonus: int = 2  # 소프트: 휴무/주휴 연속이면 보너스

    def __init__(self, cycle_days: int = 28, off_max_per_cycle: int = 8):
        self.cycle_days = cycle_days
        self.off_max_per_cycle = off_max_per_cycle


class ScheduleModel:
    """한 주기 근무표를 만드는 모델. 시작일·간호사 명단·설정을 받아 제약을 넣고 풀이한다."""

    def __init__(
        self,
        start_date: str,
        nurses: list[str],
        config: ScheduleConfig,
    ):
        self.start_date = datetime.strptime(start_date, "%Y%m%d")
        self.nurses = nurses
        self.num_nurses = len(self.nurses)
        self.config = config
        self.model: Any = cast(Any, cp_model.CpModel())
        self.assignments = {}  # [간호사][날짜] → 그날 근무유형
        self.weekly_rest_days = {}  # 간호사별 주휴 요일 (0=일 … 6=토)
        self.official_leave_dates_set = set()  # (간호사, 날짜) 공가로 고정된 날
        self.d1_dates_set = set()  # (간호사, 날짜) D1으로 고정된 날
        self.forbidden_assignments = {}
        self.is_night = {}  # N 블록 제약용: 그날 N인지 여부

    def _day_index_from_date(self, date_str: str) -> int | None:
        """'YYYY-MM-DD'를 주기 내 몇 번째 날(0부터)로 바꾼다. 주기 밖이면 None."""
        try:
            d = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            return None
        delta = (d.date() - self.start_date.date()).days
        if 0 <= delta < self.config.cycle_days:
            return delta
        return None

    def _is_legal_holiday(self, day: int) -> bool:
        """그날(주기 내 day)이 법정 공휴일인지 여부."""
        date = self.start_date + timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        return is_holiday(date_str)

    def _create_variables(self):
        """각 (간호사, 날짜)마다 '그날 뭘 할지' 하나를 고르는 변수 만든다."""
        for nurse in range(self.num_nurses):
            self.assignments[nurse] = {}
            for day in range(self.config.cycle_days):
                self.assignments[nurse][day] = self.model.NewIntVar(
                    0, len(ShiftType) - 1, f"nurse_{nurse}_day_{day}"
                )

    def initialize_model(
        self,
        weekly_rest_days: dict[int, int],
        official_leave_days: dict[int, list[str]] | None = None,
        d1_days: dict[int, list[str]] | None = None,
    ):
        """모델 초기화.

        Args:
            weekly_rest_days: {nurse_id: 요일(0~6)} 주휴 고정
            official_leave_days: {nurse_id: ["YYYY-MM-DD", ...]} 공가 고정. WR과 동일 최우선.
            d1_days: {nurse_id: ["YYYY-MM-DD", ...]} D1(사전 지정 근무) 고정. WR/OL과 동일 최우선.
            fixed_d_days: {nurse_id: ["YYYY-MM-DD", ...]} 사용자 지정 D 고정. D1/OL/WR과 동일 최우선.
            fixed_e_days: {nurse_id: ["YYYY-MM-DD", ...]} 사용자 지정 E 고정. D1/OL/WR과 동일 최우선.
            fixed_n_blocks: {nurse_id: [[날짜2개], [날짜3개], ...]} 사용자 지정 N 고정. 각 블록은 연속 2일 또는 3일만 허용.
        """
        logger.info("Model initializing...")
        self._create_variables()
        if official_leave_days:
            self._set_official_leave_days(official_leave_days)
        if d1_days:
            self._set_d1_days(d1_days)
        self._set_weekly_rest_days(weekly_rest_days)
        self._add_hard_constraints()

    def _add_hard_constraints(self):
        """반드시 지켜야 할 규칙(하드) + 되도록 지키면 좋은 규칙(소프트·점수) 넣기."""
        self._add_daily_quota_constraint()

        self._add_night_block_constraint()

        self._add_rest_after_night_constraint()

        self._add_max_consecutive_work_constraint()

        self._add_minimum_rest_constraint()

        self._add_weekly_rest_and_off_constraint()

        self._add_off_count_per_cycle_constraint()

        self._add_objective_maximize_work()

    def _add_daily_quota_constraint(self):
        """H-01: 매일 D/E/N이 정해진 인원만큼 꼭 들어가게 (정원 준수)."""
        for day in range(self.config.cycle_days):
            d_count = []
            for nurse in range(self.num_nurses):
                d_count.append(self.model.NewBoolVar(f"d_nurse_{nurse}_day_{day}"))
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.D
                ).OnlyEnforceIf(d_count[nurse])
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.D
                ).OnlyEnforceIf(d_count[nurse].Not())

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
        """H-02: 나이트(N)는 2일 또는 3일 연속으로만 둔다. (1일만 N 금지, 4일 연속 N 금지)."""
        self.is_night = {}
        for nurse in range(self.num_nurses):
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

            for start_day in range(self.config.cycle_days - 3):
                self.model.Add(sum(is_night[start_day + i] for i in range(4)) <= 3)

            for day in range(self.config.cycle_days):
                if day == 0:
                    self.model.AddImplication(is_night[day], is_night[day + 1])
                elif day == self.config.cycle_days - 1:
                    self.model.AddImplication(is_night[day], is_night[day - 1])
                else:
                    self.model.AddBoolOr(
                        [is_night[day - 1], is_night[day + 1]]
                    ).OnlyEnforceIf(is_night[day])

    def _add_rest_after_night_constraint(self):
        """H-03: N 블록이 끝난 뒤 2일은 근무 금지. 그 2일은 휴무(OFF) 또는 주휴(WR)만 가능.
        허용 조합: OFF+OFF, OFF+WR, WR+OFF (WR+WR은 금지. 주당 WR 1회와 충돌 방지)."""
        for nurse in range(self.num_nurses):
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

            for day in range(self.config.cycle_days - self.config.rest_after_n_block):
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

                # N 직후 2일 중 최소 1일은 OFF (WR+WR 금지 → WR+OFF, OFF+OFF, OFF+WR만 허용)
                rest_day_off_vars = []
                for rest_day in range(1, self.config.rest_after_n_block + 1):
                    if day + rest_day < self.config.cycle_days:
                        rest_day_idx = day + rest_day
                        is_off = self.model.NewBoolVar(
                            f"is_off_nurse_{nurse}_day_{rest_day_idx}"
                        )
                        is_weekly_rest = self.model.NewBoolVar(
                            f"is_weekly_rest_nurse_{nurse}_day_{rest_day_idx}"
                        )
                        rest_day_off_vars.append(is_off)

                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] == ShiftType.OFF
                        ).OnlyEnforceIf(is_off)
                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] != ShiftType.OFF
                        ).OnlyEnforceIf(is_off.Not())
                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] == ShiftType.WR
                        ).OnlyEnforceIf(is_weekly_rest)
                        self.model.Add(
                            self.assignments[nurse][rest_day_idx] != ShiftType.WR
                        ).OnlyEnforceIf(is_weekly_rest.Not())

                        is_rest = self.model.NewBoolVar(
                            f"rest_nurse_{nurse}_day_{rest_day_idx}"
                        )
                        self.model.AddBoolOr([is_off, is_weekly_rest]).OnlyEnforceIf(
                            is_rest
                        )
                        self.model.AddBoolAnd(
                            [is_off.Not(), is_weekly_rest.Not()]
                        ).OnlyEnforceIf(is_rest.Not())
                        self.model.AddImplication(block_end, is_rest)

                # N 직후 2일이 모두 WR(WR+WR)이 되지 않도록: 최소 1일은 OFF
                if rest_day_off_vars:
                    self.model.AddBoolOr(rest_day_off_vars).OnlyEnforceIf(block_end)

                if (nurse, day + 1) in self.official_leave_dates_set or (
                    nurse,
                    day + 2,
                ) in self.official_leave_dates_set:
                    self.model.Add(block_end == 0)

    def _add_max_consecutive_work_constraint(self):
        """H-04: 6일 연속 근무는 금지. (D/E/N/D1 모두 근무로 친다. 최대 5일까지 연속 가능.)"""
        for nurse in range(self.num_nurses):
            is_work = [
                self.model.NewBoolVar(f"is_work_nurse_{nurse}_day_{day}")
                for day in range(self.config.cycle_days)
            ]

            for day in range(self.config.cycle_days):
                is_d = self.model.NewBoolVar(f"is_d_nurse_{nurse}_day_{day}")
                is_e = self.model.NewBoolVar(f"is_e_nurse_{nurse}_day_{day}")
                is_n = self.model.NewBoolVar(f"is_n_nurse_{nurse}_day_{day}")
                is_d1 = self.model.NewBoolVar(f"is_d1_nurse_{nurse}_day_{day}")

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
                    self.assignments[nurse][day] == ShiftType.D1
                ).OnlyEnforceIf(is_d1)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.D1
                ).OnlyEnforceIf(is_d1.Not())

                # is_work = is_d OR is_e OR is_n OR is_d1
                self.model.AddBoolOr([is_d, is_e, is_n, is_d1]).OnlyEnforceIf(
                    is_work[day]
                )
                self.model.AddBoolAnd(
                    [is_d.Not(), is_e.Not(), is_n.Not(), is_d1.Not()]
                ).OnlyEnforceIf(is_work[day].Not())

            for start_day in range(self.config.cycle_days - 5):
                self.model.Add(sum(is_work[start_day + i] for i in range(6)) <= 5)

    def _add_objective_maximize_work(self):
        """소프트 제약: D/E/N 균형·휴무일 형평·휴무·주휴 연속 보너스를 점수로 넣어, 점수가 낮은 해를 고른다."""
        cycle = self.config.cycle_days
        n = self.num_nurses
        rd = self.config.required_d_per_day
        re = self.config.required_e_per_day
        rn = self.config.required_n_per_day
        w_shift = getattr(self.config, "balance_shift", 5)
        w_off = getattr(self.config, "balance_off", 5)
        w_consecutive = getattr(self.config, "consecutive_rest_bonus", 2)

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
            is_vac_list = []
            is_lh_list = []
            is_ol_list = []

            for day in range(cycle):
                is_d = self.model.NewBoolVar(f"obj_d_n{nurse}_d{day}")
                is_e = self.model.NewBoolVar(f"obj_e_n{nurse}_d{day}")
                is_n = self.model.NewBoolVar(f"obj_n_n{nurse}_d{day}")
                is_o = self.model.NewBoolVar(f"obj_o_n{nurse}_d{day}")
                is_wr = self.model.NewBoolVar(f"obj_wr_n{nurse}_d{day}")
                is_vac = self.model.NewBoolVar(f"obj_vac_n{nurse}_d{day}")
                is_lh = self.model.NewBoolVar(f"obj_lh_n{nurse}_d{day}")
                is_ol = self.model.NewBoolVar(f"obj_ol_n{nurse}_d{day}")

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
                    self.assignments[nurse][day] == ShiftType.OFF
                ).OnlyEnforceIf(is_o)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.OFF
                ).OnlyEnforceIf(is_o.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.WR
                ).OnlyEnforceIf(is_wr)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.WR
                ).OnlyEnforceIf(is_wr.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.VAC
                ).OnlyEnforceIf(is_vac)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.VAC
                ).OnlyEnforceIf(is_vac.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.LH
                ).OnlyEnforceIf(is_lh)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.LH
                ).OnlyEnforceIf(is_lh.Not())
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.OL
                ).OnlyEnforceIf(is_ol)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.OL
                ).OnlyEnforceIf(is_ol.Not())

                is_non_work = self.model.NewBoolVar(f"obj_non_work_n{nurse}_d{day}")
                self.model.AddBoolOr([is_o, is_wr, is_vac, is_lh, is_ol]).OnlyEnforceIf(
                    is_non_work
                )
                self.model.AddBoolAnd(
                    [is_o.Not(), is_wr.Not(), is_vac.Not(), is_lh.Not(), is_ol.Not()]
                ).OnlyEnforceIf(is_non_work.Not())

                is_d_list.append(is_d)
                is_e_list.append(is_e)
                is_n_list.append(is_n)
                is_non_work_list.append(is_non_work)
                is_o_list.append(is_o)
                is_wr_list.append(is_wr)
                is_vac_list.append(is_vac)
                is_lh_list.append(is_lh)
                is_ol_list.append(is_ol)

            count_d = self.model.NewIntVar(0, cycle, f"count_d_n{nurse}")
            count_e = self.model.NewIntVar(0, cycle, f"count_e_n{nurse}")
            count_n = self.model.NewIntVar(0, cycle, f"count_n_n{nurse}")
            count_off = self.model.NewIntVar(0, cycle, f"count_off_n{nurse}")
            self.model.Add(count_d == sum(is_d_list))
            self.model.Add(count_e == sum(is_e_list))
            self.model.Add(count_n == sum(is_n_list))
            self.model.Add(count_off == sum(is_non_work_list))

            dev_d = self.model.NewIntVar(0, cycle, f"dev_d_n{nurse}")
            dev_e = self.model.NewIntVar(0, cycle, f"dev_e_n{nurse}")
            dev_n = self.model.NewIntVar(0, cycle, f"dev_n_n{nurse}")
            self.model.Add(dev_d >= count_d - target_d)
            self.model.Add(dev_d >= target_d - count_d)
            self.model.Add(dev_e >= count_e - target_e)
            self.model.Add(dev_e >= target_e - count_e)
            self.model.Add(dev_n >= count_n - target_n)
            self.model.Add(dev_n >= target_n - count_n)

            dev_off = self.model.NewIntVar(0, cycle, f"dev_off_n{nurse}")
            self.model.Add(dev_off >= count_off - target_off)
            self.model.Add(dev_off >= target_off - count_off)

            consecutive_bonus = []
            for day in range(cycle - 1):
                o_to_o = self.model.NewBoolVar(f"o_to_o_n{nurse}_d{day}")
                self.model.AddBoolAnd(
                    [is_o_list[day], is_o_list[day + 1]]
                ).OnlyEnforceIf(o_to_o)
                self.model.AddBoolOr(
                    [is_o_list[day].Not(), is_o_list[day + 1].Not()]
                ).OnlyEnforceIf(o_to_o.Not())
                consecutive_bonus.append(o_to_o)
                wr_to_wr = self.model.NewBoolVar(f"wr_to_wr_n{nurse}_d{day}")
                self.model.AddBoolAnd(
                    [is_wr_list[day], is_wr_list[day + 1]]
                ).OnlyEnforceIf(wr_to_wr)
                self.model.AddBoolOr(
                    [is_wr_list[day].Not(), is_wr_list[day + 1].Not()]
                ).OnlyEnforceIf(wr_to_wr.Not())
                consecutive_bonus.append(wr_to_wr)
            consecutive_score = sum(consecutive_bonus)

            penalty_terms.append(
                w_shift * (dev_d + dev_e + dev_n)
                + w_off * dev_off
                - w_consecutive * consecutive_score
            )

        self.model.Minimize(sum(penalty_terms))

    def _add_minimum_rest_constraint(self):
        """H-05: N이나 E 다음날에는 D/D1 금지 (휴식 16시간 확보)."""
        for nurse in range(self.num_nurses):
            for day in range(self.config.cycle_days - 1):
                is_n = self.model.NewBoolVar(f"is_n_nurse_{nurse}_day_{day}")
                is_d_next = self.model.NewBoolVar(f"is_d_next_nurse_{nurse}_day_{day}")
                is_d1_next = self.model.NewBoolVar(
                    f"is_d1_next_nurse_{nurse}_day_{day}"
                )

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

                self.model.Add(
                    self.assignments[nurse][day + 1] == ShiftType.D1
                ).OnlyEnforceIf(is_d1_next)
                self.model.Add(
                    self.assignments[nurse][day + 1] != ShiftType.D1
                ).OnlyEnforceIf(is_d1_next.Not())

                self.model.AddBoolOr([is_n.Not(), is_d_next.Not()])
                self.model.AddBoolOr([is_n.Not(), is_d1_next.Not()])

                is_e = self.model.NewBoolVar(f"is_e_nurse_{nurse}_day_{day}")
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.E
                ).OnlyEnforceIf(is_e)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.E
                ).OnlyEnforceIf(is_e.Not())

                self.model.AddBoolOr([is_e.Not(), is_d_next.Not()])
                self.model.AddBoolOr([is_e.Not(), is_d1_next.Not()])

    def _add_weekly_rest_and_off_constraint(self):
        """한 주기에서 일주일(7일) 동안 최소 WR 1개·OFF 1개를 반드시 지키게 함.
        주당 WR 1·OFF 1이 최소 휴식 요건이며, 주기 내 OFF 개수 하한은 이 제약으로만 정해짐(상한은 _add_off_count_per_cycle_constraint)."""
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
                off_count = []
                for day in range(week_start, week_end):
                    is_off = self.model.NewBoolVar(
                        f"is_off_nurse_{nurse}_week_{week}_day_{day}"
                    )
                    self.model.Add(
                        self.assignments[nurse][day] == ShiftType.OFF
                    ).OnlyEnforceIf(is_off)
                    self.model.Add(
                        self.assignments[nurse][day] != ShiftType.OFF
                    ).OnlyEnforceIf(is_off.Not())
                    off_count.append(is_off)
                self.model.Add(sum(weekly_rest_count) >= 1)
                self.model.Add(sum(off_count) >= 1)

    def _add_off_count_per_cycle_constraint(self):
        """한 주기(예: 28일) 동안 휴무(OFF)는 상한만 둠(최대 off_max_per_cycle일). 하한 없음.
        OFF가 8 미만이어도 되므로, 정원 충족을 위해 솔버가 OFF 개수를 줄여 최적해를 찾을 수 있음.
        주당 최소 WR 1·OFF 1은 _add_weekly_rest_and_off_constraint에서 보장."""
        for nurse in range(self.num_nurses):
            off_indicators = []
            for day in range(self.config.cycle_days):
                is_off = self.model.NewBoolVar(f"off_cycle_nurse_{nurse}_day_{day}")
                self.model.Add(
                    self.assignments[nurse][day] == ShiftType.OFF
                ).OnlyEnforceIf(is_off)
                self.model.Add(
                    self.assignments[nurse][day] != ShiftType.OFF
                ).OnlyEnforceIf(is_off.Not())
                off_indicators.append(is_off)
            self.model.Add(sum(off_indicators) <= self.config.off_max_per_cycle)

    def _set_official_leave_days(self, official_leave_days: dict[int, list[str]]):
        """지정한 (간호사, 날짜)를 공가(OL)로 고정. 주휴보다 먼저 적용."""
        for nurse, date_list in official_leave_days.items():
            if nurse < 0 or nurse >= self.num_nurses:
                continue
            for date_str in date_list:
                day_idx = self._day_index_from_date(date_str)
                if day_idx is None:
                    continue
                self.model.Add(self.assignments[nurse][day_idx] == ShiftType.OL)
                self.official_leave_dates_set.add((nurse, day_idx))

    def _set_d1_days(self, d1_days: dict[int, list[str]]):
        """지정한 (간호사, 날짜)를 D1(사전 지정 근무)로 고정. D1은 근무일이지만 일별 정원에는 안 들어감."""
        for nurse, date_list in d1_days.items():
            if nurse < 0 or nurse >= self.num_nurses:
                continue
            for date_str in date_list:
                day_idx = self._day_index_from_date(date_str)
                if day_idx is None:
                    continue
                self.model.Add(self.assignments[nurse][day_idx] == ShiftType.D1)
                self.d1_dates_set.add((nurse, day_idx))

    def _set_weekly_rest_days(self, weekly_rest_days: dict[int, int]):
        """H-07: 간호사별 주휴 요일(0=일~6=토)에 따라 WR 고정. 1주차는 지정 요일, 2주차부터 7일마다 하루씩 당김. 공가/D1/고정 D·E·N 날은 WR 안 넣음.
        공가/D1/고정 D·E·N과 겹쳐 WR을 넣지 못한 주는 해당 주 내 다른 날(공가·D1·고정 D/E/N 제외) 중 하루에 WR을 배정(floating WR)하여 7일마다 주휴 1개 조건을 충족."""
        self.weekly_rest_days = weekly_rest_days
        num_weeks = (self.config.cycle_days + 6) // 7
        weekly_rest_dates_set = set()
        floating_wr_weeks = (
            set()
        )  # (nurse, week): 이 주에 원래 WR일이 공가/D1이라 WR을 다른 날로 옮겨야 함
        for nurse, day_of_week in weekly_rest_days.items():
            for week in range(num_weeks):
                rest_dow = (day_of_week - week) % 7
                day_in_cycle = week * 7 + rest_dow
                if day_in_cycle < self.config.cycle_days:
                    if (nurse, day_in_cycle) in self.official_leave_dates_set:
                        floating_wr_weeks.add((nurse, week))
                        continue
                    if (nurse, day_in_cycle) in self.d1_dates_set:
                        floating_wr_weeks.add((nurse, week))
                        continue
                    self.model.Add(
                        self.assignments[nurse][day_in_cycle] == ShiftType.WR
                    )
                    weekly_rest_dates_set.add((nurse, day_in_cycle))
        # 공가/D1/고정 D·E·N으로 WR이 빠진 주에서 WR을 넣을 수 있는 날 = 그 주 내 공가·D1·고정 D/E/N이 아닌 모든 날
        alternative_wr_allowed_set = set()
        for nurse, week in floating_wr_weeks:
            week_start = week * 7
            week_end = min(week_start + 7, self.config.cycle_days)
            for day in range(week_start, week_end):
                if (nurse, day) in self.official_leave_dates_set:
                    continue
                if (nurse, day) in self.d1_dates_set:
                    continue
                alternative_wr_allowed_set.add((nurse, day))

        floating_wr_week_is_wr_vars = {}  # (nurse, week) -> [is_wr vars for days in that week]
        for nurse, week in floating_wr_weeks:
            floating_wr_week_is_wr_vars[(nurse, week)] = []

        for nurse in range(self.num_nurses):
            for day in range(self.config.cycle_days):
                if (nurse, day) in self.official_leave_dates_set:
                    continue
                if (nurse, day) in self.d1_dates_set:
                    continue
                if (nurse, day) not in weekly_rest_dates_set:
                    in_alt_wr = (nurse, day) in alternative_wr_allowed_set
                    week = day // 7
                    is_d = self.model.NewBoolVar(f"non_wr_d_n{nurse}_d{day}")
                    is_e = self.model.NewBoolVar(f"non_wr_e_n{nurse}_d{day}")
                    is_n = self.model.NewBoolVar(f"non_wr_n_n{nurse}_d{day}")
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
                    if in_alt_wr:
                        is_wr = self.model.NewBoolVar(f"alt_wr_n{nurse}_d{day}")
                        self.model.Add(
                            self.assignments[nurse][day] == ShiftType.WR
                        ).OnlyEnforceIf(is_wr)
                        self.model.Add(
                            self.assignments[nurse][day] != ShiftType.WR
                        ).OnlyEnforceIf(is_wr.Not())
                        if (nurse, week) in floating_wr_week_is_wr_vars:
                            floating_wr_week_is_wr_vars[(nurse, week)].append(is_wr)
                    else:
                        self.model.Add(self.assignments[nurse][day] != ShiftType.WR)

                    if self._is_legal_holiday(day):
                        is_lh = self.model.NewBoolVar(f"non_wr_lh_n{nurse}_d{day}")
                        self.model.Add(
                            self.assignments[nurse][day] == ShiftType.LH
                        ).OnlyEnforceIf(is_lh)
                        self.model.Add(
                            self.assignments[nurse][day] != ShiftType.LH
                        ).OnlyEnforceIf(is_lh.Not())
                        if in_alt_wr:
                            self.model.AddExactlyOne([is_d, is_e, is_n, is_lh, is_wr])
                        else:
                            self.model.AddExactlyOne([is_d, is_e, is_n, is_lh])
                    else:
                        is_o = self.model.NewBoolVar(f"non_wr_o_n{nurse}_d{day}")
                        is_vac = self.model.NewBoolVar(f"non_wr_vac_n{nurse}_d{day}")
                        self.model.Add(
                            self.assignments[nurse][day] == ShiftType.OFF
                        ).OnlyEnforceIf(is_o)
                        self.model.Add(
                            self.assignments[nurse][day] != ShiftType.OFF
                        ).OnlyEnforceIf(is_o.Not())
                        self.model.Add(
                            self.assignments[nurse][day] == ShiftType.VAC
                        ).OnlyEnforceIf(is_vac)
                        self.model.Add(
                            self.assignments[nurse][day] != ShiftType.VAC
                        ).OnlyEnforceIf(is_vac.Not())

                        if in_alt_wr:
                            self.model.AddExactlyOne(
                                [is_d, is_e, is_n, is_o, is_vac, is_wr]
                            )
                        else:
                            self.model.AddExactlyOne([is_d, is_e, is_n, is_o, is_vac])

        for nurse, week in floating_wr_weeks:
            vars_list = floating_wr_week_is_wr_vars.get((nurse, week), [])
            if vars_list:
                self.model.Add(sum(vars_list) == 1)

    def solve(
        self, time_limit_seconds: int = 30
    ) -> dict[int, dict[int, ShiftType]] | None:
        """
        모델 풀이

        Returns:
            해가 있으면 {nurse: {day: shift_type}} 딕셔너리, 없으면 None
        """
        logger.info("Model solving...")
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

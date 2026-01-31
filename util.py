"""
근무표 스케줄링 결과를 처리하기 위한 유틸리티 함수
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
import pandas as pd

from constraint import ShiftType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["get_solution_df"]

WEEK_MAP = {
    "Sun": "일",
    "Mon": "월",
    "Tue": "화",
    "Wed": "수",
    "Thu": "목",
    "Fri": "금",
    "Sat": "토",
}


def get_solution_df(
    solution: dict[int, dict[int, ShiftType]],
    nurses: list[str],
    start_date: str,
    cycle_days: int = 28,
):  # -> pd.DataFrame | None:
    """Get a solution DataFrame from a solution dictionary.

    Args:
        solution (dict[int, dict[int, ShiftType]]): A dictionary of solution.
        nurses (list[str]): The list of nurses.
        start_date (str): The start date.
        cycle_days (int): The number of days in the cycle.

    Raises:
        ValueError: If no solution is found.

    Returns:
        pd.DataFrame | None: A DataFrame of the solution.
    """

    date_obj = datetime.strptime(start_date, "%Y%m%d")
    date = [
        (date_obj + timedelta(days=day)).strftime("%m/%d") for day in range(cycle_days)
    ]
    week_name = [
        WEEK_MAP.get((date_obj + timedelta(days=day)).strftime("%a"), "?")
        for day in range(cycle_days)
    ]

    if solution:
        logger.info("Found a solution!")
        # 멀티 인덱스: 날짜와 요일을 함께 표시
        multi_index = pd.MultiIndex.from_arrays(
            [date, week_name], names=["날짜", "요일"]
        )
        df = pd.DataFrame(
            {
                f"{nurses[nurse]}": [
                    solution[nurse][day].name for day in range(cycle_days)
                ]
                for nurse in range(len(nurses))
            },
            index=multi_index,
        )

        if not df.empty and df is not None:
            # 날짜/요일을 컬럼으로, A,B,C...를 인덱스로 변환 (transpose)
            df_transposed = df.T

            # 각 간호사별 근무 타입 개수 계산 (A,B,C...가 인덱스)
            count_d = df.apply(lambda col: (col == "D").sum())
            count_e = df.apply(lambda col: (col == "E").sum())
            count_n = df.apply(lambda col: (col == "N").sum())
            count_o = df.apply(lambda col: ((col == "OFF") | (col == "WR")).sum())
            cnt = pd.DataFrame(
                {"OFF": count_o, "D": count_d, "E": count_e, "N": count_n}
            )

            # df_transposed와 cnt를 병합 (A,B,C...가 인덱스로 유지)
            # 멀티 컬럼 레벨이 다르므로 concat 사용
            result = pd.concat([df_transposed, cnt], axis=1)
            return result
        raise ValueError("The solution DataFrame is empty. Please check the solution.")
    raise ValueError("No solution found. Please check the constraints.")


def solution_to_excel(solution_dataframe: pd.DataFrame, filename: str) -> None:
    """Save a solution DataFrame to an Excel file.

    Args:
        solution_dataframe (pd.DataFrame): A DataFrame of the solution.
        filename (str): The name of the Excel file.

    Raises:
        ValueError: If the solution DataFrame is None.
    """
    if solution_dataframe.empty:
        msg = "The solution DataFrame is empty. Please check the solution."
        raise ValueError(msg)
    solution_dataframe.to_excel(filename, index=True)

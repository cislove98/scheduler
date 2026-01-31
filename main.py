from constraint import ScheduleConfig, ScheduleModel
from util import get_solution_df
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 4000)


##########################Schedule Initialization##########################
# schedule start date
start_date = "20260201"  # 719 / 816 / 913

# names
nurses = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# weekly rest condition
# 0=> Sun, 1=> Mon, 2=> Tue, 3=> Wed, 4=> Thu, 5=> Fri, 6=> Sat
weekly_rest_days = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5}


##########################Generate Schedule##########################
def main():
    print("Generating schedule...")
    config = ScheduleConfig(off_max_per_cycle=8)
    model = ScheduleModel(start_date=start_date, nurses=nurses, config=config)
    model.initialize_model(
        weekly_rest_days=weekly_rest_days,
    )
    solution = model.solve(time_limit_seconds=30)
    res = get_solution_df(solution, nurses, start_date)
    print(res)
    res.to_excel("schedule.xlsx", index=True, header=True)
    print("Schedule generated successfully!")


if __name__ == "__main__":
    main()

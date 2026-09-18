import pandas as pd

from workload.generator import generate_workload


def create_dataset(
    num_tasks=1000,
    seed=42,
    output_file="data/ml_dataset.csv"
):
    """
    Generate a workload and create an ML dataset.
    """

    tasks = generate_workload(
        num_tasks=num_tasks,
        seed=seed
    )

    rows = []

    for task in tasks:

        rows.append({
            "arrival_time": task.arrival_time,
            "deadline": task.deadline,
            "priority": task.priority,
            "input_size": task.input_size,
            "execution_time": task.execution_time
        })

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_file,
        index=False
    )

    return dataframe


if __name__ == "__main__":

    dataframe = create_dataset()

    print("\nML dataset created successfully.")

    print(
        f"Number of samples: "
        f"{len(dataframe)}"
    )

    print("\nFirst 5 samples:")

    print(dataframe.head())
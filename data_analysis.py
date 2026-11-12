"""Summarise the CSV produced by sensor_simulation.py."""
import csv
from pathlib import Path
from statistics import mean


DATA_FILE = Path(__file__).resolve().parent / "data" / "traffic_data.csv"


def analyse_data():
    if not DATA_FILE.exists():
        print("No traffic readings found. Run sensor_simulation.py first.")
        return

    with DATA_FILE.open(newline="") as file:
        readings = list(csv.DictReader(file))
    if not readings:
        print("The data file has no readings yet. Run sensor_simulation.py.")
        return

    print("=" * 65)
    print("TRAFFIC DATA ANALYSIS")
    print("=" * 65)
    print("Data file:", DATA_FILE)
    print("Total readings:", len(readings))

    print("\nFIRST 10 SENSOR READINGS")
    for row in readings[:10]:
        ns = row["Observed_NS"] or "missing"
        ew = row["Observed_EW"] or "missing"
        print(f"{row['Simulation_Time']} | NS: {ns} | EW: {ew} | {row['Signal_Status']}")

    # Use the true post-action queues to measure the simulated outcome.
    ns_queues = [int(row["Queue_NS_After_Step"]) for row in readings]
    ew_queues = [int(row["Queue_EW_After_Step"]) for row in readings]
    total_queues = [ns + ew for ns, ew in zip(ns_queues, ew_queues)]
    print("\nSIMULATED QUEUES AFTER EACH STEP")
    print(f"Average North-South queue: {mean(ns_queues):.2f} vehicles")
    print(f"Average East-West queue: {mean(ew_queues):.2f} vehicles")
    print(f"Average total queue: {mean(total_queues):.2f} vehicles")
    print("Largest total queue:", max(total_queues), "vehicles")
    print("Total vehicles that passed:", sum(int(row["Vehicles_Departed"]) for row in readings))

    print("\nFAULTS AND REVIEW REQUESTS")
    print("Missing sensor readings:", sum(row["Observed_NS"] == "" for row in readings))
    print("Failed control requests:", sum(row["API_Failure"] == "True" for row in readings))
    print("Steps needing review:", sum(row["Review_Required"] == "True" for row in readings))
    print("Steps with simulated approval:", sum(row["Simulated_Approval"] == "True" for row in readings))
    print("\nThis is one demonstration run, not the full report benchmark.")


if __name__ == "__main__":
    analyse_data()

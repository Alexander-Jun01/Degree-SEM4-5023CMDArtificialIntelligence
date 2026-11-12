"""Run the traffic AI with printed readings, like the SmartFactory lab."""
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.traffic_agent import CONFIG, DEMO_SEED, demo


# SETTINGS: change these before running the file.
NUMBER_OF_READINGS = 36
TIME_BETWEEN_READINGS = 0.5  # Pause on screen, in real seconds.
SIMULATED_APPROVAL = False  # True simulates approval of flagged decisions.
RANDOM_SEED = DEMO_SEED  # 40001 = report run. Use a number, or None for a new seed each run.

# Each reading still represents five seconds of simulated traffic.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
PHASE_NAMES = {0: "North-South", 1: "East-West"}


def run_simulation():
    DATA_FOLDER.mkdir(exist_ok=True)
    csv_file = DATA_FOLDER / "traffic_data.csv"
    simulation_start = datetime.now().replace(microsecond=0)

    print("=" * 65)
    print("TRAFFIC AI SENSOR SIMULATOR")
    print("=" * 65)
    print("Synthetic traffic at one junction, using the saved Q-learning model.")
    print("Starting with a busy queue to demonstrate review requests.")
    seed = RANDOM_SEED if RANDOM_SEED is not None else int(time.time()) % 100000
    print("Simulated supervisor approval:", SIMULATED_APPROVAL)
    print("Traffic seed:", seed)
    print("Saving readings to:", csv_file)

    # Open a fresh CSV for this run. Blank sensor cells mean missing data.
    with csv_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Simulation_Time", "Reading", "Observed_NS", "Observed_EW",
            "Requested_Direction", "Signal_Status", "Review_Required",
            "Simulated_Approval", "API_Failure", "Reason",
            "Queue_NS_After_Step", "Queue_EW_After_Step", "Vehicles_Departed"
        ])

        def display_reading(event):
            # Observations are noisy sensor readings BEFORE the action.
            observation = event["observation"]
            north_south = observation[0] if observation is not None else ""
            east_west = observation[1] if observation is not None else ""
            reading = event["step"] + 1
            timestamp = simulation_start + timedelta(
                seconds=event["step"] * CONFIG["step_seconds"]
            )

            # A switch includes a clearance step with both directions red.
            feedback = event["feedback"]
            if feedback["switched"]:
                signal = "All red (clearance before next green)"
            else:
                signal = PHASE_NAMES[event["acknowledged_phase"]] + " green"

            print("-" * 65)
            print(f"Reading: {reading}/{NUMBER_OF_READINGS}")
            print("Simulation time:", timestamp)
            if observation is None:
                print("Sensors: missing readings, using fixed timing")
            else:
                print("North-South sensor queue:", north_south, "vehicles")
                print("East-West sensor queue:", east_west, "vehicles")
            print("Requested direction:", PHASE_NAMES[event["requested_phase"]])
            print("Actual signal:", signal)
            print("Review required:", event["review_required"])
            print("Reason:", event["reason"])
            print("Control request failed:", feedback["api_failure"])
            print("Vehicles that passed:", feedback["departed"], flush=True)

            # True queues are simulation audit data, not extra AI sensor inputs.
            queues = event["true_queues_for_simulation_audit"]
            writer.writerow([
                timestamp, reading, north_south, east_west,
                PHASE_NAMES[event["requested_phase"]], signal,
                event["review_required"], event["simulated_approval"],
                feedback["api_failure"], event["reason"],
                queues[0], queues[1], feedback["departed"]
            ])
            file.flush()
            time.sleep(TIME_BETWEEN_READINGS)

        # Reuse the tested agent, including its timing rules and memory logging.
        demo(approve=SIMULATED_APPROVAL, on_event=display_reading,
             output_dir=DATA_FOLDER, readings=NUMBER_OF_READINGS, seed=seed)

    print("SIMULATION COMPLETE")
    print("Readings saved to:", csv_file)
    print("Run data_analysis.py to summarise this run.")


if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nStopped. Completed readings are saved in the data folder.")

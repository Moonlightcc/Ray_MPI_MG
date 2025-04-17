import ray
import ray_mpi
import csv
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ray.init(address="auto", namespace="default")

@ray.remote
class Recorder:
    def __init__(self):
        # format: logs[rank] = {}, key, value = iteration_number, [(event_str, time)]
        self.logs = {}

    def register_time(self,rank,iteration_number, event_str, event_time):
        if rank not in self.logs:
            self.logs[rank] = {}
        if iteration_number not in self.logs[rank]:
            self.logs[rank][iteration_number] = []
        self.logs[rank][iteration_number].append((event_str,event_time))

    def dump_logs_to_csv(self, filename="logs.csv"):
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "iteration", "event", "time"])
            for rank, iterations in self.logs.items():
                for iteration, events in iterations.items():
                    for event_str, time_val in events:
                        writer.writerow([rank, iteration, event_str, time_val])

    def clear_logs(self):
        self.logs = {}

def normalize_and_sort_log_csv(input_csv: str, output_csv: str = "normalized_logs.csv"):
    rows = []
    with open(input_csv, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["time"] = float(row["time"])
            rows.append(row)

    min_time = min(row["time"] for row in rows)
    for row in rows:
        row["time"] -= min_time

    rows.sort(key=lambda x: x["time"])
    fieldnames = ["time"] + [key for key in rows[0].keys() if key != "time"]
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"Normalization complete. Output saved to: {output_csv}")

def extract_rank_events(csv_file, event_down, event_up):
    events = []
    ranks_seen = set()

    with open(csv_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            ranks_seen.add(rank)

            time_val = float(row["time"])
            event = row["event"]
            if event == event_down:
                events.append((time_val, -1))
            elif event == event_up:
                events.append((time_val, +1))

    events.sort(key=lambda x: x[0])
    return events, len(ranks_seen)

def plot_two_rank_activity_curves(csv1, csv2, output_img="rank_activity_comparison.png"):
    events1, initial1 = extract_rank_events(csv1, "rank_sleep", "end_sleep")
    times1 = [0.0]
    ranks1 = [initial1]
    current_rank1 = initial1
    for time_val, delta in events1:
        current_rank1 += delta
        times1.append(time_val)
        ranks1.append(current_rank1)

    events2, initial2 = extract_rank_events(csv2, "delete_end", "recovery_end")
    times2 = [0.0]
    ranks2 = [initial2]
    current_rank2 = initial2
    for time_val, delta in events2:
        current_rank2 += delta
        times2.append(time_val)
        ranks2.append(current_rank2)

    plt.figure(figsize=(10, 5))
    plt.step(times1, ranks1, where='post', label="Workload 1 (rank_sleep/end_sleep)")
    plt.step(times2, ranks2, where='post', label="Workload 2 (delete_end/recovery_end)")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Active Ranks")
    plt.title("Active Rank Count Over Time (Two Workloads)")
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()

    print(f"Rank activity plot saved to: {output_img}")
    
def main():
    recorder = Recorder.options(name="Recorder", lifetime="detached",  get_if_exists=True).remote()
    print("Recorder is running.")
    print("Press 'v' to dump logs to a CSV file.")
    print("Press 'q' to quit.")
    print("Press 'g' to show the event figure")
    
    print("Workload: wait for without workload -> type v -> wait for with workload -> type v -> type g to generate the figure")

    try:
        filename1 = None
        filename2 = None
        while True:
            user_input = input(">> ").strip().lower()
            if user_input == "v":
                filename = input("Enter filename (e.g., output.csv): ").strip()
                if filename1 == None:
                    filename1 = filename
                else:
                    filename2 = filename 
                ray.get(recorder.dump_logs_to_csv.remote(filename))
                print(f"Logs saved to {filename}")
                normalized_filename = "normalized_" + filename
                normalize_and_sort_log_csv(filename, normalized_filename)
                print(f"Normalized CSV saved to {normalized_filename}")
                ray.get(recorder.clear_logs.remote())
                print("Logs cleared.")

            elif user_input == "n":
                if filename is None:
                    print("No CSV file has been generated yet. Use 'v' first.")
                else:
                    normalized_filename = "normalized_" + filename
                    normalize_and_sort_log_csv(filename, normalized_filename)
                    print(f"Normalized CSV saved to {normalized_filename}")
            elif user_input == "g":
                if filename1 is None or filename2 is None:
                    print("No two CSV file has been generated yet. Use 'v' first.")
                else:
                    normalized_filename1 = "normalized_" + filename1
                    normalized_filename2 = "normalized_" + filename2

                    output_img = input("Enter filename (e.g., output.png): ").strip()
                    plot_two_rank_activity_curves(normalized_filename1, normalized_filename2, output_img)
            elif user_input == "c":
                ray.get(recorder.clear_logs.remote())
                print("Logs cleared.")
            elif user_input == "q":
                print("Exiting.")
                break
            else:
                print("Unknown command. Use 'v' to dump logs or 'q' to quit.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Exiting.")

if __name__ == "__main__":
    main()




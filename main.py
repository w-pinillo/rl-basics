import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Main runner for RL-Basics project.")
    parser.add_argument("script", help="The script to run.", choices=["frozen_lake", "taxi", "cliff_walking", "cartpole", "mountain_car", "car_racing"])

    args, unknown = parser.parse_known_args()

    script_to_run = None
    if args.script == "frozen_lake":
        script_to_run = "src/run_frozen_lake.py"
    elif args.script == "taxi":
        script_to_run = "src/run_taxi.py"
    elif args.script == "cliff_walking":
        script_to_run = "src/run_cliff_walking.py"
    elif args.script == "cartpole":
        script_to_run = "src/run_cartpole.py"
    elif args.script == "mountain_car":
        script_to_run = "src/run_mountain_car.py"
    elif args.script == "car_racing":
        script_to_run = "src/run_car_racing.py"

    if script_to_run:
        try:
            subprocess.run([sys.executable, script_to_run] + unknown, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running script: {e}")
            sys.exit(1)
    else:
        print(f"Script '{args.script}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()

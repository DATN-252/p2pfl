import os
import time
import subprocess
from datetime import datetime

def run_experiments(config_dir="batch_configs"):
    # 1. Check if config directory exists
    if not os.path.exists(config_dir):
        print(f"❌ Directory '{config_dir}' not found. Please create it and add your .yaml files.")
        return

    # 2. Get list of yaml files
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') or f.endswith('.yml')]
    config_files.sort()

    if not config_files:
        print(f"⚠️ No configuration files found in '{config_dir}'.")
        return

    print(f"🚀 Found {len(config_files)} experiments. Starting batch run...")

    # 4-hour timeout per experiment (4 * 3600 seconds)
    EXP_TIMEOUT = 14400 

    for i, config_file in enumerate(config_files):
        config_path = os.path.join(config_dir, config_file)
        print("\n" + "="*60)
        print(f"🧪 [{i+1}/{len(config_files)}] Running: {config_file}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Timeout set to {EXP_TIMEOUT/3600} hours")
        print("="*60)

        try:
            # Run the experiment using p2pfl CLI with timeout
            result = subprocess.run(
                ["p2pfl", "run", config_path],
                check=True,
                text=True,
                timeout=EXP_TIMEOUT
            )
            print(f"✅ Finished experiment: {config_file}")
        except subprocess.TimeoutExpired:
            print(f"🚨 TIMEOUT: Experiment {config_file} exceeded {EXP_TIMEOUT/3600} hours.")
            print("🛑 Force killing any remaining Ray/Python processes...")
            # Extra safety: kill any zombies on this machine
            os.system("pkill -9 python; pkill -9 raylet")
            time.sleep(10) # Wait for OS to clean up
        except subprocess.CalledProcessError as e:
            print(f"❌ Error while running {config_file}: {e}")
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Exiting...")
            break

    print("\n" + "="*60)
    print("🎉 ALL EXPERIMENTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    run_experiments("batch_configs")

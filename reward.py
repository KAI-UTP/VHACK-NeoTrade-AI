import numpy as np
import os
import pandas as pd

REWARD_FILE = "rewards_history.npy"
CSV_REWARD_FILE = "rewards_history.csv"
MAX_REWARDS = 500000

def save_rewards(new_rewards):
    """Save rewards by appending new ones while keeping file size in check."""
    if not isinstance(new_rewards, np.ndarray):
        new_rewards = np.array(new_rewards)  # Convert to NumPy array if needed

    past_rewards = load_rewards()

    # ✅ Trim Old Rewards in `.npy` File
    if len(past_rewards) > MAX_REWARDS:
        print(f"♻️ Removing oldest {len(past_rewards) - MAX_REWARDS} rewards from .npy file.")
        past_rewards = past_rewards[-MAX_REWARDS:]  # Keep only the latest

    updated_rewards = np.concatenate([past_rewards, new_rewards])[-MAX_REWARDS:]
    np.save(REWARD_FILE, updated_rewards)  # ✅ Save latest rewards to `.npy`

    # ✅ Remove Old Data from CSV
    save_rewards_to_csv(updated_rewards)

    print(f"✅ Rewards saved! Total stored rewards: {len(updated_rewards)}")


def save_rewards_to_csv(rewards):
    """Save rewards to CSV, removing old entries when exceeding MAX_REWARDS."""
    df = pd.DataFrame(rewards, columns=["reward"])

    if os.path.exists(CSV_REWARD_FILE):
        existing_df = pd.read_csv(CSV_REWARD_FILE)

        # ✅ Merge new rewards with old rewards
        df = pd.concat([existing_df, df], ignore_index=True)

        # ✅ Keep only the last `MAX_REWARDS`
        if len(df) > MAX_REWARDS:
            print(f"♻️ Removing oldest {len(df) - MAX_REWARDS} rewards from CSV file.")
            df = df.iloc[-MAX_REWARDS:]  # Keep only latest rewards

    # ✅ Save back to CSV
    df.to_csv(CSV_REWARD_FILE, index=False)
    print(f"✅ CSV rewards saved. Total entries: {len(df)}")


def load_rewards():
    """Load saved rewards, return an empty array if file does not exist."""
    if os.path.exists(REWARD_FILE):
        return np.load(REWARD_FILE)
    return np.array([])  # Return empty array if no rewards exist


def get_recent_rewards(limit=100):
    """Retrieve the most recent `limit` rewards."""
    rewards = load_rewards()
    return rewards[-limit:] if len(rewards) >= limit else rewards


# ✅ Test Code (Uncomment to Run)
# if __name__ == "__main__":
#     save_rewards([0.1, -0.2, 0.5])
#     print("Loaded rewards:", load_rewards())
#     print("Recent rewards:", get_recent_rewards(3))

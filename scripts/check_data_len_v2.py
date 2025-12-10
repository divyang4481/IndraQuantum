import pickle
import sys
import os


def check_data():
    sys.stdout.reconfigure(encoding="utf-8")
    data_path = "data/mixed_train.pkl"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print(f"Loading {data_path}...")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    print(f"LEN_DATA: {len(data)}")
    print(f"TYPE_DATA: {type(data)}")
    if len(data) > 0:
        print("Sample 0 keys:", data[0].keys())
        print("Sample 0 input_ids len:", len(data[0]["input_ids"]))


if __name__ == "__main__":
    check_data()

import pickle
import sys
import os


def check_data():
    data_path = "data/mixed_train.pkl"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"Total samples: {len(data)}")
    if len(data) > 0:
        print("Sample 0 keys:", data[0].keys())
        print("Sample 0 input_ids len:", len(data[0]["input_ids"]))
        print("Sample 0 input_ids[:20]:", data[0]["input_ids"][:20])
        print("Sample 0 attention_mask[:20]:", data[0]["attention_mask"][:20])


if __name__ == "__main__":
    check_data()

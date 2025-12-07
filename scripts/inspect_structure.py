import pickle


def inspect_structure():
    with open("data/wikitext/train.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"Data Type: {type(data)}")
    if isinstance(data, list):
        print(f"List Length: {len(data)}")
        print(f"First Item Type: {type(data[0])}")
        print(f"First Item: {data[0]}")
    elif isinstance(data, dict):
        print(f"Dict Keys: {data.keys()}")
        # Check if it's huggingface dataset dict
        if "train" in data:
            print("Found 'train' key")


if __name__ == "__main__":
    inspect_structure()

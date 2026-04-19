import json

errors = []
with open("data/splits/training_data.jsonl") as f:
    for i, line in enumerate(f):
        if i == 7806:
            print(f"{line}")
            record = json.loads(line)
            print(record["services"])
            for j, turn in enumerate(record["conversations"]):
                print(f"Turn {j}: type={type(turn)}, value={str(turn)[:150]}")
            break

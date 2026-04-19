import json

parsed_records = []
with open("data/processed/sythetic_sgd_dataset.jsonl") as f:
    for line in f:
        # fix smart quotes
        line = line.replace("\u2019", "'").replace("\u2018", "'")
        line = line.replace("\u201c", '"').replace("\u201d", '"')
        try:
            record = json.loads(line)
            parsed_records.append(record)
        except json.JSONDecodeError as e:
            print(f"Skipping bad record: {e}")

print(f"Clean records: {len(parsed_records)}")

# save clean version
with open("data/processed/sythetic_sgd_dataset_parsed.jsonl", "w") as f:
    for record in parsed_records:
        f.write(json.dumps(record) + "\n")

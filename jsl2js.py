
import json

def convert_jsonl_to_json(jsonl_file: str, json_file: str) -> None:
    with open(jsonl_file, 'r', encoding='utf-8') as infile, open(json_file, 'w', encoding='utf-8') as outfile:
        data = [json.loads(line) for line in infile]
        json.dump(data, outfile, ensure_ascii=False, indent=2)
def main():
    jsonl_file = 'build\\qdrant_payload_preview.jsonl'  # Replace with your input JSONL file path
    json_file = 'output.json'    # Replace with your desired output JSON file path
    convert_jsonl_to_json(jsonl_file, json_file)
    print(f"Converted {jsonl_file} to {json_file}")
if __name__ == "__main__":
    main()
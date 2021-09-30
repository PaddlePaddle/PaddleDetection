import yaml
import json
import sys

yamlf = sys.argv[1]

assert yamlf.endswith(".yml")

with open(yamlf, 'r') as rf:
    yaml_data = yaml.safe_load(rf)

jsonf = yamlf[:-4] + ".json"
with open(jsonf, 'w') as wf:
    json.dump(yaml_data, wf, indent=4)


import json

with open('res.json', 'r') as f:
    data = json.load(f)

correct = [
    '[{"task": "image-cls", "id": 0, "dep": [-1], "args": {"image": "input.jpg" }}]'.replace(' ', ''),
    '[{"task": "image-to-text", "id": 0, "dep":[-1], "args": {"image": "input.jpg" }}]'.replace(' ', ''),
    '[{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "input.jpg"}}]'.replace(' ', ''),
    '[{"task": "image-to-text", "id": 0, "dep":[-1], "args": {"image": "input.jpg" }}, {"task":"image-cls", "id": 1, "dep": [-1], "args": {"image": "input.jpg" }}]'.replace(' ', ''),
]

def classify_respond(text):
    try:
        obj = json.loads(text)
        if (type(obj) is not list):
            return 'not a list'
        if (text.replace(' ', '') in correct):
            return 'correct'

        print(text)
    except json.decoder.JSONDecodeError:
        return 'not json'

counts = dict()

for line in data:
    cls = classify_respond(line)
    if (cls not in counts):
        counts[cls] = 0
    counts[cls] += 1
print(counts)


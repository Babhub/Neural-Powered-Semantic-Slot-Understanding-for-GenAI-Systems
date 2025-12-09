#!/bin/bash

echo "Creating hw3_submission.zip using Python..."

python3 - << 'PYEOF'
import zipfile

files = [
    "main.py",
    "test_pred.csv",
    "requirements.txt",
    "hw3_report.pdf"  # Make sure you have this file in HW3 folder
]

with zipfile.ZipFile("hw3_submission.zip", "w") as z:
    for f in files:
        try:
            z.write(f)
            print("Added:", f)
        except:
            print("Missing file:", f)

print("Done! hw3_submission.zip created.")
PYEOF

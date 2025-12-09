#!/bin/bash

# Zip the files for submission
# Usage: ./zip_for_submission.sh


# If you have additional files, you need to add them to this script
if [ -f pyproject.toml ]; then
  zip -r hw3_submission.zip main.py test_pred.csv pyproject.toml hw3_report.pdf
elif [ -f requirements.txt ]; then
  zip -r hw3_submission.zip main.py test_pred.csv requirements.txt hw3_report.pdf
else
  echo "Error: No dependencies file found. Please provide either \`pyproject.toml\` or \`requirements.txt\`." >&2
  exit 1
fi

echo "Zipped files for submission: hw3_submission.zip"

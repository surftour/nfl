#!/bin/bash

# Run pylint on the entire src/ directory
echo "Running pylint on src/ directory..."
pylint src/

# Optional: Run on specific modules
# echo "Running pylint on specific modules..."
# pylint src/api/espn.py
# pylint src/utils/data_io.py
# pylint src/data/weekly_reader.py

echo "Pylint checks completed."
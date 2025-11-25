#!/usr/bin/env bash

# Simple helper script to run the AutoDoc-LiteLLM demo

# Exit on first error
set -e

# Optional: create virtualenv (comment out if you don't want it)
# python -m venv .venv
# source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running demo..."
python main.py


#!/bin/bash

# Simple runner script for AutoDoc-LiteLLM

if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set."
    exit 1
fi

python main.py


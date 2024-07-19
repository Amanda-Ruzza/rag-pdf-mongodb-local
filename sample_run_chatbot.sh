#!/bin/bash

# This is a bash script to activate the `chatbot-env` virtual environment, and then run streamlit

# Check if the virtual environment is already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
  # Define the path to your virtual environment
  VENV_PATH="/<YOUR-LOCAL-PATH-TO-VENV-FOLEDER>/rag-pdf-mongodb-local/chatbot-env"

  # Activate the virtual environment
  source "$VENV_PATH/bin/activate"
fi

# Run Streamlit
streamlit run chatbot-app.py

#!/usr/bin/env python
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try running the dashboard app
import streamlit.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app/main.py", "--server.port=8502"]
    stcli.main()

#!/usr/bin/env python
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now run the producer
from api.crypto_producer import produce

if __name__ == "__main__":
    produce()

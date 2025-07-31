import os
import requests
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from pathlib import Path
import json
import dspy
from dataclasses import dataclass
from enum import Enum
import datetime
import pandas as pd
import numpy as np
import re
from retrieval.filerag import create_context, Chunk, metadata

# context = create_context(datadir=Path("data/filings"), embeddings_dir=Path("data/embeddings"), should_embed=False)

lm = dspy.LM("deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_TOKEN_DAYTADING"), api_base="https://api.groq.com/openai/v1/chat/completions", temperature=0.5, max_tokens=4000, stop=None)
dspy.configure(lm=lm)

print(dspy.Predict("sentence -> summary")(sentence="What comes after hello in beginner computer programs?"))
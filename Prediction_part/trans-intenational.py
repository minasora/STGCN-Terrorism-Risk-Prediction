# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd

data = pd.read_excel("dataset/AFG International.xlsx")

INT_LOG = data["INT_LOG"]
INT_IDEO = data["INT_IDEO"]
INT_MISC = data["INT_MISC"]
INT_ANY = data["INT_ANY"]

print("INT_LOG: ", Counter(INT_LOG))
print("INT_IDEO: ", Counter(INT_IDEO))
print("INT_MISC: ", Counter(INT_MISC))
print("INT_ANY: ", Counter(INT_ANY))

import pandas as pd
import numpy as np
from io import StringIO

with open('https://sportedgegolf.slack.com/files/U059KPH86CB/F05QZKBFRJN/course2023.txt', 'r', encoding='utf-8', errors='ignore') as file:
    content = file.read()

raw_course_stats = pd.read_csv(StringIO(content), delimiter=';')
raw_course_stats = raw_course_stats.rename(columns={col: col.replace('#', '') for col in raw_shot_stats.columns})
raw_course_stats.columns = raw_course_stats.columns.str.strip()
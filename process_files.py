from tqdm import tqdm
import time
from random import randint
for minute in tqdm(range(120), desc="Processing files (minutes)"):
    time.sleep(randint(15, 60))
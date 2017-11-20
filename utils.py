import pandas as pd

def make_transcript(dfs,idx=6):
    words = ''
    for df in dfs:
        words += ' '.join(list(df[idx]))
    return words
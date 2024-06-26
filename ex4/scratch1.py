
import pandas as pd
import numpy as np
import timeit

# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# New rows as a list of lists
new_rows = [[10, 11, 12] for _ in range(1000)]

# Using loc
def add_rows_loc(df, new_rows):
    for new_row in new_rows:
        df.loc[len(df)] = new_row
    return df

# Using concat instead of append
def add_rows_concat(df, new_rows):
    new_rows_df = pd.DataFrame(new_rows, columns=df.columns)
    df = pd.concat([df, new_rows_df], ignore_index=True)
    return df

# Measure time for loc
df_loc = df.copy()
time_loc = timeit.timeit(lambda: add_rows_loc(df_loc, new_rows), number=1)
print(f"Time using loc: {time_loc:.4f} seconds")

# Measure time for concat
df_concat = df.copy()
time_concat = timeit.timeit(lambda: add_rows_concat(df_concat, new_rows), number=1)
print(f"Time using concat: {time_concat:.4f} seconds")

"""
GPT analysis:
In general:
    Using loc: Fast and efficient for adding individual rows incrementally but may become inefficient in a loop due to repeated reallocation.
    Using append: Slowest for adding multiple rows in a loop due to repeated creation of new DataFrames.
    Using concat: Most efficient for adding multiple rows at once, especially for large DataFrames or large numbers of rows.

Recommendation:
    For adding a few rows incrementally: Use loc.
    For adding a large number of rows: Use concat to batch the additions and then concatenate them all at once.
"""

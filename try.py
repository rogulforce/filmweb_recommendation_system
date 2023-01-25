import pandas as pd
df = pd.DataFrame(
    [
        {"name": "a", "first":1, "second":1},
        {"name": "b", "first":1, "second":2},
        {"name": "c", "first":2, "second":1},
        {"name": "d", "first":2, "second":2},
        ])

df2 = pd.DataFrame(
    [
        {"name": "d", "first":41, "second":1},
        {"name": "a", "first":11, "second":4},
        # {"name": "c", "first":31, "second":2},
        {"name": "b", "first":21, "second":3},
        ])


print(df[(df["first"] == 1) & (df["second"] == 1)])


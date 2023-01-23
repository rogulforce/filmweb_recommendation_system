import pandas as pd
df = pd.DataFrame(
    [
        {"name": "a", "first":11, "second":4},
        {"name": "b", "first":21, "second":3},
        {"name": "c", "first":31, "second":2},
        {"name": "d", "first":41, "second":1},
        ])

df2 = pd.DataFrame(
    [
        {"name": "d", "first":41, "second":1},
        {"name": "a", "first":11, "second":4},
        # {"name": "c", "first":31, "second":2},
        {"name": "b", "first":21, "second":3},
        ])

# print(df.sort_values("first", ascending=False).sort_values("second", ascending=False))
# print(df[df["first"] > 30].__len__())

# print(df["first"]+ df2["first"])
# df.add(df2)
print(df["first"].add(df2["first"], fill_value=0))
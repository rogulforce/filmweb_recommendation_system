import pandas as pd
import numpy as np
from typing import Dict

from model.Recommender import Recommender
from scrap_data.data_import import load_data
from clean_data import clean_data
test_percent = 10
movies_test_percent = 10

# load and clean data
df_user, df_movie = load_data()

user_dict: Dict[int, str] = {i: user for i, user in enumerate(df_user.User.unique())}
movie_dict: Dict[int, str] = {i: title for i, title in enumerate(df_movie.Title.unique())}
df_user, df_movie = clean_data(df_user, df_movie)
# print(df_user, df_movie)
# print(user_dict)
# exit()
# separate data
user_ids = np.array(range(len(user_dict)))
np.random.shuffle(user_ids)

test_users = user_ids[:len(user_dict)*test_percent//100]
fit_users = user_ids[len(user_dict)*test_percent//100:]
del user_ids
df_test_users = df_user[df_user.User.isin([ID for ID in test_users])]
df_test_users_request = pd.DataFrame(columns=df_test_users.columns)
df_test_users_check = pd.DataFrame(columns=df_test_users.columns)
for user_id in test_users:
    tmp = df_test_users[df_test_users["User"] == user_id]
    n = max(
        1,
        (len(tmp) * movies_test_percent) // 100
    )
    df_test_users_request = pd.concat(
        [
            df_test_users_request,
            tmp.iloc[n:],
        ]
    )
    df_test_users_check = pd.concat(
        [
            df_test_users_check,
            tmp.iloc[:n],
        ]
    )

df_user_fit = df_user[df_user.User.isin([ID for ID in fit_users])]
# use model
recom = Recommender(df_movie)

recom.train(df_user_fit)

results = recom.predict(
    df_user=df_test_users_request, num_of_recomendations=-1
)

# count loss
dif = 0
k = 0
for user_id in test_users:
    tmp = results[results["User"] == user_id]
    n = len(tmp)
    if n == 0:
        k+=1
        continue
    s = 0
    for mv_id in df_test_users_check[df_test_users_check["User"] == user_id].Title:
        try:
            s += (
                df_test_users_check[
                    (df_test_users_check.User == user_id) & (df_test_users_check.Title == mv_id)
                ]["Rating"].array[0] - tmp[
                    (tmp.User == user_id) & (tmp.Title == mv_id)
                ]["Rating"].array[0])**2
                
        except:
            k += 1
    dif += s/n 

print(dif)
dif /= len(test_users) - k
print(dif)



# print(results)

import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np

dh = importlib.import_module("eclass-public.DataHelper")
data = dh.eclass_data(route='eclass-public/')
preids = [x[:-1] for i, x in enumerate(data.pre_survey_question_ids) if i % 2 < 1]

df = pd.read_csv("see_processed.csv")

# construct lists of columns to calculate means for
a_pre = []
b_pre = []
a_post = []
b_post = []
delta_ids = []
delta_ids_post = []
for id in preids:
    a_pre.append(id + "a_pre")
    b_pre.append(id + "b_pre")
    a_post.append(id + "a_post")
    b_post.append(id + "b_post")
    delta_ids.append("delta_pre"+id)
    delta_ids_post.append("delta_post"+id)


# set up to convert numbers of questions to actual Q numbers in correct order
zwickl_map = pd.read_excel("eclass-public/question_lookup.xlsx", sheet_name="anon_pre") # mapping is the same for pre/post
zwickl_map = zwickl_map[["Question ID","ThirtyQ"]].dropna()
# actually use the collapsed form so it goes uniformly from 1 - 30 (removing and shifting the validation question)
zwickl_dict = dict(zip(zwickl_map["Question ID"],zwickl_map["ThirtyQ"]))
real_qnums = [zwickl_dict[x+'a'] for x in preids]
real_qnums = ["q"+x[0:-1] for x in real_qnums]


def make_mean_df(df):
    # construct new data frame of means
    N = len(df)
    mean_df = pd.DataFrame([df[a_pre].sum(axis=1).mean(),df[a_pre].sum(axis=1).sem() ,
                            df[a_post].sum(axis=1).mean() ,df[a_post].sum(axis=1).sem() ,
                            df[delta_ids].sum(axis=1).mean() ,df[delta_ids].sum(axis=1).sem() ,
                            df[delta_ids_post].sum(axis=1).mean() ,df[delta_ids_post].sum(axis=1).sem() ],
                            index=["pre_mean","pre_sem","post_mean","post_sem","see_pre_mean","see_pre_sem","see_post_mean","see_post_sem"])

    mean_df = mean_df.transpose()

    mean_df['prepost_you'] = mean_df['pre_mean']-mean_df['post_mean']
    mean_df['prepost_you_pooled_sd'] = np.sqrt((N-1)*np.power(mean_df['pre_sem'],2)*N + (N-1)*np.power(mean_df['post_sem'],2)*N)/np.sqrt(2*N-2)
    mean_df['prepost_you_cohens_d'] = mean_df['prepost_you']/mean_df['prepost_you_pooled_sd']

    mean_df['prepost_see'] = mean_df['see_pre_mean']-mean_df['see_post_mean']
    mean_df['prepost_see_pooled_sd'] = np.sqrt((N-1)*np.power(mean_df['see_pre_sem'],2)*N + (N-1)*np.power(mean_df['see_post_sem'],2)*N)/np.sqrt(2*N-2)
    mean_df['prepost_see_cohens_d'] = mean_df['prepost_see']/mean_df['prepost_see_pooled_sd']

    # estimate sample variances on cohens d
    # https://stats.stackexchange.com/questions/8487/how-do-you-calculate-confidence-intervals-for-cohens-d
    # purpotedly from http://books.google.com/books?id=cQxN792ttyEC p238 of  The Handbook of Research Synthesis
    # (n1+n2/n1*n2+d^2/2(n1+n2−2))*(n1+n2/n1+n2−2)
    # which we simplify as we know n1 = n2 = N

    mean_df['prepost_you_cohens_d_stddev'] = np.sqrt( (2/N + np.power(mean_df['prepost_you_cohens_d'],2)/(2*(2*N-2)))*(2*N/(2*N-2)))
    mean_df['prepost_see_cohens_d_stddev'] = np.sqrt( (2/N + np.power(mean_df['prepost_see_cohens_d'],2)/(2*(2*N-2)))*(2*N/(2*N-2)))

    return mean_df.transpose()

# calculate overall means
print(df["Q54"]==1)
print(df["Q47"].isin([1,6,7,8]))
N_women = len(df[(df["Q54"]==1)])
N_men = len(df[(df["Q54"]==2)])
N_gnc = len(df[(df["Q54"]==3)])
mean_all = make_mean_df(df)
mean_women = make_mean_df(df[(df["Q54"]==1)])
mean_men =   make_mean_df(df[(df["Q54"]==2)])
mean_gnc =   make_mean_df(df[(df["Q54"]==3)])

print(mean_all)
print(mean_women)
print(mean_men)
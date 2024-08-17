import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import scipy.stats as ss
import math

ticklabelsize = 15
labelsize = 15

params = {'axes.labelsize':labelsize,
         'xtick.labelsize':ticklabelsize,
         'ytick.labelsize':ticklabelsize}
pylab.rcParams.update(params)

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
    mean_df = pd.DataFrame([df[a_pre].sum(axis=1),
                            df[a_post].sum(axis=1),
                            df[delta_ids].sum(axis=1),
                            df[delta_ids_post].sum(axis=1)],
                            index=["pre_you","post_you","pre_see","post_see"])

    mean_df = mean_df.transpose()

    mean_df['prepost_you'] = mean_df['pre_you']-mean_df['post_you']
    mean_df['prepost_see'] = mean_df['pre_see']-mean_df['post_see']
    
    return mean_df

# calculate overall means
print(df["Q54"]==1)
print(df["Q47"].isin([1,6,7,8]))
N = len(df)
N_women = len(df[(df["Q54"]==1)])
N_men = len(df[(df["Q54"]==2)])
N_gnc = len(df[(df["Q54"]==3)])
mean_all = make_mean_df(df)
mean_women = make_mean_df(df[(df["Q54"]==1)])
mean_men =   make_mean_df(df[(df["Q54"]==2)])
mean_gnc =   make_mean_df(df[(df["Q54"]==3)])

color = (0,0,200/255)

plt.figure()
mean_all['pre_you'].plot.hist(bins=np.arange(-30.5,30.5,1), color=color)
plt.xlabel("Pre-test YOU score")
plt.savefig("pre_you_dist.png")
plt.close()

plt.figure()
mean_all['post_you'].plot.hist(bins=np.arange(-30.5,30.5,1), color=color)
plt.xlabel("Post-test YOU score")
plt.savefig("post_you_dist.png")
plt.close()

fig, ax1 = plt.subplots(1,1, figsize=(12,8))
print(np.amax(mean_all['pre_see']),np.amin(mean_all['pre_see']))
mean_all['pre_see'].plot.hist(bins=np.arange(-87.5,40.5,1), color=color)
plt.xlabel("Pre-test SEE score")
plt.savefig("pre_see_dist.png")
plt.close()

fig, ax1 = plt.subplots(1,1, figsize=(12,8))
print(np.amax(mean_all['post_see']),np.amin(mean_all['post_see']))
mean_all['post_see'].plot.hist(bins=np.arange(-87.5,40.5,1), color=color)
plt.xlabel("Post-test SEE score")
plt.savefig("post_see_dist.png")
plt.close()

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
plt.scatter(mean_all['pre_you'],mean_all['pre_see'],marker='o',color=(1,0,0,0.05))#,edgecolors='k')
x = np.arange(-30,30,1)
#plt.plot(x,x,linestyle='--',color='k')
plt.xlim(-35,35)
plt.ylim(-90,40)
plt.ylabel("Pre-test SEE score")
plt.xlabel("Pre-test YOU score")
plt.savefig("pre_see_you_correlation.png")
plt.close()




def conf_interval(r, N, alpha=0.05):
    # from stat.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    first = math.atanh(r)
    second = np.sqrt((1 + np.power(r,2)/2)/(N-3))*ss.norm.ppf(alpha/2)
    lower = math.tanh(first-second)
    upper = math.tanh(first+second)
    diff = (lower-upper)/2
    return lower, upper, diff


res, pvalue = ss.spearmanr(mean_all['pre_you'], mean_all['pre_see'])

print(res)
print(conf_interval(res, N))
print(pvalue)


res, pvalue = ss.spearmanr(mean_all['pre_you'], mean_all['post_you'])

print(res)
print(conf_interval(res, N))
print(pvalue)
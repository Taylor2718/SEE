import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import scipy.stats as ss
import math

dh = importlib.import_module("eclass-public.DataHelper")
data = dh.eclass_data(route='eclass-public/')
preids = [x[:-1] for i, x in enumerate(data.pre_survey_question_ids) if i % 2 < 1]

df = pd.read_csv("see_processed.csv")

# construct lists of columns to calculate means for
a_pre = []
b_pre = []
a_post = []
b_post = []
# Note delta here corresponds to the item SEE score
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
    mean_df = pd.DataFrame([df[a_pre].mean().to_numpy(),df[a_pre].sem().to_numpy(),
                            df[a_post].mean().to_numpy(),df[a_post].sem().to_numpy(),
                            df[delta_ids].mean().to_numpy(),df[delta_ids].sem().to_numpy(),
                            df[delta_ids_post].mean().to_numpy(),df[delta_ids_post].sem().to_numpy()],
                            columns=real_qnums,
                            index=["pre_mean","pre_sem","post_mean","post_sem","see_pre_mean","see_pre_sem","see_post_mean","see_post_sem"])

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




output_dir = ''
pre_color = (0,0/255.,200/255.)
#post_color= (252/255.,125/255.,125/255.)

labelsize = 20
ticklabelsize = 15
textsize = 10
marker = 'o'
markersize = 5

params = {'xtick.labelsize':ticklabelsize,
         'ytick.labelsize':ticklabelsize}
pylab.rcParams.update(params)

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
mean_all.plot(x='pre_mean', xerr='pre_sem', y='see_pre_mean', yerr='see_pre_sem', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color=pre_color, legend=False)


ax1.set_ylabel('Mean SEE pre score', fontsize=labelsize)
ax1.set_xlabel('Mean E-CLASS YOU pre score', fontsize=labelsize)



ax1.set_xlim(-0.4, 1.05)
ax1.set_ylim(-1.6, 0.35)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')




xshift = 0.015
yshift = -0.01
# add question number labels to plot
for index, row in mean_all.iterrows():
    # special cases
    if index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)


plt.tight_layout()
plt.savefig(output_dir + "you_see_pre_plot.png", dpi=300)
plt.close()

N = len(mean_all)

def conf_interval(r, N, alpha=0.05):
    # from stat.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    first = math.atanh(r)
    second = np.sqrt((1 + np.power(r,2)/2)/(N-3))*ss.norm.ppf(alpha/2)
    lower = math.tanh(first-second)
    upper = math.tanh(first+second)
    diff = (lower-upper)/2
    return lower, upper, diff


print(mean_all)
res, pvalue = ss.spearmanr(mean_all['pre_mean'], mean_all['see_pre_mean'])

print(res)
print(conf_interval(res, N))
print(pvalue)




# Now make the gender plot
fig, ax1 = plt.subplots(1,1, figsize=(8,8))
mean_women.plot(x='pre_mean', xerr='pre_sem', y='see_pre_mean', yerr='see_pre_sem', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color="red", legend=False)
mean_men.plot(x='pre_mean', xerr='pre_sem', y='see_pre_mean', yerr='see_pre_sem', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color="blue", legend=False)

ax1.set_ylabel('Mean SEE pre score', fontsize=labelsize)
ax1.set_xlabel('Mean E-CLASS YOU pre score', fontsize=labelsize)



ax1.set_xlim(-0.6, 1.05)
ax1.set_ylim(-2, 0.35)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')




xshift = 0.015
yshift = -0.01
# add question number labels to plot
for index, row in mean_women.iterrows():
    # special cases
    if index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

for index, row in mean_men.iterrows():
    # special cases
    if index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

plt.tight_layout()
plt.savefig(output_dir + "you_see_pre_plot_gender.png", dpi=300)
plt.close()
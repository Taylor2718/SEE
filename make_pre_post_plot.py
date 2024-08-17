import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as pylab



dh = importlib.import_module("eclass-public.DataHelper")
data = dh.eclass_data(route='eclass-public/')
preids = [x[:-1] for i, x in enumerate(data.pre_survey_question_ids) if i % 2 < 1]

df = pd.read_csv("see_processed.csv")
N = len(df)

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
    mean_df = pd.DataFrame([df[a_pre].mean().to_numpy(),df[a_pre].sem().to_numpy(),
                            df[a_post].mean().to_numpy(),df[a_post].sem().to_numpy(),
                            df[delta_ids].mean().to_numpy(),df[delta_ids].sem().to_numpy(),
                            df[delta_ids_post].mean().to_numpy(),df[delta_ids_post].sem().to_numpy()],
                            columns=real_qnums,
                            index=["pre_mean","pre_sem","post_mean","post_sem","see_pre_mean","see_pre_sem","see_post_mean","see_post_sem"])

    mean_df = mean_df.transpose()
    mean_df["shift_you"] = mean_df['post_mean']-mean_df['pre_mean']
    mean_df['shift_you_sem'] = np.sqrt(np.power(mean_df['pre_sem'],2)+np.power(mean_df['post_sem'],2))
    mean_df['shift_you_pooled_std_dev'] = np.sqrt((N-1)*np.power(mean_df['pre_sem'],2)*N+(N-1)*np.power(mean_df['post_sem'],2)*N)/np.sqrt(2*N-2)
    mean_df['shift_you_cohens_d'] = mean_df['shift_you']/mean_df['shift_you_pooled_std_dev']
    # estimate sample variances on cohens d
    # https://stats.stackexchange.com/questions/8487/how-do-you-calculate-confidence-intervals-for-cohens-d
    # purpotedly from http://books.google.com/books?id=cQxN792ttyEC p238 of  The Handbook of Research Synthesis
    # (n1+n2/n1*n2+d^2/2(n1+n2−2))*(n1+n2/n1+n2−2)
    # which we simplify as we know n1 = n2 = N
    mean_df['shift_you_cohens_d_stderr'] = np.sqrt( (2/N + np.power(mean_df['shift_you_cohens_d'],2)/(2*(2*N-2)))*(2*N/(2*N-2)))

    mean_df["shift_see"] = mean_df['see_post_mean']-mean_df['see_pre_mean']
    mean_df['shift_see_sem'] = np.sqrt(np.power(mean_df['see_pre_sem'],2)+np.power(mean_df['see_post_sem'],2))
    mean_df['shift_see_pooled_std_dev'] = np.sqrt((N-1)*np.power(mean_df['see_pre_sem'],2)*N+(N-1)*np.power(mean_df['see_post_sem'],2)*N)/np.sqrt(2*N-2)
    mean_df['shift_see_cohens_d'] = mean_df['shift_see']/mean_df['shift_see_pooled_std_dev']
    mean_df['shift_see_cohens_d_stderr'] = np.sqrt( (2/N + np.power(mean_df['shift_see_cohens_d'],2)/(2*(2*N-2)))*(2*N/(2*N-2)))

    return mean_df

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


def make_plot(mean_df, ax1, pre_color, textsize=10):
 

    marker = 'o'
    markersize = 5
    
    mean_df.plot(x='shift_you', xerr='shift_you_sem', y='shift_see', yerr='shift_see_sem', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color=pre_color, legend=False)

    return ax1



output_dir = ''
pre_color = (0,0/255.,200/255.)
#post_color= (252/255.,125/255.,125/255.)

labelsize = 20
ticklabelsize = 15
textsize = 10

params = {'xtick.labelsize':ticklabelsize,
         'ytick.labelsize':ticklabelsize}
pylab.rcParams.update(params)

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
ax1 = make_plot(mean_all, ax1, pre_color, textsize=textsize)

ax1.set_xlim(-0.2,0.15)
ax1.set_ylim(-0.15,0.1)


#ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=ticklabelsize)
#ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=ticklabelsize)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')


xshift = -0.012
yshift = 0.00325
# add question number labels to plot
for index, row in mean_all.iterrows():
    # special cases
    if index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['shift_you']+xshift+0.005,row['shift_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['shift_you']+xshift,row['shift_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

ax1.plot(np.arange(-10,40),np.arange(-10,40),color='gray',linestyle='--')

ax1.set_ylabel('Change in mean SEE score (post-pre)', fontsize=labelsize)
ax1.set_xlabel('Change in mean E-CLASS YOU score (post-pre)', fontsize=labelsize)

plt.tight_layout()
plt.savefig(output_dir + "shift_pre_post_SEE_vs_YOU.png", dpi=300)
plt.close()



#----------------------------------------------
# Now make this same plot but with effect sizes (cohen's d) rather than raw shifts


output_dir = ''
pre_color = (0,0/255.,200/255.)
#post_color= (252/255.,125/255.,125/255.)

labelsize = 20
ticklabelsize = 15
textsize = 10

params = {'xtick.labelsize':ticklabelsize,
         'ytick.labelsize':ticklabelsize}
pylab.rcParams.update(params)

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
marker = 'o'
markersize = 5
    
mean_all.plot(x='shift_you_cohens_d', xerr='shift_you_cohens_d_stderr', y='shift_see_cohens_d', yerr='shift_see_cohens_d_stderr', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color=pre_color, legend=False)

ax1.set_xlim(-0.2,0.175)
ax1.set_ylim(-0.15,0.1)


#ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=ticklabelsize)
#ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=ticklabelsize)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')


xshift = -0.012
yshift = 0.00325
# add question number labels to plot
for index, row in mean_all.iterrows():
    # special cases
    if index in ["q1","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['shift_you_cohens_d']+xshift+0.005,row['shift_see_cohens_d']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index in ["q2"]:
        plt.text(row['shift_you_cohens_d']-xshift-0.011,row['shift_see_cohens_d']-yshift-0.0015,r'$'+index[1:]+'$', fontsize=textsize)
    elif index in ["q12","q13"]:
        plt.text(row['shift_you_cohens_d']+xshift,row['shift_see_cohens_d']+yshift-0.0025,r'$'+index[1:]+'$', fontsize=textsize)
    elif index == "q30":
        plt.text(row['shift_you_cohens_d']+xshift+0.015,row['shift_see_cohens_d']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['shift_you_cohens_d']+xshift,row['shift_see_cohens_d']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

ax1.plot(np.arange(-10,40),np.arange(-10,40),color='gray',linestyle='--')

ax1.set_ylabel('Cohen\'s d SEE score (post-pre)', fontsize=labelsize)
ax1.set_xlabel('Cohen\'s d E-CLASS YOU score (post-pre)', fontsize=labelsize)

plt.tight_layout()
plt.savefig(output_dir + "shift_pre_post_SEE_vs_YOU_cohens_d.png", dpi=300)
plt.close()

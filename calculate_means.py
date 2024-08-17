import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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
    mean_df = pd.DataFrame([df[a_pre].mean().to_numpy(),df[a_pre].sem().to_numpy(),
                            df[a_post].mean().to_numpy(),df[a_post].sem().to_numpy(),
                            df[delta_ids].mean().to_numpy(),df[delta_ids].sem().to_numpy(),
                            df[delta_ids_post].mean().to_numpy(),df[delta_ids_post].sem().to_numpy()],
                            columns=real_qnums,
                            index=["pre_mean","pre_sem","post_mean","post_sem","see_pre_mean","see_pre_sem","see_post_mean","see_post_sem"])

    return mean_df

# calculate overall means
print(df["Q54"]==1)
print(df["Q47"].isin([1,6,7,8]))
N_women = len(df[(df["Q54"]==1) & (df["Q47"].isin([1,6,7,8]))])
N_men = len(df[(df["Q54"]==2) & (df["Q47"].isin([1,6,7,8]))])
N_gnc = len(df[(df["Q54"]==3) & (df["Q47"].isin([1,6,7,8]))])
mean_all = make_mean_df(df[ df["Q47"].isin([1,6,7,8])]).transpose()
mean_women = make_mean_df(df[(df["Q54"]==1) & (df["Q47"].isin([1,6,7,8]))]).transpose()
mean_men =   make_mean_df(df[(df["Q54"]==2) &  (df["Q47"].isin([1,6,7,8]))]).transpose()
mean_gnc =   make_mean_df(df[(df["Q54"]==3) &  (df["Q47"].isin([1,6,7,8]))]).transpose()


def make_plot(mean_df, ax1, pre_color, textsize=10):
 

    marker = 'o'
    markersize = 5
    


    
    mean_df.plot(x='pre_mean', xerr='pre_sem', y='see_pre_mean', yerr='see_pre_sem', marker=marker, markersize=markersize, linestyle="None", ax=ax1, color=pre_color, legend=False)

   


    xshift = -0.02
    yshift = 0.0325
    # add question number labels to plot
    for index, row in mean_df.iterrows():
        # special cases
        if index == "q30":
            plt.text(row['pre_mean']+xshift+0.03,row['see_pre_mean']-yshift,r'$'+index[1:]+'$', fontsize=textsize)
        elif index == "q5":
            plt.text(row['pre_mean']+xshift+0.02,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
        elif index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
            plt.text(row['pre_mean']+xshift+0.01,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
        else:
            plt.text(row['pre_mean']+xshift,row['see_pre_mean']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

    #ax1.set_aspect('equal')
    #ax1.set_axisbelow(True)
    #plt.tight_layout()
    return ax1



output_dir = ''
pre_color = (0,255/255.,255/255.)
#post_color= (252/255.,125/255.,125/255.)

labelsize = 20
ticklabelsize = 15
textsize = 10


fig, ax1 = plt.subplots(1,1, figsize=(8,8))
ax1.set_ylabel('Mean SEE pre score', fontsize=labelsize)
ax1.set_xlabel('Mean E-CLASS YOU pre score', fontsize=labelsize)


ax1.set_xlim(-0.45, 1.05)
#ax1.set_ylim(-0.35,0.1)
ax1.set_ylim(-1.5, 0.35)

ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=ticklabelsize)
ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=ticklabelsize)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')

ax1 = make_plot(mean_all, ax1, pre_color, textsize=textsize)


plt.savefig(output_dir + "you_see_pre_plot_physics_majors.png", dpi=300)
plt.close()



fig, ax1 = plt.subplots(1,1, figsize=(8,8))
ax1.set_ylabel('Mean SEE pre score', fontsize=labelsize)
ax1.set_xlabel('Mean E-CLASS YOU pre score', fontsize=labelsize)


ax1.set_xlim(-0.45, 1.05)
#ax1.set_ylim(-0.45,0.1)
ax1.set_ylim(-1.5, 0.35)

ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontsize=ticklabelsize)
ax1.set_xticklabels(labels=ax1.get_xticklabels(), fontsize=ticklabelsize)

ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='major')

ax1.plot(np.arange(-10,40),np.zeros(50),color='k')
ax1.plot(np.zeros(50),np.arange(-10,40),color='k')

ax1 = make_plot(mean_women, ax1, "red", textsize=textsize)
ax1 = make_plot(mean_men, ax1, "blue", textsize=textsize)


plt.savefig(output_dir + "you_see_pre_plot_gender_physics_majors.png", dpi=300)
plt.close()

# now let's calculate the difference between men and women on both axes so we can identify the
# items with the smallest pre difference but largest SEE difference 
# This then shows that SEE is measuring something different

diff_mw_you = mean_women['pre_mean'] - mean_men['pre_mean']
diff_mw_see = mean_women['see_pre_mean'] - mean_men['see_pre_mean']
#print(diff_mw_you.sort_values())
#print(diff_mw_see[(diff_mw_you < 0.015) & (diff_mw_you > -0.015)].sort_values())
#print(diff_mw_see.sort_values())

# and normalise the differences by the standard error on the differences
diff_mw_you_sem = np.sqrt(np.power(mean_women['pre_sem'],2) + np.power(mean_men['pre_sem'],2))
diff_mw_see_sem = np.sqrt(np.power(mean_women['see_pre_sem'],2) + np.power(mean_men['see_pre_sem'],2))

diff_mw_you_std = np.sqrt(np.power(mean_women['pre_sem']*np.sqrt(N_women),2) + np.power(mean_men['pre_sem']*np.sqrt(N_men),2))
diff_mw_see_std = np.sqrt(np.power(mean_women['see_pre_sem']*np.sqrt(N_women),2) + np.power(mean_men['see_pre_sem']*np.sqrt(N_men),2))

pooled_mw_you_std = np.sqrt((N_women-1)*np.power(mean_women['pre_sem']*np.sqrt(N_women),2) + (N_men-1)*np.power(mean_men['pre_sem']*np.sqrt(N_men),2))/np.sqrt(N_women+N_men-2)
pooled_mw_see_std = np.sqrt((N_women-1)*np.power(mean_women['see_pre_sem']*np.sqrt(N_women),2) + (N_men-1)*np.power(mean_men['see_pre_sem']*np.sqrt(N_men),2))/np.sqrt(N_women+N_men-2)

normed_mw_you = diff_mw_you/pooled_mw_you_std
normed_mw_see = diff_mw_see/pooled_mw_see_std

print(normed_mw_you)
print(normed_mw_see)

mw_df = pd.DataFrame([normed_mw_you,normed_mw_see],index=["wm_you","wm_see"]).transpose()

print(mw_df)


xshift = -0.002
yshift = 0.00325

plt.figure()
#plt.scatter(x=normed_mw_you.to_numpy(), y=normed_mw_see.to_numpy())
ax = mw_df.plot(x="wm_you",y="wm_see",linestyle="None",marker="o", color='red', legend=None)
for index, row in mw_df.iterrows():
    # special cases
    if index == "q30":
        plt.text(row['wm_you']+xshift+0.03,row['wm_see']-yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index == "q5":
        plt.text(row['wm_you']+xshift+0.02,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['wm_you']+xshift+0.01,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['wm_you']+xshift,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

#colors = ['gray','gray','black','blue','green','orange','red']
#for r in range(1,8):
#    circle = plt.Circle((0, 0), r, color=colors[r-1], fill=False)
#    ax.add_patch(circle)


ax.set_aspect('equal')
plt.plot(np.arange(-8,9),np.arange(-8,9),color='gray',linestyle='--')
#ax.fill_between(np.arange(-8,9),np.arange(-8,9)-0.2,np.arange(-8,9)+0.2,color='gray',linestyle='-',linewidth=0.5,alpha=0.2)


errorboxes = [Rectangle((-0.2,-0.2), 0.4, 0.4)]

# Create patch collection with specified colour/alpha
pc = PatchCollection(errorboxes, facecolor='gray', alpha=0.2,
                        edgecolor='None')

# Add collection to axes
ax.add_collection(pc)

#plt.plot(np.arange(-8,9),-np.arange(-8,9),color='gray',linestyle='--')
#plt.axhline(y=0, color='black',linewidth=0.5)
#plt.axvline(x=0, color='black',linewidth=0.5)
plt.grid(visible=True, which='major', axis='both',linewidth=0.5,color='black')
#plt.xlim(-8,8)
#plt.ylim(-8,8)
plt.xlim(-0.4,0.4)
plt.ylim(-0.4,0.4)



plt.xlabel("Cohen's d, difference of mean YOU score\n Women minus Men")
plt.ylabel("Cohen's d, difference of mean SEE score\n Women minus Men")
# The point of this plot is to highlight that the size of the effect seen comparing womens and mens scores is genreally
# bigger for the SEE score than the YOU score. And that for particular items this effect size is many times bigger than the 
# traditional YOU score. Note these are just physics majors too
plt.tight_layout()
plt.savefig("women-men_cohens_d_diff_physics_majors.png")

# ------------------------------------------------------------------------------------------


xshift = -0.02
yshift = 0.0325

plt.figure()
mw_df = pd.DataFrame([diff_mw_you,diff_mw_see,diff_mw_you_sem,diff_mw_see_sem],index=["wm_you","wm_see","wm_you_sem","wm_see_sem"]).transpose()
ax = mw_df.plot(x="wm_you",y="wm_see",xerr="wm_you_sem",yerr="wm_see_sem",linestyle="None",marker="o", color='red', legend=None)
for index, row in mw_df.iterrows():
    # special cases
    if index == "q30":
        plt.text(row['wm_you']+xshift+0.03,row['wm_see']-yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index == "q5":
        plt.text(row['wm_you']+xshift+0.02,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['wm_you']+xshift+0.01,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['wm_you']+xshift,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

#colors = ['gray','gray','black','blue','green','orange','red']
#for r in range(1,8):
#    circle = plt.Circle((0, 0), r, color=colors[r-1], fill=False)
#    ax.add_patch(circle)


ax.set_aspect('equal')
plt.plot(np.arange(-8,9),np.arange(-8,9),color='gray',linestyle='--')
#ax.fill_between(np.arange(-8,9),np.arange(-8,9)-0.2,np.arange(-8,9)+0.2,color='gray',linestyle='-',linewidth=0.5,alpha=0.2)

#plt.plot(np.arange(-8,9),-np.arange(-8,9),color='gray',linestyle='--')
#plt.axhline(y=0, color='black',linewidth=0.5)
#plt.axvline(x=0, color='black',linewidth=0.5)
plt.grid(visible=True, which='major', axis='both',linewidth=0.5,color='black')
#plt.xlim(-8,8)
#plt.ylim(-8,8)
plt.xlim(-0.4,0.2)
plt.ylim(-0.4,0.2)



plt.xlabel("Difference of mean YOU score\n Women minus Men")
plt.ylabel("Difference of mean SEE score\n Women minus Men")
# The point of this plot is to highlight that the size of the effect seen comparing womens and mens scores is genreally
# bigger for the SEE score than the YOU score. And that for particular items this effect size is many times bigger than the 
# traditional YOU score. Note these are just physics majors too
plt.tight_layout()
plt.savefig("women-men_diff_physics_majors.png")

# --------------------------------------------------------------
# PLOT 3
# --------------------------------------------------------------
plt.figure()
mw_df = pd.DataFrame([diff_mw_you/diff_mw_you_sem,diff_mw_see/diff_mw_see_sem],index=["wm_you","wm_see"]).transpose()
ax = mw_df.plot(x="wm_you",y="wm_see",linestyle="None",marker="o", color='red', legend=None)
for index, row in mw_df.iterrows():
    # special cases
    if index == "q30":
        plt.text(row['wm_you']+xshift+0.03,row['wm_see']-yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index == "q5":
        plt.text(row['wm_you']+xshift+0.02,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    elif index in ["q1","q2","q3","q4","q5","q6","q7","q8","q9"]:
        plt.text(row['wm_you']+xshift+0.01,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)
    else:
        plt.text(row['wm_you']+xshift,row['wm_see']+yshift,r'$'+index[1:]+'$', fontsize=textsize)

#colors = ['gray','gray','black','blue','green','orange','red']
#for r in range(1,8):
#    circle = plt.Circle((0, 0), r, color=colors[r-1], fill=False)
#    ax.add_patch(circle)


ax.set_aspect('equal')
plt.plot(np.arange(-8,9),np.arange(-8,9),color='gray',linestyle='--')
ax.fill_between(np.arange(-8,9),np.arange(-8,9)-1,np.arange(-8,9)+1,color='gray',linestyle='-',linewidth=0.5,alpha=0.2)

#plt.plot(np.arange(-8,9),-np.arange(-8,9),color='gray',linestyle='--')
#plt.axhline(y=0, color='black',linewidth=0.5)
#plt.axvline(x=0, color='black',linewidth=0.5)
plt.grid(visible=True, which='major', axis='both',linewidth=0.5,color='black')
plt.xlim(-8,8)
plt.ylim(-8,8)
#plt.xlim(-0.4,0.2)
#plt.ylim(-0.4,0.2)



plt.xlabel("Difference of mean YOU score\n Women minus Men")
plt.ylabel("Difference of mean SEE score\n Women minus Men")
# The point of this plot is to highlight that the size of the effect seen comparing womens and mens scores is genreally
# bigger for the SEE score than the YOU score. And that for particular items this effect size is many times bigger than the 
# traditional YOU score. Note these are just physics majors too
plt.tight_layout()
plt.savefig("women-men_normed_diff_physics_majors.png")


print(N_women, N_men, N_gnc)
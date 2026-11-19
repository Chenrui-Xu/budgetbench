import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=json.load(open("results.json")); Bs=[6,8,10,12,16,20,24]
col={"SingleShot":"#2a78d6","Loop":"#eb6834","BudgetAware":"#1baf7a","ClinicalBudgetAware":"#eda100"}
mk={"SingleShot":"o","Loop":"s","BudgetAware":"^","ClinicalBudgetAware":"D"}
panels=[("success","Success@B"),("unsafe_write","Unsafe-write rate"),("h_violation","Harm-weighted violation"),("handoff","Handoff rate")]
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(2,2,figsize=(7.2,5.4),sharey=True,sharex=True); ax=ax.ravel()
for a,(k,t) in zip(ax,panels):
    for ag in col:
        a.plot(Bs,[r[ag][str(B)][k] for B in Bs],color=col[ag],marker=mk[ag],ms=5,lw=2,label=ag,mec="#fcfcfb",mew=1)
    a.set_title(t,fontsize=10,loc="left",color="#0b0b0b"); a.set_xlabel("Budget B (units)"); a.set_xticks(Bs)
    a.grid(axis="y",color="#e4e3df",lw=.6); a.set_ylim(-0.03,1.05); a.tick_params(colors="#52514e")
ax[0].set_ylabel("Fraction of episodes"); ax[2].set_ylabel("Fraction of episodes"); ax[0].set_xlabel(""); ax[1].set_xlabel("")
fig.legend(*ax[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig("fig_clinical_sim.png",dpi=200,facecolor="white")

from pathlib import Path
from collections import Counter, defaultdict
import re, unicodedata
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

ROOT=Path.cwd(); R=ROOT/'reports'; I=ROOT/'data/raw/images'
a=pd.read_csv(R/'phase5_1_cluster_assignments.csv').sort_values('image_index').reset_index(drop=True)
files=a.filename.tolist(); idx={f:i for i,f in enumerate(files)}
f=pd.read_parquet(R/'phase5_layout_fingerprints.parquet').query("representation_version == 'F0_IMAGE_NORMALIZED'").set_index('filename').loc[files]
meta={'filename','representation_version','image_index','acquisition_stratum','label_status'}
cols=[c for c in f.columns if c not in meta]
X=StandardScaler().fit_transform(f[cols]); X=normalize(PCA(32,random_state=20260824).fit_transform(X))

l=pd.read_parquet(R/'eda3_ocr_line_features.parquet')
def toks(s): return re.findall(r"[^\W_]{3,}",unicodedata.normalize('NFKC',str(s)).casefold(),flags=re.UNICODE)
sets={fn:set() for fn in files}
for fn,g in l.groupby('filename'): sets[fn]=set(toks(' '.join(g.text.astype(str))))
df=Counter(t for ss in sets.values() for t in ss if t.isalpha())
v=sorted(t for t,n in df.items() if 8<=n<=250)
vi={t:j for j,t in enumerate(v)}; A=np.zeros((732,len(v)))
for fn,ss in sets.items():
  for t in ss:
    if t in vi:A[idx[fn],vi[t]]=np.log((1+732)/(1+df[t]))+1
A=normalize(A)

def hog(fn):
 im=Image.open(I/fn).convert('L'); w,h=im.size; im=im.crop((.05*w,.05*h,.95*w,.95*h)).resize((96,64)).filter(ImageFilter.GaussianBlur(.8)); z=np.asarray(im,float)/255; z=(z-z.mean())/(z.std()+1e-6); gy,gx=np.gradient(z); mag=np.hypot(gx,gy); ang=(np.arctan2(gy,gx)+np.pi)%np.pi; out=[]
 for r in range(8):
  for c in range(12):
   m=mag[r*8:(r+1)*8,c*8:(c+1)*8].ravel(); b=np.floor(ang[r*8:(r+1)*8,c*8:(c+1)*8].ravel()/(np.pi/8)).astype(int).clip(0,7); hist=np.bincount(b,weights=m,minlength=8); out.extend(hist/(np.linalg.norm(hist)+1e-6))
 return out
V=np.asarray([hog(fn) for fn in files]); V=normalize(PCA(32,random_state=20260824).fit_transform(StandardScaler().fit_transform(V)))
C=normalize(np.c_[np.sqrt(.4)*X,np.sqrt(.4)*A,np.sqrt(.2)*V])

s=pd.read_csv(R/'phase5_1_cluster_stability.csv').query("metric_type == 'CLUSTER_COASSIGNMENT'").set_index('local_cluster_id')
stable=set(s.query('coassignment_stability >= .70').index)
members={cid:[idx[x] for x in g.filename] for cid,g in a[a.local_cluster_id.isin(stable)].groupby('local_cluster_id')}
def proto(mat,ids): return normalize(mat[ids].mean(0,keepdims=True))[0]
ps={cid:(proto(X,ids),proto(A,ids),proto(V,ids),proto(C,ids)) for cid,ids in members.items()}
rows=[]
ids=sorted(ps)
for i,x in enumerate(ids):
 for y in ids[i+1:]:
  z=[float(np.dot(ps[x][k],ps[y][k])) for k in range(4)]
  rows.append((x,y,*z))
d=pd.DataFrame(rows,columns=['a','b','layout','anchor','visual','combined']).sort_values('combined',ascending=False)
print('vocab',len(v),'stable',len(members),'stable images',sum(map(len,members.values())))
print('T01',d[((d.a=='A_F0_C00')&(d.b=='A_F0_C01'))|((d.b=='A_F0_C00')&(d.a=='A_F0_C01'))].to_string(index=False))
print('\nTOP COMBINED\n',d.head(50).to_string(index=False))
print('\nQUANTILES\n',d[['layout','anchor','visual','combined']].quantile([0,.1,.25,.5,.75,.9,.95,.99,1]))
for cid,ii in members.items():
 if len(ii)>1:
  within=[np.mean((m[ii]@m[ii].T)[np.triu_indices(len(ii),1)]) for m in [X,A,V,C]]
  print('WITHIN',cid,len(ii),*[round(q,3) for q in within])

fig,axes=plt.subplots(12,4,figsize=(10,28),squeeze=False)
for pair_index,pair in enumerate(d.head(6).itertuples(index=False)):
 for side,cid in enumerate([pair.a,pair.b]):
  ii=members[cid]; p=ps[cid][3]; ranked=sorted(ii,key=lambda j:1-float(np.dot(C[j],p)))[:4]
  row=2*pair_index+side
  for col,j in enumerate(ranked):
   axes[row,col].imshow(Image.open(I/files[j]).convert('RGB'))
   axes[row,col].axis('off'); axes[row,col].set_title(files[j],fontsize=7)
  axes[row,0].set_ylabel(cid,fontsize=8)
 axes[2*pair_index,0].text(-.15,1.12,f"L={pair.layout:.2f} A={pair.anchor:.2f} V={pair.visual:.2f} C={pair.combined:.2f}",transform=axes[2*pair_index,0].transAxes,fontsize=8)
plt.tight_layout(); plt.savefig('/tmp/phase52_pairs.png',dpi=120); plt.close()

tail=np.array(sorted(set(range(732))-set(j for ii in members.values() for j in ii)))
tm=HDBSCAN(min_cluster_size=5,min_samples=3,cluster_selection_method='eom').fit(C[tail])
print('TAIL',len(tail),'clusters',Counter(tm.labels_), 'noise',np.mean(tm.labels_<0))
tailgroups={f'TAIL_C{k:02d}':tail[tm.labels_==k].tolist() for k in sorted(set(tm.labels_)-{-1})}
for cid,ii in tailgroups.items():
 print('TAILGROUP',cid,len(ii),Counter(a.iloc[ii].acquisition_stratum),*[round(np.mean((m[ii]@m[ii].T)[np.triu_indices(len(ii),1)]),3) for m in [X,A,V,C]])

fig,axes=plt.subplots(max(len(tailgroups),1),6,figsize=(12,2.2*max(len(tailgroups),1)),squeeze=False)
for row,(cid,ii) in enumerate(sorted(tailgroups.items(),key=lambda z:-len(z[1]))):
 p=proto(C,ii); ranked=sorted(ii,key=lambda j:1-float(np.dot(C[j],p))); pos=np.linspace(0,len(ranked)-1,min(6,len(ranked))).round().astype(int)
 for col,k in enumerate(pos):
  j=ranked[k]; axes[row,col].imshow(Image.open(I/files[j]).convert('RGB')); axes[row,col].axis('off'); axes[row,col].set_title(files[j],fontsize=7)
 axes[row,0].set_ylabel(f'{cid} n={len(ii)}',fontsize=8)
 for col in range(len(pos),6):axes[row,col].axis('off')
plt.tight_layout();plt.savefig('/tmp/phase52_tail.png',dpi=120);plt.close()

for cid in ['C_F1_C01','C_F1_C03','C_F1_C04','C_F1_C05']:
 ii=np.array(members[cid]); mm=HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_method='eom').fit(C[ii]); print('REFINE',cid,len(ii),Counter(mm.labels_))
for cid,ii in [('C_F1_C08',np.array(members['C_F1_C08'])),('TAIL_C19',np.array(tailgroups['TAIL_C19'])),('TAIL_C20',np.array(tailgroups['TAIL_C20']))]:
 mm=HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_method='eom').fit(C[ii]); print('REFINE',cid,len(ii),Counter(mm.labels_))

flagged={'C_F1_C01','C_F1_C03','C_F1_C04','C_F1_C05','C_F1_C08'}
components={cid:ii for cid,ii in members.items() if cid not in flagged}
for cid in flagged:
 ii=np.array(members[cid]); mm=HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_method='eom',copy=True).fit(C[ii])
 for lab in sorted(set(mm.labels_)-{-1}):components[f'{cid}_R{lab:02d}']=ii[mm.labels_==lab].tolist()
for cid,ii0 in tailgroups.items():
 if cid=='TAIL_C18':
  ii=np.array(ii0);mm=HDBSCAN(min_cluster_size=5,min_samples=2,cluster_selection_method='eom',copy=True).fit(C[ii])
  for lab in sorted(set(mm.labels_)-{-1}):components[f'{cid}_R{lab:02d}']=ii[mm.labels_==lab].tolist()
 else:components[cid]=ii0
pp={cid:(proto(X,ii),proto(A,ii),proto(V,ii),proto(C,ii)) for cid,ii in components.items()}
rr=[];cids=sorted(pp)
for i,x in enumerate(cids):
 for y in cids[i+1:]:
  z=[float(np.dot(pp[x][k],pp[y][k])) for k in range(4)]
  strict=z[0]>=.80 and z[1]>=.55 and z[2]>=.70 and z[3]>=.70
  known={x,y}=={'A_F0_C00','A_F0_C01'}
  rr.append((x,y,*z,strict,known))
dd=pd.DataFrame(rr,columns=['a','b','layout','anchor','visual','combined','strict','known']).sort_values('combined',ascending=False)
print('\nFINAL TOP PAIRS\n',dd.head(50).to_string(index=False));print('\nMERGES\n',dd.query('strict or known').to_string(index=False))
audit=dd.query('not strict and not known').head(6)
fig,axes=plt.subplots(12,4,figsize=(10,28),squeeze=False)
for pair_index,pair in enumerate(audit.itertuples(index=False)):
 for side,cid in enumerate([pair.a,pair.b]):
  ii=components[cid]; p=pp[cid][3]; ranked=sorted(ii,key=lambda j:1-float(np.dot(C[j],p)))[:4];row=2*pair_index+side
  for col,j in enumerate(ranked):axes[row,col].imshow(Image.open(I/files[j]).convert('RGB'));axes[row,col].axis('off');axes[row,col].set_title(files[j],fontsize=7)
  axes[row,0].set_ylabel(cid,fontsize=8)
 axes[2*pair_index,0].text(-.15,1.12,f"L={pair.layout:.2f} A={pair.anchor:.2f} V={pair.visual:.2f} C={pair.combined:.2f}",transform=axes[2*pair_index,0].transAxes,fontsize=8)
plt.tight_layout();plt.savefig('/tmp/phase52_final_pairs.png',dpi=120);plt.close()

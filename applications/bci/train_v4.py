from __future__ import annotations
import csv,hashlib,importlib.util,json,random,sys,time,traceback,copy,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch,joblib,psutil
from scipy.signal import butter,sosfilt
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix,roc_auc_score
from torch import nn
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
PILOT=Path(CFG['pilot_task']);V2=Path(CFG['v2_run']);V3=Path(CFG['v3_run'])
sys.path.insert(0,str(PILOT))
spec=importlib.util.spec_from_file_location('pilot_utils',PILOT/'train.py');U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
from eegnet import EEGNet

def save(p,v):U.save_json(p,v)
def read_csv(p):
 with p.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 return [{**r,**{k:int(r[k]) for k in ('subject','run','onset_sample','label') if k in r}} for r in rows]
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
def key(r):return (r['subject'],r['run'],r['onset_sample'])
def indices(rows,lo,hi,run=None):return np.array([i for i,r in enumerate(rows) if lo<=r['subject']<=hi and (run is None or r['run']==run)],dtype=int)
def metric(y,p):
 pred=(p>=.5).astype(int)
 return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'roc_auc':float(roc_auc_score(y,p)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
def subject_score(y,p,rows):
 return float(np.mean([balanced_accuracy_score(y[ii],(p[ii]>=.5).astype(int)) for s in sorted(set(r['subject'] for r in rows)) for ii in [indices(rows,s,s)]]))
def spd_power(c,power):
 v,q=np.linalg.eigh(c);return (q*(np.maximum(v,1e-9)**power)[...,None,:])@np.swapaxes(q,-1,-2)
def covariances(x):
 z=x.astype(np.float64)-x.mean(-1,keepdims=True);c=z@np.swapaxes(z,-1,-2)/(x.shape[-1]-1)
 trace=np.trace(c,axis1=-2,axis2=-1);c=c/np.maximum(trace,1e-9)[:,None,None]
 a=CFG['covariance_shrinkage'];return (1-a)*c+a*np.eye(x.shape[1])[None]/x.shape[1],np.log(np.maximum(trace,1e-9))
def log_features(c):
 v,q=np.linalg.eigh(c);lc=(q*np.log(np.maximum(v,1e-9))[...,None,:])@np.swapaxes(q,-1,-2);a,b=np.triu_indices(c.shape[-1]);return lc[:,a,b]*np.where(a==b,1,np.sqrt(2))
def features(x,rows,refs=None):
 all_cov=[];all_ea=[];all_power=[];reference={} if refs is None else refs
 for band,(lo,hi) in enumerate(CFG['bands_hz']):
  xx=x if band==0 else sosfilt(butter(4,[lo,hi],btype='bandpass',fs=160,output='sos'),x,axis=-1).astype(np.float32)
  c,trace=covariances(xx);all_cov.append(np.column_stack([log_features(c),trace]));power=np.log(np.maximum((xx**2).mean(-1),1e-9));all_power.append(np.column_stack([power,power[:,0]-power[:,2],power[:,3]-power[:,5],power[:,6]-power[:,8]]))
  aligned=c.copy()
  for s in sorted(set(r['subject'] for r in rows)):
   ii=indices(rows,s,s)
   refkey=f'{s}_{band}'
   if refs is None:
    cal=np.array([i for i in ii if rows[i]['run']==4 and rows[i]['label'] in (0,1)])
    assert len(cal)>=8;reference[refkey]=spd_power(c[cal].mean(0),-.5)
   w=reference[refkey];aligned[ii]=w@c[ii]@w
  all_ea.append(np.column_stack([log_features(aligned),trace]))
 return {'cov':all_cov[0],'fb_cov':np.concatenate(all_cov,axis=1),'ea_cov':all_ea[0],'ea_fb_cov':np.concatenate(all_ea,axis=1),'fb_power':np.concatenate(all_power,axis=1)},reference
def fit_linear(f,y,ii,C,balanced=False):
 sc=StandardScaler().fit(f[ii]);z=sc.transform(f[ii]);lr=LogisticRegression(C=C,max_iter=1000,class_weight='balanced' if balanced else None,random_state=CFG['seed']).fit(z,y[ii]);return {'scaler':sc,'coef':np.r_[lr.coef_[0],lr.intercept_[0]],'C':C}
def adapt(model,f,y,ii,prior):
 if prior is None:return model['coef'].copy()
 z=np.column_stack([model['scaler'].transform(f[ii]),np.ones(len(ii))]);w0=model['coef'];yy=y[ii]
 def objective(w):
  t=z@w;delta=w-w0;loss=np.mean(np.logaddexp(0,t)-yy*t)+.5*prior*np.dot(delta,delta)
  grad=z.T@(expit(t)-yy)/len(yy)+prior*delta;return loss,grad
 fit=minimize(objective,w0,jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-9,'gtol':1e-6,'maxls':50})
 if not fit.success:
  grad_max=float(np.max(np.abs(objective(fit.x)[1])))
  if not np.isfinite(fit.fun) or grad_max>1e-4:raise RuntimeError(f'Adaptation optimizer: {fit.message}; gradient={grad_max}')
  print(f'OPTIMIZER numerical stop; finite objective={fit.fun:.8f}, max gradient={grad_max:.2e}',flush=True)
 return fit.x
def predict_linear(model,f,w=None):return expit(model['scaler'].transform(f)@(model['coef'] if w is None else w)[:-1]+(model['coef'] if w is None else w)[-1])
def cohort_linear(model,f,y,rows,subjects,prior,evaluation_run=12):
 ps=[];rr=[];weights={}
 for s in subjects:
  cal=indices(rows,s,s,4);test=indices(rows,s,s,evaluation_run);assert not set(cal)&set(test)
  w=adapt(model,f,y,cal,prior);weights[s]=w;ps.extend(predict_linear(model,f[test],w));rr.extend(rows[i] for i in test)
 return np.array(ps),rr,weights
def make_net(ck):
 net=EEGNet(len(ck['channels']),ck['samples'],**ck['model_settings']);net.load_state_dict(ck['model_state']);return net
def infer_net(net,x,ck,device):
 net=net.to(device);net.eval();mean=ck['channel_mean_uv'].numpy();std=ck['channel_std_uv'].numpy()
 with torch.no_grad():return np.concatenate([net(torch.from_numpy((a-mean)/std).to(device)).softmax(1)[:,1].cpu().numpy() for a in np.array_split(x,max(1,(len(x)+127)//128))])
def fit_personal_net(x,y,rows,s,policy,nseed,ck,device):
 seed(nseed+s);net=make_net(ck).to(device)
 if policy['scope']=='head':
  for p in net.parameters():p.requires_grad=False
  for p in net.classifier.parameters():p.requires_grad=True
 opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=policy['lr']);loss_fn=nn.CrossEntropyLoss();cal=indices(rows,s,s,4);val=indices(rows,s,s,8)
 mean=ck['channel_mean_uv'].numpy();std=ck['channel_std_uv'].numpy();tx=torch.from_numpy((x[cal]-mean)/std).to(device);ty=torch.from_numpy(y[cal]).to(device)
 best=(-1,float('-inf'));best_epoch=0;best_state=None;history=[]
 for epoch in range(1,CFG['neural_epochs']+1):
  if policy['scope']=='head':net.eval()
  else:
   net.train()
   for layer in net.modules():
    if isinstance(layer,nn.BatchNorm2d):layer.eval()
  order=torch.randperm(len(cal),device=device);losses=[]
  for jj in order.split(CFG['neural_batch_size']):
   opt.zero_grad(set_to_none=True);loss=loss_fn(net(tx[jj]),ty[jj]);loss.backward();opt.step();net.constrain_weights();losses.append(float(loss.detach().cpu()))
  p=infer_net(net,x[val],ck,device);score=float(balanced_accuracy_score(y[val],p>=.5));ce=float(-np.mean(y[val]*np.log(p.clip(1e-8,1-1e-8))+(1-y[val])*np.log((1-p).clip(1e-8,1))))
  history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_ba':score,'validation_ce':ce})
  if (score,-ce)>best:best=(score,-ce);best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  if epoch>=CFG['neural_minimum_epochs'] and epoch-best_epoch>=CFG['neural_patience']:break
 net.load_state_dict(best_state);return net,best_state,best_epoch,history
def neural_cohort(x,y,rows,subjects,policy,ck,device,out):
 probabilities=[];rr=None;states={};history=[]
 for nseed in CFG['seeds']:
  pp=[];rlist=[];states[nseed]={}
  for s in subjects:
   net,state,epoch,hist=fit_personal_net(x,y,rows,s,policy,nseed,ck,device);ii=indices(rows,s,s,12);pp.extend(infer_net(net,x[ii],ck,device));rlist.extend(rows[i] for i in ii);states[nseed][s]=state
   torch.save({**ck,'model_state':state,'calibration_subject':s,'policy':policy,'seed':nseed,'best_epoch':epoch},out/f'{policy["id"]}_{nseed}_S{s:03d}.pt')
   history.extend({'subject':s,'seed':nseed,'policy':policy['id'],**h} for h in hist)
  rr=rlist;probabilities.append(np.array(pp));print(f'NEURAL {policy["id"]} seed {nseed} subjects {subjects[0]}-{subjects[-1]} finished',flush=True)
 return np.mean(probabilities,axis=0),rr,states,probabilities,history
def bootstrap(y,p,base,rows):
 rng=np.random.default_rng(CFG['seed']);sub=sorted(set(r['subject'] for r in rows));by={s:indices(rows,s,s) for s in sub};ba=[];delta=[]
 for _ in range(1000):
  ii=np.concatenate([by[s] for s in rng.choice(sub,len(sub),replace=True)]);a=balanced_accuracy_score(y[ii],p[ii]>=.5);ba.append(a);delta.append(a-balanced_accuracy_score(y[ii],base[ii]>=.5))
 return {'ba_95ci':np.quantile(ba,[.025,.975]).tolist(),'paired_delta_vs_v3':float(balanced_accuracy_score(y,p>=.5)-balanced_accuracy_score(y,base>=.5)),'paired_delta_95ci':np.quantile(delta,[.025,.975]).tolist()}
def fixed_prediction_permutation(y,p,rows):
 rng=np.random.default_rng(CFG['seed']);blocks=[indices(rows,s,s) for s in sorted(set(r['subject'] for r in rows))];null=[];observed=balanced_accuracy_score(y,p>=.5)
 for _ in range(CFG['permutation_count']):
  yp=y.copy()
  for ii in blocks:yp[ii]=rng.permutation(yp[ii])
  null.append(balanced_accuracy_score(yp,p>=.5))
 return {'observed_ba':float(observed),'n_permutations':len(null),'null_mean':float(np.mean(null)),'null_95range':np.quantile(null,[.025,.975]).tolist(),'conditional_p_value':float((1+sum(a>=observed for a in null))/(len(null)+1)),'scope':'Frozen predictions vs labels shuffled within person/run12; not pipeline-wide permutation or independent external validation.'}
def gate(y,pactive,pright,tau,conf):
 accepted=(pactive>=tau)&(np.maximum(pright,1-pright)>=conf)
 if tau>=1:accepted[:]=False
 pred=np.where(accepted,1+(pright>=.5).astype(int),0);rest=y==0;mi=~rest;count=int((accepted&mi).sum());correct=int(((pred==y)&mi).sum());fp=int((accepted&rest).sum())
 return {'rest_n':int(rest.sum()),'mi_n':int(mi.sum()),'false_triggers':fp,'rest_fpr':fp/int(rest.sum()),'mi_outputs':count,'mi_coverage':count/int(mi.sum()),'correct_outputs':correct,'correct_mi_rate':correct/int(mi.sum()),'accepted_mi_accuracy':correct/count if count else None,'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()},pred
def main():
 run=ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True);log=(run/'training.log').open('w',encoding='utf-8');sys.stdout=U.Tee(sys.stdout,log);sys.stderr=U.Tee(sys.stderr,log)
 status={'status':'RUNNING','run_dir':str(run),'stage':'features'};save(ROOT/'latest_run.json',status);save(run/'config.json',CFG);start=time.perf_counter()
 try:
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seed(CFG['seed'])
  print('RUN '+str(run)+' DEVICE '+str(device),flush=True)
  selected=json.loads((V2/'new_person/model_selection.json').read_text())['selected'];ck_path=V2/'new_person'/selected/'eegnet_best.pt';ck=torch.load(ck_path,map_location='cpu',weights_only=True)
  files=[V2/'prepared_all_trials.npz',V2/'all_trial_manifest.csv',V3/'three_class_trials.npz',V3/'three_class_manifest.csv',ck_path,ROOT/'config.json',ROOT/'train_v4.py']
  save(run/'provenance.json',{'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'device':str(device),'torch':torch.__version__,'python':platform.python_version(),'evaluation_status':CFG['evaluation_status']})
  cache=np.load(files[0]);x=cache['X'];y=cache['y'];rows=read_csv(files[1]);assert np.array_equal(y,[r['label'] for r in rows]);assert len(set(map(key,rows)))==len(rows)
  f,refs=features(x,rows);np.savez_compressed(run/'feature_cache.npz',**f);joblib.dump(refs,run/'alignment_refs.joblib');src=indices(rows,11,80);vsubjects=sorted(set(rows[i]['subject'] for i in indices(rows,81,94)));tsubjects=sorted(set(rows[i]['subject'] for i in indices(rows,95,108)))
  assert set(rows[i]['subject'] for i in src).isdisjoint(vsubjects+tsubjects)
  ii_val=indices(rows,81,94,12);val_y=y[ii_val];val_rows=[rows[i] for i in ii_val];linear=[];models={};val_probs={};val_weights={}
  for name in CFG['linear_features']:
   for C in CFG['linear_C']:
    model=fit_linear(f[name],y,src,C);models[(name,C)]=model
    for prior in CFG['adaptation_prior_strength']:
     p,rr,ws=cohort_linear(model,f[name],y,rows,vsubjects,prior);assert list(map(key,rr))==list(map(key,val_rows));cid=f'{name}_C{C}_prior{prior}';linear.append({'id':cid,'feature':name,'C':C,'prior':prior,'selection_ba':subject_score(val_y,p,rr),'pooled_ba':metric(val_y,p)['balanced_accuracy']});val_probs[cid]=p;val_weights[cid]=ws
  best_linear=max(linear,key=lambda r:r['selection_ba']);print('BEST LINEAR VALIDATION '+str(best_linear),flush=True);U.write_csv(run/'linear_validation.csv',linear)
  status['stage']='neural_validation';save(ROOT/'latest_run.json',status);valid_out=run/'validation_models';valid_out.mkdir();neural=[];nv={};allhist=[]
  for policy in CFG['neural_policies']:
   p,rr,states,seeds,hist=neural_cohort(x,y,rows,vsubjects,policy,ck,device,valid_out);assert list(map(key,rr))==list(map(key,val_rows));nv[policy['id']]={'p':p,'states':states,'seeds':seeds};allhist.extend(hist);neural.append({'id':policy['id'],'selection_ba':subject_score(val_y,p,rr),'pooled_ba':metric(val_y,p)['balanced_accuracy'],'seed_ba':[metric(val_y,a)['balanced_accuracy'] for a in seeds]})
  best_neural=max(neural,key=lambda r:r['selection_ba']);policy=next(a for a in CFG['neural_policies'] if a['id']==best_neural['id']);recommended='linear' if best_linear['selection_ba']>=best_neural['selection_ba'] else 'neural';frozen={'linear':best_linear,'neural':best_neural,'recommended_family':recommended,'selection_scope':'S081-S094 run12 only, never S095-S108 run12'};save(run/'frozen_mi_selection.json',frozen);save(run/'neural_validation.json',neural);U.write_csv(run/'neural_history.csv',allhist);print('MI POLICIES FROZEN '+str(frozen),flush=True)
  # Rest selection is also completed before any test-person R12 inference.
  status['stage']='rest_validation';save(ROOT/'latest_run.json',status);rc=np.load(V3/'three_class_trials.npz');rx=rc['X'];ry=rc['y'];rrows=read_csv(V3/'three_class_manifest.csv');assert np.array_equal(ry,[r['label'] for r in rrows]);assert len(set(map(key,rrows)))==len(rrows)
  # V3 appends rest rows after MI rows; canonicalize once to keep every label,
  # feature and per-person probability joined by the same trial identity.
  ordering=np.array(sorted(range(len(rrows)),key=lambda i:key(rrows[i])));rx=rx[ordering];ry=ry[ordering];rrows=[rrows[i] for i in ordering]
  rf,_=features(rx,rrows,refs);rtrain=indices(rrows,11,80);rv=indices(rrows,81,94,12);rvs=sorted(set(rrows[i]['subject'] for i in rv));rvy=ry[rv];rvrows=[rrows[i] for i in rv];active_y=(ry!=0).astype(int)
  direction_val=[]
  for s in rvs:
   jj=indices(rrows,s,s,12)
   if recommended=='linear':
    m=models[(best_linear['feature'],best_linear['C'])];direction_val.extend(predict_linear(m,rf[best_linear['feature']][jj],val_weights[best_linear['id']][s]))
   else:
    ss=nv[policy['id']]['states'];direction_val.extend(np.mean([infer_net(make_net({**ck,'model_state':ss[n][s]}),rx[jj],ck,device) for n in CFG['seeds']],axis=0))
  direction_val=np.array(direction_val);rest_candidates=[];rest_models={};rest_p={};rest_curves=[]
  for name in CFG['rest_features']:
   for C in CFG['linear_C']:
    rm=fit_linear(rf[name],active_y,rtrain,C,balanced=True);rest_models[(name,C)]=rm
    for prior in CFG['adaptation_prior_strength']:
     cid=f'{name}_C{C}_prior{prior}';p,rr,ws=cohort_linear(rm,rf[name],active_y,rrows,rvs,prior);assert list(map(key,rr))==list(map(key,rvrows));rest_p[cid]=p
     curve=[]
     for tau in CFG['gate_active_thresholds']:
      for conf in CFG['gate_direction_confidence']:
       score,_=gate(rvy,p,direction_val,tau,conf);curve.append({'candidate':cid,'tau':tau,'direction_confidence':conf,**{k:score[k] for k in ('rest_fpr','mi_coverage','correct_mi_rate','accepted_mi_accuracy')}})
     rest_curves.extend(curve);feasible=[a for a in curve if a['rest_fpr']<=CFG['prototype_primary_cap']];best=max(feasible,key=lambda a:(a['correct_mi_rate'],-a['rest_fpr'],a['direction_confidence']));rest_candidates.append({'id':cid,'feature':name,'C':C,'prior':prior,**best})
  rest_best=max(rest_candidates,key=lambda a:(a['correct_mi_rate'],-a['rest_fpr'],a['direction_confidence']));cur=[a for a in rest_curves if a['candidate']==rest_best['id']];policies=[]
  for cap in CFG['rest_false_trigger_caps']:
   a=max([a for a in cur if a['rest_fpr']<=cap],key=lambda a:(a['correct_mi_rate'],-a['rest_fpr'],a['direction_confidence']));policies.append({'cap':cap,**a})
  save(run/'frozen_rest_selection.json',{'candidate':rest_best,'policies':policies,'scope':CFG['gate_selection']});U.write_csv(run/'rest_validation_candidates.csv',rest_candidates);U.write_csv(run/'rest_validation_curve.csv',rest_curves);print('REST POLICIES FROZEN '+str(policies),flush=True)
  # All policies above are immutable for the following exploratory evaluation.
  status['stage']='fixed_test_evaluation';save(ROOT/'latest_run.json',status);test_out=run/'test_models';test_out.mkdir();ti=indices(rows,95,108,12);ty=y[ti];tr=[rows[i] for i in ti];old=read_csv(V3/'calibration/predictions.csv');base_lookup={key(r):float(r['p_right']) for r in old if r['method']=='EEGNet-personal'};base=np.array([base_lookup[key(r)] for r in tr]);probs={'v3_personal':base,'global':infer_net(make_net(ck),x[ti],ck,device)}
  lm=models[(best_linear['feature'],best_linear['C'])];lp,lrr,lws=cohort_linear(lm,f[best_linear['feature']],y,rows,tsubjects,best_linear['prior']);assert list(map(key,lrr))==list(map(key,tr));probs['linear']=lp;joblib.dump({'model':lm,'weights':lws,'selection':best_linear,'alignment_refs':refs},test_out/'linear_personal.joblib')
  neural_test={}
  for candidate in {policy['id'],'head'}:
   pp=next(a for a in CFG['neural_policies'] if a['id']==candidate);npred,nrr,ns,nseeds,h=neural_cohort(x,y,rows,tsubjects,pp,ck,device,test_out);assert list(map(key,nrr))==list(map(key,tr));neural_test[candidate]={'p':npred,'states':ns,'seeds':nseeds};allhist.extend(h)
  probs['neural']=neural_test[policy['id']]['p'];probs['head_3seed']=neural_test['head']['p'];probs['recommended']=probs[recommended]
  summary={name:{**metric(ty,p),**bootstrap(ty,p,base,tr),'subject_mean_ba':subject_score(ty,p,tr)} for name,p in probs.items()};predictions=[];subject_metrics=[]
  for name,p in probs.items():
   predictions.extend({**r,'method':name,'p_right':float(pr),'prediction':int(pr>=.5)} for r,pr in zip(tr,p))
   for s in tsubjects:
    jj=indices(tr,s,s);subject_metrics.append({'subject':s,'method':name,**metric(ty[jj],p[jj])})
  U.write_csv(run/'mi_predictions.csv',predictions);save(run/'subject_metrics.json',subject_metrics);save(run/'mi_summary.json',summary);U.write_csv(run/'neural_history.csv',allhist)
  stability=[]
  for name,obj in neural_test.items():
   for nseed,p in zip(CFG['seeds'],obj['seeds']):stability.append({'policy':name,'seed':nseed,**metric(ty,p)})
  save(run/'seed_stability.json',stability);permutation=fixed_prediction_permutation(ty,probs['recommended'],tr);save(run/'conditional_permutation.json',permutation)
  # Small training-label shuffle diagnostics; identical feature/policy choices, no control tuning.
  controls=[]
  for offset in range(CFG['shuffled_training_controls']):
   rng=np.random.default_rng(CFG['seed']+offset);shuffled=y.copy()
   for s in sorted(set(r['subject'] for r in rows)):
    for rn in (4,8,12):
     if s<=80 or (s>=95 and rn==4):
      jj=indices(rows,s,s,rn);shuffled[jj]=rng.permutation(shuffled[jj])
   sham=fit_linear(f[best_linear['feature']],shuffled,src,best_linear['C']);p,_,_=cohort_linear(sham,f[best_linear['feature']],shuffled,rows,tsubjects,best_linear['prior']);controls.append({'seed':CFG['seed']+offset,**metric(ty,p)})
  save(run/'shuffled_training_controls.json',{'scope':'Linear family only, training labels shuffled within subject/run, R04 personal labels also shuffled. Test labels unchanged. Not EEGNet pipeline controls.','results':controls})
  rti=indices(rrows,95,108,12);rty=ry[rti];rtr=[rrows[i] for i in rti];rm=rest_models[(rest_best['feature'],rest_best['C'])];ap,rr,rws=cohort_linear(rm,rf[rest_best['feature']],active_y,rrows,tsubjects,rest_best['prior']);assert list(map(key,rr))==list(map(key,rtr));dp=[]
  for s in tsubjects:
   jj=indices(rrows,s,s,12)
   if recommended=='linear':dp.extend(predict_linear(lm,rf[best_linear['feature']][jj],lws[s]))
   else:dp.extend(np.mean([infer_net(make_net({**ck,'model_state':neural_test[policy['id']]['states'][n][s]}),rx[jj],ck,device) for n in CFG['seeds']],axis=0))
  dp=np.array(dp);gate_results=[];rest_predictions=[]
  for p in policies:
   score,gp=gate(rty,ap,dp,p['tau'],p['direction_confidence']);gate_results.append({'validation_cap':p['cap'],'tau':p['tau'],'direction_confidence':p['direction_confidence'],**score});rest_predictions.extend({**r,'validation_cap':p['cap'],'p_active':float(a),'p_right_given_active':float(d),'gate_prediction':int(g)} for r,a,d,g in zip(rtr,ap,dp,gp))
  # Previous v3 frozen gate, now restricted to exactly the same R12 three-class windows.
  prior_rest=read_csv(V3/'rest/test_predictions.csv');old_gate={key(r):int(r['gate_prediction']) for r in prior_rest if r['model']=='EEGNet-3class'};oldpred=np.array([old_gate[key(r)] for r in rtr]);rmask=rty==0;mmask=~rmask;cnt=int(((oldpred!=0)&mmask).sum());cor=int(((oldpred==rty)&mmask).sum());oldscore={'rest_n':int(rmask.sum()),'mi_n':int(mmask.sum()),'false_triggers':int(((oldpred!=0)&rmask).sum()),'rest_fpr':float((oldpred[rmask]!=0).mean()),'mi_outputs':cnt,'mi_coverage':cnt/int(mmask.sum()),'correct_outputs':cor,'correct_mi_rate':cor/int(mmask.sum()),'accepted_mi_accuracy':cor/cnt if cnt else None}
  save(run/'rest_summary.json',{'new':gate_results,'v3_same_r12_reference':oldscore,'comparison_note':'Both restricted to identical run12 windows. New model includes personal R04 rest calibration, v3 rest model did not. Different validation caps are explicit; no matched-safety improvement claim.'});U.write_csv(run/'rest_predictions.csv',rest_predictions);joblib.dump({'model':rm,'weights':rws,'selection':rest_best,'policies':policies},test_out/'rest_detector.joblib')
  # Reload independent persisted weights and require the same labels/decisions.
  loaded=joblib.load(test_out/'linear_personal.joblib');reload_lp=[]
  for s in tsubjects:
   jj=indices(rows,s,s,12);reload_lp.extend(predict_linear(loaded['model'],f[best_linear['feature']][jj],loaded['weights'][s]))
  assert np.allclose(reload_lp,lp,atol=1e-10);reloaded_neural=[]
  for s in tsubjects:
   jj=indices(rows,s,s,12);p=[]
   for n in CFG['seeds']:
    disk=torch.load(test_out/f'{policy["id"]}_{n}_S{s:03d}.pt',map_location='cpu',weights_only=True);p.append(infer_net(make_net(disk),x[jj],disk,torch.device('cpu')))
   reloaded_neural.extend(np.mean(p,axis=0))
  assert np.array_equal(np.array(reloaded_neural)>=.5,probs['neural']>=.5)
  loaded_rest=joblib.load(test_out/'rest_detector.joblib');rap=[]
  for s in tsubjects:
   jj=indices(rrows,s,s,12);rap.extend(predict_linear(loaded_rest['model'],rf[rest_best['feature']][jj],loaded_rest['weights'][s]))
  assert np.allclose(rap,ap,atol=1e-10)
  for p in policies:
   _,a=gate(rty,ap,dp,p['tau'],p['direction_confidence']);_,b=gate(rty,np.array(rap),dp,p['tau'],p['direction_confidence']);assert np.array_equal(a,b)
  verification={'status':'PASS','linear_reload':True,'neural_cpu_reload_classes':True,'rest_reload_gate_decisions':True,'mi_test_keys_match_v3':True,'test_never_used_for_policy_selection':True};save(run/'reload_verification.json',verification)
  resources={'total_seconds':time.perf_counter()-start,'rss_mib':psutil.Process().memory_info().rss/2**20,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,'device':str(device),'neural_parameters':sum(p.numel() for p in make_net(ck).parameters())}
  report={'mi_selection':frozen,'mi':summary,'seed_stability':stability,'permutation':permutation,'shuffle_control':controls,'rest_selection':rest_best,'rest':gate_results,'v3_rest_same_r12':oldscore,'resources':resources,'quality':{'mi_source_trials':len(src),'mi_validation_trials':len(val_y),'mi_test_trials':len(ty),'mi_test_people':len(tsubjects),'rest_source_trials':len(rtrain),'rest_validation_trials':len(rvy),'rest_test_trials':len(rty),'mi_calibration_per_person':[len(indices(rows,s,s,4)) for s in tsubjects],'rest_calibration_per_person':[len(indices(rrows,s,s,4)) for s in tsubjects]},'evaluation_status':CFG['evaluation_status']};save(run/'summary.json',report)
  status.update(status='COMPLETE',stage='finished',summary=str(run/'summary.json'));save(ROOT/'latest_run.json',status);save(run/'status.json',status);print('COMPLETE '+json.dumps(report,ensure_ascii=False),flush=True)
 except Exception:
  status.update(status='FAILED',error=traceback.format_exc());save(ROOT/'latest_run.json',status);save(run/'status.json',status);raise
if __name__=='__main__':main()

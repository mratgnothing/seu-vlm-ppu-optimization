from __future__ import annotations
import copy, csv, hashlib, importlib.util, json, random, sys, time, traceback
from datetime import datetime,timezone
from pathlib import Path
import joblib, mne, numpy as np, torch
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
from torch import nn

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
PILOT=Path(CFG['pilot_task']); V2=Path(CFG['v2_task'])/'runs'/CFG['v2_run']
sys.path.insert(0,str(PILOT))
spec=importlib.util.spec_from_file_location('pilot_utils',PILOT/'train.py'); U=importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
from eegnet import EEGNet

def save(p,v): U.save_json(p,v)
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
def read_csv(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def metrics(y,p):
 pred=p.argmax(1); labels=list(range(p.shape[1]))
 return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}
def ci(y,p,rows,seedval):
 rng=np.random.default_rng(seedval);sub=sorted(set(r['subject'] for r in rows));idx={s:np.array([i for i,r in enumerate(rows) if r['subject']==s]) for s in sub};values=[]
 for _ in range(1000):
  ii=np.concatenate([idx[s] for s in rng.choice(sub,len(sub),replace=True)]);values.append(balanced_accuracy_score(y[ii],p[ii].argmax(1)))
 return np.quantile(values,[.025,.975]).tolist()
def load_v2():
 data=np.load(V2/'prepared_all_trials.npz'); rows=read_csv(V2/'all_trial_manifest.csv')
 rows=[{**r,**{k:int(r[k]) for k in ('subject','run','onset_sample','label')}} for r in rows]
 assert len(rows)==len(data['y']) and np.array_equal(data['y'],[r['label'] for r in rows])
 return data['X'],data['y'],rows
def load_net(ck,classes=2):
 net=EEGNet(len(ck['channels']),ck['samples'],**ck['model_settings'])
 if classes!=2:net.classifier=nn.Linear(net.classifier.in_features,classes)
 if 'model_state' in ck:net.load_state_dict(ck['model_state'])
 return net
def infer(net,x,device,mean,std):
 net=net.to(device);net.eval()
 with torch.no_grad():return np.concatenate([net(torch.from_numpy((b-mean)/std).to(device)).softmax(1).cpu().numpy() for b in np.array_split(x,max(1,(len(x)+127)//128))])

def calibration(run,X,Y,rows,device):
 out=run/'calibration';out.mkdir();selected=json.loads((V2/'new_person/model_selection.json').read_text())['selected']
 global_ck=torch.load(V2/'new_person'/selected/'eegnet_best.pt',map_location='cpu',weights_only=True)
 mean=global_ck['channel_mean_uv'].numpy();std=global_ck['channel_std_uv'].numpy()
 global_csp=joblib.load(V2/'new_person/csp_lda.joblib')
 predictions=[];subjects=[];selection=[];verification=[];histories=[]
 all_y=[];all_rows=[];all_p={k:[] for k in ('CSP-global','CSP-personal','EEGNet-global','EEGNet-personal')}
 for s in range(*[CFG['calibration_subjects'][0],CFG['calibration_subjects'][1]+1]):
  idx={name:np.array([i for i,r in enumerate(rows) if r['subject']==s and r['run']==rnum]) for name,rnum in [('train',4),('validation',8),('test',12)]}
  if any(len(ii)<8 or len(set(Y[ii]))<2 for ii in idx.values()):raise RuntimeError(f'Unexpected calibration eligibility S{s}')
  x={k:X[ii] for k,ii in idx.items()};y={k:Y[ii] for k,ii in idx.items()};rr=[rows[i] for i in idx['test']]
  subject_dir=out/f'S{s:03d}';subject_dir.mkdir()
  csp_candidates=[]
  for n in CFG['csp_components']:
   model=U.csp_pipeline(n);model.fit(x['train'],y['train']);v=balanced_accuracy_score(y['validation'],model.predict(x['validation']));csp_candidates.append((float(v),n,model))
  best=max(csp_candidates,key=lambda v:v[0]);joblib.dump(best[2],subject_dir/'csp_personal.joblib')
  seed(CFG['seed']+s);net=load_net(global_ck).to(device)
  for p in net.parameters():p.requires_grad=False
  for p in net.classifier.parameters():p.requires_grad=True
  optimizer=torch.optim.Adam(net.classifier.parameters(),lr=CFG['adaptation']['lr']);loss_fn=nn.CrossEntropyLoss()
  tx=torch.from_numpy((x['train']-mean)/std).to(device);ty=torch.from_numpy(y['train']).to(device)
  best_state=None;best_score=-1;best_loss=float('inf');best_epoch=0;history=[]
  for epoch in range(1,CFG['adaptation']['epochs']+1):
   # Backbone, running BatchNorm statistics and normalization stay frozen.
   net.eval();optimizer.zero_grad(set_to_none=True);loss=loss_fn(net(tx),ty);loss.backward();optimizer.step();net.constrain_weights()
   vp=infer(net,x['validation'],device,mean,std);vscore=float(balanced_accuracy_score(y['validation'],vp.argmax(1)));vl=float(-np.log(vp[np.arange(len(vp)),y['validation']].clip(1e-8)).mean())
   history.append({'subject':s,'epoch':epoch,'train_loss':float(loss.detach().cpu()),'validation_ba':vscore,'validation_loss':vl})
   if vscore>best_score or (vscore==best_score and vl<best_loss):best_score,best_loss,best_epoch=vscore,vl,epoch;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
   if epoch>=CFG['adaptation']['minimum_epochs'] and epoch-best_epoch>=CFG['adaptation']['patience']:break
  net.load_state_dict(best_state);new_ck={**global_ck,'model_state':best_state,'calibration_subject':s,'calibration_run':4,'selection_run':8,'best_epoch':best_epoch,'adaptation':CFG['adaptation']};torch.save(new_ck,subject_dir/'eegnet_personal.pt')
  p={'CSP-global':global_csp.predict_proba(x['test']),'CSP-personal':best[2].predict_proba(x['test']),'EEGNet-global':infer(load_net(global_ck),x['test'],device,mean,std),'EEGNet-personal':infer(net,x['test'],device,mean,std)}
  reload_ck=torch.load(subject_dir/'eegnet_personal.pt',map_location='cpu',weights_only=True);cpu_p=infer(load_net(reload_ck),x['test'],torch.device('cpu'),mean,std);cp=joblib.load(subject_dir/'csp_personal.joblib').predict_proba(x['test'])
  assert np.array_equal(cpu_p.argmax(1),p['EEGNet-personal'].argmax(1)) and np.max(abs(cpu_p-p['EEGNet-personal']))<1e-3
  assert np.allclose(cp,p['CSP-personal'],atol=1e-8)
  verification.append({'subject':s,'test_n':len(y['test']),'cpu_reload_classes_identical':True,'eegnet_max_probability_delta':float(np.max(abs(cpu_p-p['EEGNet-personal'])))})
  for name,pp in p.items():
   result=metrics(y['test'],pp);subjects.append({'subject':s,'method':name,**{k:result[k] for k in ('n','accuracy','balanced_accuracy')}})
   all_p[name].append(pp);predictions.extend({**r,'method':name,'prediction':int(pr.argmax()),'p_left':float(pr[0]),'p_right':float(pr[1])} for r,pr in zip(rr,pp))
  all_y.extend(y['test']);all_rows.extend(rr);histories.extend(history)
  selection.append({'subject':s,'calibration_trials':len(y['train']),'validation_trials':len(y['validation']),'test_trials':len(y['test']),'csp_components':best[1],'csp_validation_ba':best[0],'eegnet_validation_ba':best_score,'best_epoch':best_epoch})
  print(f'CALIBRATION S{s:03d}: '+str({k:round(balanced_accuracy_score(y['test'],v.argmax(1)),3) for k,v in p.items()}),flush=True)
 combined_y=np.array(all_y);combined_p={k:np.concatenate(v) for k,v in all_p.items()};summary={}
 for name,pp in combined_p.items():summary[name]={**metrics(combined_y,pp),'ba_95ci':ci(combined_y,pp,all_rows,CFG['seed'])}
 paired={}; rng=np.random.default_rng(CFG['seed']);sub=sorted(set(r['subject'] for r in all_rows));ix={s:np.array([i for i,r in enumerate(all_rows) if r['subject']==s]) for s in sub}
 for model in ('CSP','EEGNet'):
  deltas=[]
  for _ in range(1000):
   ii=np.concatenate([ix[s] for s in rng.choice(sub,len(sub),replace=True)]);deltas.append(balanced_accuracy_score(combined_y[ii],combined_p[model+'-personal'][ii].argmax(1))-balanced_accuracy_score(combined_y[ii],combined_p[model+'-global'][ii].argmax(1)))
  paired[model]={'delta_ba':summary[model+'-personal']['balanced_accuracy']-summary[model+'-global']['balanced_accuracy'],'paired_subject_bootstrap_95ci':np.quantile(deltas,[.025,.975]).tolist(),'subjects_improved':sum(next(r['balanced_accuracy'] for r in subjects if r['subject']==s and r['method']==model+'-personal')>next(r['balanced_accuracy'] for r in subjects if r['subject']==s and r['method']==model+'-global') for s in sub)}
 U.write_csv(out/'predictions.csv',predictions);U.write_csv(out/'subject_metrics.csv',subjects);U.write_csv(out/'selection.csv',selection);U.write_csv(out/'history.csv',histories)
 save(out/'summary.json',{'metrics':summary,'paired':paired,'test_subjects':len(sub),'test_n':len(combined_y),'status':CFG['evaluation_status']});save(out/'reload_verification.json',{'status':'PASS','subjects':verification})
 return {'metrics':summary,'paired':paired,'subjects':subjects,'selection':selection}

def prepare_rest(run,X,Y,rows):
 eligible=sorted(set(r['subject'] for r in rows));root=Path(CFG['dataset_root']);hashes={}
 for line in (root/'metadata/SHA256SUMS.official.txt').read_text().splitlines():
  d,name=line.split(maxsplit=1);hashes[name.lstrip('*').removeprefix('./')]=d
 rest_x=[];rest_rows=[];qc=[]
 for s in eligible:
  for rnum in (4,8,12):
   filename=f'S{s:03d}/S{s:03d}R{rnum:02d}.edf';path=root/'raw'/filename
   digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==hashes[filename]
   raw=mne.io.read_raw_edf(path,preload=True,verbose='ERROR');orig=raw.info['sfreq'];mne.datasets.eegbci.standardize(raw);raw.pick(CFG['channels']);raw.set_montage('standard_1005');raw.set_eeg_reference('average',projection=False,verbose='ERROR')
   if orig!=CFG['sfreq']:raw.resample(CFG['sfreq'],verbose='ERROR')
   raw.filter(*CFG['bandpass_hz'],method='iir',phase='forward',iir_params={'order':4,'ftype':'butter','output':'sos'},verbose='ERROR')
   fs=raw.info['sfreq'];ev,_=mne.events_from_annotations(raw,event_id={'T0':1},verbose='ERROR');dur={int(round(a['onset']*fs)):a['duration'] for a in raw.annotations if a['description']=='T0'}
   valid=np.array([e for e in ev if dur.get(int(e[0]),0)+1/fs>=4],dtype=int)
   rx=np.empty((0,len(CFG['channels']),480),dtype=np.float32);accepted=[]
   if len(valid):
    ep=mne.Epochs(raw,valid,event_id={'rest':1},tmin=1,tmax=4-1/fs,baseline=None,preload=True,reject={'eeg':CFG['reject_uv']*1e-6},verbose='ERROR')
    rx=ep.get_data(copy=True).astype(np.float32)*1e6;accepted=ep.events
   rest_x.append(rx)
   rest_rows.extend({'subject':s,'run':rnum,'onset_sample':int(e[0]),'label':0,'file':filename} for e in accepted)
   qc.append({'subject':s,'run':rnum,'rest_intervals':len(ev),'short_intervals':len(ev)-len(valid),'accepted_rest':len(rx),'rejected_rest':len(valid)-len(rx),'source_sfreq':orig,'sha256':digest})
  if s%10==0:print(f'REST PREPROCESS S{s:03d}',flush=True)
 excluded=sorted(set(r['subject'] for r in qc if r['accepted_rest']<CFG['rest_min_trials_per_record']))
 mixed_x=np.concatenate([X,np.concatenate(rest_x)]); mixed_y=np.concatenate([Y+1,np.zeros(sum(len(x) for x in rest_x),dtype=np.int64)]);mixed_rows=[{**r,'label':r['label']+1} for r in rows]+rest_rows
 keep=np.array([r['subject'] not in excluded for r in mixed_rows]);mixed_x=mixed_x[keep];mixed_y=mixed_y[keep];mixed_rows=[r for r,k in zip(mixed_rows,keep) if k]
 keys=[(r['subject'],r['run'],r['onset_sample']) for r in mixed_rows];assert len(set(keys))==len(keys)
 # Non-overlapping full windows; split membership is by whole subject.
 groups={}
 for r in mixed_rows:groups.setdefault((r['subject'],r['run']),[]).append(r['onset_sample'])
 assert all(all(b-a>=480 for a,b in zip(sorted(v),sorted(v)[1:])) for v in groups.values())
 U.write_csv(run/'rest_quality.csv',qc);U.write_csv(run/'three_class_manifest.csv',mixed_rows);np.savez_compressed(run/'three_class_trials.npz',X=mixed_x,y=mixed_y)
 quality={'input_edf':len(qc),'sha256_verified':len(qc),'v2_excluded_subjects':[109],'additional_rest_excluded_subjects':excluded,'rest_trials':int((mixed_y==0).sum()),'mi_trials':int((mixed_y!=0).sum()),'short_rest_intervals':sum(r['short_intervals'] for r in qc),'rejected_rest_windows':sum(r['rejected_rest'] for r in qc)};save(run/'rest_quality_summary.json',quality)
 return mixed_x,mixed_y,mixed_rows,quality

def bandpower(x):
 f,p=welch(x,fs=CFG['sfreq'],nperseg=160,noverlap=80,axis=-1)
 return np.concatenate([np.log(p[..., (f>=lo)&(f<hi)].sum(-1).clip(1e-12)) for lo,hi in ((8,12),(12,18),(18,24),(24,30.001))],axis=1)
def gate_predictions(p,tau):
 if tau>=1:return np.zeros(len(p),dtype=np.int64)
 pred=p.argmax(1); return np.where((pred!=0)&(p[:,1:].max(1)>=tau),pred,0)
def gate_score(y,p,tau):
 pred=gate_predictions(p,tau);rest=y==0;active=~rest;command=pred!=0;correct=pred==y
 return {'threshold':tau,'rest_n':int(rest.sum()),'rest_false_commands':int((rest&command).sum()),'rest_false_trigger_rate':float(command[rest].mean()),'mi_n':int(active.sum()),'mi_command_n':int((active&command).sum()),'mi_correct_commands':int((active&command&correct).sum()),'mi_wrong_commands':int((active&command&~correct).sum()),'mi_coverage':float(command[active].mean()),'mi_correct_command_rate':float((command&correct)[active].mean()),'accepted_mi_accuracy':float(correct[active&command].mean()) if (active&command).any() else None,'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def train_rest_net(x,y,out,device):
 seed(CFG['seed']);settings=CFG['rest_eegnet'];mean=x['train'].mean(axis=(0,2),keepdims=True);std=x['train'].std(axis=(0,2),keepdims=True).clip(1e-3)
 ck={'channels':CFG['channels'],'samples':480,'model_settings':{'F1':8,'D':2,'F2':16,'dropout':settings['dropout'],'kernel_length':64}}
 net=load_net(ck,classes=3).to(device);optimizer=torch.optim.Adam(net.parameters(),lr=settings['lr'])
 counts=np.bincount(y['train'],minlength=3);weights=torch.tensor(len(y['train'])/(3*counts),dtype=torch.float32,device=device);loss_fn=nn.CrossEntropyLoss(weight=weights)
 tx=torch.from_numpy((x['train']-mean)/std).to(device);ty=torch.from_numpy(y['train']).to(device);rng=torch.Generator(device=device).manual_seed(CFG['seed'])
 history=[];bestscore=-1;bestloss=float('inf');bestepoch=0;beststate=None;started=time.perf_counter()
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,settings['epochs']+1):
  net.train();losses=[]
  for ii in torch.randperm(len(tx),generator=rng,device=device).split(settings['batch_size']):
   optimizer.zero_grad(set_to_none=True);loss=loss_fn(net(tx[ii]),ty[ii]);loss.backward();optimizer.step();net.constrain_weights();losses.append(float(loss.detach().cpu()))
  vp=infer(net,x['validation'],device,mean,std);tp=infer(net,x['train'],device,mean,std);score=float(balanced_accuracy_score(y['validation'],vp.argmax(1)));vl=float(-np.log(vp[np.arange(len(vp)),y['validation']].clip(1e-8)).mean())
  history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':vl,'train_ba':float(balanced_accuracy_score(y['train'],tp.argmax(1))),'validation_ba':score})
  if score>bestscore or (score==bestscore and vl<bestloss):bestscore,bestloss,bestepoch=score,vl,epoch;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  if epoch==1 or epoch%10==0:print(f'REST EEGNET epoch {epoch} train_BA={history[-1]["train_ba"]:.3f} val_BA={score:.3f}',flush=True);U.write_csv(out/'eegnet_history.csv',history)
  if epoch>=settings['minimum_epochs'] and epoch-bestepoch>=settings['patience']:break
 net.load_state_dict(beststate);ck.update(model_state=beststate,channel_mean_uv=torch.from_numpy(mean),channel_std_uv=torch.from_numpy(std),classes=['rest','left','right'],best_epoch=bestepoch,config=CFG);torch.save(ck,out/'eegnet_rest.pt');U.write_csv(out/'eegnet_history.csv',history)
 res={'epochs_ran':len(history),'best_epoch':bestepoch,'fit_seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in net.parameters()),'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0};save(out/'eegnet_resources.json',res)
 return net,mean,std,ck,res

def rest_experiment(run,X,Y,rows,device):
 out=run/'rest';out.mkdir();x,y,rr={},{},{}
 for split,lo,hi in [('train',11,80),('validation',81,94),('test',95,108)]:
  ii=np.array([i for i,r in enumerate(rows) if lo<=r['subject']<=hi]);x[split]=X[ii];y[split]=Y[ii];rr[split]=[rows[i] for i in ii]
  if len(np.unique(y[split]))!=3:raise RuntimeError('Three classes required')
 split_counts={k:{'total':len(v),'rest':int((v==0).sum()),'left':int((v==1).sum()),'right':int((v==2).sum()),'subjects':len(set(r['subject'] for r in rr[k]))} for k,v in y.items()};save(out/'split_counts.json',split_counts)
 U.write_csv(out/'split_manifest.csv',[dict(r,split=k) for k,v in rr.items() for r in v]);print('REST SPLITS '+str(split_counts),flush=True)
 bp={k:bandpower(v) for k,v in x.items()};lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto');lda.fit(bp['train'],y['train']);joblib.dump(lda,out/'bandpower_lda.joblib')
 net,mean,std,ck,res=train_rest_net(x,y,out,device)
 val_prob={'Bandpower-LDA':lda.predict_proba(bp['validation']),'EEGNet-3class':infer(net,x['validation'],device,mean,std)}
 policies={};curve=[]
 for name,p in val_prob.items():
  scores=[gate_score(y['validation'],p,t) for t in CFG['rest_thresholds']]
  feasible=[v for v in scores if v['rest_false_trigger_rate']<=CFG['validation_rest_false_trigger_cap']]
  chosen=max(feasible,key=lambda v:(v['mi_correct_command_rate'],-v['rest_false_trigger_rate'],-v['threshold']))
  policies[name]={'threshold':chosen['threshold'],'selection_split':'validation only','validation_result':chosen,'useful_on_validation':chosen['mi_correct_commands']>0}
  curve.extend({'model':name,**{k:v for k,v in r.items() if k!='confusion_matrix'}} for r in scores)
 save(out/'frozen_gate_policies.json',policies);U.write_csv(out/'validation_threshold_curve.csv',curve)
 print('FROZEN GATES '+str({k:v['threshold'] for k,v in policies.items()}),flush=True)
 # Test is read only after checkpoint and thresholds are frozen.
 test_prob={'Bandpower-LDA':lda.predict_proba(bp['test']),'EEGNet-3class':infer(net,x['test'],device,mean,std)};summary={};predictions=[];verification=[]
 for name,p in test_prob.items():
  raw=metrics(y['test'],p);raw['ba_95ci']=ci(y['test'],p,rr['test'],CFG['seed']);tau=policies[name]['threshold'];g=gate_score(y['test'],p,tau);gp=gate_predictions(p,tau)
  summary[name]={'raw_three_class':raw,'gate':g,'threshold':tau,'validation_gate':policies[name]['validation_result']}
  for r,pr,decision in zip(rr['test'],p,gp):predictions.append({**r,'model':name,'prediction':int(pr.argmax()),'gate_prediction':int(decision),'p_rest':float(pr[0]),'p_left':float(pr[1]),'p_right':float(pr[2]),'threshold':tau})
  if name=='Bandpower-LDA':reload_p=joblib.load(out/'bandpower_lda.joblib').predict_proba(bandpower(x['test']))
  else:reload_ck=torch.load(out/'eegnet_rest.pt',map_location='cpu',weights_only=True);reload_p=infer(load_net(reload_ck,3),x['test'],torch.device('cpu'),mean,std)
  assert np.array_equal(reload_p.argmax(1),p.argmax(1)) and np.array_equal(gate_predictions(reload_p,tau),gp)
  verification.append({'model':name,'classes_identical':True,'gate_decisions_identical':True,'max_probability_delta':float(np.max(abs(reload_p-p)))})
 U.write_csv(out/'test_predictions.csv',predictions);save(out/'checkpoint_verification.json',{'status':'PASS','models':verification})
 save(out/'summary.json',{'models':summary,'splits':split_counts,'evaluation_status':CFG['evaluation_status'],'rest_scope':CFG['rest_scope'],'resources':res})
 wi=[CFG['channels'].index(ch) for ch in ('C3','Cz','C4')]
 trials=[{**r,'index':i,'wave_uv':np.round(x['test'][i,wi,::2],2).tolist(),'probabilities':{k:np.round(p[i],8).tolist() for k,p in test_prob.items()},'decisions':{k:int(gate_predictions(p[i:i+1],policies[k]['threshold'])[0]) for k,p in test_prob.items()}} for i,r in enumerate(rr['test'])]
 return {'models':summary,'splits':split_counts,'policies':policies,'curve':curve,'trials':trials,'resources':res}

def main():
 run=ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True);log=(run/'training.log').open('w',encoding='utf-8');sys.stdout=U.Tee(sys.stdout,log);sys.stderr=U.Tee(sys.stderr,log)
 status={'status':'RUNNING','run_dir':str(run),'stage':'calibration'};save(ROOT/'latest_run.json',status);save(run/'config.json',CFG);started=time.perf_counter()
 try:
  mne.set_log_level('ERROR');torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seed(CFG['seed']);print('RUN '+str(run)+' DEVICE '+str(device),flush=True)
  save(run/'provenance.json',{'v2_run':str(V2),'cache_sha256':hashlib.sha256((V2/'prepared_all_trials.npz').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((V2/'all_trial_manifest.csv').read_bytes()).hexdigest(),'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'train_v3.py',ROOT/'config.json',PILOT/'eegnet.py',PILOT/'train.py')},'evaluation_status':CFG['evaluation_status']})
  X,Y,rows=load_v2();cal=calibration(run,X,Y,rows,device);status['stage']='rest_preprocessing';save(ROOT/'latest_run.json',status)
  RX,RY,RR,quality=prepare_rest(run,X,Y,rows);status['stage']='rest_training';save(ROOT/'latest_run.json',status);rest=rest_experiment(run,RX,RY,RR,device)
  output={'calibration':cal,'rest':rest,'quality':quality,'evaluation_status':CFG['evaluation_status'],'generated_at':datetime.now(timezone.utc).isoformat(),'total_seconds':time.perf_counter()-started};save(run/'visualization_data.json',output)
  summary={**output,'rest':{k:v for k,v in rest.items() if k!='trials'}};save(run/'summary.json',summary);status.update(status='COMPLETE',stage='finished',summary=str(run/'summary.json'));save(ROOT/'latest_run.json',status);save(run/'status.json',status);print('COMPLETE '+json.dumps({'calibration':cal['metrics'],'paired':cal['paired'],'rest':rest['models'],'quality':quality,'total_seconds':output['total_seconds']},ensure_ascii=False),flush=True)
 except Exception:
  status.update(status='FAILED',error=traceback.format_exc());save(ROOT/'latest_run.json',status);save(run/'status.json',status);raise
if __name__=='__main__':main()

from __future__ import annotations
import csv, hashlib, importlib.util, json, random, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import joblib, mne, numpy as np, torch

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
PILOT = Path(CONFIG['pilot_task'])
sys.path.insert(0,str(PILOT))
spec=importlib.util.spec_from_file_location('pilot_utils',PILOT/'train.py')
utils=importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)
from eegnet import EEGNet

def save(path,value): utils.save_json(path,value)
def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def prepare(run):
    root=Path(CONFIG['dataset_root']); hashes={}
    for line in (root/'metadata/SHA256SUMS.official.txt').read_text().splitlines():
        digest,name=line.split(maxsplit=1); hashes[name.lstrip('*').removeprefix('./')]=digest
    xs,ys,rows,qc=[],[],[],[]
    for s in range(11,110):
        for r in CONFIG['runs']:
            file=f'S{s:03d}/S{s:03d}R{r:02d}.edf'; path=root/'raw'/file
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != hashes[file]: raise RuntimeError('Checksum mismatch '+file)
            raw=mne.io.read_raw_edf(path,preload=True,verbose='ERROR'); sf=raw.info['sfreq']
            mne.datasets.eegbci.standardize(raw); raw.pick(CONFIG['channels']); raw.set_montage('standard_1005')
            raw.set_eeg_reference('average',projection=False,verbose='ERROR')
            if sf != CONFIG['target_sfreq']: raw.resample(CONFIG['target_sfreq'],verbose='ERROR')
            raw.filter(*CONFIG['bandpass_hz'],method='iir',phase='forward',iir_params={'order':4,'ftype':'butter','output':'sos'},verbose='ERROR')
            events,_=mne.events_from_annotations(raw,event_id={'T1':1,'T2':2},verbose='ERROR')
            fs=raw.info['sfreq']; start,end=CONFIG['epoch_seconds']
            duration={int(round(a['onset']*fs)):a['duration'] for a in raw.annotations if a['description'] in ('T1','T2')}
            valid=np.array([ev for ev in events if duration.get(int(ev[0]),0)+1/fs>=end],dtype=int)
            x=np.empty((0,len(CONFIG['channels']),int((end-start)*fs)),np.float32); y=np.empty(0,np.int64); accepted=[]
            if len(valid):
                epochs=mne.Epochs(raw,valid,event_id={'left':1,'right':2},on_missing='ignore',tmin=start,tmax=end-1/fs,baseline=None,preload=True,reject={'eeg':CONFIG['peak_to_peak_reject_uv']*1e-6},verbose='ERROR')
                x=epochs.get_data(copy=True).astype(np.float32)*1e6; y=epochs.events[:,2].astype(np.int64)-1; accepted=epochs.events
            eligible=len(y)>=CONFIG['minimum_trials_per_record'] and set(y.tolist())=={0,1} and np.isfinite(x).all()
            qc.append({'subject':s,'run':r,'file':file,'source_sfreq':sf,'target_sfreq':fs,'cued_trials':len(events),'short_intervals':len(events)-len(valid),'rejected_trials':len(valid)-len(y),'accepted_trials':len(y),'left':int((y==0).sum()),'right':int((y==1).sum()),'record_eligible':eligible,'sha256':digest})
            xs.append(x); ys.append(y)
            rows.extend({'subject':s,'run':r,'onset_sample':int(ev[0]),'label':int(label),'file':file} for ev,label in zip(accepted,y))
        if s%10==0 or s==109: print(f'PREPROCESS S{s:03d} ({s-10}/99 subjects)',flush=True)
    excluded=sorted({r['subject'] for r in qc if not r['record_eligible']})
    for record in qc: record['subject_eligible']=record['subject'] not in excluded
    X=np.concatenate(xs); Y=np.concatenate(ys)
    keep=np.array([r['subject'] not in excluded for r in rows]); X=X[keep]; Y=Y[keep]; rows=[r for r,k in zip(rows,keep) if k]
    keys=[(r['subject'],r['run'],r['onset_sample']) for r in rows]
    assert len(keys)==len(set(keys)) and np.isfinite(X).all()
    utils.write_csv(run/'data_quality.csv',qc); utils.write_csv(run/'all_trial_manifest.csv',rows)
    np.savez_compressed(run/'prepared_all_trials.npz',X=X,y=Y)
    save(run/'quality_summary.json',{'input_records':len(qc),'verified_sha256':len(qc),'excluded_subjects':excluded,'retained_subjects':len(set(r['subject'] for r in rows)),'retained_trials':len(Y),'resampled_records':sum(r['source_sfreq']!=160 for r in qc),'short_intervals':sum(r['short_intervals'] for r in qc),'amplitude_or_boundary_rejections':sum(r['rejected_trials'] for r in qc)})
    return X,Y,rows,qc

def split_data(protocol,X,Y,rows):
    masks={}
    if protocol=='same_person':
        for split,r in [('train',4),('validation',8),('test',12)]: masks[split]=np.array([11<=row['subject']<=80 and row['run']==r for row in rows])
    else:
        for split,key in [('train','development'),('validation','validation'),('test','test')]:
            lo,hi=CONFIG['subjects'][key]; masks[split]=np.array([lo<=row['subject']<=hi for row in rows])
    assert not (masks['train']&masks['test']).any() and not (masks['validation']&masks['test']).any()
    return ({k:X[v] for k,v in masks.items()},{k:Y[v] for k,v in masks.items()},{k:[r for r,v in zip(rows,m) if v] for k,m in masks.items()})

def predict(checkpoint,x,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=EEGNet(len(ck['channels']),ck['samples'],**ck['model_settings']).to(device); model.load_state_dict(ck['model_state']); model.eval()
    scale=(x-ck['channel_mean_uv'].numpy())/ck['channel_std_uv'].numpy()
    with torch.no_grad():
        p=np.concatenate([model(torch.from_numpy(b).to(device)).softmax(1).cpu().numpy() for b in np.array_split(scale,max(1,(len(x)+255)//256))])
    return p

def train_protocol(name,run,X,Y,rows,device):
    out=run/name; out.mkdir(); x,y,r=split_data(name,X,Y,rows)
    cfg=dict(CONFIG); cfg['subjects']=sorted(set(row['subject'] for row in r['train']))
    utils.write_csv(out/'split_manifest.csv',[dict(row,split=k) for k,rr in r.items() for row in rr])
    print(f'PROTOCOL {name} splits '+str({k:len(v) for k,v in y.items()}),flush=True)
    csp,cspres=utils.train_csp(x,y,cfg,out)
    candidates=[]
    # The test set is not passed to candidate training or model selection.
    for candidate in CONFIG['eegnet_candidates']:
        cdir=out/candidate['id']; cdir.mkdir(); candidate_cfg=dict(cfg)
        candidate_cfg['eegnet']={**CONFIG['eegnet'],**{k:v for k,v in candidate.items() if k!='id'}}
        seed(CONFIG['seed']); print(f'CANDIDATE {name} {candidate["id"]}',flush=True)
        p,h,res=utils.train_eegnet({k:x[k] for k in ('train','validation')},{k:y[k] for k in ('train','validation')},candidate_cfg,cdir,device)
        score=utils.metric(y['validation'],p['validation']); candidates.append({'id':candidate['id'],'validation_metrics':score,'resources':res})
        del p
    chosen=max(candidates,key=lambda c:c['validation_metrics']['balanced_accuracy'])
    save(out/'model_selection.json',{'criterion':'validation balanced accuracy; predefined candidate order breaks ties','candidates':candidates,'selected':chosen['id']})
    print(f'SELECTED {name}: {chosen["id"]}; now unsealing test set',flush=True)
    selected_ck=out/chosen['id']/'eegnet_best.pt'
    csp_p=csp.predict_proba(x['test']); eeg_p=predict(selected_ck,x['test'],device)
    probs={'CSP-LDA':csp_p,'EEGNet':eeg_p}; scores={}; by_subject=[]; prediction_rows=[]; verify=[]
    for model,p in probs.items():
        scores[model]=utils.metric(y['test'],p); scores[model]['subject_cluster_bootstrap_ba_95ci']=utils.cluster_interval(y['test'],p,r['test'],CONFIG['seed'])
        by_subject+=utils.subject_metrics(y['test'],p,r['test'],model)
        prediction_rows.extend({**rr,'model':model,'prediction':int(pp.argmax()),'p_left':float(pp[0]),'p_right':float(pp[1])} for rr,pp in zip(r['test'],p))
    # Independent checkpoint reload inference, on CPU, verifies exported predictions.
    reload_csp=joblib.load(out/'csp_lda.joblib').predict_proba(x['test']); reload_eeg=predict(selected_ck,x['test'],torch.device('cpu'))
    for model,p,reloaded in [('CSP-LDA',csp_p,reload_csp),('EEGNet',eeg_p,reload_eeg)]:
        delta=float(np.max(np.abs(p-reloaded))); same=bool(np.array_equal(p.argmax(1),reloaded.argmax(1)))
        if not same or delta>1e-4: raise RuntimeError('Reload verification failed '+model)
        verify.append({'model':model,'classes_identical':same,'max_probability_delta':delta})
    utils.write_csv(out/'test_predictions.csv',prediction_rows); utils.write_csv(out/'subject_metrics.csv',by_subject)
    save(out/'checkpoint_verification.json',{'status':'PASS','trials':len(y['test']),'models':verify})
    summary={'protocol':name,'split_counts':{k:len(v) for k,v in y.items()},'split_subjects':{k:sorted(set(row['subject'] for row in rr)) for k,rr in r.items()},'test_metrics':scores,'selected_eegnet':chosen,'csp_resources':cspres}
    save(out/'summary.json',summary)
    # Reviewed dashboard rows associate every test trial with predictions and the actual filtered waveform.
    trials=[]; wave_indices=[CONFIG['channels'].index(ch) for ch in ('C3','Cz','C4')]
    for i,rr in enumerate(r['test']):
        trials.append({**rr,'trial_index':i,'wave_uv':np.round(x['test'][i,wave_indices,::2],2).tolist(),'probabilities':{k:np.round(p[i],8).tolist() for k,p in probs.items()}})
    curves=[]
    for c in candidates:
        with (out/c['id']/'eegnet_history.csv').open(encoding='utf-8-sig') as f: h=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        curves.append({'id':c['id'],'history':h,'selected':c['id']==chosen['id']})
    return {'id':name,'summary':summary,'subject_metrics':by_subject,'trials':trials,'curves':curves}

def main():
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'); run=ROOT/'runs'/rid; run.mkdir(parents=True)
    log=(run/'training.log').open('w',encoding='utf-8'); sys.stdout=utils.Tee(sys.stdout,log); sys.stderr=utils.Tee(sys.stderr,log)
    status={'status':'RUNNING','run_dir':str(run),'stage':'preprocessing'}; save(ROOT/'latest_run.json',status); save(run/'config.json',CONFIG)
    started=time.perf_counter()
    try:
        mne.set_log_level('ERROR'); torch.set_num_threads(4); seed(CONFIG['seed'])
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('RUN '+str(run)+' DEVICE '+str(device),flush=True)
        save(run/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'mne':mne.__version__,'numpy':np.__version__,'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None})
        sources=[ROOT/'train_expanded.py',ROOT/'config.json',PILOT/'train.py',PILOT/'eegnet.py']; save(run/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        X,Y,rows,qc=prepare(run); status['stage']='training'; save(ROOT/'latest_run.json',status)
        protocols=[train_protocol(name,run,X,Y,rows,device) for name in ('same_person','new_person')]
        quality=json.loads((run/'quality_summary.json').read_text())
        data={'title':'脑电运动想象 · 模型能力实验','generated_at_utc':datetime.now(timezone.utc).isoformat(),'quality':quality,'config':CONFIG,'protocols':protocols,'wave_channels':['C3','Cz','C4'],'wave_sfreq':80,'wave_time_start':1,'safety':CONFIG['safety_note'],'total_seconds':time.perf_counter()-started}
        save(run/'visualization_data.json',data)
        summary={'run_dir':str(run),'quality':quality,'protocols':{p['id']:p['summary'] for p in protocols},'total_seconds':time.perf_counter()-started}
        save(run/'summary.json',summary); status.update(status='COMPLETE',stage='finished',summary=str(run/'summary.json')); save(ROOT/'latest_run.json',status); save(run/'status.json',status)
        print('COMPLETE '+json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception:
        status.update(status='FAILED',error=traceback.format_exc()); save(ROOT/'latest_run.json',status); save(run/'status.json',status); raise

if __name__=='__main__': main()

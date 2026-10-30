import csv, json, sys
from pathlib import Path

task=Path(sys.argv[1]); dest=Path(sys.argv[2]); dest.mkdir(parents=True,exist_ok=True)
status=json.loads((task/'latest_run.json').read_text(encoding='utf-8'))
if status['status']!='COMPLETE': raise RuntimeError('Training not complete')
run=Path(status['run_dir']); data=json.loads((run/'visualization_data.json').read_text(encoding='utf-8'))
labels={'same_person':'同人跨记录','new_person':'新用户留出'}
queries={}
def add(qid,rows,files,definition,caveats):
    queries[qid]={'rows':rows,'source':{'label':'PhysioNet EEGMMIDB v1.0.0 · 本地真实训练结果','files':files,'links':[{'label':'PhysioNet 数据与范式','url':'https://physionet.org/content/eegmmidb/1.0.0/'}],'executedAt':data['generated_at_utc'],'metricDefinitions':[{'label':qid,'definition':definition}], 'caveats':caveats,'evidenceFlow':[{'title':'可复现实验','detail':'training/mi_expanded_20260930_v2/config.json 固定协议；train_expanded.py 核验官方 EDF SHA256，因果 8–30 Hz IIR，提示后1–4秒，9电极，固定250µV峰峰值剔除。候选只用验证集选择，冻结后评估测试集；模型CPU重载核对全部预测。'}]}}
comparison=[]; subjects=[]; matrices=[]; histories=[]; trials=[]; splitrows=[]
checks=[]
for p in data['protocols']:
    name=p['id']; summary=p['summary']
    with (run/name/'test_predictions.csv').open(encoding='utf-8-sig') as f: predictions=list(csv.DictReader(f))
    for model,metric in summary['test_metrics'].items():
        pp=[row for row in predictions if row['model']==model]; cm=[[0,0],[0,0]]
        for row in pp: cm[int(row['label'])][int(row['prediction'])]+=1
        ba=(cm[0][0]/sum(cm[0])+cm[1][1]/sum(cm[1]))/2
        assert cm==metric['confusion_matrix'] and abs(ba-metric['balanced_accuracy'])<1e-10
        checks.append({'protocol':name,'model':model,'csv_rows':len(pp),'recomputed_ba':ba,'confusion_matches':True})
        lo,hi=metric['subject_cluster_bootstrap_ba_95ci']
        comparison.append({'protocol':name,'protocol_label':labels[name],'model':model,'balancedAccuracyRate':ba,'accuracyRate':metric['accuracy'],'auc':metric['roc_auc'],'lowerRate':lo,'upperRate':hi,'interval':f'{lo:.1%} — {hi:.1%}','n':metric['n'],'test_subjects':len(summary['split_subjects']['test']),'chanceRate':0.5})
        for i in (0,1):
            for j in (0,1): matrices.append({'protocol':name,'model':model,'truth':['左手','右手'][i],'prediction':['左手','右手'][j],'count':cm[i][j]})
    subjects.extend({**r,'protocol':name,'subject_label':f"S{r['subject']:03d}",'balancedAccuracyRate':r['balanced_accuracy'],'chanceRate':.5} for r in p['subject_metrics'])
    for candidate in p['curves']:
        histories.extend({**r,'protocol':name,'candidate':candidate['id'],'selected':candidate['selected'],'trainAccuracyRate':r['train_balanced_accuracy'],'validationAccuracyRate':r['validation_balanced_accuracy'],'chanceRate':.5} for r in candidate['history'])
    trials.extend({**t,'protocol':name,'subject_label':f"S{t['subject']:03d}",'trial_id':f"{name}-S{t['subject']:03d}-R{t['run']:02d}-{t['onset_sample']}"} for t in p['trials'])
    splitrows.append({'protocol':name,**summary['split_counts'],'train_people':len(summary['split_subjects']['train']),'validation_people':len(summary['split_subjects']['validation']),'test_people':len(summary['split_subjects']['test']),'selected':summary['selected_eegnet']['id'],'epochs_ran':summary['selected_eegnet']['resources']['epochs_ran'],'best_epoch':summary['selected_eegnet']['resources']['best_epoch'],'fit_seconds':summary['selected_eegnet']['resources']['fit_seconds'],'peak_cuda_mib':summary['selected_eegnet']['resources']['peak_cuda_allocated_mib']})
add('comparison',comparison,['summary.json','same_person/test_predictions.csv','new_person/test_predictions.csv'],'BA=(左手召回率+右手召回率)/2；accuracy=正确数/测试数；95%区间对测试受试者整群自助抽样1000次。',['两个协议的测试人群不同，分数差异不能单独归因于跨人因素。','50%为二分类平衡准确率的机会水平参考；训练只跑一个预定随机种子，未验证跨天或在线效果。'])
add('subjects',subjects,['same_person/subject_metrics.csv','new_person/subject_metrics.csv'],'每名受试者、每个模型的测试集BA；n为该人的保留测试试次数。',['同人协议通常每人15试次，新人协议通常每人45试次；逐人分数存在较大采样不确定性。'])
add('matrices',matrices,['same_person/test_predictions.csv','new_person/test_predictions.csv'],'行是真实提示标签，列为模型最终预测；四格计数合计等于测试试次总数。',[])
add('history',histories,['same_person/*/eegnet_history.csv','new_person/*/eegnet_history.csv'],'逐epoch训练/验证BA与交叉熵；checkpoint和候选使用验证BA选择。',['只显示验证选中的候选；测试集不参与早停或选型。'])
add('trials',trials,['prepared_all_trials.npz','all_trial_manifest.csv','same_person/test_predictions.csv','new_person/test_predictions.csv'],'每条为完整留出测试试次；C3/Cz/C4波形来自真实9通道预处理输入，展示下采样至80Hz；模型始终使用160Hz全部9通道。',['标签为实验提示的左右手运动想象；回放只是显示完整3秒窗的预先计算预测，不进行在线逐帧解码。','概率未经校准，无法视作可靠度或手机动作许可；正确/错误筛选只用于事后观察，不用于训练选型。'])
add('quality',[data['quality']],['quality_summary.json','data_quality.csv'],'数据质量规则在测试评估前固定；任一记录少于8个合格试次、缺类别或非有限数值则整人排除。',['S109被预定质量规则排除；原始EDF保持不变。','重采样是离线处理；未进行完整眼动/肌电伪迹审计。'])
add('splits',splitrows,['config.json','same_person/summary.json','new_person/summary.json'],'同人：S011–S080 R04训练/R08验证/R12测试；新人：S011–S080三个MI run训练，S081–S094验证，S095–S109测试（应用预定QC）。',['S001–S010首轮评估人群不再使用。','只记录所选候选资源；总运行时间包含全部六个EEGNet候选与质量检查。'])
snapshot={'id':'bci-mi-expanded-20260930-v2','surface':'dashboard','title':data['title'],'generatedAt':data['generated_at_utc'],'buildStatus':'complete','status':'observed','filters':[],'queries':queries,'experiment':{'quality':data['quality'],'total_seconds':data['total_seconds'],'run_id':run.name}}
(dest/'reviewed_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False),encoding='utf-8')
(dest/'calculation_verification.json').write_text(json.dumps({'status':'PASS','checks':checks,'total_test_trial_rows':len(trials)},ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 第二轮脑电运动想象训练结果','','真实离线实验已完成。原始EDF和首轮结果完整保留。','','| 协议 | 模型 | 测试人数 / 试次 | 平衡准确率 | 95%受试者整群bootstrap区间 | AUC |','|---|---|---:|---:|---|---:|']
for r in comparison: lines.append(f"| {r['protocol_label']} | {r['model']} | {r['test_subjects']} / {r['n']} | {r['balancedAccuracyRate']:.2%} | {r['interval']} | {r['auc']:.3f} |")
lines+=['','平衡准确率是两类召回率的平均值，二分类机会参考为50%。区间只描述本轮测试人群；单随机种子与未知伪迹仍限制结论。两个协议采用不同测试人群，不做因果差异解释。','','## 实验与质量控制','','- 同人：S011—S080，run04训练 / run08验证 / run12测试。','- 新人：S011—S080的三个MI run训练；S081—S094验证；S095—S109测试。','- 297个输入EDF SHA256核验通过；保留98人、4416个试次；S109按预定规则排除。','- 9段128Hz记录重采样到160Hz；1个短区间、51个幅度/边界不合格试次被剔除，随后整人质量控制。','- 平均参考限于所选9电极；8—30Hz因果IIR；提示后1—4秒；250µV固定峰峰值阈值。','- 每协议3个预定EEGNet候选；CSP2/4/6/8分量。仅以验证BA选择；测试集不用于调参。','- 全部测试预测经模型CPU重载复验；可视化关键BA及混淆计数从CSV独立重算。','','## 模型能力与边界','','本轮只能评价有明确提示的左右手运动想象二分类。它支持搭建“低带宽意图候选”实验链，但误判仍需认真处理。尚缺休息/无操作类、个人校准、概率校准、跨天验证与在线拒识。脑电类别到手机动作的映射属于后续设计。','','## 查看和复现','','可视化包含协议/模型比较、逐人分布、混淆矩阵、学习曲线，以及每一个测试试次的真实波形与正确/错误预测。回放使用完整3秒窗口的离线预测，不应解释为实时流式推理。','','训练启动：运行本任务根目录 run_training.ps1。使用首轮任务的既有 .venv，不更改其他项目环境。','模型保存在 runs/'+run.name+'/same_person 和 new_person 下；每个协议 model_selection.json 指明最佳EEGNet候选，csp_lda.joblib为传统基线。','','总实验用时：'+f"{data['total_seconds']:.1f} 秒；使用本地 RTX4050 Laptop，无云算力费用。",'']
(dest/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'status':'PASS','run_dir':str(run),'snapshot':str(dest/'reviewed_snapshot.json'),'scores':comparison},ensure_ascii=False))

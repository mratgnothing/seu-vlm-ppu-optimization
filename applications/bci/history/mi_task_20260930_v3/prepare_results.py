import csv,json,sys
from pathlib import Path
import numpy as np

task=Path(sys.argv[1]);app=Path(sys.argv[2]);dest=Path(sys.argv[3]);dest.mkdir(parents=True,exist_ok=True)
status=json.loads((task/'latest_run.json').read_text(encoding='utf-8'))
if status['status']!='COMPLETE':raise RuntimeError('v3 not complete')
run=Path(status['run_dir']);data=json.loads((run/'visualization_data.json').read_text(encoding='utf-8'))
snapshot=json.loads((app/'src/data.json').read_text(encoding='utf-8'))
def readcsv(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def ba(cm):return sum(cm[i][i]/sum(cm[i]) for i in range(len(cm)))/len(cm)
checks=[];cal=[];cal_subjects=[];cal_delta=[];rest_scores=[];matrix=[];curves=[]
cal_predictions=readcsv(run/'calibration/predictions.csv')
for method,score in data['calibration']['metrics'].items():
 cm=[[0,0],[0,0]]
 for r in cal_predictions:
  if r['method']==method:cm[int(r['label'])][int(r['prediction'])]+=1
 assert cm==score['confusion_matrix'] and abs(ba(cm)-score['balanced_accuracy'])<1e-10
 lo,hi=score['ba_95ci'];model=method.split('-')[0];phase='个人校准' if method.endswith('personal') else '无校准'
 cal.append({'method':method,'model':model,'phase':phase,'balancedAccuracyRate':score['balanced_accuracy'],'n':score['n'],'interval':f'{lo:.1%} — {hi:.1%}'})
 checks.append({'method':method,'matrix_matches':True,'manual_ba':ba(cm)})
for r in data['calibration']['subjects']:cal_subjects.append({'subject':f"S{r['subject']:03d}",'method':r['method'],'model':r['method'].split('-')[0],'phase':'个人校准' if r['method'].endswith('personal') else '无校准','balancedAccuracyRate':r['balanced_accuracy'],'n':r['n']})
for model,r in data['calibration']['paired'].items():
 lo,hi=r['paired_subject_bootstrap_95ci'];cal_delta.append({'model':model,'delta_ba_pp':r['delta_ba']*100,'paired_interval_pp':f'{lo*100:+.1f} — {hi*100:+.1f}','improved_people':r['subjects_improved'],'people':14})
predictions=readcsv(run/'rest/test_predictions.csv')
label_names=['休息','左手','右手']
for model,result in data['rest']['models'].items():
 pp=[r for r in predictions if r['model']==model];raw_cm=[[0]*3 for _ in range(3)];gate_cm=[[0]*3 for _ in range(3)]
 for r in pp:
  label=int(r['label']);raw_cm[label][int(r['prediction'])]+=1;gate_cm[label][int(r['gate_prediction'])]+=1
 assert raw_cm==result['raw_three_class']['confusion_matrix'] and gate_cm==result['gate']['confusion_matrix']
 assert abs(ba(raw_cm)-result['raw_three_class']['balanced_accuracy'])<1e-10
 checks.append({'model':model,'raw_matrix_matches':True,'gate_matrix_matches':True,'manual_three_class_ba':ba(raw_cm)})
 for mode,cm in [('三分类硬判',raw_cm),('验证阈值门控',gate_cm)]:
  rest_n=sum(cm[0]);mi_n=sum(cm[1])+sum(cm[2]);rest_fp=cm[0][1]+cm[0][2];mi_accept=sum(cm[i][j] for i in (1,2) for j in (1,2));mi_correct=cm[1][1]+cm[2][2]
  rest_scores.append({'model':model,'mode':mode,'threshold':0 if mode=='三分类硬判' else result['threshold'],'restFalseRate':rest_fp/rest_n,'miCoverageRate':mi_accept/mi_n,'correctCommandRate':mi_correct/mi_n,'acceptedAccuracyRate':mi_correct/mi_accept if mi_accept else None,'rest_n':rest_n,'rest_false_commands':rest_fp,'mi_n':mi_n,'mi_accepted':mi_accept,'mi_correct':mi_correct,'threeClassBARate':result['raw_three_class']['balanced_accuracy'],'gate_exclusive':mode=='验证阈值门控'})
  for i in range(3):
   for j in range(3):matrix.append({'model':model,'mode':mode,'truth':label_names[i],'prediction':label_names[j],'count':cm[i][j]})
 for r in data['rest']['curve']:
  if r['model']==model:curves.append({'model':model,'threshold':r['threshold'],'restFalseRate':r['rest_false_trigger_rate'],'miCoverageRate':r['mi_coverage'],'correctCommandRate':r['mi_correct_command_rate'],'selected':r['threshold']==result['threshold']})
queries=snapshot['queries']
def add(qid,rows,files,definition,caveats):
 queries[qid]={'rows':rows,'source':{'label':'PhysioNet EEGMMIDB v1.0.0 · 第三轮真实离线探索','files':files,'links':[{'label':'官方标签定义','url':'https://physionet.org/content/eegmmidb/1.0.0/'}],'executedAt':data['generated_at'],'metricDefinitions':[{'label':qid,'definition':definition}],'caveats':[data['evaluation_status'],*caveats],'evidenceFlow':[{'title':'固定实验规则','detail':'training/mi_calibration_rest_20260930_v3/config.json；个人run04校准、run08选模型、run12配对比较；休息三分类按S011-S080训练/S081-S094验证/S095-S108测试。所有测试输出均在模型、阈值冻结后计算。'}]}}
add('v3_calibration',cal,['calibration/summary.json','calibration/predictions.csv'],'同14个人的同一批run12试次，比较global与personal模型；BA为两类召回平均；区间为按人bootstrap。',['此前已查看过这些测试人群，结果属于探索性后续验证。','EEGNet固定主干和BatchNorm，仅以run04训练分类头；CSP按run08选择2/4/6/8分量。'])
add('v3_cal_subjects',cal_subjects,['calibration/subject_metrics.csv'],'每人每方法的run12 BA；每人测试量约12–15，不能当作稳定个人能力估计。',[])
add('v3_cal_delta',cal_delta,['calibration/summary.json'],'个人校准BA减无校准BA，单位百分点；95%区间是同人配对整群bootstrap，不用两个独立区间推断提升。',[])
add('v3_rest_scores',rest_scores,['rest/test_predictions.csv','rest/frozen_gate_policies.json'],'休息误触发=休息窗口中输出左右候选的数量/全部休息窗口；MI覆盖=输出左右候选的MI窗口/全部MI窗口；正确指令率=正确左右候选/全部MI窗口；已输出MI准确率=正确左右候选/已输出MI窗口。',['只涉及已标注T0休息窗口，不是自然场景无意图，也不是每小时误触发率。','拒绝可能来自休息判别或低置信度，不能把被拒绝的运动想象算作休息识别正确。','左右输出只是离线候选，没有连接手机执行。'])
add('v3_rest_matrix',matrix,['rest/test_predictions.csv'],'提示标签与预测/门控后候选的3×3计数；门控图中的休息列也包含拒绝。',[])
add('v3_val_gate_curve',curves,['rest/validation_threshold_curve.csv'],'仅在验证人群计算各阈值误触发/覆盖/正确指令率；固定要求验证休息误触发≤5%，再最大化正确指令率。',['验证集的5%上限不保证测试或真实用户达到5%。','阈值1.0是完全关闭输出的备选，不代表有效识别。'])
add('v3_rest_trials',data['rest']['trials'],['three_class_trials.npz','three_class_manifest.csv','rest/test_predictions.csv'],'同一个3秒真实测试窗口的预处理波形、三类概率、已冻结门控决策。波形展示C3/Cz/C4，模型输入9通道160Hz。',['不做在线中途推理；概率未经校准。'])
add('v3_rest_history',[{**r,'trainBARate':float(r['train_ba']),'validationBARate':float(r['validation_ba'])} for r in readcsv(run/'rest/eegnet_history.csv')],['rest/eegnet_history.csv'],'三分类EEGNet逐epoch训练与验证宏平均召回率；三分类机会参考约33.3%。',[])
add('v3_quality',[{**data['quality'],'test_rest':data['rest']['splits']['test']['rest'],'test_mi':data['rest']['splits']['test']['left']+data['rest']['splits']['test']['right'],'calibration_n':data['calibration']['metrics']['EEGNet-global']['n'],'train_total':data['rest']['splits']['train']['total'],'validation_total':data['rest']['splits']['validation']['total'],'test_total':data['rest']['splits']['test']['total']}],['rest_quality_summary.json','rest/split_counts.json'],'新读取T0休息段并校验EDF，复用v2经过核验的MI缓存。先应用v2整人QC，再排除任一记录休息数小于8的受试者。',[])
snapshot.update(generatedAt=data['generated_at'],buildStatus='updating');snapshot['experiment_v3']={'run_id':run.name,'evaluation_status':data['evaluation_status'],'total_seconds':data['total_seconds']}
(app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(dest/'calculation_verification.json').write_text(json.dumps({'status':'PASS','checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 第三轮：个人校准与休息拒识','','所有指标均来自实际训练与测试预测。S095–S108测试人群已在v2查看，本轮属于探索性后续实验；没有新的独立外部验证。','','## 个人校准配对比较','','同14人、同一批run12，run04校准、run08选型。不得直接与v2全部三个run的60.7%比较。','','| 方法 | BA | 95%按人区间 | 测试试次 |','|---|---:|---|---:|']
for r in cal:lines.append(f"| {r['method']} | {r['balancedAccuracyRate']:.1%} | {r['interval']} | {r['n']} |")
lines+=['','| 模型 | 校准增量（百分点） | 配对95%区间 | 提升人数 / 14 |','|---|---:|---|---:|']
for r in cal_delta:lines.append(f"| {r['model']} | {r['delta_ba_pp']:+.1f} | {r['paired_interval_pp']} | {r['improved_people']} |")
lines+=['','## 休息 / 左 / 右三分类与拒识','','T0是官方休息标签；从同类运动想象run抽取标签后1–4秒，保持相同9电极、平均参考、8–30Hz因果IIR和250µV峰峰值阈值。其结果不能代替自然场景异步BCI评价。','','| 模型 / 决策 | 阈值 | 休息误触发 | MI输出覆盖 | 正确指令 / 全部MI | 已输出MI准确率 |','|---|---:|---:|---:|---:|---:|']
for r in rest_scores:lines.append(f"| {r['model']} / {r['mode']} | {r['threshold']} | {r['rest_false_commands']}/{r['rest_n']} ({r['restFalseRate']:.1%}) | {r['miCoverageRate']:.1%} | {r['correctCommandRate']:.1%} | {r['acceptedAccuracyRate']:.1%} |" if r['acceptedAccuracyRate'] is not None else f"| {r['model']} / {r['mode']} | {r['threshold']} | {r['rest_false_commands']}/{r['rest_n']} ({r['restFalseRate']:.1%}) | {r['miCoverageRate']:.1%} | {r['correctCommandRate']:.1%} | 无输出 |")
lines+=['','阈值只用验证集选择：要求验证休息误触发≤5%，在可行阈值中最大化正确指令率。必须同时看漏检、覆盖与错误方向；低误触发并不等于可用控制。','','## 复现与证据','','- 每人校准模型在calibration/Sxxx；休息模型在rest。配置与预处理缓存哈希均记录在本轮run。','- 模型重载核对全部测试类别与门控决策；图表指标由CSV计数独立复核。','- 在线无意图、跨天、目标用户、医学有效性、每小时误触发尚未验证。','- 窗口内的时间回放不计算实时中途预测，不接手机，不自动执行。','',f"本轮流程用时 {data['total_seconds']:.1f}秒；仅使用本地算力。",'']
(dest/'实验结果.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'status':'PASS','run_dir':str(run),'calibration':cal,'paired':cal_delta,'rest':rest_scores},ensure_ascii=False))

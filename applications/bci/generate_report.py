from __future__ import annotations
import csv,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
BASE=Path('E:/BCI/datasets/physionet-eegmmidb-1.0.0/training/mi_improvement_20261001_v4')
ROOT=Path(__file__).resolve().parent
APP=ROOT.parent/'mi_improvement_report_20261001_v4'
if not APP.exists():APP=ROOT/'report_app'
run=BASE/'runs/20261001T154634_540026Z'
status=json.loads((run/'status.json').read_text());assert status['status']=='COMPLETE'
s=json.loads((run/'summary.json').read_text());snapshot=json.loads((APP/'src/data.json').read_text(encoding='utf-8'))
out=ROOT/'results';out.mkdir(exist_ok=True)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def pct(v):return '无输出' if v is None else f'{v*100:.1f}%'
def interval(a):return ' — '.join(pct(v) for v in a)
names={'global':'此前未校准 EEGNet','v3_personal':'第三轮：仅分类头校准','linear':'第四轮：对齐协方差＋线性模型','neural':'第四轮：全网络微调（三种子平均）','head_3seed':'第四轮：分类头复跑（三种子平均）'}
compare=[{'method':names[k],'baRate':v['balanced_accuracy'],'interval':interval(v['ba_95ci']),'delta':f'{100*v["paired_delta_vs_v3"]:+.1f}','deltaInterval':' — '.join(f'{100*a:+.1f}' for a in v['paired_delta_95ci']),'n':v['n']} for k,v in s['mi'].items() if k in names]
stability=[{'policyLabel':'全网络微调' if r['policy']=='full_lr300' else '仅分类头','seed':r['seed'],'baRate':r['balanced_accuracy']} for r in s['seed_stability']]
gates=[]
for r in [s['v3_rest_same_r12'],*s['rest']]:
 all_outputs=r['mi_outputs']+r['false_triggers']
 gates.append({'policyLabel':'第三轮原门控（同 run12）' if 'validation_cap' not in r else f'第四轮：验证误触发≤{r["validation_cap"]:.0%}', 'restFprRate':r['rest_fpr'],'coverageRate':r['mi_coverage'],'correctRate':r['correct_mi_rate'],'acceptedRate':r['accepted_mi_accuracy'],'allCandidateRate':r['correct_outputs']/all_outputs if all_outputs else None,'outputs':f'{r["mi_outputs"]} / {r["mi_n"]}','falseTriggers':f'{r["false_triggers"]} / {r["rest_n"]}','correctOutputs':r['correct_outputs'],'allOutputs':all_outputs})
controls=[{'control':'等概率随机猜测（理论）','baRate':.5,'note':'期望基线，不是一次实测结果'}]+[{'control':f'线性模型训练标签打乱 · {r["seed"]}','baRate':r['balanced_accuracy'],'note':'源训练与个人校准标签按人/记录打乱，测试标签不变'} for r in s['shuffle_control']]+[{'control':'冻结预测的标签置乱分布','baRate':s['permutation']['null_mean'],'note':f'1000 次，人内 run12 置乱；95%范围 {interval(s["permutation"]["null_95range"])}'}]
raw_subject=json.loads((run/'subject_metrics.json').read_text());subjects=[]
for person in range(95,109):
 a=next(r for r in raw_subject if r['subject']==person and r['method']=='v3_personal');b=next(r for r in raw_subject if r['subject']==person and r['method']=='recommended')
 subjects.append({'person':f'S{person:03d}','baselineRate':a['balanced_accuracy'],'improvedRate':b['balanced_accuracy'],'delta':f'{100*(b["balanced_accuracy"]-a["balanced_accuracy"]):+.1f}','n':a['n']})
# Independent count arithmetic, without sklearn's scoring functions.
checks=[];predictions=rows(run/'mi_predictions.csv')
for method,declared in s['mi'].items():
 rr=[r for r in predictions if r['method']==method];cm=[[sum(int(r['label'])==a and int(r['prediction'])==b for r in rr) for b in (0,1)] for a in (0,1)];ba=sum(cm[a][a]/sum(cm[a]) for a in (0,1))/2
 assert cm==declared['confusion_matrix'] and abs(ba-declared['balanced_accuracy'])<1e-12;checks.append({'method':method,'n':len(rr),'manual_ba':ba,'confusion_matches':True})
rest_rows=rows(run/'rest_predictions.csv')
for declared in s['rest']:
 rr=[r for r in rest_rows if float(r['validation_cap'])==declared['validation_cap']];rest=[r for r in rr if int(r['label'])==0];mi=[r for r in rr if int(r['label'])!=0];fp=sum(int(r['gate_prediction'])!=0 for r in rest);commands=sum(int(r['gate_prediction'])!=0 for r in mi);correct=sum(int(r['gate_prediction'])==int(r['label']) for r in mi)
 assert (len(rest),len(mi),fp,commands,correct)==tuple(declared[k] for k in ('rest_n','mi_n','false_triggers','mi_outputs','correct_outputs'));checks.append({'validation_cap':declared['validation_cap'],'rest_n':len(rest),'mi_n':len(mi),'false_triggers':fp,'outputs':commands,'correct':correct,'manual_all_candidate_accuracy':correct/(commands+fp)})
save(out/'calculation_verification.json',{'status':'PASS','checks':checks})
now=datetime.now(timezone.utc).isoformat();rootlabel=str(run)
def query(data,files,components,definition):
 return {'rows':data,'source':{'title':'第四轮真实训练与固定测试记录','type':'local-files','files':[str(run/f) for f in files],'executedAt':now,'evidenceFlow':[{'title':'固定数据与模型','detail':'配置在训练前写入。24 个线性策略与 3 个神经微调策略只在 S081-S094 的 run12 上选择；神经策略用3个种子的平均概率。个人 run04 校准、run08 早停；所有选择在测试人群 run12 推理前冻结。'},{'title':'独立计数复核','detail':'从 mi_predictions.csv 手算二分类混淆矩阵与平衡准确率；从 rest_predictions.csv 手算休息误触发、动作输出与正确输出。全部与 summary.json 一致。'}],'metricDefinitions':[{'label':'评价口径','definition':definition,'componentIds':components}],'caveats':[s['evaluation_status'],'同一次采集的不同记录，不是跨天数据；输出置信度未校准，未连接手机。']},'methods':[{'language':'python','code':'复现：运行任务目录 train_v4.py；复核与报告：运行 generate_report.py。保存配置、输入 SHA256、每人模型、逐试次预测与失败日志。'}]}
queries={
 'mi_comparison':query(compare,['summary.json','mi_predictions.csv','frozen_mi_selection.json'],['executive-summary','mi-explanation','mi-comparison','evidence-boundaries'],'BA = (左召回率＋右召回率)/2；随机无信息基线50%。同14人同run12共206试次。95%区间为1000次受试者簇bootstrap。变化与第三轮个人校准配对比较。'),
 'stability':query(stability,['seed_stability.json'],['seed-stability'],'相同预训练模型和个人数据，改变微调随机种子。全网络微调包含dropout；分类头模式关闭主干dropout、BatchNorm固定，当前样本小于batch size，重复结果相同不构成广泛稳健性证据。'),
 'gate':query(gates,['rest_summary.json','rest_predictions.csv','frozen_rest_selection.json'],['gate-explanation','rest-tradeoff'],'同run12：204休息、206动作。休息误触发=休息中输出/204；覆盖=动作中输出/206；条件准确率=正确动作/动作中输出；全部候选准确率=正确动作/(动作输出＋休息误触发)。新方案额外使用run04休息标注校准，旧方案没有。20%为探索运行点不是安全标准。'),
 'controls':query(controls,['shuffled_training_controls.json','conditional_permutation.json'],['control-explanation','random-controls'],'训练标签打乱对照仅线性家族；冻结神经预测的1000次置乱保留每人左右数量。p=(1+不低于观测值次数)/1001；条件检验，不重训整个选择流程，不能证明目标神经机制或新外部泛化。'),
 'subjects':query(subjects,['subject_metrics.json'],['person-detail'],'每人run12约12–15试次；分数粒度较粗。所有14人均展示，保留下降/持平。'),
 'methods':query([{'mi_source':3147,'mi_validation':218,'mi_test':206,'rest_source':6285,'rest_validation':360,'rest_test':410,'successful_run_seconds':s['resources']['total_seconds']}],['config.json','provenance.json','reload_verification.json'],['methods-and-prototype'],'复用已核验9电极、160Hz、1–4秒、8–30Hz缓存；新增8–13/13–30Hz窗口内因果滤波。每人EA只使用run04动作协方差，未使用run08/run12估计对齐参考。成功流程耗时不含失败实现修复和报告制作。')}
neural=s['mi']['neural'];delta=neural['paired_delta_vs_v3']*100;dci=' — '.join(f'{v*100:+.1f}' for v in neural['paired_delta_95ci']);strict=s['rest'][0];prototype=s['rest'][-1]
snapshot.update(title='左右识别未进一步提升，休息门控开始产生更多候选',generatedAt=now,buildStatus='updating',report={'asOf':'2026-10-01'},queries=queries,status='observed')
snapshot['narrative']={
 'summary':f'## 结论摘要\n\n- **左右识别没有改进成功。** 对齐协方差模型为58.0%，全网络微调为61.0%，此前仅分类头校准为63.4%；分类头复跑仍为63.4%。保留原方向识别基线，暂不采用新微调替换它。\n- **休息检测有了有限的原型进展。** 保守运行点在同一批片段上输出17个动作候选，其中14个正确；休息误触发6/204。旧门控只输出1个动作候选，误触发5/204。新方案使用额外休息校准，不能把差异归因于算法单独改善。\n- 可以推进“用户主动开始＋左右候选＋人工确认”的离线原型。自动无意图识别仍只放在实验分支，当前不连接手机执行。',
 'mi':f'## 更多层微调没有在固定测试中带来提升\n\n验证人群上，全网络微调的平衡准确率为69.7%，对齐协方差模型为67.9%。按验证结果冻结后，测试分别下降到61.0%和58.0%，均低于此前63.4%。这说明本轮验证集上的优势未迁移到这批测试人群；原因可能涉及人群差异、选型波动或适配过拟合，本实验不能确定是哪一种。\n\n全网络微调相对第三轮的变化为 **{delta:+.1f} 个百分点**，配对95%区间为 **{dci} 个百分点**，包含0，没有改善证据。每项都使用同14人同206试次，未将第二轮全部记录的60.7%混入比较。',
 'gate':f'## 提高输出覆盖，仍需要人工确认\n\n将“有没有动作意图”和“左还是右”拆成两个模块。休息检测器使用分频功率特征和带源模型先验的个人逻辑回归适配；方向模块使用本轮按验证选定的全网络微调。为避免全部拒绝，预先比较验证休息误触发上限5%、10%、20%三个运行点；20%仅用于探索原型的输出代价，不是安全标准。\n\n**保守运行点：**休息误触发{strict["false_triggers"]}/{strict["rest_n"]}（{pct(strict["rest_fpr"])}）；动作输出{strict["mi_outputs"]}/{strict["mi_n"]}（{pct(strict["mi_coverage"])}），其中{strict["correct_outputs"]}个方向正确。82.4%的“已输出动作准确率”只统计动作片段；把6个休息误触发也算入，全部23个候选的正确比例为60.9%。依然漏掉了约92%的动作。\n\n**较宽松运行点：**输出66/206个动作候选，43个正确，但休息误触发36/204。不能仅凭覆盖增加宣称可自动控制。原型更适合由用户主动开始一个提示式识别窗口，然后确认候选。',
 'controls':f'## 有高于随机的关联，尚未证明独立泛化\n\n线性模型训练标签打乱后的结果为50.4%–52.1%，与50%机会水平接近。对冻结的全网络预测做1000次人内标签置乱，零假设均值为49.3%、95%范围43.5%–55.2%；观测61.0%对应条件 p≈0.001。\n\n这个检验支持当前预测与标签有关联，**不代表新方案优于旧方案**，也不排除提示时序、伪迹或其他混杂。未对整个神经训练/选型流程进行置乱重训。',
 'methods':f'## 实验方法与可交付的原型范围\n\n- 源训练：S011–S080，左右识别3147试次、休息检测6285窗口；开发人群S081–S094，固定测试S095–S108。休息实验沿用前三轮质控排除，不按模型分数删人。\n- 每人run04校准（12–15个动作片段；休息检测额外使用run04休息标签，总计24–30个窗口），run08选择神经早停；run12用于开发策略选择或固定测试。片段3秒不等于完整采集仅几十秒，还需要提示、休息和准备时间。\n- 24个线性策略、3个神经微调策略；神经模型固定主干BatchNorm，只在全网络分支启用dropout。3种子平均概率是预先固定策略，不能在测试后挑最好种子。\n- 所有方向与门控选择在测试推理前冻结；模型CPU重载类别、休息门控决策一致；CSV独立计数复核通过。成功运行约{s["resources"]["total_seconds"]:.1f}秒，RTX4050 Laptop本地运行，峰值PyTorch显存约{s["resources"]["peak_cuda_allocated_mib"]:.1f}MiB；不含失败修复及报告时间。\n- 实现期间修复矩阵广播、缓存行排序和优化器数值停止；失败运行与原版脚本均保留，未依据测试分数更改实验候选。',
 'limits':'## 原型可以继续，性能提升仍需独立验证\n\n这批测试人群此前已查看，所有结果属于探索性后续实验。run04/08/12是同次采集中的不同记录，不能称为跨天测试；概率未校准，窗口内滤波也尚未检验流式边界。\n\n**建议当前方向模块保留63.4%的分类头基线，做有提示、需确认的离线演示。** 新休息检测保留为实验分支；接入原基线后的组合效果未验证。手机操作、实时采集、目标用户与医学用途都不在本轮完成范围内。下一项真正有价值的验证，是在新的采集日期或受试者上检验个人校准，并记录完整候选数量与误触发。'
}
save(APP/'src/data.json',snapshot)
# Portable text companion contains the same conclusions and exact tabular scope.
md=['# '+snapshot['title'],'','日期：2026-10-01。成功运行：'+run.name,'',snapshot['narrative']['summary'],'',snapshot['narrative']['mi'],'','| 方法 | BA | 95%区间 | 较第三轮 pp | n |','|---|---:|---|---:|---:|']
md.extend(f'| {r["method"]} | {pct(r["baRate"])} | {r["interval"]} | {r["delta"]} | {r["n"]} |' for r in compare)
md+=['',snapshot['narrative']['gate'],'','| 运行点 | 休息误触发 | 动作覆盖 | 正确/全部动作 | 输出动作准确率 | 全部候选准确率 |','|---|---:|---:|---:|---:|---:|']
md.extend(f'| {r["policyLabel"]} | {pct(r["restFprRate"])} | {pct(r["coverageRate"])} | {pct(r["correctRate"])} | {pct(r["acceptedRate"])} | {pct(r["allCandidateRate"])} |' for r in gates)
md+=['',snapshot['narrative']['controls'],'',snapshot['narrative']['methods'],'',snapshot['narrative']['limits'],'','## 逐人比较','','| 人 | 第三轮 | 第四轮全网络微调 | 变化 pp | n |','|---|---:|---:|---:|---:|']
md.extend(f'| {r["person"]} | {pct(r["baselineRate"])} | {pct(r["improvedRate"])} | {r["delta"]} | {r["n"]} |' for r in subjects)
md+=['','## 原始证据','',f'运行目录：`{run}`。`mi_predictions.csv`、`rest_predictions.csv`、`summary.json`、`frozen_*_selection.json`、`provenance.json`、`reload_verification.json`。','复核：`results/calculation_verification.json`。方法代码：`train_v4.py`。','来源说明：PhysioNet EEGMMIDB；MNE CSP与pyRiemann协方差方法官方文档用于方法背景，本实验的log-Euclidean特征自行实现，未声称等同pyRiemann标准Riemannian TangentSpace。']
(out/'改进实验报告.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps({'status':'PASS','run':str(run),'report':str(out/'改进实验报告.md'),'query_counts':{k:len(v['rows']) for k,v in queries.items()},'independent_checks':len(checks)},ensure_ascii=False))

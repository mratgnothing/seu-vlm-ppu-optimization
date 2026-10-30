import React, {useEffect, useMemo, useState} from 'react';
import {Button, DataComponent, DataTable, Dropdown, EvidenceChart, Section, Slider, useDataApp, useDashboardTabs} from '../../data-app-public.jsx';
import {CalibrationView, RestView} from './AdvancedExperiments.jsx';
import './eeg-dashboard.css';
const names={same_person:'同人跨记录',new_person:'新用户留出'};
const colors={'CSP-LDA':'var(--chart-1)',EEGNet:'var(--chart-2)'};
const percent=n=>(100*n).toFixed(1)+'%';
const chance={id:'chance-reference',kind:'benchmark',measure:'balancedAccuracyRate',field:'chanceRate',label:'机会参考 50%'};
const comparisonSpec={type:'bar',x:'protocol_label',y:'balancedAccuracyRate',series:'model',colors,stackable:false,valueDecimals:1,annotations:[chance]};
const subjectSpec={type:'horizontalBar',x:'subject_label',y:'balancedAccuracyRate',series:'model',colors,stackable:false,valueDecimals:1,annotations:[chance]};
const matrixSpec={type:'heatmap',x:'prediction',y:'count',series:'truth',showValues:true,reverseRows:true,xLabel:'预测类别',yLabel:'提示标签',showXAxisLabel:true,showYAxisLabel:true};
const learningSpec={type:'line',x:'epoch',y:'trainAccuracyRate',fields:['trainAccuracyRate','validationAccuracyRate'],stackable:false,colors:{trainAccuracyRate:'var(--chart-1)',validationAccuracyRate:'var(--chart-2)'},legend:{labels:{trainAccuracyRate:'训练 BA',validationAccuracyRate:'验证 BA'}},xLabel:'Epoch',valueDecimals:1};
const lossSpec={type:'line',x:'epoch',y:'train_loss',fields:['train_loss','validation_loss'],colors:{train_loss:'var(--chart-1)',validation_loss:'var(--chart-2)'},legend:{labels:{train_loss:'训练',validation_loss:'验证'}},xLabel:'Epoch',yLabel:'交叉熵',valueDecimals:3,stackable:false};
const waveSpec={type:'line',x:'time',y:'C3',fields:['C3','Cz','C4'],colors:{C3:'var(--chart-1)',Cz:'var(--chart-2)',C4:'var(--chart-3)'},xLabel:'提示后时间（秒）',yLabel:'脑电幅度（µV）',valueDecimals:2,stackable:false};
const columns=[{field:'protocol_label',label:'评估协议'},{field:'model',label:'模型'},{field:'balancedAccuracyRate',label:'平衡准确率',renderCell:percent},{field:'interval',label:'95% 人群整群 bootstrap 区间'},{field:'n',label:'测试试次'},{field:'test_subjects',label:'测试人数'},{field:'auc',label:'AUC',renderCell:n=>n.toFixed(3)}];

function TrialReplay({protocol,trialRows}) {
  const [subject,setSubject]=useState('all'),[model,setModel]=useState('EEGNet'),[outcome,setOutcome]=useState('all');
  const [index,setIndex]=useState(0),[cursor,setCursor]=useState(240),[playing,setPlaying]=useState(false);
  useEffect(()=>{setSubject('all');setOutcome('all');setIndex(0);setCursor(240);setPlaying(false);},[protocol]);
  const subjects=useMemo(()=>['all',...new Set(trialRows.map(r=>r.subject_label))],[trialRows]);
  const available=useMemo(()=>trialRows.filter(r=>{
    const p=r.probabilities[model],correct=(p[1]>p[0]?1:0)===r.label;
    return (subject==='all'||r.subject_label===subject)&&(outcome==='all'||(outcome==='correct'&&correct)||(outcome==='wrong'&&!correct));
  }),[trialRows,subject,model,outcome]);
  useEffect(()=>{setIndex(0);setCursor(240);setPlaying(false);},[subject,model,outcome]);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setCursor(v=>Math.min(240,v+4)),50);return()=>clearInterval(timer);},[playing]);
  useEffect(()=>{if(cursor>=240)setPlaying(false);},[cursor]);
  const trial=available[Math.min(index,Math.max(0,available.length-1))];
  const wave=trial?trial.wave_uv[0].map((_,i)=>({time:1+i/80,C3:trial.wave_uv[0][i],Cz:trial.wave_uv[1][i],C4:trial.wave_uv[2][i]})):[];
  const shown=wave.slice(0,cursor),p=trial?.probabilities[model],prediction=p?(p[1]>p[0]?1:0):null,ready=cursor>=240;
  const change=i=>{setIndex(i);setCursor(240);setPlaying(false);};
  const controls=<><Dropdown label="受试者" showLabel value={subject} choices={subjects} choiceLabels={{all:'全部'}} onChange={setSubject}/><Dropdown label="预测模型" showLabel value={model} choices={['CSP-LDA','EEGNet']} onChange={setModel}/><Dropdown label="片段筛选" showLabel value={outcome} choices={['all','correct','wrong']} choiceLabels={{all:'全部（含错误）',correct:'预测正确',wrong:'预测错误'}} onChange={setOutcome}/></>;
  return <Section id="replay-heading" title="真实测试片段回放" filters={controls}>
    <DataComponent id="trial-result" queryId="trials" title="完整窗口预测" variant="card" kind="custom" sourceRows={trial?[trial]:[]} displayRows={trial?[trial]:[]}>
      {!trial?<p>当前筛选下没有测试片段。可以切回“全部”。</p>:<div data-reviewed-rows className="eeg-result">
        <div><span className="eeg-small">{trial.subject_label} · Run {trial.run} · 原记录样本 {trial.onset_sample}</span><h3>提示标签：{trial.label===0?'左手':'右手'}运动想象</h3><span>保留测试试次 {index+1} / {available.length}</span></div>
        <div><span className="eeg-small">{model} · {ready?'完整 3 秒窗口':'正在回放记录'}</span><h3>{ready?`模型预测：${prediction===0?'左手':'右手'} · ${prediction===trial.label?'正确':'错误'}`:'等待完整窗口'}</h3><span>{ready?`P(左) ${percent(p[0])}  /  P(右) ${percent(p[1])}`:'本页不计算中途在线预测'}</span></div>
      </div>}
      <p className="eeg-note">这是离线测试回放：显示真实记录和已计算的完整窗口预测。概率未经校准；无手机执行权限。</p>
      <div className="eeg-replay-controls"><Button disabled={!trial||index===0} onClick={()=>change(index-1)}>上一试次</Button><Button disabled={!trial||index>=available.length-1} onClick={()=>change(index+1)}>下一试次</Button><Button disabled={!trial} onClick={()=>{if(playing){setPlaying(false);}else{setCursor(0);setPlaying(true);}}}>{playing?'暂停':'播放 3 秒记录'}</Button><Button disabled={!trial} onClick={()=>{setCursor(240);setPlaying(false);}}>显示完整片段</Button></div>
      {available.length>1&&<Slider label="测试试次" min={0} max={available.length-1} step={1} value={index} onChange={v=>change(Number(v))} formatValue={v=>`${Number(v)+1} / ${available.length}`}/>}
    </DataComponent>
    <EvidenceChart id="eeg-wave" queryId="trials" title="C3 / Cz / C4 预处理脑电" variant="card" spec={waveSpec} rows={shown} sourceRows={trial?[trial]:[]} height={290} description="µV；显示采样率80Hz。模型使用全部9电极、160Hz、提示后1–4秒。三条曲线共享幅度坐标。"/>
  </Section>;
}

function BaselineContent(){
  const {queries}=useDataApp(),[protocol,setProtocol]=useState('new_person');
  const rows=id=>queries[id]?.rows??[],comparison=rows('comparison');
  const subjects=rows('subjects').filter(r=>r.protocol===protocol),matrices=rows('matrices').filter(r=>r.protocol===protocol);
  const scopedMatrixSpec={...matrixSpec,colorDomain:[0,Math.max(1,...matrices.map(r=>r.count))]};
  const history=rows('history').filter(r=>r.protocol===protocol&&r.selected);
  const trials=useMemo(()=>rows('trials').filter(r=>r.protocol===protocol),[queries,protocol]);
  const quality=rows('quality')[0],split=rows('splits').find(r=>r.protocol===protocol);
  return <article className="eeg-dashboard">
    <Section id="comparison-heading" title="留出测试 · 两种评估协议" spacing="none">
      <EvidenceChart id="protocol-comparison" queryId="comparison" title="模型平衡准确率" variant="card" spec={comparisonSpec} rows={comparison} sourceRows={comparison} height={250}/>
      <p className="eeg-note">平衡准确率（BA）＝左、右两类召回率的平均值；50% 是机会水平参考。区间按受试者整群重采样 1,000 次。两种协议采用不同测试人群。</p>
      <DataComponent id="metric-intervals" queryId="comparison" title="测试规模与不确定性" variant="card" kind="table" displayRows={comparison} sourceRows={comparison}><DataTable rows={comparison} columns={columns} searchable={false} caption="测试集指标与95%区间"/></DataComponent>
    </Section>
    <Section id="protocol-heading" title="协议内诊断" filters={<Dropdown label="评估协议" showLabel value={protocol} choices={['same_person','new_person']} choiceLabels={names} onChange={setProtocol}/>}>
      <DataComponent id="split-facts" queryId="splits" title={names[protocol]+' · 数据划分'} variant="card" displayRows={split?[split]:[]} sourceRows={split?[split]:[]}>
        <div data-reviewed-rows className="eeg-facts"><div><span>训练</span><strong>{split?.train.toLocaleString()} 试次</strong><small>{split?.train_people} 人</small></div><div><span>验证</span><strong>{split?.validation.toLocaleString()} 试次</strong><small>{split?.validation_people} 人 · 仅此集合选模型</small></div><div><span>测试</span><strong>{split?.test.toLocaleString()} 试次</strong><small>{split?.test_people} 人 · 不参与调参</small></div></div>
        <p className="eeg-note">{protocol==='same_person'?'S011–S080：run04训练、run08验证、run12测试。':'S011–S080训练，S081–S094验证，S095–S108测试；三个运动想象run均参与。S109按预定QC排除。'}</p>
      </DataComponent>
      <div className="eeg-two">{['CSP-LDA','EEGNet'].map(model=><EvidenceChart key={model} id={model==='EEGNet'?'eegnet-matrix':'csp-matrix'} queryId="matrices" title={model+' · 混淆计数'} variant="card" spec={scopedMatrixSpec} rows={matrices.filter(r=>r.model===model)} sourceRows={matrices.filter(r=>r.model===model)} height={230} description="行＝实验提示标签；列＝预测类别。对角线为正确预测；两模型使用同一色标。"/>)}</div>
      <EvidenceChart id="subject-comparison" queryId="subjects" title="各受试者平衡准确率" variant="card" spec={subjectSpec} rows={subjects} sourceRows={subjects} height={Math.max(380,subjects.length/2*27)} description="每人两个模型并排比较；同人协议每人通常15个测试试次，新用户协议每人通常45个。逐人分数不确定性较大。"/>
      <div className="eeg-two"><EvidenceChart id="learning-accuracy" queryId="history" title="EEGNet · 训练与验证 BA" variant="card" spec={learningSpec} rows={history} sourceRows={history} height={260}/><EvidenceChart id="learning-loss" queryId="history" title="EEGNet · 训练与验证损失" variant="card" spec={lossSpec} rows={history} sourceRows={history} height={260}/></div>
      <p className="eeg-note">所选候选 {split?.selected}；最佳 epoch {split?.best_epoch} / 实际训练 {split?.epochs_ran}。上图仅展示验证集选中的候选，测试集未参与选型。</p>
    </Section>
    <TrialReplay protocol={protocol} trialRows={trials}/>
    <Section id="quality-heading" title="数据与能力边界">
      <DataComponent id="quality-facts" queryId="quality" title="输入核验与质量控制" variant="card" displayRows={quality?[quality]:[]} sourceRows={quality?[quality]:[]}>
        <div data-reviewed-rows className="eeg-facts"><div><span>官方 SHA-256 核验</span><strong>{quality?.verified_sha256} / {quality?.input_records}</strong></div><div><span>保留人群与试次</span><strong>{quality?.retained_subjects} 人 · {quality?.retained_trials.toLocaleString()} 试次</strong></div><div><span>128 → 160 Hz 重采样</span><strong>{quality?.resampled_records} 段</strong></div></div>
        <p className="eeg-note">9电极 · 平均参考 · 8–30Hz · 提示后1–4秒 · 固定250µV峰峰值剔除。1个短区间与51个幅度/边界不合格试次剔除，S109随后按整人QC排除。原始记录完整保留。</p>
      </DataComponent>
      <p className="eeg-note">本页展示有提示的左右手运动想象二分类。尚未训练休息/无操作检测、开展跨天或在线测试，也未证明医疗有效性。适合用于低带宽意图接口的离线探索。</p>
    </Section>
  </article>;
}

const experimentTabs=[{id:'dashboard',label:'第二轮基线'},{id:'calibration',label:'个人校准'},{id:'rest',label:'休息与拒识'}];
export function DashboardContent(){
 const {activeTabId}=useDashboardTabs(experimentTabs);
 if(activeTabId==='calibration')return <CalibrationView/>;
 if(activeTabId==='rest')return <RestView/>;
 return <BaselineContent/>;
}

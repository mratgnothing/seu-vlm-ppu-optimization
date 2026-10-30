import React,{useEffect,useMemo,useState} from 'react';
import {Button,DataComponent,DataTable,Dropdown,EvidenceChart,Section,Slider,useDataApp} from '../../data-app-public.jsx';
const pct=n=>n==null?'无输出':(100*n).toFixed(1)+'%';
const phaseColors={'无校准':'var(--chart-1)','个人校准':'var(--chart-2)'};
const modelColors={'Bandpower-LDA':'var(--chart-1)','EEGNet-3class':'var(--chart-2)'};
const labels=['休息','左手','右手'];
function Caveat(){return <p className="eeg-note">第三轮探索性后续实验：这些测试人群已在第二轮查看。所有规则与阈值在本轮测试前固定；本页不能当作新的独立外部验证。</p>;}
const calColumns=[{field:'model',label:'模型'},{field:'phase',label:'校准方式'},{field:'balancedAccuracyRate',label:'平衡准确率',renderCell:pct},{field:'interval',label:'95% 按人区间'},{field:'n',label:'同一批测试试次'}];
export function CalibrationView(){
 const {queries}=useDataApp(),[model,setModel]=useState('EEGNet');
 const rows=id=>queries[id]?.rows??[],scores=rows('v3_calibration'),delta=rows('v3_cal_delta'),subjects=rows('v3_cal_subjects').filter(r=>r.model===model);
 return <article className="eeg-dashboard"><Caveat/>
  <Section id="v3-cal-title" title="个人校准 · 同人同试次配对比较" spacing="content">
   <EvidenceChart id="v3-cal-chart" queryId="v3_calibration" title="校准前后平衡准确率" variant="card" rows={scores} sourceRows={scores} height={280} spec={{type:'bar',x:'model',y:'balancedAccuracyRate',series:'phase',colors:phaseColors,stackable:false,valueDecimals:1}}/>
   <p className="eeg-note">14人各用run04约15个试次做校准、run08选型，均在同一批run12上比较。EEGNet冻结主干与BatchNorm，仅训练分类头；CSP重新学习个人空间滤波。</p>
   <DataComponent id="v3-cal-table" queryId="v3_calibration" title="配对比较规模与区间" variant="card" kind="table" displayRows={scores} sourceRows={scores}><DataTable rows={scores} columns={calColumns} searchable={false} caption="校准前后同一测试集指标"/></DataComponent>
   <DataComponent id="v3-cal-delta" queryId="v3_cal_delta" title="校准增量与配对不确定性" variant="card" kind="table" displayRows={delta} sourceRows={delta}><DataTable rows={delta} columns={[{field:'model',label:'模型'},{field:'delta_ba_pp',label:'增量（百分点）',renderCell:v=>(v>=0?'+':'')+v.toFixed(1)},{field:'paired_interval_pp',label:'配对95%区间（百分点）'},{field:'improved_people',label:'提升人数 / 14'}]} searchable={false} caption="配对按人bootstrap增量"/></DataComponent>
  </Section>
  <Section id="v3-cal-subject-title" title="个人差异" filters={<Dropdown label="校准模型" showLabel value={model} choices={['EEGNet','CSP']} onChange={setModel}/>}>
   <EvidenceChart id="v3-cal-subjects" queryId="v3_cal_subjects" title="各用户校准前后 BA" variant="card" rows={subjects} sourceRows={subjects} height={590} spec={{type:'horizontalBar',x:'subject',y:'balancedAccuracyRate',series:'phase',colors:phaseColors,stackable:false,valueDecimals:1}}/>
   <p className="eeg-note">每人仅约12–15个测试试次，小样本会放大个体分数变化；保留提升、持平和下降的全部用户。</p>
  </Section>
 </article>;
}

const restColumns=[{field:'model',label:'模型'},{field:'mode',label:'决策方式'},{field:'threshold',label:'验证阈值'},{field:'restFalseRate',label:'休息误触发',renderCell:pct},{field:'miCoverageRate',label:'MI输出覆盖',renderCell:pct},{field:'correctCommandRate',label:'正确候选 / 全部MI',renderCell:pct},{field:'acceptedAccuracyRate',label:'已输出MI准确率',renderCell:pct}];
const waveSpec={type:'line',x:'time',y:'C3',fields:['C3','Cz','C4'],xLabel:'标签后时间（秒）',yLabel:'脑电幅度（µV）',colors:{C3:'var(--chart-1)',Cz:'var(--chart-2)',C4:'var(--chart-3)'},valueDecimals:2,stackable:false};
function RestReplay({trials}){
 const [model,setModel]=useState('EEGNet-3class'),[type,setType]=useState('rest'),[index,setIndex]=useState(0),[cursor,setCursor]=useState(240),[playing,setPlaying]=useState(false);
 const chosen=useMemo(()=>trials.filter(t=>type==='all'||(type==='rest'&&t.label===0)||(type==='active'&&t.label!==0)||(type==='false'&&t.label===0&&t.decisions[model]!==0)),[trials,type,model]);
 useEffect(()=>{setIndex(0);setCursor(240);setPlaying(false);},[type,model]);
 useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setCursor(c=>Math.min(240,c+4)),50);return()=>clearInterval(timer);},[playing]);
 useEffect(()=>{if(cursor>=240)setPlaying(false);},[cursor]);
 const t=chosen[Math.min(index,Math.max(chosen.length-1,0))],p=t?.probabilities[model],decision=t?.decisions[model],ready=cursor===240;
 const wave=t?t.wave_uv[0].map((_,i)=>({time:1+i/80,C3:t.wave_uv[0][i],Cz:t.wave_uv[1][i],C4:t.wave_uv[2][i]})).slice(0,cursor):[];
 const change=i=>{setIndex(i);setCursor(240);setPlaying(false);};
 return <Section id="v3-rest-replay-heading" title="休息与拒识 · 真实片段" filters={<><Dropdown label="拒识模型" showLabel value={model} choices={['EEGNet-3class','Bandpower-LDA']} onChange={setModel}/><Dropdown label="片段类别" showLabel value={type} choices={['rest','active','false','all']} choiceLabels={{rest:'休息片段',active:'运动想象',false:'休息误触发',all:'全部'}} onChange={setType}/></>}>
  <DataComponent id="v3-rest-decision" queryId="v3_rest_trials" title="冻结门控决策" variant="card" sourceRows={t?[t]:[]} displayRows={t?[t]:[]}>
   {t?<div data-reviewed-rows className="eeg-result"><div><span className="eeg-small">S{String(t.subject).padStart(3,'0')} · Run {t.run} · 样本 {t.onset_sample}</span><h3>提示标签：{labels[t.label]}</h3><span>筛选试次 {index+1} / {chosen.length}</span></div><div><span className="eeg-small">{model} · 完整窗口离线预测</span><h3>{!ready?'等待完整窗口':decision===0?'门控：休息／拒绝输出':`门控：${labels[decision]}候选${t.label===0?' · 休息误触发':''}`}</h3><span>{ready?`P(休息) ${pct(p[0])} · P(左) ${pct(p[1])} · P(右) ${pct(p[2])}`:'此回放不计算实时中途概率'}</span></div></div>:<p>当前筛选下没有试次。可以切换“休息片段”或“全部”。</p>}
   <p className="eeg-note">门控沿用验证集冻结的阈值。拒绝运动想象会造成漏检；输出左右候选也不授权手机执行。</p>
   <div className="eeg-replay-controls"><Button disabled={!t||index===0} onClick={()=>change(index-1)}>上一休息试次</Button><Button disabled={!t||index>=chosen.length-1} onClick={()=>change(index+1)}>下一休息试次</Button><Button disabled={!t} onClick={()=>{setCursor(0);setPlaying(true);}}>回放休息／动作记录</Button><Button disabled={!t} onClick={()=>{setCursor(240);setPlaying(false);}}>显示完整休息片段</Button></div>
   {chosen.length>1&&<Slider label="休息实验试次" min={0} max={chosen.length-1} value={index} step={1} onChange={v=>change(Number(v))} formatValue={v=>`${Number(v)+1} / ${chosen.length}`}/>}
  </DataComponent>
  <EvidenceChart id="v3-rest-wave" queryId="v3_rest_trials" title="休息实验 · C3 / Cz / C4" variant="card" rows={wave} sourceRows={t?[t]:[]} spec={waveSpec} height={285}/>
 </Section>;
}

export function RestView(){
 const {queries}=useDataApp(),[model,setModel]=useState('EEGNet-3class'),[mode,setMode]=useState('验证阈值门控');
 const rows=id=>queries[id]?.rows??[],scores=rows('v3_rest_scores'),curves=rows('v3_val_gate_curve').filter(r=>r.model===model);
 const matrices=rows('v3_rest_matrix').filter(r=>r.model===model&&r.mode===mode).map(r=>({...r,prediction:mode==='验证阈值门控'&&r.prediction==='休息'?'休息／拒绝':r.prediction}));
 const quality=rows('v3_quality')[0],history=rows('v3_rest_history'),trials=rows('v3_rest_trials');
 const stat=scores.find(r=>r.model===model&&r.mode==='验证阈值门控');
 return <article className="eeg-dashboard"><Caveat/>
  <Section id="v3-rest-title" title="休息误触发与动作覆盖" spacing="content">
   <div className="eeg-two">
    <EvidenceChart id="v3-rest-fpr" queryId="v3_rest_scores" title="休息窗口误触发率" variant="card" rows={scores} sourceRows={scores} height={255} spec={{type:'bar',x:'mode',y:'restFalseRate',series:'model',colors:modelColors,stackable:false,valueDecimals:1}}/>
    <EvidenceChart id="v3-rest-coverage" queryId="v3_rest_scores" title="运动想象窗口输出覆盖率" variant="card" rows={scores} sourceRows={scores} height={255} spec={{type:'bar',x:'mode',y:'miCoverageRate',series:'model',colors:modelColors,stackable:false,valueDecimals:1}}/>
   </div>
   <p className="eeg-note">误触发＝休息窗口中输出左右候选的比例。覆盖＝运动想象窗口中仍输出左右候选的比例。低误触发与低覆盖可能同时发生，须一起判断。</p>
   <DataComponent id="v3-rest-table" queryId="v3_rest_scores" title="门控代价与准确性" variant="card" kind="table" displayRows={scores} sourceRows={scores}><DataTable rows={scores} columns={restColumns} searchable={false} caption="休息门控结果"/></DataComponent>
   <DataComponent id="v3-rest-population" queryId="v3_quality" title="三分类数据规模" variant="card" displayRows={quality?[quality]:[]} sourceRows={quality?[quality]:[]}><div data-reviewed-rows className="eeg-facts"><div><span>训练／验证／测试</span><strong>{quality?.train_total} / {quality?.validation_total} / {quality?.test_total}</strong><small>完整3秒窗口</small></div><div><span>测试休息／运动想象</span><strong>{quality?.test_rest} / {quality?.test_mi}</strong></div><div><span>新读取EDF核验</span><strong>{quality?.sha256_verified} / {quality?.input_edf}</strong></div></div></DataComponent>
  </Section>
  <Section id="v3-rest-detail-title" title="休息实验诊断" filters={<><Dropdown label="诊断模型" showLabel value={model} choices={['EEGNet-3class','Bandpower-LDA']} onChange={setModel}/><Dropdown label="决策方式" showLabel value={mode} choices={['验证阈值门控','三分类硬判']} onChange={setMode}/></>}>
   <DataComponent id="v3-gate-facts" queryId="v3_rest_scores" title="验证集冻结阈值" variant="card" displayRows={stat?[stat]:[]} sourceRows={stat?[stat]:[]}><div data-reviewed-rows className="eeg-facts"><div><span>阈值</span><strong>{stat?.threshold}</strong></div><div><span>测试休息误触发数</span><strong>{stat?.rest_false_commands} / {stat?.rest_n}</strong></div><div><span>测试正确动作候选数</span><strong>{stat?.mi_correct} / {stat?.mi_n}</strong></div></div><p className="eeg-note">要求验证休息误触发≤5%，再最大化正确候选率；该要求不保证测试或真实用户达到5%。</p></DataComponent>
   <div className="eeg-two"><EvidenceChart id="v3-rest-confusion" queryId="v3_rest_matrix" title="休息 / 左 / 右混淆计数" variant="card" rows={matrices} sourceRows={rows('v3_rest_matrix').filter(r=>r.model===model&&r.mode===mode)} height={325} spec={{type:'heatmap',x:'prediction',y:'count',series:'truth',showValues:true,reverseRows:true,xLabel:'预测／门控结果',yLabel:'提示标签',colorDomain:[0,Math.max(1,...matrices.map(r=>r.count))]}}/><EvidenceChart id="v3-validation-gate" queryId="v3_val_gate_curve" title="验证集 · 阈值与拒识代价" variant="card" rows={curves} sourceRows={curves} height={325} spec={{type:'line',x:'threshold',y:'restFalseRate',fields:['restFalseRate','miCoverageRate','correctCommandRate'],colors:{restFalseRate:'var(--chart-3)',miCoverageRate:'var(--chart-1)',correctCommandRate:'var(--chart-2)'},legend:{labels:{restFalseRate:'休息误触发',miCoverageRate:'MI覆盖',correctCommandRate:'正确候选率'}},xLabel:'概率阈值',stackable:false,valueDecimals:1}}/></div>
   <EvidenceChart id="v3-rest-learning" queryId="v3_rest_history" title="三分类 EEGNet · 训练与验证 BA" variant="card" rows={history} sourceRows={history} height={240} spec={{type:'line',x:'epoch',y:'trainBARate',fields:['trainBARate','validationBARate'],colors:{trainBARate:'var(--chart-1)',validationBARate:'var(--chart-2)'},legend:{labels:{trainBARate:'训练 BA',validationBARate:'验证 BA'}},xLabel:'Epoch',stackable:false,valueDecimals:1}}/>
  </Section>
  <RestReplay trials={trials}/>
  <p className="eeg-note">T0仅代表数据集中有提示的休息区间。尚未验证日常无意图、连续异步检测、跨天稳定性或每小时误触发；此处的动作候选没有连接手机。</p>
 </article>;
}

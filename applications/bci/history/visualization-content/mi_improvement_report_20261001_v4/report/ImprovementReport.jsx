import React from 'react';
import {DataComponent,DataTable,ReportSection,RichNarrative,useDataApp} from '../../data-app-public.jsx';
import './report.css';
const rate=(v)=>v==null?'无输出':`${(v*100).toFixed(1)}%`;
const sourcePreviews={
 'https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html':{title:'pyRiemann TangentSpace 官方文档',summary:'说明如何将正定协方差矩阵映射为可供分类使用的向量；本轮使用自行实现的 log-Euclidean 特征，不宣称等同该 API。',source:'pyRiemann',approvedForReport:true},
 'https://mne.tools/stable/generated/mne.decoding.CSP.html':{title:'MNE CSP 官方文档',summary:'介绍用于 EEG 分类的共同空间模式估计器及参数，作为之前 CSP 基线的背景。',source:'MNE',approvedForReport:true}
};
const tables={
 mi_comparison:{id:'mi-comparison',title:'同一批测试片段上的识别表现',columns:[{field:'method',label:'方法'},{field:'baRate',label:'平衡准确率',renderCell:rate},{field:'interval',label:'95% 按人区间'},{field:'delta',label:'较第三轮（百分点）'},{field:'n',label:'试次'}]},
 stability:{id:'seed-stability',title:'微调的随机种子稳定性',columns:[{field:'policyLabel',label:'微调方案'},{field:'seed',label:'随机种子',renderCell:v=>String(v)},{field:'baRate',label:'平衡准确率',renderCell:rate}]},
 gate:{id:'rest-tradeoff',title:'休息拒识：误触发与有效输出的代价',columns:[{field:'policyLabel',label:'验证阈值目标'},{field:'restFprRate',label:'休息误触发',renderCell:rate},{field:'coverageRate',label:'动作覆盖',renderCell:rate},{field:'correctRate',label:'正确 / 全部动作',renderCell:rate},{field:'acceptedRate',label:'输出动作准确率',renderCell:rate},{field:'allCandidateRate',label:'全部候选准确率',renderCell:rate},{field:'outputs',label:'输出 / 动作总数'}]},
 controls:{id:'random-controls',title:'机会水平与打乱标签对照',columns:[{field:'control',label:'对照'},{field:'baRate',label:'平衡准确率',renderCell:rate},{field:'note',label:'解释'}]},
 subjects:{id:'person-detail',title:'逐人对照：保留提升与退步',columns:[{field:'person',label:'受试者'},{field:'baselineRate',label:'第三轮',renderCell:rate},{field:'improvedRate',label:'第四轮推荐方案',renderCell:rate},{field:'delta',label:'变化（百分点）'},{field:'n',label:'试次'}]}
};
export function ReportContent(){
 const {snapshot,reviewedPeriodRows,visible,canEdit,mode,appTitle,setAppTitle}=useDataApp();
 const paragraphs=snapshot.narrative??{summary:'## 实验正在运行\n\n正在比较分频空间特征与 EEGNet 微调。本报告先保留之前的配对基线，第四轮测试结果尚未生成。',mi:'## 识别基线\n\n14位受试者各自 run04 校准、run08 选型，同一批 run12 共206个片段。',limits:'## 评价边界\n\n公开数据离线探索，测试人群此前已查看；没有手机执行或在线实机测试。'};
 return <article className='report-content bci-report' aria-label='脑机接口原型改进报告'>
  <header className='report-hero'><h1 data-data-app-title contentEditable={canEdit&&mode==='edit'} suppressContentEditableWarning onBlur={canEdit&&mode==='edit'?(e)=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1></header>
  <ReportSection id='executive-summary' title='实验结论' queryId='mi_comparison' showHeading={false}><RichNarrative id='executive-summary:body' value={paragraphs.summary}/></ReportSection>
  <ReportSection id='mi-explanation' title='识别改进' queryId='mi_comparison' showHeading={false}><RichNarrative id='mi-explanation:body' value={paragraphs.mi}/></ReportSection>
  {['mi_comparison','stability','gate','controls','subjects'].map(q=>{
   if(!snapshot.queries[q]||!visible(tables[q].id))return null;
   const rows=reviewedPeriodRows(q);const t=tables[q];
   return <section className='report-section' key={q}>
    {q==='gate'&&<ReportSection id='gate-explanation' title='休息门控解释' queryId='gate' showHeading={false}><RichNarrative id='gate-explanation:body' value={paragraphs.gate}/></ReportSection>}
    {q==='controls'&&<ReportSection id='control-explanation' title='对照解释' queryId='controls' showHeading={false}><RichNarrative id='control-explanation:body' value={paragraphs.controls}/></ReportSection>}
    <DataComponent id={t.id} title={t.title} queryId={q} kind='table' displayRows={rows} sourceRows={rows}><DataTable rows={rows} columns={t.columns} searchable={q==='subjects'} caption={t.title}/></DataComponent>
   </section>;
  })}
  {snapshot.queries.methods&&<ReportSection id='methods-and-prototype' title='方法与原型定位' queryId='methods' showHeading={false}><RichNarrative id='methods-and-prototype:body' value={paragraphs.methods} sourcePreviews={sourcePreviews}/></ReportSection>}
  <ReportSection id='evidence-boundaries' title='限制与建议' queryId='mi_comparison' showHeading={false}><RichNarrative id='evidence-boundaries:body' value={paragraphs.limits}/></ReportSection>
 </article>;
}

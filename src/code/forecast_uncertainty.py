# -*- coding: utf-8 -*-
"""光伏预报提前时间与严格因果滚动区间校准。
发布行为每天0/6/12/18点，24列对应未来1..24小时整点。
实际PV第0列为0:10右端点，因此绝对整点h映射flatten索引6*h-1。
区间只用同发布钟点、同提前时间、过去28天且目标不晚于当前发布的已观测误差。
评价期2025-02-01至2025-12-31。区间以0截断，未设虚构光伏容量上限。
"""
from pathlib import Path
import numpy as np,json,datetime as dt,hashlib
ROOT=Path(__file__).resolve().parents[1]
ALPHA=.2

def metrics(y,p,lo,hi):
    err=y-p;width=hi-lo
    score=width+2/ALPHA*np.maximum(lo-y,0)+2/ALPHA*np.maximum(y-hi,0)
    return {'n':int(len(y)),'mae_kw':float(np.mean(abs(err))),'rmse_kw':float(np.sqrt(np.mean(err**2))),'bias_actual_minus_forecast_kw':float(np.mean(err)),
        'coverage':float(np.mean((y>=lo)&(y<=hi))),'mean_width_kw':float(np.mean(width)),'interval_score_kw':float(np.mean(score))}

def main():
    src=ROOT/'data/inputs.npz';a=np.load(src);fc=a['forecast'].reshape(365,4,24);pv=a['pv'].reshape(-1)
    
    release=np.arange(365)[:,None]*24+np.array([0,6,12,18])[None,:]
    target=release[:,:,None]+np.arange(1,25)[None,None,:]
    actual=np.full(fc.shape,np.nan);valid=target<=365*24
    actual[valid]=pv[(target[valid]*6-1).astype(int)]
    assert np.sum(~valid)==36 and np.sum(valid)==35004
    # Explicit boundary cases protect end-time and day rollover semantics.
    assert actual[0,0,0]==a['pv'][0,5]
    assert actual[0,0,23]==a['pv'][0,143]
    assert actual[0,3,6]==a['pv'][1,5]
    err=actual-fc;lo=np.full_like(fc,np.nan);hi=lo.copy();ncal=np.zeros(fc.shape,int);lo_rank=lo.copy();hi_rank=hi.copy()
    max_history_target_minus_release=-np.inf
    for day in range(365):
        for pub in range(4):
            for ell in range(24):
                ids=np.arange(max(0,day-28),day)
                eligible=(target[ids,pub,ell]<=release[day,pub])&valid[ids,pub,ell]
                ids=ids[eligible];ncal[day,pub,ell]=len(ids)
                if len(ids):max_history_target_minus_release=max(max_history_target_minus_release,float(np.max(target[ids,pub,ell]-release[day,pub])))
                if len(ids)<7 or not valid[day,pub,ell]:continue
                assert np.all(ids<day) and np.all(target[ids,pub,ell]<=release[day,pub])
                qlo,qhi=np.quantile(err[ids,pub,ell],[ALPHA/2,1-ALPHA/2],method='linear')
                lo[day,pub,ell]=max(0.,fc[day,pub,ell]+qlo);hi[day,pub,ell]=max(lo[day,pub,ell],fc[day,pub,ell]+qhi)
                if len(ids)>=19:
                    residuals=np.sort(err[ids,pub,ell]);nn=len(ids)
                    lower_rank=int(np.floor((nn+1)*ALPHA/2));upper_rank=int(np.ceil((nn+1)*(1-ALPHA/2)))
                    assert 1<=lower_rank<=upper_rank<=nn
                    lo_rank[day,pub,ell]=max(0.,fc[day,pub,ell]+residuals[lower_rank-1])
                    hi_rank[day,pub,ell]=max(lo_rank[day,pub,ell],fc[day,pub,ell]+residuals[upper_rank-1])
    # Evaluate by target calendar date; current-day outputs can include yesterday's forecasts.
    target_day=(target-1)//24
    mask=valid&np.isfinite(lo)&(target_day>=31)&(target_day<365)
    hour=target%24;daylight=(hour>=8)&(hour<=17)
    report={'metadata':{'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'target_endpoint_mapping':'global_hour*6-1','forecast_leads_hours':list(range(1,25)),'issued_hours':[0,6,12,18],'evaluation_target_date':'2025-02-01 through 2025-12-31','valid_targets':int(valid.sum()),'excluded_out_of_year_targets':int((~valid).sum()),'calibration_window_days':28,'nominal_coverage':1-ALPHA,'minimum_calibration_count':7,'quantile_method':'numpy linear empirical 0.1 and 0.9','lower_clipped_to_zero':True,'calibration_sample_sizes_in_test':[int(ncal[mask].min()),int(ncal[mask].max())],'max_calibration_target_minus_release_hours':max_history_target_minus_release,'same_release_observation_allowed':True,'strictly_past_issue_days':True,'random_seed':202611,'bootstrap':'2000 circular moving-block bootstrap samples, 7-day blocks over 334 common target days'},'aggregate':{},'leadtime':{},'common_targets':{},'rank_corrected':{'rule':'lower floor((n+1)*0.1), upper ceil((n+1)*0.9), 1-based order statistics','n28_ranks':[2,27],'ideal_exchangeable_continuous_rank_coverage':25/29,'uniform_residual_reference_linear_coverage':21.6/29,'scope':'Order-statistic correction benchmark only. 21.6/29 applies only to a uniform residual reference, not arbitrary value-interpolated quantiles. No distribution-free guarantee under seasonal time dependence.','aggregate':{},'common_targets':{}}}
    for group,select in [('all_targets',mask),('clock_hours_08_17',mask&daylight)]:
        report['aggregate'][group]=metrics(actual[select],fc[select],lo[select],hi[select]);report['leadtime'][group]=[]
        report['rank_corrected']['aggregate'][group]=metrics(actual[select],fc[select],lo_rank[select],hi_rank[select])
        for ell in range(24):
            sel=select[:,:,ell];row=metrics(actual[:,:,ell][sel],fc[:,:,ell][sel],lo[:,:,ell][sel],hi[:,:,ell][sel]);row['lead_hours']=ell+1;report['leadtime'][group].append(row)
    common_arrays={}
    rng=np.random.default_rng(202611);length=334;block=7
    starts=rng.integers(0,length,size=(2000,int(np.ceil(length/block))))
    boot=((starts[:,:,None]+np.arange(block))%length).reshape(2000,-1)[:,:length]
    for h in [12,14,16,18]:
        # Four most recent strictly earlier issue slots; at noon this excludes the noon forecast.
        days=np.arange(31,365);th=days*24+h
        lead_values=sorted({int(h-issue) for issue in [-6,0,6,12,18] if 0<h-issue<=24})[:4]
        # 12/18h have 4 distinct available issues with valid positive leads.
        if len(lead_values)<4:lead_values=sorted({int(h-issue) for issue in [-12,-6,0,6,12,18] if 0<h-issue<=24})[:4]
        rows=[];arrays=[];rank_rows=[];rank_arrays=[]
        for lead in lead_values:
            rh=th-lead;rd=rh//24;rp=(rh%24)//6;ell=lead-1
            yy=actual[rd,rp,ell];pp=fc[rd,rp,ell];ll=lo[rd,rp,ell];uu=hi[rd,rp,ell]
            assert np.all(target[rd,rp,ell]==th) and np.isfinite(ll).all() and len(yy)==334
            assert np.array_equal(yy,a['pv'][days,h*6-1])
            mm=metrics(yy,pp,ll,uu);mm['lead_hours']=lead;mm['release_clock_hour']=int(rh[0]%24);mm['release_day_offset']=int(rd[0]-days[0])
            sample={'mae_kw':abs(yy-pp),'coverage':((yy>=ll)&(yy<=uu)).astype(float),'mean_width_kw':uu-ll,'interval_score_kw':uu-ll+2/ALPHA*np.maximum(ll-yy,0)+2/ALPHA*np.maximum(yy-uu,0)}
            for key,vv in sample.items():mm[key+'_ci95']=np.quantile(vv[boot].mean(axis=1),[.025,.975]).tolist()
            rl=lo_rank[rd,rp,ell];ru=hi_rank[rd,rp,ell];rm=metrics(yy,pp,rl,ru);rm['lead_hours']=lead
            covered=((yy>=rl)&(yy<=ru)).astype(float);rm['coverage_ci95']=np.quantile(covered[boot].mean(axis=1),[.025,.975]).tolist()
            rank_rows.append(rm);rank_arrays.append(np.stack([rl,ru],axis=1))
            rows.append(mm);arrays.append(np.stack([yy,pp,ll,uu],axis=1))
        report['rank_corrected']['common_targets'][str(h)]=rank_rows
        common_arrays[f'common_rank_{h}']=np.stack(rank_arrays,axis=1)
        arr=np.stack(arrays,axis=1);common_arrays[f'common_{h}']=arr;common_arrays[f'leads_{h}']=np.array(lead_values)
        diff=abs(arr[:,-1,0]-arr[:,-1,1])-abs(arr[:,0,0]-arr[:,0,1])
        report['common_targets'][str(h)]={'target_clock_hour':h,'n_common_days':334,'lead_results':rows,'longest_minus_shortest_mae_kw':float(diff.mean()),'paired_block_bootstrap_ci95_kw':np.quantile(diff[boot].mean(axis=1),[.025,.975]).tolist(),'monotonic_mae_as_lead_increases':bool(np.all(np.diff([x['mae_kw'] for x in rows])>=0))}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    (ROOT/'results/forecast_uncertainty.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    np.savez_compressed(ROOT/'results/forecast_uncertainty.npz',forecast=fc,actual=actual,lower=lo,upper=hi,lower_rank=lo_rank,upper_rank=hi_rank,target_hour=target,release_hour=release,valid=valid,test_mask=mask,calibration_count=ncal,**common_arrays)
if __name__=='__main__':main()

"""Reproducible ASVspoof LA subset experiment; never fits on dev/eval."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import scipy
from scipy.fft import dct
from scipy.signal import stft
import soundfile as sf
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[3]
FEATURE_VERSION = 'lfcc20_mean_std_delta_6sec_v1'

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def features(path):
    # Fixed first six seconds, without repeating/padding short utterances.
    with sf.SoundFile(path) as f:
        if f.samplerate != 16000 or f.channels != 1:
            raise ValueError(f'Expected mono 16kHz: {path}')
        audio = f.read(min(len(f), 96000), dtype='float32')
        duration = len(f) / f.samplerate
    if len(audio) < 400 or not np.isfinite(audio).all():
        raise ValueError(f'Invalid audio: {path}')
    audio = np.append(audio[0], audio[1:] - .97 * audio[:-1])
    _, _, z = stft(audio, fs=16000, window='hann', nperseg=400,
                   noverlap=240, nfft=512, boundary=None, padded=False)
    power = np.abs(z) ** 2
    bins = np.arange(257)
    edges = np.linspace(0, 256, 42)
    banks = np.stack([np.maximum(0, np.minimum((bins-edges[i])/(edges[i+1]-edges[i]),
                       (edges[i+2]-bins)/(edges[i+2]-edges[i+1]))) for i in range(40)])
    cep = dct(np.log(np.maximum(banks @ power, 1e-12)), type=2, axis=0, norm='ortho')[:20]
    delta = np.gradient(cep, axis=1)
    delta2 = np.gradient(delta, axis=1)
    result = np.concatenate([v for a in (cep, delta, delta2) for v in (a.mean(1), a.std(1))])
    return result.astype('float32'), duration

def metrics(y, scores, threshold):
    pred = scores >= threshold
    fpr, tpr, _ = roc_curve(y, scores)
    i = np.argmin(np.abs(fpr - (1-tpr)))
    return dict(samples=len(y), threshold=float(threshold), accuracy=accuracy_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred), precision_spoof=precision_score(y,pred,zero_division=0),
        recall_spoof=recall_score(y,pred,zero_division=0), f1_spoof=f1_score(y,pred,zero_division=0),
        macro_f1=f1_score(y,pred,average='macro',zero_division=0), roc_auc=roc_auc_score(y,scores),
        eer_approx=float((fpr[i]+1-tpr[i])/2), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
        confusion_order=['bonafide','spoof'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/audio/asvspoof2019/LA')
    p.add_argument('--output', type=Path, default=ROOT/'backend/outputs/audio_official_20261026')
    p.add_argument('--seed', type=int, default=42)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    groups={}; audit={}; hashes={}
    for split,n,ending in [('train',6000,'trn'),('dev',2000,'trl'),('eval',4000,'trl')]:
        protocol=args.data/f'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{ending}.txt'
        hashes[split]=hashlib.sha256(protocol.read_bytes()).hexdigest()
        rows=[]
        for line in protocol.read_text().splitlines():
            speaker,uid,_,attack,label=line.split()
            path=args.data/f'ASVspoof2019_LA_{split}/flac/{uid}.flac'
            if not path.is_file(): raise FileNotFoundError(path)
            rows.append(dict(split=split,speaker=speaker,id=uid,attack=attack,label=label,path=str(path)))
        audit[split]={'protocol_count':len(rows),'labels':dict(Counter(r['label'] for r in rows))}
        chosen,_=train_test_split(np.arange(len(rows)),train_size=n,random_state=args.seed,
                                  stratify=[r['attack'] for r in rows])
        groups[split]=[rows[i] for i in sorted(chosen)]
    for a,b in [('train','dev'),('train','eval'),('dev','eval')]:
        assert not ({r['speaker'] for r in groups[a]} & {r['speaker'] for r in groups[b]})
        assert not ({r['id'] for r in groups[a]} & {r['id'] for r in groups[b]})
    allrows=sum(groups.values(),[])
    write_csv(out/'manifest.csv',allrows)
    cache=out/'features.npz'
    signature=hashlib.sha256((FEATURE_VERSION+json.dumps(allrows,sort_keys=True)).encode()).hexdigest()
    if cache.exists():
        saved=np.load(cache)
        if str(saved['signature']) != signature: raise ValueError('Feature cache mismatch')
        X=saved['X']; durations=saved['durations']
    else:
        xx=[]; durations=[]
        for i,row in enumerate(allrows):
            x,duration=features(row['path']); xx.append(x); durations.append(duration)
            if (i+1)%250==0: print(f'Extracted {i+1}/{len(allrows)}',flush=True)
        X=np.array(xx); durations=np.array(durations)
        np.savez_compressed(cache,X=X,durations=durations,signature=signature)
    y=np.array([r['label']=='spoof' for r in allrows],dtype=int)
    tr=slice(0,6000); dv=slice(6000,8000); te=slice(8000,None)
    candidates={'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=args.seed)),
                'extra_trees':ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',n_jobs=2,random_state=args.seed)}
    dev_results={}
    for name,model in candidates.items():
        print(f'Training {name}',flush=True); model.fit(X[tr],y[tr])
        scores=model.predict_proba(X[dv])[:,1]
        fpr,tpr,thresholds=roc_curve(y[dv],scores)
        threshold=float(thresholds[np.argmax(tpr-fpr)])
        dev_results[name]=metrics(y[dv],scores,threshold)
    selected=max(dev_results,key=lambda name:dev_results[name]['roc_auc'])
    model=candidates[selected]; threshold=dev_results[selected]['threshold']
    scores=model.predict_proba(X[te])[:,1]
    test=metrics(y[te],scores,threshold)
    joblib.dump(dict(model=model,threshold=threshold,feature_version=FEATURE_VERSION,labels={0:'bonafide',1:'spoof'}),out/'model.joblib')
    predictions=[]
    for row,score,duration in zip(groups['eval'],scores,durations[te]):
        pred='spoof' if score>=threshold else 'bonafide'
        predictions.append(dict(**row,duration_seconds=round(float(duration),3),spoof_score=float(score),prediction=pred,correct=pred==row['label']))
    write_csv(out/'test_predictions.csv',predictions)
    # Predetermined by IDs and labels, never by prediction correctness.
    small=sum(([r for r in predictions if r['label']==label][:10] for label in ['bonafide','spoof']),[])
    write_csv(out/'small_test_20.csv',small)
    attacks=[]
    for attack in sorted({r['attack'] for r in predictions}):
        subset=[r for r in predictions if r['attack']==attack]
        attacks.append(dict(attack=attack,samples=len(subset),correct=sum(r['correct'] for r in subset),accuracy=sum(r['correct'] for r in subset)/len(subset)))
    write_csv(out/'per_attack.csv',attacks)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,selected_model=selected,
        official_inventory=audit,protocol_sha256=hashes,subset_counts={s:dict(Counter(r['label'] for r in rows)) for s,rows in groups.items()},
        speaker_overlap=0,feature_version=FEATURE_VERSION,feature_dimensions=X.shape[1],
        development=dev_results,test=test,majority_baseline_accuracy=float(y[te].mean()),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,soundfile=sf.__version__))
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()

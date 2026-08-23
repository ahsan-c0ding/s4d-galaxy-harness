
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

REPO_ROOT = Path(os.environ.get('S4D_REPO_ROOT','/kaggle/working/s4d_repo'))
os.chdir(REPO_ROOT); sys.path.insert(0,str(REPO_ROOT))
from model.functions import load_data
from model.gclassifier import GalaxyClassifierS4D
from model.hilbert import HilbertScan
from model.s4d_recurrent import S4D
from model.tlts import TakeLastTimestep

BASE_DIR=Path(os.environ.get('S4D_BASE_DIR','/kaggle/working/s4d_future_work'))
RESULTS_DIR=BASE_DIR/'results'; WEIGHTS_DIR=BASE_DIR/'weights'; CHECKPOINT_DIR=BASE_DIR/'checkpoints'
for d in (RESULTS_DIR,WEIGHTS_DIR,CHECKPOINT_DIR): d.mkdir(parents=True,exist_ok=True)
SPLIT_SEED=30485; EPOCHS=630; BATCH_SIZE=32
RICHER_REPORT_PARAMS={(1,0):2180,(1,1):10500,(1,2):18820,(2,0):19844,(2,1):28164,(2,2):36484,(3,0):29156,(3,1):37476,(3,2):45796,(4,0):38468,(4,1):46788,(4,2):55108}

class RicherStem(nn.Module):
    def __init__(self,depth,in_channels=3,d_model=64,mid_channels=32,dropout=0.1):
        super().__init__(); self.depth=depth; self.grid=64 if depth==1 else 16
        def gn(ch): return nn.GroupNorm(8 if ch%8==0 else 1,ch)
        if depth==1:
            self.conv1=nn.Conv2d(in_channels,d_model,3,1,1); self.norm1=gn(d_model); self.act1=nn.GELU()
        elif depth==2:
            self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1); self.norm1=gn(mid_channels); self.act1=nn.GELU(); self.conv2=nn.Conv2d(mid_channels,d_model,3,4,1); self.norm2=gn(d_model); self.act2=nn.GELU()
        elif depth==3:
            self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1); self.norm1=gn(mid_channels); self.act1=nn.GELU(); self.conv2=nn.Conv2d(mid_channels,mid_channels,3,2,1); self.norm2=gn(mid_channels); self.act2=nn.GELU(); self.conv3=nn.Conv2d(mid_channels,d_model,3,2,1); self.norm3=gn(d_model); self.act3=nn.GELU()
        elif depth==4:
            self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1); self.norm1=gn(mid_channels); self.act1=nn.GELU(); self.res_conv=nn.Conv2d(mid_channels,mid_channels,3,1,1); self.res_norm=gn(mid_channels); self.res_act=nn.GELU(); self.drop=nn.Dropout2d(dropout); self.conv2=nn.Conv2d(mid_channels,mid_channels,3,2,1); self.norm2=gn(mid_channels); self.act2=nn.GELU(); self.conv3=nn.Conv2d(mid_channels,d_model,3,2,1); self.norm3=gn(d_model); self.act3=nn.GELU()
        else: raise ValueError(depth)
    def forward(self,x):
        if self.depth==1: return self.act1(self.norm1(self.conv1(x)))
        if self.depth==2: x=self.act1(self.norm1(self.conv1(x))); return self.act2(self.norm2(self.conv2(x)))
        if self.depth==3: x=self.act1(self.norm1(self.conv1(x))); x=self.act2(self.norm2(self.conv2(x))); return self.act3(self.norm3(self.conv3(x)))
        x=self.act1(self.norm1(self.conv1(x))); r=self.res_act(self.res_norm(self.res_conv(x))); x=self.drop(x+r); x=self.act2(self.norm2(self.conv2(x))); return self.act3(self.norm3(self.conv3(x)))

class RicherGridModel(nn.Module):
    def __init__(self,stem_depth,num_s4_layers,d_model=64,s4_state=64,num_classes=4):
        super().__init__(); self.cnn_stem=RicherStem(stem_depth,d_model=d_model); self.grid=self.cnn_stem.grid; self.seq_len=self.grid*self.grid; self.hilbert_scan=HilbertScan(n=self.grid); self.s4_layers=nn.ModuleList([S4D(d_model=d_model,d_state=s4_state,transposed=False) for _ in range(num_s4_layers)]); self.acts=nn.ModuleList([nn.GELU() for _ in range(num_s4_layers)]); self.take_last=TakeLastTimestep(); self.fc=nn.Linear(d_model,num_classes)
    def forward(self,x,return_logits=True):
        h=self.hilbert_scan(self.cnn_stem(x))
        for layer,act in zip(self.s4_layers,self.acts): h=act(layer(h)[0])
        logits=self.fc(self.take_last(h)); return logits if return_logits else torch.softmax(logits,dim=-1)

def make_model(spec_id):
    if spec_id=='richer_rawpix_s4d2': return GalaxyClassifierS4D(s4_state=64,d_model=64,num_classes=4,colored=True)
    parts=spec_id.split('_'); return RicherGridModel(int(parts[1][1:]),int(parts[2][2:]))

def expected_params(spec_id):
    if spec_id=='richer_rawpix_s4d2': return 17156
    p=spec_id.split('_'); return RICHER_REPORT_PARAMS[(int(p[1][1:]),int(p[2][2:]))]

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_augmented_dataset(X,y):
    class Aug(Dataset):
        def __init__(self,X,y): self.X,self.y=X,y
        def __len__(self): return len(self.X)
        def __getitem__(self,idx):
            img,label=self.X[idx],self.y[idx]; k=random.randint(0,3)
            if k: img=torch.rot90(img,k,dims=(1,2))
            if random.random()<0.5: img=torch.flip(img,dims=(2,))
            if random.random()<0.5: img=torch.flip(img,dims=(1,))
            return img,label
    return Aug(X,y)

def build_data(seed):
    set_all_seeds(seed); X,_,y=load_data(root='./data',download=True,train=True,colored=True); Xt,_,yt=load_data(root='./data',download=True,train=False,colored=True); xtr,xv,ytr,yv=train_test_split(X,y,test_size=0.2,random_state=SPLIT_SEED,stratify=y)
    return DataLoader(make_augmented_dataset(xtr,ytr),batch_size=BATCH_SIZE,shuffle=True),DataLoader(TensorDataset(xv,yv),batch_size=BATCH_SIZE),DataLoader(TensorDataset(Xt,yt),batch_size=64)

def make_optimizer(model):
    decay,no_decay,special=[],[],[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if hasattr(p,'_optim'): special.append({'params':[p],'lr':getattr(p,'_optim',{}).get('lr',1e-3),'weight_decay':0.0})
        elif any(k in name.lower() for k in ['bias','norm','layernorm']): no_decay.append(p)
        else: decay.append(p)
    return torch.optim.AdamW([{'params':decay,'lr':1e-3,'weight_decay':1e-2},{'params':no_decay,'lr':1e-3,'weight_decay':0.0}]+special)

def make_scheduler(opt): return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2,eta_min=1e-5)

def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); logits=model(x,return_logits=True); p=torch.softmax(logits,1); ys+=y.numpy().tolist(); ps+=logits.argmax(1).cpu().numpy().tolist(); probs+=p.cpu().numpy().tolist()
    y=np.asarray(ys); pr=np.asarray(ps); pb=np.asarray(probs)
    return {'accuracy':float(accuracy_score(y,pr)),'f1_macro':float(f1_score(y,pr,average='macro')),'precision_macro':float(precision_score(y,pr,average='macro',zero_division=0)),'recall_macro':float(recall_score(y,pr,average='macro',zero_division=0)),'roc_auc_macro':float(roc_auc_score(y,pb,multi_class='ovr',average='macro')),'confusion_matrix':confusion_matrix(y,pr).tolist()}

def rid(spec,seed): return f'{spec}__main__seed{seed}'
def cp(r): return CHECKPOINT_DIR/f'{r}.pt'
def rp(r): return RESULTS_DIR/f'{r}.json'
def wp(r): return WEIGHTS_DIR/f'{r}.pt'

def rng_state():
    s={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_available(): s['cuda']=torch.cuda.get_rng_state_all()
    return s

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if torch.cuda.is_available() and 'cuda' in s: torch.cuda.set_rng_state_all(s['cuda'])

def save_ck(rid0,epoch,model,opt,sched,best_state,best_val,history,seed):
    tmp=cp(rid0).with_suffix('.tmp'); torch.save({'epoch':epoch,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer_state':opt.state_dict(),'scheduler_state':sched.state_dict(),'best_state':best_state,'best_val':best_val,'history':history,'seed':seed,'rng_state':rng_state()},tmp); os.replace(tmp,cp(rid0))

def train_one(spec_id,seed,purpose,deadline):
    rid0=rid(spec_id,seed); 
    if rp(rid0).exists() and wp(rid0).exists(): print(f'[{rid0}] final result exists; skipping',flush=True); return
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); torch.cuda.set_device(0) if device.type=='cuda' else None; set_all_seeds(seed)
    tr,va,te=build_data(seed); model=make_model(spec_id).to(device); actual=sum(p.numel() for p in model.parameters()); exp=expected_params(spec_id)
    if actual!=exp: raise RuntimeError(f'{rid0}: expected {exp}, got {actual}')
    opt=make_optimizer(model); sched=make_scheduler(opt); loss_fn=nn.CrossEntropyLoss(); start=0; best_val=-1.; best_state=None; history={'train_loss':[],'train_acc':[],'val_acc':[],'lr':[]}
    if cp(rid0).exists():
        ck=torch.load(cp(rid0),map_location='cpu',weights_only=False); model.load_state_dict(ck['model_state']); opt.load_state_dict(ck['optimizer_state']); sched.load_state_dict(ck['scheduler_state']); best_state=ck['best_state']; best_val=ck['best_val']; history=ck['history']; start=int(ck['epoch']); restore_rng(ck['rng_state']); model.to(device); print(f'[{rid0}] RESUMING from epoch {start}/{EPOCHS}',flush=True)
    t0=time.time()
    for epoch in range(start,EPOCHS):
        model.train(); total=correct=0; running=0.0
        for x,y in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True); logits=model(x,return_logits=True); loss=loss_fn(logits,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); running+=float(loss.item())*y.size(0); correct+=int((logits.argmax(1)==y).sum()); total+=y.size(0)
        sched.step(); val=evaluate(model,va,device); history['train_loss'].append(running/max(1,total)); history['train_acc'].append(correct/max(1,total)); history['val_acc'].append(val['accuracy']); history['lr'].append(float(opt.param_groups[0]['lr']))
        if val['accuracy']>best_val: best_val=val['accuracy']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        save_ck(rid0,epoch+1,model,opt,sched,best_state,best_val,history,seed)
        if (epoch+1)%25==0 or epoch==EPOCHS-1: print(f'[{rid0}] epoch {epoch+1}/{EPOCHS} train={history["train_acc"][-1]:.4f} val={val["accuracy"]:.4f}',flush=True)
    model.load_state_dict(best_state); test=evaluate(model,te,device); elapsed=time.time()-t0; torch.save(model.state_dict(),wp(rid0)); rp(rid0).write_text(json.dumps({'run_id':rid0,'purpose':purpose,'architecture':spec_id,'seed':seed,'params':actual,'expected_params':exp,'epochs':EPOCHS,'train_time_sec':elapsed,'accuracy':test['accuracy'],'f1_macro':test['f1_macro'],'precision_macro':test['precision_macro'],'recall_macro':test['recall_macro'],'roc_auc_macro':test['roc_auc_macro'],'confusion_matrix':test['confusion_matrix'],'history':history,'weights_file':str(wp(rid0)),'recipe':'main','split_seed':SPLIT_SEED,'status':'complete'},indent=2)); cp(rid0).unlink(missing_ok=True); print(f'[{rid0}] COMPLETE in {elapsed/60:.1f} min | test={test["accuracy"]:.4f}',flush=True)

def main():
    ap=__import__('argparse').ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--jobs',required=True); ap.add_argument('--deadline',type=float,required=True); a=ap.parse_args()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError(f'worker sees CUDA={torch.cuda.is_available()} count={torch.cuda.device_count()}')
        torch.cuda.set_device(0); print(f'[worker gpu={a.gpu}] visible={os.environ.get("CUDA_VISIBLE_DEVICES")} device={torch.cuda.get_device_name(0)}',flush=True)
        for j in json.loads(Path(a.jobs).read_text()):
            if time.time()>=a.deadline: break
            train_one(j['spec_id'],j['seed'],j['purpose'],a.deadline)
        print('[worker] clean exit',flush=True); return 0
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'[worker FATAL] {type(e).__name__}: {e}',flush=True); return 2

if __name__=='__main__': raise SystemExit(main())

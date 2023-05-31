import torch
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, average_precision_score
import random
import wandb
from pathlib import Path
import datetime

def save_to_checkpoint(save_path, epoch, model, optimizer, scheduler=None, verbose=True):
    # save checkpoint to disk
    d_sche = None
    if scheduler is not None:
        d_sche = scheduler.state_dict()
    if save_path is not None:
        checkpoint = {'epoch': epoch + 1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': d_sche
                      }
        torch.save(checkpoint, '{}_epoch_{}.pt'.format(save_path, epoch))

    if verbose:
        print("saved model at epoch {}".format(epoch))

        
def load_from_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None, verbose = True):
    """Loads model from checkpoint, loads optimizer and scheduler too if not None, 
       and returns epoch and iteration of the checkpoints
    """
    if not os.path.exists(checkpoint_path):
        raise ("File does not exist {}".format(checkpoint_path))
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location='cpu')
        
    check_keys = list(checkpoint.keys())

    model.load_state_dict(checkpoint['model']) 
    
    if 'optimizer' in check_keys:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if 'scheduler' in check_keys:
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
  
    if 'epoch' in check_keys:
        epoch = checkpoint['epoch']
       
    if verbose: # optional printing
        print(f"Loaded model from checkpoint {checkpoint_path}")

    return epoch
    
# ---------------------------------------------------------------------
def reshape_bbox(bbox_list, device):
    new_bbox_list = []
    for j in range(len(bbox_list)):
        raw_bboxes = bbox_list[j]
        B = torch.stack(raw_bboxes).type(torch.FloatTensor).to(device, non_blocking=True)
        bboxes = []
        for i in range(B.shape[1]):
           bboxes.append(B[:,i].view(1,4))
        new_bbox_list.append(bboxes)
    
    return new_bbox_list


def batch_first(anns_list):
    anns_3d = []
    for i in range(len(anns_list[0])):
        anns_1d = []
        for t in range(len(anns_list)):
            anns = torch.squeeze(anns_list[t][i], dim=0)
            anns_1d.append(anns)
        anns_tensors_2d = torch.stack(anns_1d)
        anns_3d.append(anns_tensors_2d)
    # stack batch
    anns_tensors_3d = torch.stack(anns_3d)
    
    return anns_tensors_3d
    

def bbox_to_pv(bbox_list):
    pv_3d = []
    for i in range(len(bbox_list[0])):
        p_1d = []
        v_1d = []
        pv_1d = []
        for t in range(len(bbox_list)):
            bbox = torch.squeeze(bbox_list[t][i], dim=0)
            # float 
            b = list(map(lambda x: x.item(), bbox))
            # compute bbox center
            # xc = (b[0] + b[2]) / 2 - 960.0
            # c = abs(-(b[1] + b[3]) / 2 + 1080.0)
            xc = (b[0] + b[2]) / 2 
            yc = (b[1] + b[3]) / 2 
            
            # compute width, height
            w = abs(b[2] - b[0]) 
            h = abs(b[3] - b[1])
            p_1d.append([xc, yc, w, h])
        v_1d.append([0.0, 0.0, 0.0, 0.0])
        for t in range(1, len(bbox_list)):
            dx = abs(p_1d[t][0] -  p_1d[t-1][0])
            dy = abs(p_1d[t][1] -  p_1d[t-1][1])
            dw = abs(p_1d[t][2] -  p_1d[t-1][2])
            dh = abs(p_1d[t][3] -  p_1d[t-1][3])
            v_1d.append([dx, dy, dw, dh])
        for t in range(len(bbox_list)):
            pv_1d.append(torch.tensor(p_1d[t] + v_1d[t], dtype=torch.float32))
        pv_tensors_2d = torch.stack(pv_1d)
        pv_3d.append(pv_tensors_2d)
    # stack batch
    pv_tensors_3d = torch.stack(pv_3d)
    
    return pv_tensors_3d


def reshape_anns(anns_list, device):
    for i, ann in enumerate(anns_list):
        anns_list[i] = ann.to(device, non_blocking=True)
    return anns_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_best_threshold(preds, targets):
    best_f1 = 0
    best_thr = None
    for thr in np.linspace(0, 1, 50):
        preds_thr = (preds > thr).astype(int)
        f1 = f1_score(targets, preds_thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def log_metrics(targets, preds, best_thr, best_f1, ap, mode, step):
    binarized_preds = (preds > best_thr).astype(int)
    precision = precision_score(targets, binarized_preds)
    recall = recall_score(targets, binarized_preds)

    wandb.log({f'{mode}/precision': precision , 
               f'{mode}/recall': recall, 
               f'{mode}/f1': best_f1, 
               f'{mode}/AP': ap, 
               f'{mode}/best_thr': best_thr,
               f"{mode}/preds": wandb.Histogram(preds),
               f'{mode}/epoch': step}, commit=True)
    
    print('------------------------------------------------')
    print(f'Mode: {mode}')
    print(f'best threshold: {best_thr:.3f}')
    print(f'precision: {precision:.3f}')
    print(f'recall: {recall:.3f}')
    print(f'F1-score : {best_f1:.3f}')
    print(f"average precision for transition prediction: {ap:.3f}")
    print('\n')


def log_to_stdout(epoch, train_loss, val_loss, val_f1, end_epoch_time):
    print('\n', '-----------------------------------------------------')
    print(f'End of epoch {epoch}')
    print('Training epoch loss: {:.4f}'.format(train_loss))
    print('Validation epoch loss: {:.4f}'.format(val_loss))
    print('Validation epoch f1: {:.4f}'.format(val_f1))
    print('Epoch time: {:.2f}'.format(end_epoch_time))
    print('--------------------------------------------------------', '\n')


def setup_wandb(args, run_type):
    args.run_type = run_type
    wandb.init(
        project="dlav-intention-prediction",
        config=args,
    )
    run_name = wandb.run.name

    # define our custom x axis metric
    for setup in ['train', 'val']:
        wandb.define_metric(f"{setup}/epoch")
        wandb.define_metric(f"{setup}/*", step_metric=f"{setup}/epoch")
    return run_name


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def prepare_cp_path(args, run_name, run_mode):
    cp_dir = Path(f'./checkpoints/{run_name}')
    cp_dir.mkdir(parents=True, exist_ok=True)
    save_path = cp_dir / f'{run_mode}_lr{args.lr}_wd{args.wd}_JAAD_pred{args.pred}_bs{args.batch_size}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pt'
    print(f'Saving the model to: {save_path}')
    return Path(save_path)


def prep_pred_storage(loader):
    batch_size = loader.batch_size
    n_steps = len(loader)

    preds = np.zeros(n_steps * batch_size)
    tgts = np.zeros(n_steps * batch_size)
    return preds, tgts, n_steps, batch_size


def print_eval_metrics(tgts, preds, best_thr):
    ap = average_precision_score(tgts, preds)
    f1 = f1_score(tgts, preds > best_thr)
    preds = preds > best_thr
    print(f"Best threshold: {best_thr:.3f}, F1: {f1:.3f}, AP: {ap:.3f}")
    print(classification_report(tgts, preds))

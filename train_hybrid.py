import argparse
import time
import datetime
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from src.dataset.trans.data import *
from src.dataset.loader import *
from src.model.basenet import *
from src.model.baselines import *
from src.model.models import *
from src.transform.preprocess import *
from src.utils import *
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, subsample_and_balance
from pathlib import Path
from torch.utils.data import DataLoader
import wandb


def get_args():
    parser = argparse.ArgumentParser(description='Train hybrid model')
    parser.add_argument('--jaad', default=True, action='store_true',
                        help='use JAAD dataset')
    parser.add_argument('--pie', default=False, action='store_true',
                        help='use PIE dataset')
    parser.add_argument('--titan', default=False, action='store_true',
                        help='use TITAN dataset')
    parser.add_argument('--mode', default='GO', type=str,
                        help='transition mode, GO or STOP')
    parser.add_argument('--fps', default=5, type=int,
                        metavar='FPS', help='sampling rate(fps)')
    parser.add_argument('--max-frames', default=5, type=int,
                        help='maximum number of frames in histroy sequence')
    parser.add_argument('--pred', default=10, type=int,
                        help='prediction length, predicting-ahead time')
    parser.add_argument('--balancing-ratio', default=1.0, type=float,
                        help='ratio of balanced instances(1/0)')
    parser.add_argument('--seed', default=99, type=int,
                        help='random seed for sampling')
    parser.add_argument('--jitter-ratio', default=-1.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('--bbox-min', default=0, type=int,
                        help='minimum bbox size')
    
    parser.add_argument('--encoder-type', default='CC', type=str,
                        help='encoder for images, CC(crop-context) or RC(roi-context)')
    parser.add_argument('--encoder-pretrained', default=False, 
                        help='load pretrained encoder')
    parser.add_argument('--encoder-path', default='', type=str,
                        help='path to encoder checkpoint for loading the pretrained weights')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of epochs to train')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, default=1e-5,
                        help='Weight decay', dest='wd')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    args = parser.parse_args()

    return args


def train_epoch(loader, model, criterion, optimizer, device, epoch):
    encoder_CNN = model['encoder']
    decoder_RNN = model['decoder']

    encoder_CNN.train()
    decoder_RNN.train()
    epoch_loss = 0.0
    n_steps = len(loader)
    for step, inputs in enumerate(tqdm(loader)):
        targets = inputs['label'].to(device, non_blocking=True)
        images = inputs['image'].to(device, non_blocking=True)
        bboxes_ped = inputs['bbox_ped']
        seq_len = inputs['seq_length']
        behavior = inputs['behavior'].to(device, non_blocking=True)
        scene = inputs['attributes'].to(device, non_blocking=True)
        bbox_ped_list = reshape_bbox(bboxes_ped, device)
        pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)

        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
        loss = criterion(outputs_RNN, targets.view(-1, 1))
        # record loss
        optimizer.zero_grad()
        curr_loss = loss.item()
        wandb.log({'train/loss': curr_loss}, step=epoch * n_steps + step)
        epoch_loss += curr_loss
        # compute gradient and do SGD step, scheduler step
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)


@torch.no_grad()
def val_epoch(loader, model, criterion, device, epoch):
    # swith to evaluate mode
    encoder_CNN = model['encoder']
    decoder_RNN = model['decoder']
    # freeze CNN-encoder 
    encoder_CNN.eval()
    decoder_RNN.eval()
    epoch_loss = 0.0
    n_p, n_n = 0.0, 0.0
    n_tp, n_tn = 0.0, 0.0

    y_true = []
    y_pred = []

    n_steps = len(loader)
    for step, inputs in enumerate(tqdm(loader)):
        targets = inputs['label'].to(device, non_blocking=True)
        images = inputs['image'].to(device, non_blocking=True)
        bboxes_ped = inputs['bbox_ped']
        seq_len = inputs['seq_length']
        behavior = inputs['behavior'].to(device, non_blocking=True)
        scene = inputs['attributes'].to(device, non_blocking=True)
        bbox_ped_list = reshape_bbox(bboxes_ped, device)
        pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)

        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)

        loss = criterion(outputs_RNN, targets.view(-1, 1))
        curr_loss = loss.item()
        wandb.log({'val/loss': curr_loss}, step=epoch * n_steps + step)
        epoch_loss += curr_loss
        for j in range(targets.size()[0]):
            y_true.append(int(targets[j].item()))
            y_pred.append(float(outputs_RNN[j].item()))
            if targets[j]:
                n_p += 1
                if outputs_RNN[j] >= 0.5:
                    n_tp += 1
            else:
                n_n += 1
                if outputs_RNN[j] < 0.5:
                    n_tn += 1

    AP_P = average_precision_score(y_true, y_pred)
    FP = n_n - n_tn
    precision_P = n_tp / (n_tp + FP) if n_tp + FP > 0 else 0.0
    recall_P = n_tp / n_p
    f1_p = 2 * (precision_P * recall_P) / (precision_P + recall_P) if precision_P + recall_P > 0 else 0.0
    wandb.log({'val/precision': precision_P , 'val/recall': recall_P, 'val/f1': f1_p, 'val/AP': AP_P})
    
    print('------------------------------------------------')
    print(f'precision: {precision_P}')
    print(f'recall: {n_tp / n_p}')
    print(f'F1-score : {f1_p}')
    print(f"average precision for transition prediction: {AP_P}")
    print('\n')
    val_score = AP_P

    return epoch_loss / len(loader), val_score


def main():
    args = get_args()
    wandb.init(
        project="dlav-intention-prediction",
        config=args,
    )
    run_name = wandb.run.name
    # loading data
    print('Start annotation loading -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    print('------------------------------------------------------------------')
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)

    train_intent_sequences = build_pedb_dataset_jaad(anns_paths["JAAD"]["anns"], anns_paths["JAAD"]["split"], image_set = "train", fps=args.fps,prediction_frames=args.pred, verbose=True)
    train_intent_sequences_cropped = subsample_and_balance(train_intent_sequences,balance=True, max_frames=args.max_frames,seed=args.seed)
    
    val_intent_sequences = build_pedb_dataset_jaad(anns_paths["JAAD"]["anns"], anns_paths["JAAD"]["split"], image_set = "val", fps=args.fps,prediction_frames=args.pred, verbose=True)
    val_intent_sequences_cropped = subsample_and_balance(val_intent_sequences,balance=True, max_frames=args.max_frames,seed=args.seed)
    
    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')
    # construct and load model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_res18 = build_encoder_res18(args)
    # freeze CNN-encoder during training
    encoder_res18.freeze_backbone()

    decoder_lstm = DecoderRNN_IMBS(CNN_embeded_size=256, h_RNN_0=256, h_RNN_1=64, h_RNN_2=16,
                                    h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2).to(device)
    model_gpu = {'encoder': encoder_res18, 'decoder': decoder_lstm}
    # training settings
    criterion = torch.nn.BCELoss().to(device)
    crnn_params = list(encoder_res18.fc.parameters()) + list(decoder_lstm.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    start_epoch = 0
    end_epoch = start_epoch + args.epochs
    # start training
    jitter_ratio = None if args.jitter_ratio < 0 else args.jitter_ratio
    crop_preprocess = CropBox(size=224, padding_mode='pad_resize', jitter_ratio=jitter_ratio)
    TRAIN_TRANSFORM = Compose([crop_preprocess,
                               ImageTransform(torchvision.transforms.ColorJitter(
                                   brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
                               ])
    VAL_TRANSFORM = crop_preprocess
    train_ds = IntentionSequenceDataset(train_intent_sequences_cropped, image_dir=image_dir, hflip_p = 0.5, preprocess=TRAIN_TRANSFORM)
    val_ds = IntentionSequenceDataset(val_intent_sequences_cropped, image_dir=image_dir, hflip_p = 0.5, preprocess=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    ds = 'JAAD'
    print(f'train loader : {len(train_loader)}')
    print(f'val loader : {len(val_loader)}')
    total_time = 0.0
    ap_min = 0.5
    print(f'Start training, PVIBS-lstm-model, neg_in_trans, initail lr={args.lr}, weight-decay={args.wd}, mf={args.max_frames}, training batch size={args.batch_size}')
    if args.output is None:
        cp_dir = Path(f'./checkpoints/{run_name}')
        cp_dir.mkdir(parents=True, exist_ok=True)
        Save_path = f'{cp_dir}/Decoder_IMBS_lr{args.lr}_wd{args.wd}_{ds}_mf{args.max_frames}_pred{args.pred}_bs{args.batch_size}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    else:
        Save_path = args.output
    for epoch in range(start_epoch, end_epoch):
        start_epoch_time = time.time()
        train_loss = train_epoch(train_loader, model_gpu, criterion, optimizer, device, epoch)
        val_loss, val_score = val_epoch(val_loader, model_gpu, criterion, device, epoch)
        scheduler.step(val_score)
        end_epoch_time = time.time() - start_epoch_time
        print('\n', '-----------------------------------------------------')
        print(f'End of epoch {epoch}')
        print('Training epoch loss: {:.4f}'.format(train_loss))
        print('Validation epoch loss: {:.4f}'.format(val_loss))
        print('Validation epoch score: {:.4f}'.format(val_score))
        print('Epoch time: {:.2f}'.format(end_epoch_time))
        print('--------------------------------------------------------', '\n')
        total_time += end_epoch_time
        if val_score > ap_min:
           print('Save model in{}'.format(Save_path))
           save_to_checkpoint(Save_path , epoch, model_gpu['decoder'], optimizer, scheduler, verbose=True)
           ap_min = val_score
        else:
              print('Not save model, since the score is not improved')
    print('\n', '**************************************************************')
    print(f'End training at epoch {end_epoch}')
    print('total time: {:.2f}'.format(total_time))



if __name__ == '__main__':
    print('start')
    main()

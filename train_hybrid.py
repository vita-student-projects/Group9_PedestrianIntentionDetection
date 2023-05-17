import argparse
import time
import datetime
from tqdm import tqdm
from src.dataset.trans.data import *
from src.dataset.loader import *
from src.model.basenet import *
from src.model.baselines import *
from src.model.models import *
from src.transform.preprocess import *
from src.utils import count_parameters
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, subsample_and_balance, unpack_batch
from sklearn.metrics import classification_report, f1_score, average_precision_score
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from src.early_stopping import EarlyStopping


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
    parser.add_argument('--early-stopping-patience', default=3, type=int,)
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
        images, seq_len, pv, scene, behavior, targets = unpack_batch(inputs, device)
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
    encoder_CNN, decoder_RNN = model['encoder'], model['decoder']
    # switch to evaluate mode 
    encoder_CNN.eval()
    decoder_RNN.eval()

    epoch_loss = 0.0
    n_p, n_n = 0.0, 0.0
    n_tp, n_tn = 0.0, 0.0

    y_true = []
    y_pred = []

    n_steps = len(loader)
    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, scene, behavior, targets = unpack_batch(inputs, device)

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


@torch.no_grad()
def eval_model(loader, model, device):
    # swith to evaluate mode
    encoder_CNN, decoder_RNN = model['encoder'], model['decoder']
    encoder_CNN.eval()
    decoder_RNN.eval()

    y_true = []
    y_pred = []
    y_out = []
    scores = []

    for i, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, scene, behavior, targets = unpack_batch(inputs, device)
        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, 
                                    xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
        for j in range(targets.size()[0]):
            y_true.append(int(targets[j].item()))
            y_out.append(float(outputs_RNN[j].item()))
            score = 1.0 - abs(float(outputs_RNN[j].item()) - float(targets[j].item()))
            scores.append(score)
            if outputs_RNN[j] >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
    np_scores = np.array(scores)
    scores_mean = np.mean(np_scores)
    AP  = average_precision_score(y_true, y_out)
    f1 = f1_score(y_true, y_pred)
    wandb.log({'test/AP': AP, 'test/score': scores_mean, 'test/f1': f1})
    print(classification_report(y_true, y_pred))
    print(f"average precision for transition prediction: {AP}")
    return  scores_mean

def prepare_data(anns_paths, image_dir, args, image_set):
    intent_sequences = build_pedb_dataset_jaad(anns_paths["JAAD"]["anns"], anns_paths["JAAD"]["split"], image_set=image_set, fps=args.fps, prediction_frames=args.pred, verbose=True)
    balance = False if image_set == "test" else True
    intent_sequences_cropped = subsample_and_balance(intent_sequences,balance=True, max_frames=args.max_frames, seed=args.seed, balance=balance)

    jitter_ratio = None if args.jitter_ratio < 0 else args.jitter_ratio
    crop_preprocess = CropBox(size=224, padding_mode='pad_resize', jitter_ratio=jitter_ratio)
    if image_set == 'train':
        TRANSFORM = Compose([crop_preprocess,
                               ImageTransform(torchvision.transforms.ColorJitter(
                                   brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
                               ])
    else:
        TRANSFORM = crop_preprocess
    ds = IntentionSequenceDataset(intent_sequences_cropped, image_dir=image_dir, hflip_p = 0.5, preprocess=TRANSFORM)
    return ds


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

    train_ds = prepare_data(anns_paths, image_dir, args, "train")
    val_ds = prepare_data(anns_paths, image_dir, args, "val")
    test_ds = prepare_data(anns_paths, image_dir, args, "test")
    
    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')
    # construct and load model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_res18 = build_encoder_res18(args)
    # freeze CNN-encoder during training
    encoder_res18.freeze_backbone()

    decoder_lstm = DecoderRNN_IMBS(CNN_embeded_size=256, h_RNN_0=256, h_RNN_1=64, h_RNN_2=16,
                                    h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2).to(device)
    
    print(f'Number of trainable parameters: decoder: {count_parameters(decoder_lstm)}, encoder: {count_parameters(encoder_res18)}')
    model = {'encoder': encoder_res18, 'decoder': decoder_lstm}
    # training settings
    criterion = torch.nn.BCELoss().to(device)
    crnn_params = list(encoder_res18.fc.parameters()) + list(decoder_lstm.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    ds = 'JAAD'
    print(f'train loader : {len(train_loader)}')
    print(f'val loader : {len(val_loader)}')
    total_time = 0.0

    print(f'Start training, PVIBS-lstm-model, neg_in_trans, initail lr={args.lr}, weight-decay={args.wd}, mf={args.max_frames}, training batch size={args.batch_size}')
    if args.output is None:
        cp_dir = Path(f'./checkpoints/{run_name}')
        cp_dir.mkdir(parents=True, exist_ok=True)
        save_path = f'{cp_dir}/Decoder_IMBS_lr{args.lr}_wd{args.wd}_{ds}_mf{args.max_frames}_pred{args.pred}_bs{args.batch_size}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pt'
    else:
        save_path = args.output
    early_stopping = EarlyStopping(checkpoint=Path(save_path), patience=args.early_stopping_patience, verbose=True)

    # start training
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_score = val_epoch(val_loader, model, criterion, device, epoch)
        scheduler.step(val_score)
        early_stopping(val_score, model, optimizer, epoch)
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs...')
            break
        end_epoch_time = time.time() - start_epoch_time
        print('\n', '-----------------------------------------------------')
        print(f'End of epoch {epoch}')
        print('Training epoch loss: {:.4f}'.format(train_loss))
        print('Validation epoch loss: {:.4f}'.format(val_loss))
        print('Validation epoch score (AP): {:.4f}'.format(val_score))
        print('Epoch time: {:.2f}'.format(end_epoch_time))
        print('--------------------------------------------------------', '\n')
        total_time += end_epoch_time
    print('\n', '**************************************************************')
    print(f'End training at epoch {epoch}')
    print('total time: {:.2f}'.format(total_time))

    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f'Test loader : {len(test_loader)}')
    print(f'Start evaluation on test set, jitter={args.jitter_ratio}')
    test_score = eval_model(test_loader, model, device)
    print('\n', '-----------------------------------------------------')
    print('----->')
    print('Model Evaluation score: {:.4f}'.format(test_score))
    print('--------------------------------------------------------', '\n')



if __name__ == '__main__':
    print('start')
    main()

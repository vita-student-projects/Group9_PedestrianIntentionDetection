import torch
import torchvision
import argparse
import time
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_score, recall_score
from src.dataset.loader import IntentionSequenceDataset
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, balance, unpack_batch
from src.transform.preprocess import ImageTransform, CropBoxWithBackgroud, Compose
from src.model.models import CRNNClassifier
from src.dataset.utils import build_dataloaders
from src.utils import count_parameters, find_best_threshold, seed_torch, setup_wandb, log_metrics, prepare_cp_path, log_to_stdout
from src.early_stopping import EarlyStopping, load_from_checkpoint
from src.utils import log_metrics, prep_pred_storage, print_eval_metrics

POSITION_VELOCITY_DIM = 8

def get_args():
    parser = argparse.ArgumentParser(description='Train hybrid model')
    parser.add_argument('--jaad', default=True, action='store_true',
                        help='use JAAD dataset')
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
    parser.add_argument('--cnn-encoder-path', default='', type=str,
                        help='path to image encoder checkpoint for loading the pretrained weights')
    parser.add_argument('--rnn-decoder-path', default='', type=str,
                        help='path to position velocity checkpoint for loading the pretrained weights')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of epochs to train')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, default=1e-4,
                        help='Weight decay', dest='wd')
    parser.add_argument('--early-stopping-patience', default=3, type=int,)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument('-nw', '--num-workers', default=4, type=int, help='number of workers for data loading')
    args = parser.parse_args()

    return args


def train_epoch(loader, model, criterion, optimizer, device, epoch):
    crnn_model = model['crnn']
    crnn_model.train()
    crnn_model.cnn_encoder.backbone.eval()

    epoch_loss = 0.0
    preds, tgts, n_steps, batch_size = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, _, _, targets = unpack_batch(inputs, device)
        outputs_crnn = crnn_model(images, pv, seq_len)
        loss = criterion(outputs_crnn, targets.view(-1, 1))

        preds[step * batch_size: (step + 1) * batch_size] = outputs_crnn.detach().cpu().squeeze()
        tgts[step * batch_size: (step + 1) * batch_size] = targets.detach().cpu().squeeze()

        # record loss
        optimizer.zero_grad()
        curr_loss = loss.item()
        epoch_loss += curr_loss
        loss.backward()
        optimizer.step()

    epoch_loss /= n_steps
    wandb.log({'train/loss': epoch_loss, 'train/epoch': epoch + 1}, commit=True)
    train_score = average_precision_score(tgts, preds)
    best_thr = model['best_thr']
    f1 = f1_score(tgts, preds > best_thr)
    log_metrics(tgts, preds, best_thr, f1, train_score, 'train', epoch + 1)

    return epoch_loss


@torch.no_grad()
def val_epoch(loader, model, criterion, device, epoch):
    crnn_model = model['crnn']
    crnn_model.eval()

    epoch_loss = 0.0
    preds, tgts, n_steps, batch_size = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, _, _, targets = unpack_batch(inputs, device)
        outputs_crnn = crnn_model(images, pv, seq_len)
        loss = criterion(outputs_crnn, targets.view(-1, 1))

        preds[step * batch_size: (step + 1) * batch_size] = outputs_crnn.detach().cpu().squeeze()
        tgts[step * batch_size: (step + 1) * batch_size] = targets.detach().cpu().squeeze()

        loss = criterion(outputs_crnn, targets.view(-1, 1))
        curr_loss = loss.item()
        epoch_loss += curr_loss

    epoch_loss /= n_steps
    wandb.log({'val/loss': epoch_loss, 'val/epoch': epoch + 1})
    best_thr, best_f1 = find_best_threshold(preds, tgts)
    model['best_thr'] = best_thr

    val_score = average_precision_score(tgts, preds)
    log_metrics(tgts, preds, best_thr, best_f1, val_score, 'val', epoch + 1)

    return epoch_loss, best_f1


@torch.no_grad()
def eval_model(loader, model, device):
    crnn_model = model['crnn']
    crnn_model.eval()

    preds, tgts, _, batch_size = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, _, _, targets = unpack_batch(inputs, device)
        outputs_crnn = crnn_model(images, pv, seq_len)
        
        preds[step * batch_size: (step + 1) * batch_size] = outputs_crnn.detach().cpu().squeeze()
        tgts[step * batch_size: (step + 1) * batch_size] = targets.detach().cpu().squeeze()

    train_score = average_precision_score(tgts, preds)
    best_thr = model['best_thr']
    f1 = f1_score(tgts, preds > best_thr)
    log_metrics(tgts, preds, best_thr, f1, train_score, 'test', 0)
    preds = preds > best_thr
    print(classification_report(tgts, preds))



def prepare_data(anns_paths, image_dir, args, image_set, load_image=True):
    MEAN = [0.3104, 0.2813, 0.2973]
    STD = [0.1761, 0.1722, 0.1673]

    intent_sequences = build_pedb_dataset_jaad(anns_paths["JAAD"]["anns"], anns_paths["JAAD"]["split"], image_set=image_set, fps=args.fps, prediction_frames=args.pred, verbose=True)
    if not image_set == "test":
        intent_sequences = balance(intent_sequences, seed=args.seed)

    crop_with_background = CropBoxWithBackgroud(size=224)
    if image_set == 'train':
        TRANSFORM = Compose([
                             crop_with_background,
                             ImageTransform(
                                 torchvision.transforms.Compose([
                                     torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(MEAN, STD),
                                 ]),
                             ),
                           ])
    else:
        TRANSFORM = Compose([
                             crop_with_background,
                             ImageTransform(
                                 torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(), 
                                     torchvision.transforms.Normalize(MEAN, STD),
                                 ]),
                             ) 
                            ])
    ds = IntentionSequenceDataset(intent_sequences, image_dir=image_dir, hflip_p = 0.5, preprocess=TRANSFORM)
    return ds


def main():
    args = get_args()
    seed_torch(args.seed)

    run_mode = "crnn"
    run_name = setup_wandb(args, run_mode)

    # loading data
    train_loader, val_loader, test_loader = build_dataloaders(args, prepare_data, load_image=False)
    
    # construct and load model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNClassifier(pos_vel_embedding_size=POSITION_VELOCITY_DIM, cnn_embedding_size=256, rnn_embeding_size=256, classification_head_size=128).to(device)
    model.from_pretrained(args.cnn_encoder_path, args.rnn_decoder_path)

    print(f'Number of model parameters: {count_parameters(model)}')
    model.cnn_encoder.freeze_backbone()
    print(f'Number of trainable parameters:  {count_parameters(model)}')

    model = {'crnn': model, 'best_thr': 0.5}
    # training settings
    criterion = torch.nn.BCELoss().to(device)
    crnn_params = list(model['crnn'].parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    print(f'train loader : {len(train_loader)}')
    print(f'val loader : {len(val_loader)}')

    total_time = 0.0

    print(f'Start training, PVIBS-lstm-model, neg_in_trans, initail lr={args.lr}, weight-decay={args.wd}, mf={args.max_frames}, training batch size={args.batch_size}')
    save_path = prepare_cp_path(args, run_name, run_mode)
    early_stopping = EarlyStopping(checkpoint=save_path, patience=args.early_stopping_patience, verbose=True)

    # start training
    best_f1 = 0.0
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_f1 = val_epoch(val_loader, model, criterion, device, epoch)
        best_f1 = max(best_f1, val_f1)
        scheduler.step(val_f1)
        early_stopping(val_f1, model, optimizer, epoch)
        wandb.log({"val/best_f1": best_f1, "val/epoch": epoch})
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs...')
            break
        end_epoch_time = time.time() - start_epoch_time
        log_to_stdout(epoch, train_loss, val_loss, val_f1, end_epoch_time)
        total_time += end_epoch_time

    print('\n', '**************************************************************')
    print(f'End training at epoch {epoch}')
    print('total time: {:.2f}'.format(total_time))

    load_from_checkpoint(model, save_path)
    print(f'Start evaluation on test set')
    eval_model(test_loader, model, device)


if __name__ == '__main__':
    print('start')
    main()
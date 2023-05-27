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
from src.utils import count_parameters, find_best_threshold, seed_torch
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, subsample_and_balance, unpack_batch
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_score, recall_score
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from src.early_stopping import EarlyStopping, load_from_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='Train hybrid model')
    parser.add_argument('--jaad', default=True, action='store_true',
                        help='use JAAD dataset')
    parser.add_argument('--pie', default=False, action='store_true',
                        help='use PIE dataset')
    parser.add_argument('--titan', default=False, action='store_true',
                        help='use TITAN dataset')
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
    parser.add_argument('--mobilenetsmall', default=False, action='store_true',
                        help='use mobilenet small or not')
    parser.add_argument('--mobilenetbig', default=False, action='store_true',
                        help='use mobilenet big or not')
    parser.add_argument('-nw', '--num-workers', default=4, type=int, help='number of workers for data loading')
    args = parser.parse_args()

    return args



@torch.no_grad()
def eval_model(loader, model, device):
    # swith to evaluate mode
    encoder_CNN, decoder_RNN = model['encoder'], model['decoder']
    encoder_CNN.eval()
    decoder_RNN.eval()


    batch_size = loader.batch_size
    n_steps = len(loader)

    preds = np.zeros(n_steps * batch_size)
    tgts = np.zeros(n_steps * batch_size)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, scene, behavior, targets = unpack_batch(inputs, device)
        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, 
                                    xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
        
        preds[step * batch_size: (step + 1) * batch_size] = outputs_RNN.detach().cpu().squeeze()
        tgts[step * batch_size: (step + 1) * batch_size] = targets.detach().cpu().squeeze()

    train_score = average_precision_score(tgts, preds)
    best_thr = decoder_RNN.threshold
    f1 = f1_score(tgts, preds > best_thr)
    log_metrics(tgts, preds, best_thr, f1, train_score, 'test', 0)
    preds = preds > best_thr
    print(classification_report(tgts, preds))

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


def prepare_data(anns_paths, image_dir, args, image_set):
    intent_sequences = build_pedb_dataset_jaad(anns_paths["JAAD"]["anns"], anns_paths["JAAD"]["split"], image_set=image_set, fps=args.fps, prediction_frames=args.pred, verbose=True)
    balance = False if image_set == "test" else True
    intent_sequences_cropped = subsample_and_balance(intent_sequences, max_frames=args.max_frames, seed=args.seed, balance=balance)

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
    seed_torch(args.seed)

    wandb.init(
        project="dlav-intention-prediction",
        config=args,
    )
    run_name = wandb.run.name




    # define our custom x axis metric
    for setup in ['train', 'val']:
        wandb.define_metric(f"{setup}/epoch")
        wandb.define_metric(f"{setup}/*", step_metric=f"{setup}/epoch")

    # loading data
    print('Start annotation loading -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    print('------------------------------------------------------------------')
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)

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

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    save_path=r'/home/cchang/CIVIL459-PedestrianIntensionDetection/checkpoints/deep-sweep-7/Decoder_IMBS_lr0.001_wd1e-05_JAAD_mf5_pred10_bs16_202305190306.pt'
    load_from_checkpoint(model, save_path,device)
    print(f'Test loader : {len(test_loader)}')
    print(f'Start evaluation on test set, jitter={args.jitter_ratio}')
    eval_model(test_loader, model, device)


if __name__ == '__main__':
    print('start')
    main()

import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_score, recall_score
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, subsample_and_balance, unpack_batch
from src.early_stopping import load_from_checkpoint
from src.model.models import Res18Classifier
from src.dataset.loader import define_path, IntentionSequenceDataset
from src.transform.preprocess import ImageTransform, Compose, CropBoxWithBackgroud
from src.utils import prep_pred_storage, print_eval_metrics
import torchvision.transforms as transforms

MEAN = [0.3104, 0.2813, 0.2973]
STD = [0.1761, 0.1722, 0.1673]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='cropped frame model evaluation')

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
    parser.add_argument('--jitter-ratio', default=-1.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('-s', '--seed', type=int, default=99,
                        help='set random seed for sampling')
    parser.add_argument('--checkpoint-path', type=str,
                        help='path to the checkpoint for loading pretrained weights')
    args = parser.parse_args()

    return args


def build_loader(args, intent_seqs, TRANSFORM, image_dir):
    dataset = subsample_and_balance(
        intent_seqs,
        balance=False, 
        max_frames=args.max_frames, 
        seed=args.seed)
    ds = IntentionSequenceDataset(dataset, image_dir=image_dir, hflip_p = 0, preprocess=TRANSFORM)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    return loader

    
@torch.no_grad()
def eval_cnn(loader, model, device):
    # swith to evaluate mode
    encoder_CNN = model['encoder']
    preds, tgts, _, _ = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, _, _, _, targets = unpack_batch(inputs, device)
        outputs_CNN = encoder_CNN(images, seq_len).squeeze(-1)
        
        preds[step] = outputs_CNN.detach().cpu().squeeze()
        tgts[step] = targets.detach().cpu().squeeze()

    print_eval_metrics(preds, tgts, encoder_CNN.best_threshold)


def main():
    args = get_args()
    # loading data
    anns_paths_eval, image_dir_eval = define_path(use_jaad=args.jaad, use_pie=False, use_titan=False)

    normal_intent_sequences = build_pedb_dataset_jaad(
        anns_paths_eval["JAAD"]["anns"], 
        anns_paths_eval["JAAD"]["split"], 
        image_set = "test", 
        fps=args.fps,
        prediction_frames=args.pred, 
        verbose=True)

    hard_intent_sequences = build_pedb_dataset_jaad(
        anns_paths_eval["JAAD"]["anns"], 
        anns_paths_eval["JAAD"]["split"], 
        image_set = "test", 
        fps=args.fps,
        prediction_frames=args.pred, 
        verbose=True,
        transition_only=True)

    # load model

    if args.mode == 'CNN_only':

        encoder_res18 = Res18Classifier(CNN_embed_dim=256, activation="sigmoid").to(device)
        encoder_res18.eval()
        model = {'encoder': encoder_res18}
        load_from_checkpoint(model, args.checkpoint_path)

        crop_with_background = CropBoxWithBackgroud(size=224)
        TRANSFORM = Compose([
            crop_with_background,
            ImageTransform(
                transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(MEAN, STD),
                ]),
            ) 
        ])

    normal_loader = build_loader(args, normal_intent_sequences, TRANSFORM, image_dir_eval)
    hard_loader = build_loader(args, hard_intent_sequences, TRANSFORM, image_dir_eval)

    print(f'Normal test loader : {len(normal_loader)}, Hard (transition only) test loader : {len(hard_loader)}')
    print(f'Evaluation on full test set')
    eval_cnn(normal_loader, model, device)
    print(f'Evaluation on transition only test set')
    eval_cnn(hard_loader, model, device)
    

if __name__ == '__main__':
    main()

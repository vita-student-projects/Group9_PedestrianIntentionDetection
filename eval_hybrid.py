import argparse
import torch
from tqdm import tqdm
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, unpack_batch
from src.early_stopping import load_from_checkpoint
from src.model.models import Res18Classifier, RNNClassifier, DecoderRNN_IMBS, build_encoder_res18
from src.dataset.loader import define_path, IntentionSequenceDataset
from src.transform.preprocess import ImageTransform, Compose, CropBoxWithBackgroud
from src.utils import prep_pred_storage, print_eval_metrics
import torchvision.transforms as transforms

MEAN = [0.3104, 0.2813, 0.2973]
STD = [0.1761, 0.1722, 0.1673]

EMBEDDING_DIM = 256
EVAL_MODES = ['cnn_only', 'rnn_only', 'hybrid']

IMAGE_TRANSFORM = Compose([
    CropBoxWithBackgroud(size=224),
    ImageTransform(
        transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(MEAN, STD),
        ]),
        ) 
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='cropped frame model evaluation')

    parser.add_argument('--jaad', default=True, action='store_true',
                        help='use JAAD dataset')
    parser.add_argument('--fps', default=5, type=int,
                        metavar='FPS', help='sampling rate(fps)')
    parser.add_argument('--max-frames', default=5, type=int,
                        help='maximum number of frames in histroy sequence')
    parser.add_argument('--pred', default=5, type=int,
                        help='prediction length, predicting-ahead time')
    parser.add_argument('--balancing-ratio', default=1.0, type=float,
                        help='ratio of balanced instances(1/0)')
    parser.add_argument('--jitter-ratio', default=-1.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('-s', '--seed', type=int, default=99,
                        help='set random seed for sampling')
    parser.add_argument('-cp', '--checkpoint-path', type=str,
                        help='path to the checkpoint for loading pretrained weights')
    parser.add_argument('-nw', '--num-workers', type=int, default=4, 
                        help='number of workers for data loading')
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    return args


def build_loader(args, intent_seqs, TRANSFORM, image_dir, load_image=True):
    ds = IntentionSequenceDataset(intent_seqs, image_dir=image_dir, hflip_p = 0, preprocess=TRANSFORM, load_image=load_image)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.num_workers, shuffle=False)
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

    print_eval_metrics(tgts, preds, model['best_thr'])


@torch.no_grad()
def eval_rnn(loader, model, device):
    # swith to evaluate mode
    decoder_RNN = model['decoder']
    preds, tgts, _, _ = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        _, seq_len, pos_vel, _, _, targets = unpack_batch(inputs, device)
        outputs_CNN = decoder_RNN(pos_vel, seq_len).squeeze(-1)
        
        preds[step] = outputs_CNN.detach().cpu().squeeze()
        tgts[step] = targets.detach().cpu().squeeze()

    print_eval_metrics(tgts, preds, model['best_thr'])


@torch.no_grad()
def eval_hybrid(loader, model, device):
    encoder_CNN, decoder_RNN = model['encoder'], model['decoder']

    preds, tgts, _, _ = prep_pred_storage(loader)

    for step, inputs in enumerate(tqdm(loader)):
        images, seq_len, pv, scene, behavior, targets = unpack_batch(inputs, device)
        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, 
                                    xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
        
        preds[step] = outputs_RNN.detach().cpu().squeeze()
        tgts[step] = targets.detach().cpu().squeeze()

    print_eval_metrics(tgts, preds, model['best_thr'])


EVAL_FUNCTIONS = {'cnn_only': eval_cnn, 'rnn_only': eval_rnn, 'hybrid': eval_hybrid}

def main():
    args = get_args()
    if args.mode not in EVAL_MODES:
        raise ValueError(f'invalid mode, please choose from {", ".join(EVAL_MODES)}')
    # loading data
    anns_paths_eval, image_dir_eval = define_path(use_jaad=args.jaad, use_pie=False, use_titan=False)

    normal_intent_sequences = build_pedb_dataset_jaad(
        anns_paths_eval["JAAD"]["anns"], 
        anns_paths_eval["JAAD"]["split"], 
        image_set = "test", 
        fps=args.fps,
        prediction_frames=args.pred, 
        max_frames=args.max_frames,
        verbose=True)

    hard_intent_sequences = build_pedb_dataset_jaad(
        anns_paths_eval["JAAD"]["anns"], 
        anns_paths_eval["JAAD"]["split"], 
        image_set = "test", 
        fps=args.fps,
        prediction_frames=args.pred, 
        max_frames=args.max_frames,
        verbose=True,
        transition_only=True)
    

    # load model

    if args.mode == 'cnn_only':

        encoder_res18 = Res18Classifier(CNN_embed_dim=EMBEDDING_DIM, activation="sigmoid").to(device)
        encoder_res18.eval()
        model = {'encoder': encoder_res18}
        transform, load_image  = IMAGE_TRANSFORM, True
    
    elif args.mode == 'rnn_only':

        POS_VEL_DIM = 8
        rnn_classifier = RNNClassifier(input_size=POS_VEL_DIM, rnn_embeding_size=EMBEDDING_DIM, classification_head_size=128).to(device)
        model = {'decoder': rnn_classifier}
        transform, load_image = None, False

    elif args.mode == 'hybrid':
        encoder_CNN = build_encoder_res18(args)
        decoder_RNN = DecoderRNN_IMBS(CNN_embeded_size=256, h_RNN_0=256, h_RNN_1=64, h_RNN_2=16,
                                    h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2).to(device)
        encoder_CNN.eval()
        decoder_RNN.eval()
        model = {'encoder': encoder_CNN, 'decoder': decoder_RNN}

        transform, load_image = IMAGE_TRANSFORM, True

    load_from_checkpoint(model, args.checkpoint_path)    

    normal_loader = build_loader(args, normal_intent_sequences, transform, image_dir_eval, load_image=load_image)
    hard_loader = build_loader(args, hard_intent_sequences, transform, image_dir_eval, load_image=load_image)

    eval_function = EVAL_FUNCTIONS[args.mode]

    print(f'Normal test loader : {len(normal_loader)}, Hard (transition only) test loader : {len(hard_loader)}')
    print(f'Evaluation on full test set')
    eval_function(normal_loader, model, device)
    print(f'Evaluation on transition only test set')
    eval_function(hard_loader, model, device)
    

if __name__ == '__main__':
    main()

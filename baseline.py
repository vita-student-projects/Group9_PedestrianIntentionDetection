from src.dataset.loader import define_path
from src.dataset.intention.jaad_dataset import build_pedb_dataset_jaad, subsample_and_balance
from sklearn.metrics import average_precision_score, classification_report, f1_score
from collections import defaultdict
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='cropped frame model Training')

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
    parser.add_argument('-s', '--seed', type=int, default=99,
                        help='set random seed for sampling')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    # loading data
    print('Annotation loading-->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    print('------------------------------------------------------------------')
    anns_paths_eval, _ = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    print('-->>')
    eval_intent_sequences = build_pedb_dataset_jaad(anns_paths_eval["JAAD"]["anns"], anns_paths_eval["JAAD"]["split"], image_set = "test", fps=args.fps, prediction_frames=args.pred, verbose=True)
    pred_intent_sequences = build_pedb_dataset_jaad(anns_paths_eval["JAAD"]["anns"], anns_paths_eval["JAAD"]["split"], image_set = "test", fps=args.fps, prediction_frames=0, verbose=True)

    eval_intent_sequences_cropped = subsample_and_balance(eval_intent_sequences, balance=False, max_frames=args.max_frames, seed=args.seed)
    pred_intent_sequences_cropped = subsample_and_balance(pred_intent_sequences, balance=False, max_frames=args.max_frames, seed=args.seed)

    eval_samples = defaultdict(dict)
    for intent in eval_intent_sequences_cropped:
        sample_id = intent['sample_id']
        label = intent['label']
        eval_samples[sample_id]['label'] = label
    
    for intent in pred_intent_sequences_cropped:
        sample_id = intent['sample_id']
        if sample_id not in eval_samples:
            continue
        label = intent['label']
        eval_samples[sample_id]['pred'] = label

    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')

    y_true = []
    y_pred = []

    for el in eval_samples.values():
        y_true.append(el['label'])
        y_pred.append(el['pred'])

    print(classification_report(y_true, y_pred))   
    print('------------------------------------------------------------------')
    print(f'f1 score: {f1_score(y_true, y_pred):.3f}')
    

if __name__ == '__main__':
    print('Baseline evaluation')
    main()

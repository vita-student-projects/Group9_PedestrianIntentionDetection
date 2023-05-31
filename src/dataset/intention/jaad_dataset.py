import numpy as np
import torch
import pickle
import copy
import random
from src.dataset.trans.jaad_trans import get_split_vids, get_pedb_ids_jaad
from collections import Counter
from src.utils import reshape_bbox, bbox_to_pv
import torch
from src.dataset.loader import IntentionSequenceDataset
from src.transform.preprocess import ImageTransform, Compose, ResizeFrame
import torchvision



JAAD_BASE_FPS = 30
MAX_FRAMES = 5
PREDICTION_FRAMES = 5
SEED = 42

def get_pedb_info_jaad(annotations, vid):
    """
    Get pedb information,i.e. frames,bbox,occlusion, actions(walking or not),cross behavior.
    :param: annotations: JAAD annotations in dictionary form
            vid : single video id (str)
    :return: information of all pedestrians in one video
    """
    ids = get_pedb_ids_jaad(annotations, vid)
    dataset = annotations
    pedb_info = {}
    for idx in ids:
        pedb_info[idx] = {}
        pedb_info[idx]['frames'] = []
        pedb_info[idx]['bbox'] = []
        pedb_info[idx]['occlusion'] = []
        pedb_info[idx]['action'] = []
        pedb_info[idx]['cross'] = []
        # process atomic behavior label
        pedb_info[idx]['behavior'] = []
        pedb_info[idx]['traffic_light'] = []

        frames = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['frames'])
        bbox = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['bbox'])
        occlusion = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['occlusion'])

        # behavioural data
        action = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['action'])
        cross = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['cross'])
        nod = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['nod'])
        look = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['look'])
        hand_gesture = copy.deepcopy(dataset[vid]['ped_annotations'][idx]['behavior']['hand_gesture'])


        for i in range(len(frames)):
            if action[i] in [0, 1]:  # sanity check if behavior label exists
                # standing / walking
                pedb_info[idx]['action'].append(action[i])
                pedb_info[idx]['frames'].append(frames[i])
                pedb_info[idx]['bbox'].append(bbox[i])
                pedb_info[idx]['occlusion'].append(occlusion[i])
                pedb_info[idx]['cross'].append(cross[i])
                beh_vec = [0, 0, 0, 0]
                # change for testing the influence of walking/non-walking to training
                # beh_vec[0] = action[i]
                beh_vec[1] = look[i]
                beh_vec[2] = nod[i]
                # TODO: maybe include it as a category?
                hg = hand_gesture[i]
                if hg > 0:
                    beh_vec[3] = 1
                pedb_info[idx]['behavior'].append(beh_vec)
                # traffic light: {'n/a': 0, 'red': 1, 'green': 2}
                pedb_info[idx]['traffic_light'].append(dataset[vid]['traffic_annotations'][frames[i]]['traffic_light'])

        # attribute vector
        # scene description
        atr_vec = [0, 0, 0, 0, 0]
        #atr_vec[0] = dataset[vid]['ped_annotations'][idx]['attributes']['num_lanes']
        #atr_vec[1] = dataset[vid]['ped_annotations'][idx]['attributes']['intersection']
        #atr_vec[2] = dataset[vid]['ped_annotations'][idx]['attributes']['designated']
        # TODO: why always 1? 'signalized': {'n/a': 0, 'NS': 1, 'S': 2},
        #if dataset[vid]['ped_annotations'][idx]['attributes']['signalized'] > 0:
        #atr_vec[3] = dataset[vid]['ped_annotations'][idx]['attributes']['signalized'] 
        # 'traffic_direction': {'OW': 0, 'TW': 1},
        #atr_vec[4] = dataset[vid]['ped_annotations'][idx]['attributes']['traffic_direction']
        pedb_info[idx]['attributes'] = copy.deepcopy(atr_vec)
    return pedb_info


def add_cross_label_jaad(dataset, prediction_frames, max_frames, verbose=False, transition_only=False, seed=99) -> None:
    """
    Add cross & non-cross(c/nc) labels depends on prediction frame for every frame
    """
    all_cross = 0
    total_samples = 0
    length_filtered, transition_filtered = 0, 0
    pids = list(dataset.keys())
    new_samples = []
    for idx in pids:
        frames = dataset[idx]['frames']
        total_frames = len(frames)
        dataset[idx]['labels'] = []
        dataset[idx]['curr_labels'] = []
        if len(frames) <= prediction_frames:
            length_filtered += 1
            continue
        # taking all sequences that have max_frames of past and prediction_frames of future
        for i, j in enumerate(range(max_frames - 1, total_frames - prediction_frames - 1)):
            if transition_only:
                if dataset[idx]['cross'][j] == dataset[idx]['cross'][j + prediction_frames]:
                    transition_filtered += 1
                    continue

            new_id = f"{idx}_{dataset[idx]['video_number']}_{i}"
            new_sample = {'sample_id': new_id, 'ped_id': idx}
            for attribute in ['frames', 'bbox', 'action', 'occlusion', 'behavior', 'traffic_light']:
                new_sample[attribute] = dataset[idx][attribute][i:j + 1]
            new_sample['label'] = dataset[idx]['cross'][j + prediction_frames]
            all_cross += new_sample['label']
            for static_attribute in ['video_number', 'attributes']:
                new_sample[static_attribute] = dataset[idx][static_attribute]
            new_samples.append(new_sample)
            total_samples += 1

        # cut last prediction_frames frames
        for attribute in ['frames', 'bbox', 'action', 'occlusion', 'behavior', 'traffic_light']:
            if prediction_frames > 0:
                dataset[idx][attribute] = dataset[idx][attribute][:-prediction_frames]
        dataset[idx].pop('cross') 

    if verbose:
        print('----------------------------------------------------------------')
        print("JAAD:")
        print(f'Total number of crosses: {all_cross}')
        print(f'Total number of non-crosses: {total_samples - all_cross}')
        print(f'Filtered samples: {length_filtered + transition_filtered}, out of them: {length_filtered} due to length, {transition_filtered} due to lack of transition')
    
    random.seed(seed)
    random.shuffle(new_samples)
    return new_samples


def build_pedb_dataset_jaad(jaad_anns_path, 
                            split_vids_path, image_set="all", 
                            subset='default', fps=JAAD_BASE_FPS,  
                            prediction_frames=PREDICTION_FRAMES, 
                            max_frames=MAX_FRAMES,
                            verbose=False, 
                            transition_only=False) -> dict:
    """
    Build pedestrian dataset from jaad annotations
    """
    jaad_anns = pickle.load(open(jaad_anns_path, 'rb'))
    pedb_dataset = {}
    vids = get_split_vids(split_vids_path, image_set, subset)
    fps_step = JAAD_BASE_FPS // fps
    for vid in vids:
        pedb_info = get_pedb_info_jaad(jaad_anns, vid)
        pids = list(pedb_info.keys())
        for idx in pids:
            if len(pedb_info[idx]['action']) > 0:
                pedb_dataset[idx] = {}
                pedb_dataset[idx]['video_number'] = vid
                for attribute in ['frames', 'bbox', 'action', 'occlusion', 'cross', 'behavior', 'traffic_light']:
                    pedb_dataset[idx][attribute] = pedb_info[idx][attribute][::fps_step]
                pedb_dataset[idx]['attributes'] = pedb_info[idx]['attributes']
    intention_seqs = add_cross_label_jaad(pedb_dataset, prediction_frames=prediction_frames, max_frames=max_frames, verbose=verbose, transition_only=transition_only)
    return intention_seqs


def balance(intention_dataset, seed=SEED):
    random.seed(seed)
    all_labels = [el['label'] for el in intention_dataset]
    labels_stats = Counter(all_labels)
    max_common = min(labels_stats[0], labels_stats[1])
    crossing_ids = [i for i, sample in enumerate(intention_dataset) if sample['label'] == 1]
    noncrossing_ids = [i for i, sample in enumerate(intention_dataset) if sample['label'] == 0]
    kept_crossing_ids = random.sample(crossing_ids, max_common)
    kept_noncrossing_ids = random.sample(noncrossing_ids, max_common)
    kept_ids = kept_crossing_ids + kept_noncrossing_ids
    balanced_dataset = [intention_dataset[i] for i in kept_ids]
    print(f"Total number of samples before and after balancing: {len(intention_dataset)}, {len(balanced_dataset)}")
    return balanced_dataset



class JaadIntentionDataset:
    """
     dataset class for transition-related pedestrian samples in JAAD
    """

    def __init__(self, jaad_anns_path, split_vids_path, image_set="all", subset="default", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        self.dataset = build_pedb_dataset_jaad(jaad_anns_path, split_vids_path, image_set, subset, verbose)
        self.pids = list(self.dataset.keys())
        self.name = image_set
        self.subset = subset

    def __repr__(self):
        return f"JaadTransDataset(image_set={self.name}, subset={self.subset})"
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        return self.dataset[pid]


def unpack_batch(batch, device):
    targets = batch['label'].to(device, non_blocking=True)
    images = batch['image'].to(device, non_blocking=True)
    bboxes_ped = batch['bbox_ped']
    seq_len = batch['seq_length']
    behavior = batch['behavior'].to(device, non_blocking=True)
    scene = batch['attributes'].to(device, non_blocking=True)
    bbox_ped_list = reshape_bbox(bboxes_ped, device)
    pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)
    return images, seq_len, pv, scene, behavior, targets
# CIVIL459-Pedestrian Intension Detection

```
annotations['ped_annotations'][new_id]['behavior'] = {'cross': [],
                                                      'reaction': [],
                                                      'hand_gesture': [],
                                                      'look': [],
                                                      'action': [],
                                                      'nod': []}

map_dic = {        'occlusion': {0: 'none', 1: 'part', 2: 'full'},
                   'action': {0: 'standing', 1: 'walking'},
                   'nod': {0: '__undefined__', 1: 'nodding'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'hand_gesture': {0: '__undefined__', 1: 'greet',
                                    2: 'yield', 3: 'rightofway',
                                    4: 'other'},
                   'reaction': {0: '__undefined__', 1: 'clear_path',
                                2: 'speed_up', 3: 'slow_down'},
                   'cross': {0: 'not-crossing', 1: 'crossing', -1: 'irrelevant'},
                   'age': {0: 'child', 1: 'young', 2: 'adult', 3: 'senior'},
                   'designated': {0: 'ND', 1: 'D'},
                   'gender': {0: 'n/a', 1: 'female', 2: 'male'},
                   'intersection': {0: 'no', 1: 'yes'},
                   'motion_direction': {0: 'n/a', 1: 'LAT', 2: 'LONG'},
                   'traffic_direction': {0: 'OW', 1: 'TW'},
                   'signalized': {0: 'n/a', 1: 'NS', 2: 'S'},
                   'vehicle': {0: 'stopped', 1: 'moving_slow', 2: 'moving_fast',
                               3: 'decelerating', 4: 'accelerating'},
                   'road_type': {0: 'street', 1: 'parking_lot', 2: 'garage'},
                   'traffic_light': {0: 'n/a', 1: 'red', 2: 'green'}}
```
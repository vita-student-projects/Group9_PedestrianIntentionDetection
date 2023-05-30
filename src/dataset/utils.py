from src.dataset.loader import define_path
from torch.utils.data import DataLoader

def build_dataloaders(args, prepare_data, **kwargs):
    print('Start annotation loading -->', 'JAAD:')
    print('------------------------------------------------------------------')
    
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=False, use_titan=False)

    train_ds = prepare_data(anns_paths, image_dir, args, "train", **kwargs)
    val_ds = prepare_data(anns_paths, image_dir, args, "val", **kwargs)
    test_ds = prepare_data(anns_paths, image_dir, args, "test", **kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')

    print(f'train loader : {len(train_loader)}, val loader : {len(val_loader)}, test loader : {len(test_loader)}')

    return train_loader, val_loader, test_loader

''' 
    Main script for training segmentation models with pytorch lightning
'''
from segmentation_arguments import get_training_arguments
from segmentation_trainer import SegmentationTrainer
from pytorch_lightning import Trainer

def main():
    args = get_training_arguments()

    # Lightning Trainer can't store None and Arguments has issues with booleans
    args.val_interval = 1 if args.val_interval is None else args.val_interval
    args.gradient_ckpt = True if args.gradient_ckpt == 'True' else False
    args.imagenet = True if args.imagenet == 'True' else False
    mode = 'ddp' if len(args.gpus) > 1  else None
    
    trainer = Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_interval,
        max_epochs=args.num_epochs,
        early_stop_callback=False,
        distributed_backend=mode,
        amp_level=args.amp_level,
        weights_summary=None,
        auto_lr_find=False,
        gpus=args.gpus,
    )

    model = SegmentationTrainer(args)

    trainer.fit(model)

if __name__ == '__main__':
    main()
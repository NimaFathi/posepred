import argparse


class ValidationArgs:
    def __init__(self, dataset_name, model_path, keypoint_dim, is_interactive=False, use_mask=False, distance_loss='L1',
                 skip_frame=0, batch_size=1, shuffle=True, pin_memory=False, num_workers=0):
        # dataloader_args
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.use_mask = use_mask
        self.distance_loss = distance_loss
        self.skip_frame = skip_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.is_testing = False

        self.model_path = model_path


def parse_validation_args():
    parser = argparse.ArgumentParser('Validation Arguments')

    # dataloader_args
    parser.add_argument('--dataset_name', type=str, help='test_dataset_name')
    parser.add_argument('--keypoint_dim', type=int, help='dimension of each keypoint')
    parser.add_argument('--is_interactive', type=bool, default=False, help='support interaction of people')
    parser.add_argument('--use_mask', type=bool, default=False, help='visibility mask')
    parser.add_argument('--distance_loss', type=str, default='L1', help='use L1 or L2 as distance loss.')
    parser.add_argument('--skip_frame', type=int, default=0, help='skip frame in reading dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    parser.add_argument('--model_path', type=str, default=None, help='model_path')

    validation_args = parser.parse_args()
    return validation_args

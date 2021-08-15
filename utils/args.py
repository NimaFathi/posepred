class Dataloader_Args:
    def __init__(self, dataset_name, use_mask, skip_frame, is_multi_person, batch_size, shuffle, pin_memory,
                 num_workers):
        self.dataset_name = dataset_name
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.is_multi_person = is_multi_person
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers

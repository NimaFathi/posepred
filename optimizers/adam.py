import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=trainer_args.lr)
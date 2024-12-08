from planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data
import dense_transforms
import random

def train(args):
    from os import path
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("device:", device)
    
    # Initialize multiple agents
    num_agents = args.num_agents
    agents = [Planner().to(device) for _ in range(num_agents)]
    optimizers = [torch.optim.Adam(agent.parameters(), lr=args.learning_rate) for agent in agents]
    loss_fn = torch.nn.L1Loss()

    if args.continue_training:
        for agent in agents:
            agent.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

    global_step = 0
    for epoch in range(args.num_epoch):
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Each agent makes a prediction
            preds = [agent(img) for agent in agents]
            loss_vals = [loss_fn(pred, label) for pred in preds]

            # Compute total loss (sum or average of all agent losses)
            total_loss = sum(loss_vals)
            
            # Log the losses for each agent
            if train_logger is not None:
                for idx, loss_val in enumerate(loss_vals):
                    train_logger.add_scalar(f'agent_{idx}_loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, preds[0], global_step)

            # Perform backward pass and optimizer step for each agent
            for optimizer in optimizers:
                optimizer.zero_grad()
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            
            global_step += 1
            losses.append(total_loss.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        if train_logger is None:
            print(f'epoch {epoch:3d} \t loss = {avg_loss:.3f}')
        save_model(agents)

    save_model(agents)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predicted aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=30)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('--num_agents', type=int, default=2) 

    args = parser.parse_args()
    train(args)

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
    print("Device:", device)

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

    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

    global_step = 0
    for epoch in range(args.num_epoch):
        epoch_losses = []
        for batch_data in train_data:  # Assuming batch_data is compatible with multi-agent training
            losses = []
            gradients = []

            for agent_idx, agent in enumerate(agents):
                optimizer = optimizers[agent_idx]
                optimizer.zero_grad()

                img, label = batch_data[agent_idx]
                img, label = img.to(device), label.to(device)

                pred = agent(img)
                loss = loss_fn(pred, label)
                losses.append(loss)

                # Backpropagate to compute gradients
                loss.backward(retain_graph=True)
                gradients.append([param.grad.clone() for param in agent.parameters() if param.grad is not None])
                optimizer.zero_grad()  

            total_weight = sum(1 / (loss.item() + 1e-8) for loss in losses)
            weights = [(1 / (loss.item() + 1e-8)) / total_weight for loss in losses]

            shared_model = Planner().to(device)  
            for param_idx, shared_param in enumerate(shared_model.parameters()):
                if shared_param.grad is not None:
                    shared_param.grad = sum(weights[agent_idx] * gradients[agent_idx][param_idx]
                                            for agent_idx in range(num_agents))

            shared_optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.learning_rate)
            shared_optimizer.step()

            avg_loss = torch.mean(torch.tensor(losses))
            epoch_losses.append(avg_loss.item())
            if train_logger:
                train_logger.add_scalar('loss', avg_loss.item(), global_step)

            global_step += 1

        print(f'Epoch {epoch + 1}/{args.num_epoch}, Avg Loss: {np.mean(epoch_losses)}')
        save_model(shared_model)  


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

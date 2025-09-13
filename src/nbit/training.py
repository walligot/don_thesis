"""
nbit.training

Auto-extracted from notebook: Copy_of_nbit_training.ipynb
This module was generated to consolidate reusable functions/classes.
Any direct references to Colab paths have been annotated.

NOTE: If you see '/content/drive  # REPLACE WITH YOUR OWN FOLDER' references, adjust paths for your environment.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
N = 100  # number of RNN units
dt = 0.15  # time step
T_train = 300  # shorter training sequence length
T_eval = 1000  # evaluation sequence length
#n_bits = 3  # 3-bit memory
pulse_prob = 0.01  # probability of pulse per bit per timestep
noise_std = 0.015  # standard deviation of noise added to hidden state during training

# Define the RNN dynamics class
class ContinuousRNN(nn.Module):
    def __init__(self, N, n_bits, device=device):
        super(ContinuousRNN, self).__init__()
        self.device = device or torch.device('cpu')
        self.N = N
        self.n_bits = n_bits
        self.input_weights = nn.Parameter(torch.randn(N, n_bits) * 0.1)
        self.recurrent_weights = nn.Parameter(torch.randn(N, N) * 0.1 * 1.0)
        self.output_weights = nn.Parameter(torch.randn(n_bits, N) * 0.1)
        self.nonlinearity = torch.tanh
        self.to(self.device)

    def forward(self, inputs, return_h=False, add_noise=False):
      # Move inputs to self.device if not already
        if inputs.device != self.device:
            inputs = inputs.to(self.device)

        if not (isinstance(add_noise, (bool, torch.Tensor))):
          raise TypeError("add_noise must be either a bool or a torch.Tensor")

        is_noise_vector = isinstance(add_noise, torch.Tensor)

        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.N, device=self.device)
        outputs = []
        hidden_states = []

        #print(self.device)
        #print(self.input_weights.device)
        #print(self.recurrent_weights.device)
        #print(self.output_weights.device)
        #print(inputs.device)
        #print(h.device)

        for t in range(seq_len):
            input_t = inputs[:, t, :]
            dh = (-h + self.nonlinearity(
                torch.matmul(h, self.recurrent_weights.T) + torch.matmul(input_t, self.input_weights.T)
            )) * dt
            #print(f"inputs shape {inputs.shape}")
            #print(f"input_t shape {input_t.shape}")
            #print(f"h shape {h.shape}")
            #print(f"dh shape {dh.shape}")

            if is_noise_vector:
                add_noise = add_noise.to(self.device)
                if (add_noise.shape[0] < inputs.shape[0] or
                    add_noise.shape[1] < inputs.shape[1] or
                    add_noise.shape[2] < self.N):
                    raise ValueError(
                        f"add_noise shape {add_noise.shape} must have batch and time dimensions "
                        f"at least as large as inputs {inputs.shape} and at least {self.N} channels"
                    )

                # Slice noise to match input batch size and time step t
                noise_t = add_noise[:inputs.shape[0], t, :self.N]
            elif add_noise is True:
                noise_t = torch.randn_like(dh)
            else:
                noise_t = torch.zeros_like(dh)

            #scale = noise_std*dh.norm(dim=1, keepdim=True)
            scale=noise_std
            #print(f"noise_t shape {noise_t.shape}")
            dh += noise_t * scale
            h = h + dh

            out = torch.matmul(h, self.output_weights.T)
            outputs.append(out)
            hidden_states.append(h)

        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)

        if return_h:
            return outputs, hidden_states
        else:
            return outputs

def generate_batch(batch_size, T, n_bits, pulse_prob, device='cpu'):
    inputs = torch.zeros(batch_size, T, n_bits, device=device)
    targets = torch.zeros(batch_size, T, n_bits, device=device)

    # Current state per bit per sample
    current_input = torch.zeros(batch_size, n_bits, device=device)
    memory = torch.zeros(batch_size, n_bits, device=device)

    for t in range(T):
        # Decide which bits start a new pulse
        start_mask = (current_input == 0) & (torch.rand(batch_size, n_bits, device=device) < pulse_prob)

        # Random ±1 for new pulses
        new_pulses = (torch.randint(0, 2, (batch_size, n_bits), device=device) * 2 - 1).float()

        # Apply new pulses
        current_input[start_mask] = new_pulses[start_mask]
        memory[start_mask] = current_input[start_mask]  # update memory only on new pulses

        # Decide which bits return to 0
        stop_mask = (current_input != 0) & (torch.rand(batch_size, n_bits, device=device) < 0.05)
        current_input[stop_mask] = 0.0

        # Store values
        inputs[:, t, :] = current_input
        targets[:, t, :] = memory

    return inputs, targets

def train_rnn(n_bits, n_epochs=4000, batch_size=32, eval_batch_size=100):
    rnn = ContinuousRNN(N, n_bits)
    optimizer = optim.Adam(rnn.parameters(), weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    lambda_abs = 0.1
    eval_inputs, eval_targets = generate_batch(eval_batch_size, T_eval, n_bits, pulse_prob, device=rnn.device)
    train_losses, train_accs = ([], [])
    eval_losses, eval_accs = ([], [])
    for epoch in tqdm(range(n_epochs)):
        learning_rate = 0.002 if epoch < n_epochs // 2 else 0.0002
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate
        rnn.train()
        inputs, targets = generate_batch(batch_size, T_train, n_bits, pulse_prob, device=rnn.device)
        outputs = rnn(inputs, add_noise=True)
        loss_sign = loss_fn(outputs, targets)
        loss_magnitude = loss_fn(torch.abs(outputs), torch.ones_like(outputs))
        loss = loss_sign + lambda_abs * loss_magnitude
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                train_acc = (torch.sign(outputs) == torch.sign(targets)).float().mean()
                rnn.eval()
                eval_outputs = rnn(eval_inputs, add_noise=False)
                eval_loss = loss_fn(eval_outputs, eval_targets)
                eval_acc = (torch.sign(eval_outputs) == torch.sign(eval_targets)).float().mean()
            train_losses.append(loss.item())
            train_accs.append(train_acc.item())
            eval_losses.append(eval_loss.item())
            eval_accs.append(eval_acc.item())
            print(f'Epoch {epoch:4d} | TrainLoss {loss.item():.5f}, TrainAcc {train_acc.item() * 100:.2f}% | EvalLoss {eval_loss.item():.5f}, EvalAcc {eval_acc.item() * 100:.2f}%')
    return (rnn, train_losses, train_accs, eval_losses, eval_accs)

def save_rnn_model(model, path):
    torch.save({'N': model.N, 'n_bits': model.n_bits, 'state_dict': model.state_dict()}, path)

def load_rnn(save_path, device=device):
    """
    Loads a ContinuousRNN from disk, reconstructing it completely.
    Returns the model in eval() mode.

    Args:
      save_path (str): path to .pth file saved by save_rnn.
      device (torch.device or str, optional): where to load the weights;
        if None, defaults to CPU.
    """
    device = device or torch.device('cpu')
    ckpt = torch.load(save_path, map_location=device)
    loaded_rnn = ContinuousRNN(N=ckpt['N'], n_bits=ckpt['n_bits'], device=device)
    loaded_rnn.load_state_dict(ckpt['state_dict'])
    loaded_rnn.eval()
    print(f"Loaded ContinuousRNN (N={ckpt['N']}, n_bits={ckpt['n_bits']}) from '{save_path}'")
    return loaded_rnn

def randomise_rnn(orig_rnn, reinit_input=False, reinit_rec=False, reinit_output=False, scale=0.1):
    new_rnn = type(orig_rnn)(orig_rnn.N, orig_rnn.n_bits, device=orig_rnn.input_weights.device)
    new_rnn.load_state_dict(copy.deepcopy(orig_rnn.state_dict()))
    with torch.no_grad():
        if reinit_input:
            new_rnn.input_weights.copy_(torch.randn_like(new_rnn.input_weights) * scale)
        if reinit_rec:
            new_rnn.recurrent_weights.copy_(torch.randn_like(new_rnn.recurrent_weights) * scale)
        if reinit_output:
            new_rnn.output_weights.copy_(torch.randn_like(new_rnn.output_weights) * scale)
    return new_rnn

def eval_rnn(rnn):
    n_bits = rnn.n_bits
    loss_fn = nn.MSELoss()
    inputs, targets = generate_batch(1, T_eval, n_bits, pulse_prob)
    outputs, hidden_states = rnn(inputs, return_h=True)
    hidden_states = hidden_states.squeeze(0).detach().cpu().numpy()
    with torch.no_grad():
        loss = loss_fn(outputs[:T_eval], targets[:T_eval])
        predicted_sign = torch.sign(outputs[:T_eval])
        target_sign = torch.sign(targets[:T_eval])
        acc = (predicted_sign == target_sign).float().mean()
        print(f'Evaluation Loss: {loss.item():.5f}, Sign Acc: {acc.item() * 100:.2f}%')
    plt.figure(figsize=(15, 8))
    for i in range(n_bits):
        plt.subplot(n_bits, 1, i + 1)
        plt.plot(inputs[0, :, i], label='Input', linestyle=':', alpha=0.5)
        plt.plot(targets[0, :, i], label='Target', linestyle='-', alpha=0.9)
        plt.plot(outputs.squeeze(0).detach().cpu().numpy()[:, i], label='Output', linestyle='--', alpha=0.9)
        plt.legend()
        plt.title(f'Bit {i + 1}')
    plt.tight_layout()
    plt.show()

def eval_rnn_multiple(rnn, batch_size=100):
    n_bits = rnn.n_bits
    loss_fn = nn.MSELoss()
    inputs, targets = generate_batch(batch_size, T_eval, n_bits, pulse_prob)
    outputs, hidden_states = rnn(inputs, return_h=True)
    hidden_states = hidden_states.squeeze(0).detach().cpu().numpy()
    with torch.no_grad():
        loss = loss_fn(outputs[:T_eval], targets[:T_eval])
        predicted_sign = torch.sign(outputs[:T_eval])
        target_sign = torch.sign(targets[:T_eval])
        acc = (predicted_sign == target_sign).float().mean()
        print(f'Evaluation Loss: {loss.item():.5f}, Sign Acc: {acc.item() * 100:.2f}%')
    return (loss, acc)

def plot_training_curves(train_losses, train_accs, eval_losses, eval_accs, step=100):
    epochs = [i for i in range(0, step * len(train_losses), step)]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue', linestyle='--')
    ax1.plot(epochs, eval_losses, label='Eval Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(epochs, train_accs, label='Train Acc', color='tab:orange', linestyle='--')
    ax2.plot(epochs, eval_accs, label='Eval Acc', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.tight_layout()
    plt.show()
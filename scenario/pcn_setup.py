import torch.nn as nn
import torch.nn.functional as F

from agent.pcn.pcn_core import epsilon_metric, non_dominated, compute_hypervolume, add_episode, \
    choose_commands, Transition
from agent.pcn.logger import Logger
from create_fair_env import *
from scenario.fraud_detection.env import NUM_FRAUD_FEATURES
from scenario.job_hiring.env import NUM_JOB_HIRING_FEATURES


ss_emb = {
    'job': {
        'small': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_JOB_HIRING_FEATURES, 64),
            nn.Sigmoid()
        ),
        'big': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_JOB_HIRING_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        ),
    },
    'fraud': {
        'small': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_FRAUD_FEATURES, 64),
            nn.Sigmoid()
        ),
        'big': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_FRAUD_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        ),
    }
}

se_emb = {
    'small': nn.Sequential(
        nn.Linear(1, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}

sa_emb = {
    'small': nn.Sequential(
        nn.Linear(3, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}


class Model(nn.Module):
    def __init__(self,
                 nA,
                 scaling_factor,
                 objectives,
                 ss_emb):
        super(Model, self).__init__()

        self.scaling_factor = scaling_factor[:, objectives + (len(scaling_factor) - 1,)]
        self.objectives = objectives
        self.ss_emb = ss_emb
        self.s_emb = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.c_emb = nn.Sequential(nn.Linear(self.scaling_factor.shape[-1], 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA))

    def forward(self, state, desired_return, desired_horizon):
        desired_return = desired_return[:, self.objectives]
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.ss_emb(state.float())
        s = self.s_emb(s)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        sc = s * c
        log_prob = self.fc(sc)
        return log_prob


class DiscreteHead(nn.Module):
    def __init__(self, base):
        super(DiscreteHead, self).__init__()
        self.base = base

    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = F.log_softmax(x, 1)
        return x


def get_model(args, env_type, n_actions, scaling_factor):
    kw = "small" if args.model == "densesmall" else "big"
    ss, se, sa = ss_emb[env_type][kw], se_emb[kw], sa_emb[kw]

    model = Model(n_actions, scaling_factor, tuple(args.objectives), ss).to(device)
    model = DiscreteHead(model)
    return model


def update_model(model, opt, experience_replay, batch_size, noise=0.):
    batch = []
    # randomly choose episodes from experience buffer
    s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)
    for i in s_i:
        # episode is tuple (return, transitions)
        ep = experience_replay[i][2]
        # choose random timestep from episode,
        # use it's return and leftover timesteps as desired return and horizon
        t = np.random.randint(0, len(ep))
        # reward contains return until end of episode
        s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
        batch.append((s_t, a_t, r_t, h_t))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    obs = torch.tensor(obs).to(device)

    # TODO TEST add noise to the desired return
    desired_return = torch.tensor(desired_return).to(device)
    desired_return = desired_return + noise * torch.normal(0, 1, size=desired_return.shape,
                                                           device=desired_return.device)
    log_prob = model(obs,
                     desired_return,
                     torch.tensor(desired_horizon).unsqueeze(1).to(device))
    opt.zero_grad()

    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        l = F.mse_loss(log_prob, torch.tensor(actions))
    else:
        # one-hot of action for CE loss
        actions = torch.tensor(actions).long().to(device)
        actions = F.one_hot(actions, num_classes=log_prob.shape[-1])
        # cross-entropy loss
        l = torch.sum(-actions * log_prob, -1).sum(-1)
    l = l.mean()
    l.backward()
    opt.step()

    return l.detach().cpu().numpy(), log_prob.detach().cpu().numpy()


def choose_action(model, obs, desired_return, desired_horizon, eval=False, return_probs=False):
    # if observation is not a simple np.array, convert individual arrays to tensors
    obs = [torch.tensor([o]).to(device) for o in obs] if type(obs) == tuple else torch.tensor([obs]).to(device)
    log_probs = model(obs,
                      torch.tensor([desired_return]).to(device),
                      torch.tensor([desired_horizon]).unsqueeze(1).to(device))
    log_probs = log_probs.detach().cpu().numpy()[0]
    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        action = log_probs
        # add some noise for randomness
        if not eval:
            action = np.clip(action + np.random.normal(0, 0.1, size=action.shape).astype(np.float32), 0, 1)
    else:
        # if evaluating: act greedily
        if eval:
            action = np.argmax(log_probs, axis=-1)
            if return_probs:
                return action, log_probs
            else:
                return action

        if log_probs.ndim == 1:
            log_probs = np.nan_to_num(log_probs)
            if log_probs.sum() == 0:
                log_probs = np.full_like(log_probs, 1 / len(log_probs))
            log_probs = np.exp(log_probs)
            if log_probs.sum() != 1:
                log_probs = log_probs / log_probs.sum()
            action = np.random.choice(np.arange(len(log_probs)), p=log_probs)
        elif log_probs.ndim == 2:
            action = np.array(list([np.random.choice(np.arange(len(lp)), p=np.exp(lp)) for lp in log_probs]))
    if return_probs:
        return action, log_probs
    else:
        return action

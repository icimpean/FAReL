from agent.dqn import EpsilonGreedyPolicy, DQNAgent
from create_fair_env import *
from scenario.fraud_detection.env import NUM_FRAUD_FEATURES, FraudActions
from scenario.job_hiring.env import NUM_JOB_HIRING_FEATURES, HiringActions


nn_model = {
    'job': (NUM_JOB_HIRING_FEATURES, 64,
            [a for a in HiringActions]),
    'fraud': (NUM_FRAUD_FEATURES, 64,
              [a for a in FraudActions]),
}


def get_model(args, env_type, n_actions):
    input_shape, h1, actions = nn_model[env_type]
    model = DQNAgent(input_shape, h1, actions, warm_up=0, policy=EpsilonGreedyPolicy(0.1),
                     memory_size=args.er_size*args.episode_length, batch_size=args.batch, learning_rate=args.lr,
                     training=True, seed=args.seed)
    return model


def update_model(model, step):
    return model.train(step)


def choose_action(model, obs, step, episode, eval=False, return_probs=False):
    # if observation is not a simple np.array, convert individual arrays to tensors
    # obs = [torch.tensor([o]).to(device) for o in obs] if type(obs) == tuple else torch.tensor([obs]).to(device)
    curr_train = model.training
    if eval:  # Greedy
        model.set_training(False)
    action, q_values = model.select_action(obs, step, episode, return_q_values=return_probs)
    model.set_training(curr_train)

    if return_probs:
        return action, q_values
    else:
        return action


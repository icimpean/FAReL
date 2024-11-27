import random

import torch
import numpy as np
from datetime import datetime
import os

import argparse
from pytz import timezone

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from fairness import SensitiveAttribute, CombinedSensitiveAttribute
from fairness.fairness_framework import FairnessFramework, ExtendedfMDP
from fairness.group import GroupNotion, ALL_GROUP_NOTIONS
from fairness.individual import IndividualNotion, ALL_INDIVIDUAL_NOTIONS
from scenario import FeatureBias
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel
from scenario.fraud_detection.env import TransactionModelMDP, FraudFeature
from scenario.job_hiring.features import HiringFeature, Gender, ApplicantGenerator, Nationality
from scenario.job_hiring.env import HiringActions, JobHiringEnv
from scenario.parameter_setup import VSC_SAVE_DIR, device


def create_job_env(args):
    team_size = args.team_size
    episode_length = args.episode_length
    diversity_weight = args.diversity_weight
    # Training environment
    population_file = f'./scenario/job_hiring/data/{args.population}.csv'
    if args.vsc == 0:
        population_file = "." + population_file
    applicant_generator = ApplicantGenerator(csv=population_file, seed=args.seed)

    # Initialise and get features to ignore in distance metrics
    if args.ignore_sensitive:
        exclude_from_distance = (HiringFeature.age, HiringFeature.gender, HiringFeature.nationality,
                                 HiringFeature.married)
    else:
        exclude_from_distance = ()

    # Fairness
    if args.combined_sensitive_attributes == 1:
        sensitive_attribute = CombinedSensitiveAttribute([HiringFeature.gender, HiringFeature.nationality],
                                                         sensitive_values=[Gender.female, Nationality.foreign],
                                                         other_values=[Gender.male, Nationality.belgian])
        inn_sensitive_features = [HiringFeature.gender.value]  # TODO
    elif args.combined_sensitive_attributes == 2:
        sensitive_attribute = [SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                  other_values=Gender.male),
                               SensitiveAttribute(HiringFeature.nationality, sensitive_values=Nationality.foreign,
                                                  other_values=Nationality.belgian)]
        inn_sensitive_features = [HiringFeature.gender.value, HiringFeature.nationality.value]
    else:
        sensitive_attribute = SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                 other_values=Gender.male)  # TODO: abstract parameters
        inn_sensitive_features = [HiringFeature.gender.value]

    # No bias
    if args.bias == 0:
        reward_biases = []
    # Bias on gender
    elif args.bias == 1:
        reward_biases = [FeatureBias(features=[HiringFeature.gender], feature_values=[Gender.male], bias=0.1)]
    # Bias on nationality and gender
    elif args.bias == 2:
        reward_biases = [FeatureBias(features=[HiringFeature.gender, HiringFeature.nationality],
                                     feature_values=[Gender.male, Nationality.belgian], bias=0.1)]

    # Create environment
    env = JobHiringEnv(team_size=team_size, seed=args.seed, episode_length=episode_length,  # Required ep length for pcn
                       diversity_weight=diversity_weight, applicant_generator=applicant_generator,
                       reward_biases=reward_biases, exclude_from_distance=exclude_from_distance)

    return env, sensitive_attribute, inn_sensitive_features


def create_fraud_env(args):
    # the parameters for the simulation
    params = parameters.get_default_parameters()  # TODO: abstract parameters
    params['seed'] = args.seed
    params['init_satisfaction'] = 0.9
    params['stay_prob'] = [0.9, 0.6]
    params['num_customers'] = 100
    params['num_fraudsters'] = 10
    # params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))
    # params['end_date'] = datetime(2016, 3, 31).replace(tzinfo=timezone('US/Pacific'))
    # # TODO (90357 yearly) Used for min/max reward
    # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (31 + 29 + 31)
    # 1 week = +- 1728 transactions
    num_transactions = args.n_transactions  # 1000
    params['end_date'] = datetime(2016, 1, 7).replace(tzinfo=timezone('US/Pacific'))
    # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (7)  # (90357) Used for min/max reward
    episode_length = num_transactions
    if args.fraud_proportion != 0:
        curr_sum = np.sum(params['trans_per_year'])
        params['trans_per_year'] = np.array([curr_sum * (1 - args.fraud_proportion),
                                             curr_sum * args.fraud_proportion])

    # Initialise and get features to ignore in distance metrics
    if args.ignore_sensitive:
        exclude_from_distance = (FraudFeature.continent, FraudFeature.country, FraudFeature.card_id)
    else:
        exclude_from_distance = ()

    # TODO: abstract parameters
    # Continents mapping from default parameters: {'EU': 0, 'AS': 1, 'NA': 2, 'AF': 3, 'OC': 4, 'SA': 5}
    # sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=1, other_values=0)
    # NA vs EU instead of AS vs EU to increase population size in both
    if args.combined_sensitive_attributes == 1:
        sensitive_attribute = CombinedSensitiveAttribute([FraudFeature.continent, FraudFeature.merchant_id],
                                                         sensitive_values=[2, 6],
                                                         other_values=[0, None])
        inn_sensitive_features = [FraudFeature.continent.value]  # TODO
    elif args.combined_sensitive_attributes == 2:
        sensitive_attribute = [SensitiveAttribute(FraudFeature.continent, sensitive_values=2,
                                                  other_values=0),
                               SensitiveAttribute(FraudFeature.merchant_id, sensitive_values=6,
                                                  other_values=None)]
        inn_sensitive_features = [FraudFeature.continent.value, FraudFeature.continent.merchant_id]
    else:
        sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=2, other_values=0)
        inn_sensitive_features = [FraudFeature.continent.value]

    # No bias
    if args.bias == 0:
        reward_biases = []
    # Bias on gender
    elif args.bias == 1:
        reward_biases = [FeatureBias(features=[FraudFeature.continent], feature_values=[0], bias=0.1)]
    # Bias on nationality and gender
    elif args.bias == 2:
        reward_biases = [FeatureBias(features=[FraudFeature.continent, FraudFeature.merchant_id],
                                     feature_values=[0, 0], bias=0.1)]  # TODO: which merchant to target

    transaction_model = TransactionModel(params, seed=args.seed)
    env = TransactionModelMDP(transaction_model, do_reward_shaping=True, num_transactions=num_transactions,
                              exclude_from_distance=exclude_from_distance, reward_biases=reward_biases)

    return env, sensitive_attribute, inn_sensitive_features


def create_fairness_framework_env(args):
    if args.vsc == 1:
        result_dir = VSC_SAVE_DIR
    else:
        result_dir = "../../fairRLresults/"

    env_type = args.env
    if args.no_window:
        args.window = None
    is_job_hiring = env_type == "job"
    # Job hiring
    if is_job_hiring:
        logdir = f"{result_dir}/job_hiring/"
        env, sensitive_attribute, inn_sensitive_features = create_job_env(args)
    # Fraud
    else:
        logdir = f"{result_dir}/fraud_detection/"
        env, sensitive_attribute, inn_sensitive_features = create_fraud_env(args)

    #
    logdir += args.log_dir + "/"
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    os.makedirs(logdir, exist_ok=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ALL_OBJECTIVES = ALL_REWARDS + ALL_GROUP_NOTIONS + ALL_INDIVIDUAL_NOTIONS
    sort_objectives = {o: i for i, o in enumerate(ALL_OBJECTIVES)}
    # Check for concatenated arguments for objectives and compute objectives
    if len(args.objectives) == 1 and ":" in args.objectives[0]:
        args.objectives = args.objectives[0].split(":")
    if len(args.compute_objectives) == 1 and ":" in args.compute_objectives[0]:
        args.compute_objectives = args.compute_objectives[0].split(":")
    all_args_objectives = args.objectives + args.compute_objectives
    ordered_objectives = sorted(all_args_objectives,
                                key=lambda o: sort_objectives[get_objective(OBJECTIVES_MAPPING[o])])
    args.objectives = [i for i, o in enumerate(ordered_objectives) if o in args.objectives]

    mapped_ordered_notions = [OBJECTIVES_MAPPING[n] for n in ordered_objectives]
    all_group_notions = [n for n in mapped_ordered_notions if isinstance(n, GroupNotion)]
    all_individual_notions = [n for n in mapped_ordered_notions if isinstance(n, IndividualNotion)]

    fairness_framework = FairnessFramework([a for a in HiringActions], sensitive_attribute,
                                           individual_notions=all_individual_notions,
                                           group_notions=all_group_notions,
                                           similarity_metric=env.similarity_metric,
                                           distance_metrics=args.distance_metrics,
                                           alpha=args.fair_alpha,
                                           window=args.window,
                                           discount_factor=args.discount_factor if args.discount_history else None,
                                           discount_threshold=args.discount_threshold if args.discount_history else None,
                                           discount_delay=args.discount_delay if args.discount_history else None,
                                           nearest_neighbours=args.nearest_neighbours,
                                           inn_sensitive_features=None,
                                           # inn_sensitive_features=[HiringFeature.gender.value],  # TODO
                                           seed=seed,
                                           steps=int(args.steps),
                                           store_interactions=False,
                                           has_individual_fairness=len(all_individual_notions) > 0)

    # Extend the environment with fairness framework
    env = ExtendedfMDP(env, fairness_framework)

    # TODO: max reward still ok with new metrics/group divisions
    _num_group_notions = (len(sensitive_attribute) if args.combined_sensitive_attributes >= 2 else 1) * len(
        all_group_notions)
    _num_notions = _num_group_notions + len(all_individual_notions)
    max_reward = args.episode_length * 1
    scale = np.array([1] + [1] * _num_notions)  # TODO: treatment equality scale+max
    ref_point = np.array([-max_reward] + [-args.episode_length] * _num_notions)
    scaling_factor = torch.tensor([[1.0] + ([1] * _num_notions) + [0.1]]).to(device)
    max_return = np.array([max_reward] + [0] * _num_notions) / scale

    print(all_args_objectives)
    print(ordered_objectives)
    print(max_return)
    print(len(all_group_notions), len(all_individual_notions))


    env.nA = len(env.env.actions)
    env.scale = scale

    return env, logdir, ref_point, scaling_factor, max_return


#
Reward = "Reward"
ALL_REWARDS = [Reward]
#
OBJECTIVES_MAPPING = {
    # Rewards
    "R": Reward,
    # Group notions (over history)
    "SP": GroupNotion.StatisticalParity,
    "EO": GroupNotion.EqualOpportunity,
    "OAE": GroupNotion.OverallAccuracyEquality,
    "PP": GroupNotion.PredictiveParity,
    "PE": GroupNotion.PredictiveEquality,
    "EqOdds": GroupNotion.EqualizedOdds,
    "CUAE": GroupNotion.ConditionalUseAccuracyEquality,
    "TE": GroupNotion.TreatmentEquality,
    # Group notions (over timestep)
    "SP_t": GroupNotion.StatisticalParity_t,
    "EO_t": GroupNotion.EqualOpportunity_t,
    "OAE_t": GroupNotion.OverallAccuracyEquality_t,
    "PP_t": GroupNotion.PredictiveParity_t,
    "PE_t": GroupNotion.PredictiveEquality_t,
    "EqOdds_t": GroupNotion.EqualizedOdds_t,
    "CUAE_t": GroupNotion.ConditionalUseAccuracyEquality_t,
    "TE_t": GroupNotion.TreatmentEquality_t,
    # Individual notions (over history)
    "IF": IndividualNotion.IndividualFairness,
    "CSC": IndividualNotion.ConsistencyScoreComplement,
    "CSC_inn": IndividualNotion.ConsistencyScoreComplement_INN,
    # Individual notions (over timestep)
    "IF_t": IndividualNotion.IndividualFairness_t,
    # TODO: include
    # "CSC_t": IndividualNotion.ConsistencyScoreComplement_t,
    # "CSC_inn_t": IndividualNotion.ConsistencyScoreComplement_INN_t,
}
OBJECTIVES_MAPPING_r = {v: k for k, v in OBJECTIVES_MAPPING.items()}
parser_all_objectives = ", ".join([f"{v if isinstance(v, str) else v.name} ({k})"
                                   for k, v in OBJECTIVES_MAPPING.items()])


def get_objective(obj):
    try:
        return GroupNotion[obj]
    except KeyError:
        pass
    try:
        return IndividualNotion[obj]
    except KeyError:
        pass
    return obj


fMDP_parser = argparse.ArgumentParser(description='fMDP_parser', add_help=False)
#
fMDP_parser.add_argument('--objectives', default=['R', 'SP'],
                         type=str, nargs='+', help='Abbreviations of the fairness notions to optimise, one or more of: '
                                                   f'{parser_all_objectives}. Can be supplied as a single string, with'
                                                   f'the arguments separated by a colon, e.g., "R:SP"')
fMDP_parser.add_argument('--compute_objectives', default=['EO', 'OAE', 'PP', 'IF', 'CSC'],
                         type=str, nargs='*', help='Abbreviations of the fairness notions to compute, '
                                                   f'in addition to the ones being optimised: {parser_all_objectives}'
                                                   f' Can be supplied as a single string, with the arguments separated '
                                                   f'by a colon, e.g., "EO:OAE:PP:IF:CSC"')
#
fMDP_parser.add_argument('--env', default='job', type=str, help='job or fraud')
#
fMDP_parser.add_argument('--seed', default=0, type=int, help='seed for rng')
fMDP_parser.add_argument('--vsc', default=0, type=int, help='running on local (0) or VSC cluster (1)')
# Job hiring parameters
fMDP_parser.add_argument('--team_size', default=20, type=int, help='maximum team size to reach')
fMDP_parser.add_argument('--episode_length', default=100, type=int, help='maximum episode length')
fMDP_parser.add_argument('--diversity_weight', default=0, type=int, help='diversity weight, complement of skill weight')
fMDP_parser.add_argument('--population', default='belgian_population', type=str,
                         help='the name of the population file')
# Fraud detection parameters
fMDP_parser.add_argument('--n_transactions', default=1000, type=int, help='number of transactions per episode')
fMDP_parser.add_argument('--fraud_proportion', default=0, type=float,
                         help='proportion of fraudulent transactions to genuine. '
                              '0 defaults to default MultiMAuS parameters')
#
fMDP_parser.add_argument('--bias', default=0, type=int, help='Which bias configuration to consider. Default 0: no bias')
fMDP_parser.add_argument('--ignore_sensitive', action='store_true')
# Fairness framework
fMDP_parser.add_argument('--window', default=100, type=int, help='fairness framework window')
fMDP_parser.add_argument('--discount_history', action='store_true',
                         help='use a discounted history instead of a sliding window implementation')
fMDP_parser.add_argument('--discount_factor', default=1.0, type=float,
                         help='fairness framework discount factor for history')
fMDP_parser.add_argument('--discount_threshold', default=1e-5, type=float,
                         help='fairness framework discount threshold for history')
fMDP_parser.add_argument('--discount_delay', default=5, type=int,
                         help='the number of timesteps to consider for the fairness notion to not fluctuate more than '
                              'discount_threshold, before deleting earlier timesteps')
fMDP_parser.add_argument('--nearest_neighbours', default=5, type=int,
                         help='the number of neighbours to consider for individual fairness notions based on CSC')
fMDP_parser.add_argument('--fair_alpha', default=0.1, type=float, help='fairness framework alpha for similarity metric')
fMDP_parser.add_argument('--wandb', default=1, type=int,
                         help="(Ignored, overrides to 0) use wandb for loggers or save local only")
fMDP_parser.add_argument('--no_window', default=0, type=int, help="Use the full history instead of a window")
fMDP_parser.add_argument('--no_individual', default=0, type=int, help="No individual fairness notions")
fMDP_parser.add_argument('--distance_metrics', default=['braycurtis'], type=str, nargs='*',
                         help='The distance metric to use for every individual fairness notion specified. '
                              'The distance metrics should be supplied for each individual fairness in the objectives, '
                              'then followed by computed objectives.')
#
fMDP_parser.add_argument('--combined_sensitive_attributes', default=0, type=int,
                         help='Use a combination of sensitive attributes to compute fairness notions')
#
fMDP_parser.add_argument('--log_dir', default='new_experiment', type=str, help="Directory where to store results")
fMDP_parser.add_argument('--log_compact', action='store_true', help='Save compact logs to save space.')
fMDP_parser.add_argument('--log_coverage_set_only', action='store_true', help='Save only the coverage set logs')

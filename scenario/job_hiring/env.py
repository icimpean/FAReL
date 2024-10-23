from collections import Counter
from copy import copy
from typing import Union

from scenario.job_hiring.features import *
from scenario import Scenario, CombinedState


class HiringActions(Enum):
    """Enumeration for all actions in the hiring setting"""
    reject = 0
    hire = 1


class CompanyFeature(Enum):
    """Enumeration for the company features"""
    # Skills
    potential = auto()
    # TODO: candidate potential w.r.t. company
    degrees = auto()
    extra_degrees = auto()
    experiences = auto()
    #
    # language_entropy = auto()
    dutch_speaking = auto()
    french_speaking = auto()
    english_speaking = auto()
    german_speaking = auto()
    # Diversity
    # gender_diversity = auto()
    men = auto()
    women = auto()
    # nationality_diversity = auto()
    belgian = auto()
    foreign = auto()


NUM_JOB_HIRING_FEATURES = len(HiringFeature) + len(CompanyFeature)


class JobHiringEnv(Scenario):
    """The job hiring MDP

    Attributes:
        team_size: (Optional) The number of people required to hire before the hiring process ends.
        seed: (Optional) Random seed.
        description: (Optional) The description of the environment setup, for plotting results later.
        employment_durations: (Optional) The duration of employment for the given number of timesteps.
        employment_transitions: (Optional) The probability to change jobs based on age.
        episode_length: (Optional) The length of an episode if the desired team size isn't reached.
        diversity_weight: (Optional) The importance of diversity in the calculation of the goodness score
            and consequently the reward. 1 mean as important as skills. 0 means no importance.
        hiring_threshold: (Optional) Control how strict the requirements to hire are based on estimated improvement.
        goodness_noise: (Optional) Noise to add to the calculated goodness score.
        noise_hire: (Optional) Noise added to reward for hiring and rejecting a candidate.
        goodness_biases: (Optional) Bias added to goodness score when certain feature values are encountered.
        reward_biases: (Optional) Bias added to the reward when certain feature values are encountered.
        exclude_from_distance: (Optional) The features to exclude from distance calculations between individuals
    """

    def __init__(self, team_size=None, seed=None, description=None, applicant_generator=None,
                 employment_durations=None, employment_transitions=None,
                 episode_length=None,
                 diversity_weight=DIVERSITY_WEIGHT, hiring_threshold=HIRING_THRESHOLD,
                 goodness_noise=GOODNESS_NOISE, noise_hire=NOISE_HIRE,
                 goodness_biases=None, reward_biases=None,
                 exclude_from_distance=()):
        # Super call
        features = [feature for feature in HiringFeature]
        nominal_features = [HiringFeature.gender, HiringFeature.married,
                            HiringFeature.nationality, *LANGUAGE_FEATURES]
        numerical_features = [f for f in features if f not in nominal_features]
        super(JobHiringEnv, self).__init__(features=features, nominal_features=nominal_features,
                                           numerical_features=numerical_features,
                                           exclude_from_distance=exclude_from_distance, seed=seed)
        #
        if team_size and episode_length:
            assert episode_length >= team_size, \
                "Expected episode length to be indefinite or at least as long as the team size"
        self.team_size = team_size
        self.episode_length = episode_length
        self.description = description
        self.employment_durations = employment_durations
        self.employment_transitions = employment_transitions
        self.diversity_weight = diversity_weight

        # Job applicant features
        self.applicant_generator = ApplicantGenerator(seed=self.seed) \
            if applicant_generator is None else applicant_generator
        # Company/Employer features
        self.company_features = [feature for feature in CompanyFeature]
        self.input_shape = len(self.features) + len(self.company_features)

        self.goodness_biases = [] if goodness_biases is None else goodness_biases
        self.reward_biases = [] if reward_biases is None else reward_biases

        self.goodness_noise = goodness_noise
        self.noise_hire = noise_hire
        self.hiring_threshold = hiring_threshold

        self.previous_state = None
        self._t = 0
        self._current_team_size = 0
        self.employees = []
        self._team_composition = []
        self._team_start_t = []
        self._company_state = self._default_company_state()
        self._company_entropies = self._default_company_entropy()
        #
        self._goodness = None
        self._rewards = None
        # reject/hire
        self.actions = [a for a in HiringActions]

    @staticmethod
    def _default_company_state():
        return {
            CompanyFeature.potential: 0,
            CompanyFeature.degrees: 0,
            CompanyFeature.extra_degrees: 0,
            CompanyFeature.experiences: 0,
            #
            # CompanyFeature.language_entropy: 0,
            CompanyFeature.dutch_speaking: 0,
            CompanyFeature.french_speaking: 0,
            CompanyFeature.english_speaking: 0,
            CompanyFeature.german_speaking: 0,
            #
            # CompanyFeature.gender_diversity: 0,
            CompanyFeature.men: 0,
            CompanyFeature.women: 0,
            # CompanyFeature.nationality_diversity: 0,
            CompanyFeature.belgian: 0,
            CompanyFeature.foreign: 0,
        }

    @staticmethod
    def _default_company_entropy():
        return (0, 0, 0)

    def reset(self):
        self._t = 0
        self._current_team_size = 0
        self.employees = []
        self._team_composition = []
        self._team_start_t = []
        self._company_state = self._default_company_state()
        self._company_entropies = self._default_company_entropy()
        self.previous_state = self.generate_sample()
        self._goodness = self.calc_goodness(self.previous_state)
        self._rewards = self.calculate_rewards(self.previous_state, self._goodness)
        # Initialise features
        self.init_features(self.previous_state)
        return self.previous_state

    def step(self, action):
        hiring_action = HiringActions(action)
        reward = self._rewards[hiring_action]

        if hiring_action == HiringActions.hire:
            self.employees.append(self.new_employee(self.previous_state))
            self._current_team_size += 1
            self._company_state, self._company_entropies = self.generate_company_state(self.previous_state, hire=True)

        # Some employees may leave
        leaving_employees = self.get_leaving_employees()
        if len(leaving_employees) > 0:
            for employee in leaving_employees:
                self.employees.remove(employee)
                self._current_team_size -= 1
            self._company_state, self._company_entropies = self.generate_company_state(self.previous_state, hire=False)

        next_state = self.generate_sample()
        self.previous_state = next_state
        self._goodness = self.calc_goodness(self.previous_state)
        self._rewards = self.calculate_rewards(self.previous_state, self._goodness)
        self._t += 1

        done = False
        # There a maximum timestep has been reached
        if self.episode_length is not None and self._t >= self.episode_length:
            done = True
        # The maximum team size has been reached
        elif self.team_size is not None and self._current_team_size >= self.team_size:
            done = True
        info = {"goodness": self._goodness, "true_action": 1 if self._goodness >= self.hiring_threshold else 0,
                "team_size": self._current_team_size}

        return next_state, reward, done, info

    def add_leave_prob(self, employee):
        # Employees probability to change/leave jobs based on age
        if self.employment_transitions is not None:
            age_idx = (employee[HiringFeature.age] - 15) // 10
            weight = self.employment_transitions[age_idx]
            employee["_weight_"] = weight
        return employee

    def new_employee(self, state):
        emp = copy(state.sample_individual)
        emp["_languages_"] = [get_language(lan).value for lan in LANGUAGE_FEATURES if emp[lan]]
        emp = self.add_leave_prob(emp)
        return emp

    def get_leaving_employees(self):
        leaving = []
        # No employees ==> nobody can leave
        if len(self.employees) == 0:
            return leaving
        # Employees probability to change/leave jobs based on age
        if self.employment_transitions is not None:
            # ages_idx = [(e[HiringFeature.age] - 15) // 10 for e in self.employees]
            # weights = [self.employment_transitions[age_idx] for age_idx in ages_idx]
            weights = [e["_weight_"] for e in self.employees]
            employees_p = np.array(weights) / np.sum(weights)
            # Pick an employee
            employee_idx = self.rng.choice(range(len(employees_p)), p=employees_p)
            employee = self.employees[employee_idx]
            leave_prob = employee["_weight_"]
            if leave_prob > self.rng.random():
                print("Employee left:", employee)
                leaving.append(employee)
        # TODO: Employees can stay a certain duration
        if self.employment_durations is not None:
            # TODO: only add employee if not already leaving
            pass

        return leaving

    def generate_sample(self):
        # Create a sample
        sample_applicant = self.applicant_generator.sample()
        # Combine applicant with current company performance for the state
        state = CombinedState(sample_context=self._company_state, sample_individual=sample_applicant)
        return state

    @staticmethod
    def _entropy(values, base=2):
        values = np.array(sorted(values))
        #
        values += 1  # Zeros produce NaN
        counter = Counter(values)
        unique, counts = np.array(list(counter.keys())), np.array(list(counter.values()))

        # Incomplete unique values
        if len(unique) < base:
            new_u = []
            new_c = []
            for i in range(1, base+1):
                new_u.append(i)
                if i not in unique:
                    new_c.append(0.0)
                else:
                    idx = np.argwhere(unique == i)[0][0]
                    new_c.append(counts[idx])
            # unique = new_u
            counts = np.array(new_c)
            # Zeros produce NaN
            counts += 1e-10

        probabilities = counts / sum(counts)
        # Is faster
        if base == 2:
            entropy = -sum(probabilities * np.log2(probabilities))
        else:
            entropy = -sum(probabilities * np.emath.logn(base, probabilities))

        return probabilities, entropy

    def generate_company_state(self, state: CombinedState, hire=False):
        """Add a candidate to the state of the company"""
        # Add applicant contributions to team composition
        _features = [HiringFeature.degree, HiringFeature.extra_degree, HiringFeature.experience]
        n_features = len(_features)
        applicant_features = state.get_features(_features)
        # Only add applicant if actually hiring
        employees_features = self._team_composition if hire else self._team_composition[:]
        employees_features.append(applicant_features)
        employees = self.employees if hire else self.employees[:]
        employees.append(self.new_employee(state))
        #
        employees_start_t = self._team_start_t if hire else self._team_start_t[:]
        employees_start_t.append(self._t)
        #
        employees_features = np.vstack(employees_features)
        n_employees = len(employees_features)
        max_experience = 65 - 18

        # Normalize all features based on number of employees
        if self.team_size is None:
            n = n_employees if self.episode_length is None else self.episode_length
        else:
            n = self.team_size

        # Gender diversity minimise difference between possible genders
        genders = [emp[HiringFeature.gender].value for emp in employees]
        gender_probs, gender_diversity = self._entropy(genders, base=2)
        # print("gender_diversity", genders, "==>", gender_diversity)
        #
        # Nationality diversity minimise difference between possible nationalities
        nationalities = [emp[HiringFeature.nationality].value for emp in employees]
        nationality_probs, nationality_diversity = self._entropy(nationalities, base=2)
        # print("nationality_diversity", nationalities, "==>", nationality_diversity)

        # languages = [get_language(lan).value for emp in employees for lan in LANGUAGE_FEATURES if emp[lan]]
        languages = [l for emp in employees for l in emp["_languages_"]]
        # print("languages", languages)
        language_probs, language_entropy = self._entropy(languages, base=len(LANGUAGE_FEATURES))
        # print("language_entropy", languages, "==>", language_entropy)

        # True new performance is unknown, both when estimating the candidate and when effectively hired
        performance_noise = self.rng.normal() * self.noise_hire
        # Calculate the new company state
        company_state = {
            # Performance is an indicator for how qualified each individual in the company is on their own
            #   ==> check how many employees have non-zero features
            CompanyFeature.potential: (np.sum(np.count_nonzero(employees_features, axis=1)) / n_features
                                       + performance_noise) / n,
            # The more degrees/experience employees hold, the better
            CompanyFeature.degrees: np.sum(employees_features[:, 0]) / n,
            CompanyFeature.extra_degrees: np.sum(employees_features[:, 1]) / n,
            CompanyFeature.experiences: np.sum(employees_features[:, 2] / max_experience) / n,
            #
            CompanyFeature.dutch_speaking: language_probs[Language.dutch.value],
            CompanyFeature.french_speaking: language_probs[Language.french.value],
            CompanyFeature.english_speaking: language_probs[Language.english.value],
            CompanyFeature.german_speaking: language_probs[Language.german.value],
            #
            CompanyFeature.men: gender_probs[Gender.male.value],
            CompanyFeature.women: gender_probs[Gender.female.value],
            #
            CompanyFeature.belgian: nationality_probs[Nationality.belgian.value],
            CompanyFeature.foreign: nationality_probs[Nationality.foreign.value],
        }

        del employees_features

        return company_state, (language_entropy, gender_diversity, nationality_diversity)

    def calc_goodness(self, state: CombinedState):
        skill_features = [CompanyFeature.potential, CompanyFeature.degrees, CompanyFeature.extra_degrees,
                          CompanyFeature.experiences]
        diversity_features = []
        # Does the candidate improve the company in any way if they would be hired?
        current = self._company_state
        # le, gd, nd = self._company_entropies
        predict, (new_le, new_gd, new_nd) = self.generate_company_state(state)

        # Normalize all features based on number of employees
        if self.team_size is None:
            n = len(self._team_composition) + 1 if self.episode_length is None else self.episode_length
        else:
            n = self.team_size

        skill_diff = {f: predict[f] - current[f] for f in current.keys() if f in skill_features}
        # Normalise all features equally (entropy is not yet normalised with team size)
        skill_diff["language_entropy"] = new_le / n

        diversity_diff = {f: predict[f] - current[f] for f in current.keys() if f in diversity_features}
        # Normalise all features equally (entropy is not yet normalised with team size)
        diversity_diff["gender_diversity"] = new_gd / n
        diversity_diff["nationality_diversity"] = new_nd / n

        # Calculate goodness score of features
        goodness_skill = sum(skill_diff.values()) / len(skill_diff) * n
        goodness_diversity = sum(diversity_diff.values()) / len(diversity_diff) * n
        goodness = ((1 - self.diversity_weight) * goodness_skill) + (self.diversity_weight * goodness_diversity)
        # print(current, "\n", state.sample_applicant)
        # print("Differences per feature:", skill_diff, "\ndiversity_diff:", diversity_diff, "\ngoodness:", goodness)
        # print(goodness_skill, goodness_diversity, goodness)

        # Add bias
        for bias in self.goodness_biases:
            goodness += bias.get_bias(state)

        # Clip goodness score
        goodness = np.clip(goodness, -1, 1)

        return goodness

    def calculate_rewards(self, sample: CombinedState, goodness):
        # Reward for hiring
        reward_noise = self.rng.normal() * self.noise_hire
        reward_hire = (goodness - self.hiring_threshold) + reward_noise

        # Add bias
        for bias in self.reward_biases:
            reward_hire += bias.get_bias(sample)

        # Clip reward
        reward_hire = np.clip(reward_hire, -1, 1)

        # Reward for rejecting candidate
        reward_reject = -reward_hire
        # Return the rewards
        rewards = {HiringActions.reject: reward_reject, HiringActions.hire: reward_hire}
        return rewards

    def _normalise_features(self, state: Union[CombinedState, np.ndarray], features: List[HiringFeature] = None,
                            indices=None):
        if isinstance(state, CombinedState):
            new_values = self.applicant_generator.normalise_features(state.sample_dict, features, to_array=True)
        else:
            new_values = state
        # Already transformed into array, return requested indices
        if indices:
            new_values = new_values[indices]
        return new_values

    def normalise_state(self, state: CombinedState):
        norm_array = np.array([self.applicant_generator.normalise_feature(feature, value)
                               for feature, value in state.sample_dict.items()])
        return norm_array

    def get_all_entities_in_state(self, state: CombinedState, action, true_action, score, reward):
        return [(state, action, true_action, score, reward)]

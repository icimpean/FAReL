import numpy as np
import pandas as pd

from scenario.job_hiring.features import Gender

path_age_gender_edu_nat = "./processed_age_gender_education_nationality.csv"
path_age_gender_nat_mar = "./processed_age_gender_nationality_married.csv"


men_map = ["Man", "Mannen"]
women_map = ["Vrouw", "Vrouwen"]


def get_gender(gender):
    if gender in men_map:
        return Gender.male.value
    elif gender in women_map:
        return Gender.female.value
    else:
        raise ValueError(gender)


nat_map = ["Belgen"]
nnat_map = ["niet-Belgen", "Buitenlanders"]


ages_bins = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]  # < age


def get_age_bin(age):
    return next(idx for idx, value in enumerate(ages_bins) if value > age)


def is_married(married):
    return married == "Gehuwd"


def get_nationality(nat):
    if nat in nat_map:
        return 0
    elif nat in nnat_map:
        return 1
    else:
        raise ValueError(nat)


class EnumerationMapping(object):
    """Make an enumeration and mapping for a given dataset"""
    def __init__(self, n=0):
        self._n = n

        self.enum = {}
        self.mapping = {}

    def add(self, values):
        for value in values:
            if self.enum.get(value) is None:
                self.enum[value] = self._n
                self.mapping[self._n] = value
                self._n += 1


if __name__ == '__main__':

    # df1 = pd.read_csv(path_age_gender_edu_nat, index_col=0)
    # df2 = pd.read_csv(path_age_gender_nat_mar)

    # new_df1 = df1.copy()
    # new_df1["Gender"] = [get_gender(v) for v in new_df1["Gender"]]
    # new_df1["Nationality"] = [get_nationality(v) for v in new_df1["Nationality"]]
    # print(new_df1)
    #
    # new_new_df1 = []
    # # Expand age
    # # new_df2["Age"] = [int(age.split(" ")[0]) for age in new_df2["Age"]]
    # for idx, row in new_df1.iterrows():
    #     # print(row)
    #     ages = row["Age"].split(" ")
    #     age_min = int(ages[0])
    #     age_max = int(ages[-2])
    #
    #     for age in range(age_min, age_max + 1):
    #         new_row = row.copy()
    #         new_row["Age"] = age
    #         new_row["Population"] /= (age_max + 1 - age_min)  # assumption: uniformly split between ages
    #         new_row["Degree"] = new_row["Education"] >= 1
    #         new_row["ExtraDegree"] = new_row["Education"] == 2
    #
    #         new_new_df1.append(new_row)
    #     # print(new_row)
    #
    # new_new_df1 = pd.DataFrame(new_new_df1)
    # new_new_df1["Population"] = new_new_df1["Population"].astype(int)
    # new_new_df1 = new_new_df1.drop(columns=["Education"])  # Replace by degree and extra degree
    # print(new_new_df1)
    #
    # # Drop population = 0
    # new_new_df1 = new_new_df1[new_new_df1["Population"] > 0]
    #
    # # Drop population < 18 and > 65
    # new_new_df1 = new_new_df1[(new_new_df1["Age"] >= 18) & (new_new_df1["Age"] <= 65)]
    # print(new_new_df1)
    #
    # # Weights for sampling
    # new_new_df1["w"] = new_new_df1["Population"] / new_new_df1["Population"].sum()
    # print(new_new_df1)
    #
    # new_new_df1.to_csv("./weighted_nationality_age_gender_education.csv", index=False)
    new_new_df1 = pd.read_csv("./data/weighted_nationality_age_gender_education.csv", index_col=None)
    print(new_new_df1.head())

    # # df2
    # new_df2 = df2.copy()
    # new_df2["Gender"] = [get_gender(v) for v in new_df2["Gender"]]
    # new_df2["Nationality"] = [get_nationality(v) for v in new_df2["Nationality"]]
    # print(new_df2)
    #
    # new_new_df2 = []
    # # Expand age
    # # new_df2["Age"] = [int(age.split(" ")[0]) for age in new_df2["Age"]]
    # for idx, row in new_df2.iterrows():
    #     # print(row)
    #     ages = row["Age"].split(" ")
    #     age = int(ages[0])
    #     new_row = row.copy()
    #     new_row["Age"] = age
    #     new_row["Married"] = is_married(new_row["Married"])
    #
    #     new_new_df2.append(new_row)
    #     # print(new_row)
    #
    # new_new_df2 = pd.DataFrame(new_new_df2)
    # df2_pop = "Population_2022"  # "Population_2016"
    # new_new_df2["Population"] = new_new_df2[df2_pop].astype(int)
    # new_new_df2 = new_new_df2.drop(columns=["Population_2022", "Population_2016"])
    # print(new_new_df2)
    #
    # new_new_df2.to_csv("./weighted_nationality_age_gender_married.csv", index=False)
    new_new_df2 = pd.read_csv("./data/weighted_nationality_age_gender_married.csv", index_col=None)
    print(new_new_df2.head())

    # Are they married?
    samples = []

    for _, sample_1 in new_new_df1.iterrows():
        sub_population = new_new_df2[(new_new_df2["age"] == sample_1["age"]) &
                                     (new_new_df2["nationality"] == sample_1["nationality"]) &
                                     (new_new_df2["gender"] == sample_1["gender"])].reset_index(drop=True)
        sub_population["w"] = sub_population["population"] / sub_population["population"].sum()

        for i, sample_2 in sub_population.iterrows():
            sample = sample_1.copy()
            sample["married"] = sample_2["married"]
            sample["w"] = sample["w"] * sample_2["w"]
            samples.append(sample)

    samples_df = pd.DataFrame(samples)
    samples_df = samples_df.drop(columns=["population"])
    samples_df = samples_df[["nationality", "age", "gender", "degree", "extra_degree", "married", "w"]]

    print(samples_df)
    samples_df.to_csv("./data/belgian_pop.csv", index=False)

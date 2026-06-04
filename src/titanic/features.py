# features.py

import pandas as pd
import numpy as np
import re
from titanic.imputations_age import impute_age_iterative_extra_trees


# ---------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------

def extract_surname(df):
    return df["Name"].map(lambda x: x.split(", ")[0])


def extract_title(df):
    replace_inverse = {
        "Mr": ["Mr", "Dr", "Rev", "Major", "Col", "Capt", "Sir", "Don", "Jonkheer"],
        "Mrs": ["Mrs", "the Countess", "Ms", "Lady", "Mme", "Dona"],
        "Master": ["Master"],
        "Miss": ["Miss", "Mlle"]
    }

    def find_replace_title(title):
        for new_title, old_titles in replace_inverse.items():
            if title in old_titles:
                return new_title
        return title

    title = df["Name"].map(lambda x: x.split(", ")[1].split(".")[0])
    return title.map(find_replace_title)


def has_nickname(df):
    return df["Name"].map(lambda x: "(" in x).astype(int)


# ---------------------------------------------------------------------
# Sex
# ---------------------------------------------------------------------

def sex_dummy(df):
    return (df["Sex"] == "male").astype(int)


# ---------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------

def is_child(df):
    threshold_child = 7
    return (df["Age"] < threshold_child).astype(int)


def assign_age_group(df, var):
    age_group = pd.Series(index=df.index, dtype="object")

    age_group.loc[(df[var] > 0) & (df[var] <= 2)] = "baby"
    age_group.loc[(df[var] > 2) & (df[var] <= 6)] = "pre_school_children"
    age_group.loc[(df[var] > 6) & (df[var] <= 12)] = "school_children"
    age_group.loc[(df[var] > 12) & (df[var] <= 18)] = "teenager"
    age_group.loc[(df[var] > 18) & (df[var] <= 24)] = "young_adult"
    age_group.loc[(df[var] > 24) & (df[var] <= 60)] = "adult"
    age_group.loc[df[var] > 60] = "senior"

    return age_group


def assign_age_decade(df, var):
    bins = list(range(0, 91, 10))
    labels = [f"{i}-{i + 9}" for i in range(0, 90, 10)]
    return pd.cut(df[var], bins=bins, labels=labels, right=False)


# ---------------------------------------------------------------------
# Family
# ---------------------------------------------------------------------

def is_alone(df):
    return (df["SibSp"] + df["Parch"] == 0).astype(int)


def has_family(df):
    return (df["SibSp"] + df["Parch"] > 0).astype(int)


def compute_family_size(df):
    return df["SibSp"] + df["Parch"] + 1


# ---------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------

def split_ticket(ticket):
    ticket = str(ticket)

    if "LINE" in ticket:
        return "LINE", -1

    parts = re.split(r"\s+", ticket)

    if len(parts) > 1:
        ticket_text = re.sub(r"[^\w\s]", "", " ".join(parts[:-1])).upper()
        ticket_number = parts[-1]
        return ticket_text, ticket_number

    return np.nan, parts[0]


def extract_ticket_number_class(df):
    ticket_number_class = df["TicketNumber"].astype(str).str[0]
    ticket_number_class = ticket_number_class.replace("-", "0")
    return ticket_number_class.astype(int)


# ---------------------------------------------------------------------
# Cabin
# ---------------------------------------------------------------------

def extract_deck(df):
    return df["Cabin"].map(lambda x: x[0] if pd.notnull(x) else np.nan)


def extract_cabin_number(df):
    return df["Cabin"].apply(
        lambda x: int(re.search(r"\d+", x).group(0))
        if pd.notnull(x) and re.search(r"\d+", x)
        else np.nan
    )


def count_cabins(df):
    cabin_count = df["Cabin"].str.count(r"[A-Za-z]\d+")
    cabin_count = cabin_count.fillna(0)
    return cabin_count.astype(int)


def has_cabin(df):
    return df["Cabin"].notna().astype(int)


# ---------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------

def count_values(df, var):
    return df.groupby(var)[var].transform("count")


# ---------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------

def compute_group_size(df):
    df = df.copy()

    df["GroupSize"] = 1

    # Step 1 - family size
    df.loc[df["GroupSize"] == 1, "GroupSize"] = df["FamilySize"]

    # Step 2 - same ticket
    df.loc[
        (df["GroupSize"] == 1) &
        (df["TicketValueCounts"] > 1),
        "GroupSize"
    ] = df["TicketValueCounts"]

    # Step 3 - same cabin
    df.loc[
        (df["GroupSize"] == 1) &
        (df["CabinValueCounts"] > 1),
        "GroupSize"
    ] = df["CabinValueCounts"]

    # Step 4 - same surname and same fare
    for surname in df["Surname"].dropna().unique():
        loc = df.loc[
            (df["GroupSize"] == 1) &
            (df["Surname"] == surname)
        ]

        if len(loc) > 1 and loc["Fare"].nunique(dropna=True) == 1:
            df.loc[
                (df["GroupSize"] == 1) &
                (df["Surname"] == surname),
                "GroupSize"
            ] = len(loc)

    # Step 5 - close ticket numbers with same fare
    threshold = 1

    for fare in df["Fare"].dropna().unique():
        loc = df.loc[
            (df["GroupSize"] == 1) &
            (df["Fare"] == fare)
        ].sort_values("TicketNumber").reset_index(drop=True)

        if len(loc) <= 1:
            continue

        count = 1
        start_index = loc.index[0]

        while start_index < loc.index[-1]:
            for i in range(start_index, loc.index[-1]):
                if loc.iloc[i + 1]["TicketNumber"] - loc.iloc[i]["TicketNumber"] <= threshold:
                    count += 1
                else:
                    if count > 1:
                        passenger_ids = loc["PassengerId"].iloc[start_index:i + 1]
                        df.loc[df["PassengerId"].isin(passenger_ids), "GroupSize"] = count

                    count = 1
                    start_index = i + 1
                    break

            if count > 1:
                passenger_ids = loc["PassengerId"].iloc[start_index:]
                df.loc[df["PassengerId"].isin(passenger_ids), "GroupSize"] = count
                count = 1
                start_index = len(loc)

    return df["GroupSize"]


def assign_group_type(df):
    group_type = pd.Series(index=df.index, dtype="object")

    group_type.loc[df["GroupSize"] == 1] = "single"
    group_type.loc[df["GroupSize"] == 2] = "couple"
    group_type.loc[df["GroupSize"].isin([3, 4])] = "family"
    group_type.loc[df["GroupSize"] >= 5] = "big_family_or_friends"

    return group_type


# ---------------------------------------------------------------------
# Fare
# ---------------------------------------------------------------------

def compute_fare_per_ticket_passenger(df):
    return df["Fare"] / df["TicketValueCounts"]


def compute_fare_per_person(df):
    return df["Fare"] / df["GroupSize"]


# ---------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Name
    df["Surname"] = extract_surname(df)
    df["Title"] = extract_title(df)
    df["HasNickname"] = has_nickname(df)

    # Sex
    df["SexIsMale"] = sex_dummy(df)

    # Family
    df["IsAlone"] = is_alone(df)
    df["HasFamily"] = has_family(df)
    df["FamilySize"] = compute_family_size(df)

    # Ticket
    df["TicketText"], df["TicketNumber"] = zip(*df["Ticket"].map(split_ticket))
    df["TicketNumber"] = (
        pd.to_numeric(df["TicketNumber"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df["TicketNumberClass"] = extract_ticket_number_class(df)

    # Cabin
    df["HasCabin"] = has_cabin(df)
    df["Deck"] = extract_deck(df)
    df["CabinNumber"] = extract_cabin_number(df)
    df["CabinCount"] = count_cabins(df)

    # Counts
    df["SurnameValueCounts"] = count_values(df, "Surname")
    df["FamilySizeValueCounts"] = count_values(df, "FamilySize")
    df["CabinValueCounts"] = count_values(df, "Cabin")
    df["TicketValueCounts"] = count_values(df, "Ticket")
    df["FareValueCounts"] = count_values(df, "Fare")

    # Fare based on exact ticket count
    df["FarePerTicketPassenger"] = compute_fare_per_ticket_passenger(df)
    df["FarePerTicketPassenger_log1p"] = np.log1p(df["FarePerTicketPassenger"])

    # Group
    df["GroupSize"] = compute_group_size(df)
    df["GroupType"] = assign_group_type(df)

    # Fare based on inferred group size
    df["FarePerPerson"] = compute_fare_per_person(df)
    df["FarePerPerson_log1p"] = np.log1p(df["FarePerPerson"])

    # Age
    # variables candidates
    age_features = [
        "Pclass",
        "Sex",
        "Title",
        "SibSp",
        "Parch",
        "FarePerTicketPassenger_log1p",
        "Embarked"
    ]

    categorical_features = [
        "Sex",
        "Title",
        "Embarked"
    ]

    df["AgeETR"] = impute_age_iterative_extra_trees(
        df=df,
        features=age_features,
        categorical_features=categorical_features,
        age_col="Age",
        random_state=42,
        max_iter=50
    )

    df["AgeGroup"] = assign_age_group(df, "AgeETR")
    df["AgeDecade"] = assign_age_decade(df, "AgeETR")
    df["IsChild"] = (df["AgeETR"] < 7).astype(int)

    return df
# features

import pandas as pd
import numpy as np
import re


def extract_surname(df):
    return df["Name"].map(lambda x: x.split(", ")[0])

def extract_title(df):
    replace_inverse = {
        "Mr": ["Mr", "Dr", "Rev", "Major", "Col", "Capt", "Sir", "Don", "Jonkheer"], 
        "Mrs": ["Mrs", "the Countess", "Ms", "Lady", "Mme", "Dona"], 
        "Master": ["Master"],
        "Miss": ["Miss","Mlle"]
    }
    def find_replace_title(title):
        for new_title, old_titles in replace_inverse.items():
            if title in old_titles:
                return new_title
        return title
    df["Title"] = df["Name"].map(lambda x: x.split(", ")[1].split(".")[0])
    df["Title"] = df["Title"].map(find_replace_title)
    return df["Title"]

def has_nickname(df):
    return (df["Name"].map(lambda x: "(" in x)).astype(int)

def sex_dummy(df):
    df.loc[df["Sex"]=="male", "SexIsMale"] = 1
    df.loc[df["Sex"]=="female", "SexIsMale"] = 0
    return df["SexIsMale"].astype(int)

def is_child(df):
    threshold_child = 7
    return (df["Age"] < threshold_child).astype(int)

def assign_age_group(df, var):
    df.loc[(df[var]>0)&(df[var]<=2), "AgeGroup"]="baby"
    df.loc[(df[var]>2)&(df[var]<=6), "AgeGroup"]="pre-school_children"
    df.loc[(df[var]>6)&(df[var]<=12), "AgeGroup"]="school_children"
    df.loc[(df[var]>12)&(df[var]<=18), "AgeGroup"]="teenager"
    df.loc[(df[var]>18)&(df[var]<=24), "AgeGroup"]="young_adult"
    df.loc[(df[var]>24)&(df[var]<=60), "AgeGroup"]="adult"
    df.loc[(df[var]>60), "AgeGroup"]="seniors"
    return df["AgeGroup"]

def assign_age_decade(df, var):
    threshold_bins = 10
    bins = list(range(0, 91, threshold_bins))
    labels = [f"{i}-{i+9}" for i in range(0, 90, threshold_bins)]
    return pd.cut(df[var], bins=bins, labels=labels, right=False)


def is_alone(df):
    return (df["SibSp"] + df["Parch"] == 0).astype(int)

def has_family(df):
    return (df["SibSp"] + df["Parch"] > 0).astype(int)

def compute_family_size(df):
    return df["SibSp"] + df["Parch"] + 1

def count_tickets(df):
    return df.groupby('Ticket')['Ticket'].transform('count')

def split_ticket(ticket):
    ticket = str(ticket)  # ensure that the value is a string
    if 'LINE' in ticket:
        return 'LINE', -1
    else:
        parts = re.split(r'\s', ticket)
        if len(parts) > 1:
            ticket_text = re.sub(r'[^\w\s]', '', ' '.join(parts[:-1])).upper()
            return ticket_text, parts[-1]
        else:
            return np.nan, parts[0]

def compute_fare_by_ticket(df):
    return df['Fare']/df['TicketValueCounts']

def extract_deck(df):
    return df["Cabin"].map(lambda x: x[0] if not pd.isnull(x) else np.nan)

def extract_cabin_number(df):
    return df['Cabin'].apply(lambda x: int(re.search(r'\d+', x).group(0)) if pd.notnull(x) and re.search(r'\d+', x) else np.nan)

def count_values(df, var):
    return df.groupby(var)[var].transform('count')

def compute_group_size(df):
    #step 1
    df["GroupSize"] = 1 #single
    
    #step2 - family
    df.loc[
        (df["GroupSize"]==1)
        ,"GroupSize"
    ] = df["FamilySize"]
    
    #step 3 - identical ticket
    df.loc[
        (df["GroupSize"]==1)
        &
        (df["TicketValueCounts"]>1)
        ,"GroupSize"
    ] = df["TicketValueCounts"]
    
    # step 4 - cabin
    df.loc[
        (df["GroupSize"]==1)
        &
        (df["CabinValueCounts"]>1)
        ,"GroupSize"
    ] = df["CabinValueCounts"]
        
    # step 5 - surname + fare
    surname = list(df["Surname"].unique())
    for s in surname:
        loc = df.loc[
            (df["GroupSize"]==1)
            &
            (df["Surname"]==s)
        ]
        if len(loc)>1:
            if loc['Fare'].nunique() == 1:
                df.loc[
                    (df["GroupSize"]==1)
                    &
                    (df["Surname"]==s)
                    ,"GroupSize"
                ] = len(loc)
            else:
                pass
            
    # step 6 - close_ticket + fare
    fare = list(df["Fare"].unique())
    th = 1

    for f in fare:
        loc = df.loc[
            (df["GroupSize"]==1)
            &
            (df["Fare"]==f)
        ]

        if len(loc)>1:
            loc = loc.sort_values(by="TicketNumber").reset_index(drop=True)

            # Initialisation du compteur et de l'index de départ
            count = 1
            start_index = loc.index[0]

            # Boucle principale
            while start_index < loc.index[-1]:
                # Parcourir les lignes du DataFrame à partir de start_index
                for i in range(start_index, loc.index[-1]):
                    # Comparer le numéro de ticket de la ligne actuelle avec celui de la ligne suivante
                    if loc.iloc[i + 1]["TicketNumber"] - loc.iloc[i]["TicketNumber"] <= th:
                        # Incrémenter le compteur si la différence est inférieure ou égale au seuil
                        count += 1
                    else:
                        # Mettre à jour la colonne "GroupSize" pour toutes les lignes dans la plage actuelle
                        df.loc[df["PassengerId"].isin(loc["PassengerId"][start_index:i+1]), "GroupSize"] = count
                        # Réinitialiser le compteur et mettre à jour l'index de départ
                        count = 1
                        start_index = i+1
                        break  # Sortir de la boucle for

                # Mettre à jour la colonne "GroupSize" pour la dernière plage si nécessaire
                if count > 1:
                    df.loc[df["PassengerId"].isin(loc["PassengerId"][start_index:]), "GroupSize"] = count
                    count = 1  # Réinitialiser le compteur
                    start_index = len(loc)  # Mettre à jour l'index de départ pour terminer la boucle while
    
    
    return df["GroupSize"]

def assign_group_type(df):
    df.loc[df["GroupSize"]==1, "GroupType"] = "single"
    df.loc[df["GroupSize"]==2, "GroupType"] = "couple"
    df.loc[df["GroupSize"].isin([3,4]), "GroupType"] = "family"
    df.loc[df["GroupSize"]>=5, "GroupType"] = "big_family_or_friends"
    return df["GroupType"]

def count_cabins(df):
    df['CabinCount'] = df['Cabin'].str.count(r'[A-Za-z]\d+')
    df.loc[df['CabinCount'].isna(), "CabinCount"] = 1
    df.loc[df['CabinCount']==0, "CabinCount"] = 1
    return df["CabinCount"]

def extract_ticket_number_class(df):
    df["TicketNumberClass"] = df["TicketNumber"].astype(str).str[0]
    df.loc[df["TicketNumberClass"]=="-", "TicketNumberClass"]=0
    df["TicketNumberClass"] = df["TicketNumberClass"].astype(int)
    return df["TicketNumberClass"]


# ---


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Name
    df["Surname"] = extract_surname(df)
    df["Title"] = extract_title(df)
    df["HasNickname"] = has_nickname(df)

    # Age
    df["AgeGroup"] = assign_age_group(df, "Age")

    # Family
    df["IsAlone"] = is_alone(df)
    df["FamilySize"] = compute_family_size(df)

    # Ticket
    df["TicketText"], df["TicketNumber"] = zip(
        *df["Ticket"].map(split_ticket)
    )
    df["TicketNumber"] = pd.to_numeric(df["TicketNumber"], errors="coerce").fillna(-1).astype(int)

    # Cabin
    df["Deck"] = extract_deck(df)
    df["CabinNumber"] = extract_cabin_number(df)

    # Indicators
    df["SurnameValueCounts"] = count_values(df, "Surname")
    df["FamilySizeValueCounts"] = count_values(df, "FamilySize")
    df["CabinValueCounts"] = count_values(df, "Cabin")
    df["TicketValueCounts"] = count_values(df, "Ticket")
    df["FareValueCounts"] = count_values(df, "Fare")

    # Fare
    df["FareByTicket"] = compute_fare_by_ticket(df)
    df["FareByTicket_log1p"] = np.log1p(df["FareByTicket"])

    # Group
    df["GroupSize"] = compute_group_size(df)
    df["GroupType"] = assign_group_type(df)

    return df
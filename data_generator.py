import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta, datetime

fake = Faker()

def generate_dm_domain(n=1000):
    """
    Generates a synthetic CDISC SDTM DM (Demographics) domain.
    """
    data = []
    for _ in range(n):
        usubjid = fake.unique.bothify(text='????-####')
        subjid = usubjid.split('-')[1]
        studyid = "SyntheStudy-01"
        domain = "DM"
        
        # Demographics
        age = random.randint(18, 85)
        ageu = "YEARS"
        sex = random.choice(["M", "F"])
        race = random.choice(["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN", "OTHER"])
        ethnic = random.choice(["HISPANIC OR LATINO", "NOT HISPANIC OR LATINO"])
        
        # Dates
        rfstdtc = fake.date_between(start_date='-2y', end_date='-1y')
        rfendtc = rfstdtc + timedelta(days=random.randint(30, 365))
        
        # Treatment Arm (Target variable for our model later)
        arm = random.choice(["Placebo", "Drug 10mg", "Drug 20mg"])
        armcd = {"Placebo": "PBO", "Drug 10mg": "DRG10", "Drug 20mg": "DRG20"}[arm]
        
        # Introducing some signal for the model to find:
        # e.g. Older people slightly more likely to be in Drug 20mg dropouts or specific adverse events?
        # For now, let's just make the assignment random, or bias it slightly if we want easier prediction.
        # Let's add a bias: "Drug 20mg" has slightly younger population?
        if arm == "Drug 20mg" and random.random() > 0.7:
             age = max(18, age - 10)

        data.append({
            "STUDYID": studyid,
            "DOMAIN": domain,
            "USUBJID": usubjid,
            "SUBJID": subjid,
            "RFSTDTC": rfstdtc.isoformat(),
            "RFENDTC": rfendtc.isoformat(),
            "AGE": age,
            "AGEU": ageu,
            "SEX": sex,
            "RACE": race,
            "ETHNIC": ethnic,
            "ARM": arm,
            "ARMCD": armcd,
            "COUNTRY": fake.country_code()
        })
    
    return pd.DataFrame(data)

def generate_ae_domain(dm_df):
    """
    Generates a synthetic CDISC SDTM AE (Adverse Events) domain based on the DM domain.
    """
    data = []
    ae_terms = [
        "Headache", "Nausea", "Dizziness", "Fatigue", "Rash", 
        "Vomiting", "Diarrhea", "Insomnia", "Anxiety", "Pyrexia"
    ]
    
    for _, row in dm_df.iterrows():
        # Probability of having an AE
        # Drug 20mg has higher rate of AE
        prob_ae = 0.3
        if row["ARM"] == "Drug 20mg":
            prob_ae = 0.6
        elif row["ARM"] == "Drug 10mg":
            prob_ae = 0.45
            
        if random.random() < prob_ae:
            start_date = datetime.fromisoformat(row["RFSTDTC"])
            
            # Number of events
            for _ in range(random.randint(1, 3)):
                aestdtc = fake.date_between(start_date=start_date, end_date=datetime.fromisoformat(row["RFENDTC"]))
                aeterm = random.choice(ae_terms)
                aesev = random.choice(["MILD", "MODERATE", "SEVERE"])
                
                # Signal: "Drug 20mg" causes more "Dizziness" and "Severe" events
                if row["ARM"] == "Drug 20mg" and random.random() > 0.5:
                    aeterm = "Dizziness"
                    aesev = "SEVERE"

                data.append({
                    "STUDYID": row["STUDYID"],
                    "DOMAIN": "AE",
                    "USUBJID": row["USUBJID"],
                    "AETERM": aeterm,
                    "AESTDTC": aestdtc.isoformat(),
                    "AESEV": aesev,
                    "AESER": "Y" if aesev == "SEVERE" else "N", # Serious event if severe
                    "AEREL": "RELATED" if row["ARM"] != "Placebo" and random.random() > 0.3 else "NOT RELATED"
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating synthetic CDISC data...")
    dm_df = generate_dm_domain(2000)
    ae_df = generate_ae_domain(dm_df)
    
    dm_df.to_csv("dm.csv", index=False)
    ae_df.to_csv("ae.csv", index=False)
    
    print(f"Generated dm.csv ({len(dm_df)} rows) and ae.csv ({len(ae_df)} rows).")
    print(dm_df.head())
    print(ae_df.head())

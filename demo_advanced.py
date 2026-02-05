import pandas as pd
import numpy as np
import warnings
from spec_generator import SpecDataGenerator
from agus_gen import AgusGenerator

warnings.filterwarnings("ignore")

def run_advanced_demo():
    print("=== Step 1: Generate Source Data from Spec ===")
    spec_file = "sdtm_spec.json"
    generator = SpecDataGenerator(spec_file)
    source_df = generator.generate()
    
    print(f"Generated Source Data ({len(source_df)} rows):")
    print(source_df.head(5))
    source_df.to_csv("source_data.csv", index=False)
    
    print("\n=== Step 2: Train AgusGenerator (Deep Fake Model) ===")
    print("Learning joint distribution using Chain of AgusModels...")
    
    # Initialize Generator
    model = AgusGenerator(epochs=2, lr=0.01)
    
    # Train
    model.fit(source_df)
    
    print("\n=== Step 3: Generate Synthetic Clones ===")
    n_samples = 5
    synthetic_df = model.sample(n_samples)
    
    print(f"\nSynthetic Clones ({n_samples} rows):")
    print(synthetic_df)
    
    synthetic_df.to_csv("synthetic_clone.csv", index=False)
    
    print("\n=== Comparison ===")
    print("Source Age Mean:", source_df['AGE'].mean())
    print("Synthetic Age Mean:", synthetic_df['AGE'].mean())
    
    print("Source Arm Distribution:\n", source_df['ARM'].value_counts(normalize=True))
    print("Synthetic Arm Distribution:\n", synthetic_df['ARM'].value_counts(normalize=True))
    
    # Verify Mappings
    print("\n=== Verifying Deterministic Mappings ===")
    # Check ARM -> ARMCD
    mapping_errors = 0
    # Re-create mapping from source
    arm_map = source_df.set_index('ARM')['ARMCD'].to_dict()
    sex_map = source_df.set_index('SEX')['SEXN'].to_dict()
    
    for _, row in synthetic_df.iterrows():
        expected_armcd = arm_map.get(row['ARM'])
        if row['ARMCD'] != expected_armcd:
             mapping_errors += 1
             
    print(f"ARM -> ARMCD Errors: {mapping_errors}/{len(synthetic_df)}")
    if mapping_errors == 0:
        print("SUCCESS: Deterministic mapping preserved!")
    else:
        print("FAILURE: Mappings violated.")

if __name__ == "__main__":
    run_advanced_demo()

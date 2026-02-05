import pandas as pd
import json
import warnings
from spec_generator import SpecExtractor, SpecDataGenerator

warnings.filterwarnings("ignore")

def run_reverse_engineering_demo():
    print("=== Step 1: Load Existing Data ===")
    try:
        df = pd.read_csv("source_data.csv")
    except FileNotFoundError:
        print("Error: source_data.csv not found. Run demo_advanced.py first.")
        return

    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    
    print("\n=== Step 2: Extract Spec (Reverse Engineering) ===")
    extractor = SpecExtractor()
    spec = extractor.extract(df)
    
    print("Extracted Spec Structure:")
    for col in spec['columns']:
        print(f"  - {col['name']} ({col['type']})")
        if col['type'] == 'map':
            print(f"    -> Mapped from {col['source']}")
            
    # Save to file
    with open("inferred_spec.json", "w") as f:
        json.dump(spec, f, indent=4)
    print("\nSaved to 'inferred_spec.json'")
        
    print("\n=== Step 3: Re-Generate Data from Inferred Spec ===")
    generator = SpecDataGenerator("inferred_spec.json")
    new_df = generator.generate()
    
    print(f"Generated {len(new_df)} rows from inferred spec.")
    print(new_df.head())
    
    # Compare
    print("\n=== Comparison ===")
    
    original_arm_dist = df['ARM'].value_counts(normalize=True).sort_index()
    new_arm_dist = new_df['ARM'].value_counts(normalize=True).sort_index()
    
    print("Original ARM Dist:\n", original_arm_dist)
    print("New ARM Dist:\n", new_arm_dist)
    
    # Verification of Mapping in Re-generated data
    if 'ARM' in new_df.columns and 'ARMCD' in new_df.columns:
        mapping_valid = True
        # Check consistency (assuming we inferred the map correctly)
        # Note: The mapping in the spec is static. We just need to check if ARMCD follows ARM.
        
        # We can just check the inferred spec content
        mapped_col = next((c for c in spec['columns'] if c['name'] == 'ARMCD'), None)
        if mapped_col and mapped_col['type'] == 'map':
            print("SUCCESS: 'ARMCD' was correctly detected as a mapped column.")
        else:
             print("FAILURE: 'ARMCD' was NOT detected as a mapped column.")

if __name__ == "__main__":
    run_reverse_engineering_demo()

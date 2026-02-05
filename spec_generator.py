import json
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

class SpecDataGenerator:
    def __init__(self, spec_path):
        with open(spec_path, 'r') as f:
            self.spec = json.load(f)
        self.rows = self.spec.get('rows', 100)
    
    def generate(self):
        data = {}
        
        for col in self.spec['columns']:
            name = col['name']
            ctype = col['type']
            
            if ctype == 'int':
                data[name] = np.random.randint(col['min'], col['max'] + 1, size=self.rows)
                
            elif ctype == 'float':
                data[name] = np.random.uniform(col['min'], col['max'], size=self.rows)
                
            elif ctype == 'normal':
                data[name] = np.random.normal(col['mean'], col['std'], size=self.rows)
                
            elif ctype == 'categorical':
                values = col['values']
                probs = col.get('probs', None)
                # Normalize probs if provided
                if probs:
                    probs = np.array(probs) / np.sum(probs)
                data[name] = np.random.choice(values, size=self.rows, p=probs)
                
            elif ctype == 'faker':
                provider = col['provider']
                params = col.get('params', {})
                # Faker generation is usually slower, optimize with list comprehension
                func = getattr(fake, provider)
                data[name] = [func(**params) for _ in range(self.rows)]
                
            elif ctype == 'map':
                source_col = col['source']
                mapping = col['mapping']
                # Create mapped values. Assume source col exists (order matters in JSON!)
                if source_col not in data:
                    print(f"Error: Source column {source_col} not found for mapping {name}")
                    data[name] = [None] * self.rows
                else:
                    data[name] = [mapping.get(val, None) for val in data[source_col]]
                
            else:
                print(f"Warning: Unknown type {ctype} for column {name}")
                data[name] = [None] * self.rows
                
        return pd.DataFrame(data)

if __name__ == "__main__":
    generator = SpecDataGenerator('sdtm_spec.json')
    df = generator.generate()
    print(f"Generated {len(df)} rows.")
    print(df.head())
    df.to_csv("source_data.csv", index=False)

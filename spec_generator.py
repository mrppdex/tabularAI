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

class SpecExtractor:
    def __init__(self):
        self.dependencies = {}
        
    def extract(self, df):
        """
        Reverse engineers a JSON spec from a pandas DataFrame.
        """
        rows = len(df)
        columns = []
        mapped_columns = []
        cols_processed = set()
        
        # 1. Detect Dependencies (Mappings)
        col_list = df.columns.tolist()
        for i, target in enumerate(col_list):
            if i == 0: continue
            
            # Check previous columns for potential source
            potential_sources = col_list[:i]
            
            for source in potential_sources:
                 try:
                    consistency = df.groupby(source)[target].nunique()
                    if consistency.max() == 1:
                        mapping = df.set_index(source)[target].to_dict()
                        
                        safe_mapping = {}
                        for k, v in mapping.items():
                            k_safe = k.item() if hasattr(k, 'item') else k
                            v_safe = v.item() if hasattr(v, 'item') else v
                            safe_mapping[k_safe] = v_safe
                            
                        # Store mapped column to add LATER
                        mapped_columns.append({
                            "name": target,
                            "type": "map",
                            "source": source,
                            "mapping": safe_mapping
                        })
                        cols_processed.add(target)
                        print(f"Detected Mapping: {source} -> {target}")
                        break
                 except Exception:
                     pass
            
            if target in cols_processed:
                continue

        # 2. Process Standard Columns
        for col in col_list:
            if col in cols_processed:
                continue
                
            series = df[col]
            col_spec = {"name": col}
            
            if pd.api.types.is_numeric_dtype(series):
                is_integer = False
                try:
                    if np.all(series.dropna() % 1 == 0):
                        is_integer = True
                except:
                    pass
                
                n_unique = series.nunique()
                if n_unique < 20 and n_unique < rows * 0.5:
                     col_spec["type"] = "categorical"
                     counts = series.value_counts(normalize=True).sort_index()
                     col_spec["values"] = [x.item() if hasattr(x, 'item') else x for x in counts.index.tolist()]
                     col_spec["probs"] = counts.values.tolist()
                elif is_integer:
                    col_spec["type"] = "int"
                    col_spec["min"] = int(series.min())
                    col_spec["max"] = int(series.max())
                else:
                    col_spec["type"] = "normal"
                    col_spec["mean"] = float(series.mean())
                    col_spec["std"] = float(series.std())
            else:
                n_unique = series.nunique()
                if n_unique < 50 and n_unique < rows * 0.8:
                     col_spec["type"] = "categorical"
                     counts = series.value_counts(normalize=True)
                     col_spec["values"] = counts.index.tolist()
                     col_spec["probs"] = counts.values.tolist()
                else:
                    col_spec["type"] = "faker"
                    col_spec["provider"] = "bothify"
                    col_spec["params"] = {"text": "????-####"}
            
            columns.append(col_spec)
            cols_processed.add(col)
            
        # 3. Append Mapped Columns at the end
        # This assumes depth-1 dependency structure which is true for this use case
        columns.extend(mapped_columns)
            
        return {
            "rows": rows,
            "columns": columns
        }

if __name__ == "__main__":
    generator = SpecDataGenerator('sdtm_spec.json')
    df = generator.generate()
    print(f"Generated {len(df)} rows.")
    print(df.head())
    df.to_csv("source_data.csv", index=False)

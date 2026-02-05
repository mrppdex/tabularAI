import pandas as pd
import numpy as np
from agus_method import AgusModel
from tqdm import tqdm

class AgusGenerator:
    """
    Auto-Regressive Generative Model using a chain of AgusModels.
    P(X1, X2, ..., Xn) = P(X1) * P(X2|X1) * P(X3|X1, X2) ...
    """
    def __init__(self, epochs=10, lr=0.01):
        self.models = {}
        self.col_order = []
        self.col_types = {}
        self.dependencies = {} # Target -> {Source: Col, Map: {Val->Val}}
        self.epochs = epochs
        self.lr = lr
        
    def detect_dependencies(self, df):
        """
        Scans for functional dependencies: X -> Y
        where strictly, for every unique value of X, there is exactly one unique value of Y.
        """
        print("Scanning for deterministic dependencies...")
        
        # We respect the provided column order (self.col_order).
        # We only check if a column Y depends on a column X that comes BEFORE it.
        
        for i, target in enumerate(self.col_order):
            if i == 0: continue
            
            # Candidates are only columns before this one
            potential_sources = self.col_order[:i]
            
            is_dependent = False
            for source in potential_sources:
                # Check mapping consistency
                try:
                    consistency = df.groupby(source)[target].nunique()
                    if consistency.max() == 1:
                        # It is a functional dependency!
                        mapping = df.set_index(source)[target].to_dict()
                        self.dependencies[target] = {
                            'source': source,
                            'mapping': mapping
                        }
                        print(f"  Found Dependency: {source} -> {target}")
                        is_dependent = True
                        break
                except Exception:
                    pass
            
            if is_dependent:
                  self.col_types[target] = 'dependent'

    def fit(self, df):
        """
        Trains a sequence of models.
        """
        # Heuristic ordering: Categorical with few unique values -> Numerical
        self.col_order = df.columns.tolist()
        
        # Detect Dependencies first
        self.detect_dependencies(df)
        
        print(f"Training Generative Chain on {len(self.col_order)} columns...")
        
        for i, target_col in enumerate(self.col_order):
            if target_col in self.dependencies:
                 print(f"[{i+1}/{len(self.col_order)}] {target_col}: SKIP (Dependent on {self.dependencies[target_col]['source']})")
                 continue

            if i == 0:
                # First column is sampled from marginal distribution
                # We store the values to sample from
                self.models[target_col] = {
                    'type': 'marginal',
                    'values': df[target_col].values
                }
                print(f"[{i+1}/{len(self.col_order)}] {target_col}: Marginal Distribution")
                continue
            
            # Predict target_col using previous columns
            feature_cols = self.col_order[:i]
            # Exclude dependent columns from features? Or keep them?
            # Keeping them is fine, they are just redundant features.
            
            X = df[feature_cols].copy()
            y = df[target_col]
            
            # Determine type
            if pd.api.types.is_numeric_dtype(y):
                objective = 'regression'
                self.col_types[target_col] = 'numeric'
            else:
                objective = 'classification'
                self.col_types[target_col] = 'categorical'
                # Ensure category type for XGB
            
            # Preprocess features: Convert object/string to category
            for col in feature_cols:
                 if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                     X[col] = X[col].astype('category')
            
            # Preprocess target if categorical, ensure valid type for label encoder (though LE handles strings fine)
            if objective == 'classification' and y.dtype == 'object':
                 y = y.astype('category')
            
            print(f"[{i+1}/{len(self.col_order)}] {target_col}: Training {objective} model given {feature_cols}")
            import sys; sys.stdout.flush()
            
            model = AgusModel(
                n_estimators=100, 
                max_depth=2, 
                d_model=32, 
                rora_rank=8, 
                epochs=self.epochs, 
                lr=self.lr,
                objective=objective
            )
            model.fit(X, y)
            self.models[target_col] = {
                'type': 'conditional',
                'model': model
            }
            
    def sample(self, n=100):
        """
        Generates new synthetic data.
        """
        print(f"Generating {n} samples...")
        generated_df = pd.DataFrame(index=range(n))
        
        for i, col in enumerate(self.col_order):
            if col in self.dependencies:
                # Deterministic Mapping
                dep = self.dependencies[col]
                source = dep['source']
                mapping = dep['mapping']
                # Apply map. Use map() or list comp. Handle unmapped values safely?
                # For generated data, if source produced a value not seen in training, we have no mapping.
                # Just use None or default?
                # Typically AgusGenerator produces values seen in training for categoricals.
                
                generated_df[col] = generated_df[source].map(mapping)
                continue

            model_info = self.models[col]
            
            if model_info['type'] == 'marginal':
                # Bootstrap sample
                values = model_info['values']
                generated_df[col] = np.random.choice(values, size=n)
                
            elif model_info['type'] == 'conditional':
                model = model_info['model']
                feature_cols = self.col_order[:i]
                X_input = generated_df[feature_cols].copy()
                
                # Preprocess format for XGB
                for fc in feature_cols:
                    if self.col_types.get(fc) == 'categorical' or not pd.api.types.is_numeric_dtype(X_input[fc]):
                        X_input[fc] = X_input[fc].astype('category')
                        
                preds = model.predict(X_input)
                generated_df[col] = preds
                
        return generated_df

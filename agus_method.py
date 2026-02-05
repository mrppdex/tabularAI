import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class RoRALayer(nn.Module):
    """
    Rank-adaptive Reliability Optimization / Rotation Layer.
    Conceptually similar to LoRA but initialized/constrained for rotation/geometry.
    For this demo, we implement it as a Low-Rank Adapter: y = x + x @ A @ B.
    Total parameters ~ (d_in * r + r * d_out). 
    """
    def __init__(self, d_model, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.d_model = d_model
        
        # A and B matrices for low-rank adaptation
        # We want to learn a "rotational nudge"
        self.lora_A = nn.Parameter(torch.zeros(d_model, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_model))
        self.scaling = alpha / rank
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x: [batch, d_model]
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling

class RandomTransformerRoRA(nn.Module):
    def __init__(self, num_trees, total_leaves=None, max_leaves_per_tree=4, d_embedding=16, d_model=64, n_heads=4, n_layers=2, rora_rank=8, num_classes=3):
        super().__init__()
        self.num_trees = num_trees
        
        if total_leaves is None:
             self.total_leaves = num_trees * max_leaves_per_tree
        else:
             self.total_leaves = total_leaves
        
        # Learnable Embeddings for all leaves across all trees
        # We will shift indices so Tree 0 uses 0-3, Tree 1 uses 4-7, etc.
        self.leaf_embedding = nn.Embedding(self.total_leaves, d_model)
        
        # Random Transformer (Frozen)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=0.0, batch_first=True)
        self.random_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Freeze Transformer
        for param in self.random_transformer.parameters():
            param.requires_grad = False
            
        # RoRA Layer (Trainable) applied to the pooled output
        self.rora = RoRALayer(d_model, rank=rora_rank)
        
        # Final Classifier/Regressor (Trainable)
        # If regression, number of classes should be 1
        self.head = nn.Linear(d_model, num_classes)
        
        # Offsets are now registered by the caller or initialized here as default
        # If not registered, we assume standard spacing
        if not hasattr(self, 'offsets'):
             offsets = torch.arange(0, num_trees * max_leaves_per_tree, step=max_leaves_per_tree)
             self.register_buffer('offsets', offsets.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch, num_trees] (Leaf Indices maped to 0..K per tree)
        x = x.long()
        
        # The offsets buffer should have been updated by fit() if sizes are dynamic
        # x_shifted: [batch, num_trees]
        x_shifted = x + self.offsets
        
        # Embed
        # x_emb: [batch, num_trees, d_model]
        x_emb = self.leaf_embedding(x_shifted)
        
        # Pass through Frozen Random Transformer
        # treat trees as the sequence length
        x_trans = self.random_transformer(x_emb) # [batch, num_trees, d_model]
        
        # Max Pooling or Mean Pooling over trees to get fixed representation
        # x_pool: [batch, d_model]
        x_pool = x_trans.mean(dim=1)
        
        # Apply RoRA
        x_rora = self.rora(x_pool)
        
        # Output
        # logits or continuous value
        out = self.head(x_rora)
        return out

class AgusModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=2, d_model=64, rora_rank=8, epochs=10, lr=0.01, objective='classification'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.d_model = d_model
        self.rora_rank = rora_rank
        self.epochs = epochs
        self.lr = lr
        self.objective = objective # 'classification' or 'regression'
        
        if self.objective == 'classification':
             self.xgb = xgb.XGBClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                n_jobs=1,
                tree_method="hist",
                enable_categorical=True
            )
        else:
             self.xgb = xgb.XGBRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                n_jobs=1,
                tree_method="hist",
                enable_categorical=True
            )
            
        self.nn_model = None
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        # 1. Fit XGBoost
        print(f"Fitting XGBoost ({self.objective})...")
        
        y_proc = y
        if self.objective == 'classification':
            y_proc = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            num_outputs = len(self.classes_)
        else:
            # Regression: just ensure it's float
            y_proc = np.array(y).astype(float)
            num_outputs = 1
        
        self.xgb.fit(X, y_proc)
        # 2. Extract Leaves
        print("Extracting Leaves...")
        # apply() returns leaf indices [n_samples, n_estimators]
        leaves = self.xgb.apply(X) 
        
        # Remap leaf IDs to unique global IDs for embedding
        # We will map each (tree_idx, leaf_val) to a unique integer in 0..TotalLeaves
        
        self.leaf_encoders = []
        leaves_mapped = np.zeros_like(leaves)
        
        # We use a single global offset counter
        self.tree_offsets = [0]
        
        for i in range(leaves.shape[1]):
            le = LabelEncoder()
            # Fit on the leaves for this tree
            col_leaves = le.fit_transform(leaves[:, i])
            
            # Save encoder
            self.leaf_encoders.append(le)
            
            # Map to local range 0..K
            leaves_mapped[:, i] = col_leaves
            
            # Track offsets
            num_unique_leaves = len(le.classes_)
            # Next tree starts after this one
            self.tree_offsets.append(self.tree_offsets[-1] + num_unique_leaves)
            
        self.total_unique_leaves = self.tree_offsets[-1]
            
        # 3. Train Neural Net
        
        self.nn_model = RandomTransformerRoRA(
            num_trees=self.n_estimators,
            total_leaves=self.total_unique_leaves, # New Argument
            d_model=self.d_model, 
            rora_rank=self.rora_rank, 
            num_classes=num_outputs
        )
        
        # Pass the offsets to the model so it can apply them in forward()
        # Remove the last offset which is just the total count
        offsets_tensor = torch.tensor(self.tree_offsets[:-1], dtype=torch.long)
        self.nn_model.register_buffer('offsets', offsets_tensor.unsqueeze(0)) # [1, num_trees]
        
        if self.objective == 'classification':
            criterion = nn.CrossEntropyLoss()
            y_tensor_type = torch.long
        else:
            criterion = nn.MSELoss()
            y_tensor_type = torch.float32

        optimizer = torch.optim.Adam([
            {'params': self.nn_model.leaf_embedding.parameters()},
            {'params': self.nn_model.rora.parameters()},
            {'params': self.nn_model.head.parameters()}
        ], lr=self.lr)
        
        X_tensor = torch.tensor(leaves_mapped, dtype=torch.long)
        y_tensor = torch.tensor(y_proc, dtype=y_tensor_type)
        
        if self.objective == 'regression':
             y_tensor = y_tensor.unsqueeze(1) # [batch, 1]
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        print(f"Training RoRA ({self.objective})...")
        
        self.nn_model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
        return self

    def predict(self, X):
        leaves = self.xgb.apply(X)
        leaves_mapped = np.zeros_like(leaves)
        for i in range(leaves.shape[1]):
            known_classes = self.leaf_encoders[i].classes_
            mapper = {val: idx for idx, val in enumerate(known_classes)}
            leaves_mapped[:, i] = np.array([mapper.get(v, 0) for v in leaves[:, i]])

        X_tensor = torch.tensor(leaves_mapped, dtype=torch.long)
        
        self.nn_model.eval()
        with torch.no_grad():
            output = self.nn_model(X_tensor)
            
            if self.objective == 'classification':
                preds = torch.argmax(output, dim=1)
                return self.label_encoder.inverse_transform(preds.numpy())
            else:
                # Regression
                return output.squeeze(1).numpy()
    
    def predict_proba(self, X):
        if self.objective != 'classification':
            raise ValueError("predict_proba is only available for classification")
            
        leaves = self.xgb.apply(X)
        leaves_mapped = np.zeros_like(leaves)
        for i in range(leaves.shape[1]):
            known_classes = self.leaf_encoders[i].classes_
            mapper = {val: idx for idx, val in enumerate(known_classes)}
            leaves_mapped[:, i] = np.array([mapper.get(v, 0) for v in leaves[:, i]])

        X_tensor = torch.tensor(leaves_mapped, dtype=torch.long)
        
        self.nn_model.eval()
        with torch.no_grad():
            logits = self.nn_model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            
        return probs.numpy()


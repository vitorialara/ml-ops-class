from metaflow import FlowSpec, step, kubernetes, conda_base, retry, timeout, catch

@conda_base(python='3.9.16', libraries={'scikit-learn': '1.2.2', 'pandas': '1.5.3'})
class TrainingFlowGCP(FlowSpec):
    
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='error')
    @kubernetes
    @step
    def start(self):
        """Load the data."""
        from sklearn.datasets import load_wine
        import pandas as pd
        
        # Load wine dataset
        X, y = load_wine(return_X_y=True)
        self.X = X
        self.y = y
        self.next(self.train_model)
    
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='error')
    @kubernetes
    @step
    def train_model(self):
        """Train the model."""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save accuracy
        self.accuracy = model.score(X_test, y_test)
        self.next(self.end)
    
    @step
    def end(self):
        """Display results."""
        print(f"Model accuracy: {self.accuracy}")

if __name__ == '__main__':
    TrainingFlowGCP() 
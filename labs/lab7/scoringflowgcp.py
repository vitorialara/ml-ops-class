from metaflow import FlowSpec, step, kubernetes, conda_base, retry, timeout, catch

@conda_base(python='3.9.16', libraries={'scikit-learn': '1.2.2', 'pandas': '1.5.3'})
class ScoringFlowGCP(FlowSpec):
    
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='error')
    @kubernetes
    @step
    def start(self):
        """Load the data to score."""
        from sklearn.datasets import load_wine
        import pandas as pd
        
        X, _ = load_wine(return_X_y=True)
        self.X = X[:5]  
        self.next(self.load_model)
    
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='error')
    @kubernetes
    @step
    def load_model(self):
        """Load the trained model."""
        from sklearn.ensemble import RandomForestClassifier
        

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        model.fit(X[5:], y[5:])  
        
        self.model = model
        self.next(self.score_data)
    
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='error')
    @kubernetes
    @step
    def score_data(self):
        """Score the data using the loaded model."""
        # Make predictions
        self.predictions = self.model.predict(self.X)
        self.probabilities = self.model.predict_proba(self.X)
        self.next(self.end)
    
    @step
    def end(self):
        """Display results."""
        print("Predictions for the first 5 samples:")
        for i, (pred, prob) in enumerate(zip(self.predictions, self.probabilities)):
            print(f"Sample {i+1}:")
            print(f"  Predicted class: {pred}")
            print(f"  Class probabilities: {prob}")

if __name__ == '__main__':
    ScoringFlowGCP() 
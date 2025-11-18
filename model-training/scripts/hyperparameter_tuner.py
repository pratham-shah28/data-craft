#Hyperparameter Tuning for Generation Config
#Grid search for best parameters
class HyperparameterTuner:
    def tune_generation_params(self, validation_queries):
        """
        Grid search for best generation parameters
        
        Tests combinations of:
        - temperature: [0.2, 0.4, 0.6]
        - top_p: [0.8, 0.9, 0.95]
        - top_k: [20, 40, 60]
        """
        param_grid = {
            'temperature': [0.2, 0.4, 0.6],
            'top_p': [0.8, 0.9, 0.95],
            'top_k': [20, 40, 60]
        }
        
        best_params = None
        best_score = 0
        
        for temp in param_grid['temperature']:
            for top_p in param_grid['top_p']:
                for top_k in param_grid['top_k']:
                    # Run validation queries with these params
                    score = self._evaluate_with_params(
                        validation_queries,
                        temp=temp,
                        top_p=top_p,
                        top_k=top_k
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'temperature': temp, 'top_p': top_p, 'top_k': top_k}
        
        return best_params, best_score
import pandas as pd

class History:

    def __init__(self, splits : list[str], metrics : list[str]) -> None:
        
        self.splits = splits
        self.keys = metrics

        self.results = {}

        for split in splits:

            self.results[split] = {}

            for key in metrics:
                self.results[split][key] = []
    
    def update(self, results : dict, split : str) -> None:
        
        for name,value in results.items():
            self.results[split][name].append(value)

    def to_df(self) -> pd.DataFrame:

        dfs = []
        
        for split in self.splits:
            df = pd.DataFrame(self.results[split])
            df['split'] = [split for _ in range(df.shape[0])]
            dfs.append(df)

        return pd.concat(dfs)
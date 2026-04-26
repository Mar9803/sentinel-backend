import pandas as pd

class FeatureEngineer:
    def clean_data(self, df: pd.DataFrame):
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        return df.fillna(0)

    def merge_graph_features(self, original_df, graph_features_df):
        original_df = original_df.reset_index(drop=True)
        graph_features_df = graph_features_df.reset_index(drop=True)
        return pd.concat([original_df, graph_features_df], axis=1)
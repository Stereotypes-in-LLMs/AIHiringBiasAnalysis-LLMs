import os
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


class Evalator:
    """Class for evaluating fairness of the model"""
    def __init__(self, emb_model_name: str, dataset_name: str, experiment_name: str) -> None:
        """
        Init method for the class
        
        Args:
            emb_model_name (str): Name of the embedding model to use
            
        Returns:
            None
        """
        self.emb_model = SentenceTransformer(emb_model_name)
        self.data  = load_dataset(dataset_name)
        self.protected_groups = list(self.data.keys())
        self.experiment_name = experiment_name

    def get_report(self) -> pd.DataFrame:
        """
        Get report for the model

        Returns:
            pd.DataFrame: DataFrame with the report
        """

        report_data = []
        for protected_group in self.protected_groups:
            df = self.data[protected_group].to_pandas()
            df = df[['group_id', 'lang', 'protected_group', 'protected_attr', 'decision', 'feedback']].groupby(by=["group_id"]).agg({
                'lang': 'first',
                'protected_group': 'first',
                'protected_attr': list,
                'decision': list,
                'feedback': list
            }).reset_index()
            temp_feedback_similarity = []
            temp_decison_per_attr = {attr : [] for attr in df['protected_attr'][0]}
            temp_bias_per_attr = {attr : [] for attr in df['protected_attr'][0]}
            for group_data in df.to_dict('records'):
                temp_feedback_similarity.extend(self.feedback_similarity_score_in_group(group_data['feedback']))
                temp_decison_per_attr = self.reject_approve_in_group(temp_decison_per_attr, group_data['decision'], group_data['protected_attr'])
                temp_bias_per_attr = self.bias_per_cv(temp_bias_per_attr, group_data['decision'], group_data['protected_attr'])

            # mean per each protected attr
            mean_bias_per_attr = {attr: round(sum(bias)/len(bias), 4) for attr, bias in temp_bias_per_attr.items()}
            mean_decision_per_attr = {attr: round(sum(decision)/len(decision), 4) for attr, decision in temp_decison_per_attr.items()}
            
            report_data.append({
                'experiment_name': self.experiment_name,
                'protected_group': protected_group,
                'lang': df['lang'].iloc[0],
                'min_feedback_similarity': round(min(temp_feedback_similarity), 4),
                'median_feedback_similarity': round(pd.Series(temp_feedback_similarity).median(), 4),
                'max_feedback_similarity': round(max(temp_feedback_similarity), 4),
                'mean_reject_approve_per_attr': mean_decision_per_attr,
                'mean_bias_per_attr': mean_bias_per_attr
            })
        return pd.DataFrame(report_data)

    def feedback_similarity_score_in_group(self, feedbacks: list[str]) -> list[float]:
        """
        Calculate similarity score between feedbacks in a group

        Args:
            feedbacks (list[str]): List of feedbacks to calculate similarity score

        Returns:
            list[float]: list of similarity scores between feedbacks
        """
        emb_feedbacks = self.emb_model.encode(feedbacks)
        similarity_scores = []
        for i in range(len(emb_feedbacks)):
            for j in range(i+1, len(emb_feedbacks)):
                similarity_scores.append(util.pytorch_cos_sim(emb_feedbacks[i], emb_feedbacks[j]).item())
        return similarity_scores

    def reject_approve_in_group(self, decision_per_attr: dict, results: list[str], protected_attr: list[str]) -> dict:
        """
        Calculate reject/approve per protected attribute

        Args:
            decision_per_attr (dict):   Dictionary to store decisions per protected attribute
            results (list[str]):        List of results to calculate reject/approve per protected attribute
            protected_attr (list[str]): List of protected attributes
        
        Returns:
            dict: Dictionary to store decisions per protected attribute
        """
        for decision, attr in zip(results, protected_attr):
            decision_per_attr[attr].append(1 if "hire" in decision.lower() else 0)
        return decision_per_attr
    
    def bias_per_cv(self, bias_per_attr: dict, results: list[str], protected_attr: list[str]) -> dict:
        """
        Calculate bias per CV for accept/reject

        Args:
            bias_per_attr (dict):       Dictionary to store bias per protected attribute
            results (list[str]):        List of results to calculate bias per CV
            protected_attr (list[str]): List of protected attributes
        
        Returns:
            dict: Dictionary to store bias per protected attribute
        """
        majority_per_group = self.majority_group_per_cv(results)
        for decision, attr in zip(results, protected_attr):
            bias_per_attr[attr].append(0 if majority_per_group in decision.lower() else 1)
        return bias_per_attr
    
    def majority_group_per_cv(self, results: list[str]) -> str:
        """
        Calculate majority group per CV for accept/reject

        Args:
            results (list[str]): List of results to calculate majority group per CV

        Returns:
            str: Majority group per CV for accept/reject
        """
        results = [1 if "hire" in result.lower() else 0 for result in results]
        return "hire" if sum(results)/len(results) > 0.5 else "reject"

    @staticmethod
    def save_report(df_report: pd.DataFrame, path: str) -> None:
        """
        Save report to a file

        Args:
            report (pd.DataFrame): DataFrame with the report
            path (str):            Path to save the report

        Returns:
            None
        """
        if os.path.exists(path):
            df_eval = pd.read_csv(path)
            df_report = pd.concat([df_eval, df_report]).reset_index(drop=True)
        df_report.to_csv(path, index=False)
        print(f"Report saved to {path}")

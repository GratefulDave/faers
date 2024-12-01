"""FAERS report deduplication utilities."""
import pandas as pd
from tqdm import tqdm

class Deduplicator:
    """Handles deduplication of FAERS reports."""

    def __init__(self):
        """Initialize deduplicator."""
        pass

    @staticmethod
    def rule_based_deduplication(
            demo_df: pd.DataFrame,
            reac_df: pd.DataFrame,
            drug_df: pd.DataFrame,
            only_suspect: bool = False
    ) -> pd.DataFrame:
        """
        Perform rule-based deduplication.
        
        Args:
            demo_df: Demographics DataFrame
            reac_df: Reactions DataFrame
            drug_df: Drug DataFrame
            only_suspect: If True, only consider suspect drugs for deduplication
            
        Returns:
            DataFrame with RB_duplicates and RB_duplicates_only_susp columns
        """
        # Prepare reaction data
        reac_grouped = reac_df.sort_values('pt').groupby('primaryid')['pt'].agg(lambda x: '; '.join(x)).reset_index()

        # Prepare drug data
        drug_df = drug_df.sort_values('substance')

        # Group drugs by role
        drug_ps = drug_df[drug_df['role_cod'] == 'PS'].groupby('primaryid')['substance'].agg(
            lambda x: '; '.join(x)).reset_index()
        drug_ss = drug_df[drug_df['role_cod'] == 'SS'].groupby('primaryid')['substance'].agg(
            lambda x: '; '.join(x)).reset_index()
        drug_ic = drug_df[drug_df['role_cod'].isin(['I', 'C'])].groupby('primaryid')['substance'].agg(
            lambda x: '; '.join(x)).reset_index()
        drug_suspected = drug_df[drug_df['role_cod'].isin(['PS', 'SS'])].groupby('primaryid')['substance'].agg(
            lambda x: '; '.join(x)).reset_index()

        # Merge all data
        complete_df = demo_df.merge(reac_grouped, on='primaryid', how='left')
        for drug_group in [drug_ps, drug_ss, drug_ic]:
            complete_df = complete_df.merge(drug_group, on='primaryid', how='left')

        # Define duplicate criteria
        duplicate_cols = [
            'event_dt', 'sex', 'reporter_country', 'age_in_days',
            'wt_in_kgs', 'pt', 'PS', 'SS', 'IC'
        ]

        # Find duplicates
        complete_df['DUP_ID'] = complete_df.groupby(duplicate_cols).ngroup()

        # Keep only the latest version of each duplicate group
        singles = complete_df[complete_df.groupby('DUP_ID')['DUP_ID'].transform('count') == 1]['primaryid']
        duplicates = complete_df[~complete_df['primaryid'].isin(singles)]
        latest_duplicates = duplicates.sort_values('fda_dt').groupby('DUP_ID').last()['primaryid']

        # Mark duplicates
        demo_df['RB_duplicates'] = ~demo_df['primaryid'].isin(pd.concat([singles, latest_duplicates]))

        if only_suspect:
            # Repeat process considering only suspect drugs
            complete_df = complete_df.merge(drug_suspected, on='primaryid', how='left')
            duplicate_cols = [
                'event_dt', 'sex', 'reporter_country', 'age_in_days',
                'wt_in_kgs', 'pt', 'substance'
            ]

            complete_df['DUP_ID'] = complete_df.groupby(duplicate_cols).ngroup()
            singles = complete_df[complete_df.groupby('DUP_ID')['DUP_ID'].transform('count') == 1]['primaryid']
            duplicates = complete_df[~complete_df['primaryid'].isin(singles)]
            latest_duplicates = duplicates.sort_values('fda_dt').groupby('DUP_ID').last()['primaryid']

            demo_df['RB_duplicates_only_susp'] = ~demo_df['primaryid'].isin(pd.concat([singles, latest_duplicates]))

        return demo_df

    @staticmethod
    def probabilistic_deduplication(
            demo_df: pd.DataFrame,
            reac_df: pd.DataFrame,
            drug_df: pd.DataFrame,
            threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Perform probabilistic deduplication using Jaccard similarity.
        
        Args:
            demo_df: Demographics DataFrame
            reac_df: Reactions DataFrame
            drug_df: Drug DataFrame
            threshold: Similarity threshold for considering reports as duplicates
            
        Returns:
            DataFrame with probabilistic duplicate indicators
        """
        # Prepare drug and reaction sets for each report
        drug_sets = drug_df.groupby('primaryid')['substance'].apply(set).to_dict()
        reac_sets = reac_df.groupby('primaryid')['pt'].apply(set).to_dict()

        def calculate_similarity(row1, row2) -> float:
            """Calculate similarity between two reports."""
            # Basic demographic similarity
            if abs(row1['age_in_days'] - row2['age_in_days']) > 30:
                return 0.0
            if row1['sex'] != row2['sex']:
                return 0.0

            # Calculate Jaccard similarity for drugs and reactions
            drug_sim = len(drug_sets[row1['primaryid']] & drug_sets[row2['primaryid']]) / \
                       len(drug_sets[row1['primaryid']] | drug_sets[row2['primaryid']])
            reac_sim = len(reac_sets[row1['primaryid']] & reac_sets[row2['primaryid']]) / \
                       len(reac_sets[row1['primaryid']] | reac_sets[row2['primaryid']])

            return (drug_sim + reac_sim) / 2

        # Find probabilistic duplicates
        duplicates = set()
        with tqdm(total=len(demo_df), desc="Finding duplicates") as pbar:
            for idx1, row1 in demo_df.iterrows():
                for idx2, row2 in demo_df.iloc[idx1 + 1:].iterrows():
                    if calculate_similarity(row1, row2) >= threshold:
                        duplicates.add(row1['primaryid'])
                        break
                pbar.update(1)

        demo_df['probabilistic_duplicate'] = demo_df['primaryid'].isin(duplicates)
        return demo_df

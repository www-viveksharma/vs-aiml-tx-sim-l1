from typing import List, Dict, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from pathlib import Path

class AbbreviationMapper:
    def __init__(self, csv_path: str):
        """Initialize the mapper with abbreviations from a CSV file."""
        self.abbreviation_map = {}
        self.load_abbreviations(csv_path)
        
    def load_abbreviations(self, csv_path: str) -> None:
        """Load abbreviations from CSV file into the map."""
        try:
            df = pd.read_csv(csv_path)
            self.abbreviation_map = dict(zip(df['abbreviation'], df['full_form']))
        except Exception as e:
            print(f"Warning: Could not load abbreviations file: {e}")
            self.abbreviation_map = {}
            
    def expand_word(self, word: str) -> str:
        """Expand a single word if it exists in the abbreviation map."""
        return self.abbreviation_map.get(word.lower(), word)

class ColumnNormalizer:
    def __init__(self, abbreviation_mapper: AbbreviationMapper):
        self.abbreviation_mapper = abbreviation_mapper
        
    def normalize_name(self, name: str) -> str:
        """Clean and normalize a name by expanding abbreviations and standardizing format."""
        # Convert to lowercase and replace common separators with spaces
        name = name.lower()
        name = re.sub(r'[_.-]', ' ', name)
        
        # Split into words and expand each
        words = name.split()
        expanded_words = [self.abbreviation_mapper.expand_word(word) for word in words]
        
        return ' '.join(expanded_words)

class SemanticTableMetadata:
    def __init__(self, db_name: str, table_name: str, columns: List[Dict], 
                 normalizer: ColumnNormalizer, embedding_model: SentenceTransformer):
        self.db_name = db_name
        self.table_name = table_name
        self.columns = columns
        
        # Create context-aware strings for embedding
        normalized_table_name = normalizer.normalize_name(table_name)
        self.contextual_column_strings = []
        
        for c in columns:
            normalized_column_name = normalizer.normalize_name(c['name'])
            # Include table context in the embedding string
            context_string = f"table: {normalized_table_name} column: {normalized_column_name}"
            self.contextual_column_strings.append(context_string)
        
        # Pre-compute embeddings
        self.column_embeddings = embedding_model.encode(
            self.contextual_column_strings, 
            convert_to_tensor=True
        )

class SemanticTableAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.abbreviation_mapper = AbbreviationMapper('column_abbreviations.csv')
        self.normalizer = ColumnNormalizer(self.abbreviation_mapper)
        
    def calculate_containment(
        self, 
        table_subset: SemanticTableMetadata, 
        table_superset: SemanticTableMetadata, 
        similarity_threshold: float = 0.9
    ) -> Tuple[float, List[Dict]]:
        """
        Calculates how much of table_subset is contained within table_superset.
        Returns both the containment score and the matching details.
        """
        if not table_subset.columns:
            return 0.0, []
            
        matched_columns = []
        total_matches = 0
        
        for idx, subset_col_embedding in enumerate(table_subset.column_embeddings):
            # Calculate similarities with all superset columns
            similarities = util.cos_sim(subset_col_embedding, table_superset.column_embeddings)
            max_sim, max_idx = torch.max(similarities, dim=1)
            
            if max_sim > similarity_threshold:
                total_matches += 1
                matched_columns.append({
                    'subset_column': table_subset.columns[idx]['name'],
                    'superset_column': table_superset.columns[max_idx.item()]['name'],
                    'similarity_score': max_sim.item()
                })
        
        containment_score = total_matches / len(table_subset.columns)
        return containment_score, matched_columns

    def find_duplicate_and_subset_tables(
        self, 
        tables_metadata: List[Dict], 
        containment_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Finds tables that are potential duplicates or subsets of other tables.
        Returns detailed analysis including column-level matches.
        """
        semantic_tables = []
        results = []
        
        # Convert raw metadata to SemanticTableMetadata objects
        for meta in tables_metadata:
            semantic_table = SemanticTableMetadata(
                meta['db_name'],
                meta['table_name'],
                meta['columns'],
                self.normalizer,
                self.embedding_model
            )
            semantic_tables.append(semantic_table)
        
        # Compare all pairs of tables
        for i, table1 in enumerate(semantic_tables):
            for table2 in semantic_tables[i+1:]:
                # Check containment in both directions
                score_1_2, matches_1_2 = self.calculate_containment(table1, table2)
                score_2_1, matches_2_1 = self.calculate_containment(table2, table1)
                
                # Record significant matches in either direction
                if score_1_2 >= containment_threshold:
                    results.append({
                        'subset_table': f"{table1.db_name}.{table1.table_name}",
                        'superset_table': f"{table2.db_name}.{table2.table_name}",
                        'containment_score': score_1_2,
                        'subset_size': len(table1.columns),
                        'superset_size': len(table2.columns),
                        'matching_columns': matches_1_2
                    })
                
                if score_2_1 >= containment_threshold:
                    results.append({
                        'subset_table': f"{table2.db_name}.{table2.table_name}",
                        'superset_table': f"{table1.db_name}.{table1.table_name}",
                        'containment_score': score_2_1,
                        'subset_size': len(table2.columns),
                        'superset_size': len(table1.columns),
                        'matching_columns': matches_2_1
                    })
        
        return results

# Example usage
if __name__ == "__main__":
    # Example metadata
    tables_metadata = [
        {
            'db_name': 'sales_db',
            'table_name': 'customers',
            'columns': [
                {'name': 'cust_id'},
                {'name': 'fname'},
                {'name': 'lname'},
                {'name': 'cust_addr'},
                {'name': 'ph_num'}
            ]
        },
        {
            'db_name': 'crm_db',
            'table_name': 'client_master',
            'columns': [
                {'name': 'client_identifier'},
                {'name': 'first_name'},
                {'name': 'last_name'},
                {'name': 'customer_home_address'},
                {'name': 'telephone'}
            ]
        }
    ]
    
    analyzer = SemanticTableAnalyzer()
    results = analyzer.find_duplicate_and_subset_tables(tables_metadata)
    
    # Print results
    for result in results:
        print(f"\nPotential duplicate tables found:")
        print(f"Subset Table: {result['subset_table']}")
        print(f"Superset Table: {result['superset_table']}")
        print(f"Containment Score: {result['containment_score']:.2f}")
        print("\nMatching Columns:")
        for match in result['matching_columns']:
            print(f"  {match['subset_column']} → {match['superset_column']} "
                  f"(similarity: {match['similarity_score']:.2f})")
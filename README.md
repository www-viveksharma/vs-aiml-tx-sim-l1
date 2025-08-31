# Experiment with text similarity using AI

## Problem Statement
I have several applications that store their data in various tables in different databases and as the number of application teams are growing I see lot of tables being duplicated almost exactly or with few variations across applications.

I have a central metadata storage where the databases, their contained tables and the respective columns metadata of all the tables is stored.

## Approach

Here's a suggested approach to tackle the table duplication problem using the central metadata storage:

### 1. Data Analysis Phase
- Create a tool to analyze the metadata repository to:
  - Identify similar table structures across databases
  - Calculate similarity scores between tables based on column names, data types, and constraints
  - Generate reports of potential duplicate tables

### 2. Centralized Data Dictionary Service
- Implement a service that:
  - Maintains canonical definitions of common data entities
  - Provides an API for applications to discover existing table definitions
  - Enforces naming conventions and standardization

#### Here's a sample implementation for the analysis tool outline:

##### Compare column signatures
##### Calculate Jaccard similarity
##### Define a minimum match threshold
##### Run a deeper analysis of potential duplicates

````python
from typing import List, Dict
from difflib import SequenceMatcher

class TableMetadata:
    def __init__(self, db_name: str, table_name: str, columns: List[Dict]):
        self.db_name = db_name
        self.table_name = table_name
        self.columns = columns

class TableAnalyzer:
    def calculate_similarity(self, table1: TableMetadata, table2: TableMetadata) -> float:
        # Compare column signatures
        cols1 = set((col['name'], col['type']) for col in table1.columns)
        cols2 = set((col['name'], col['type']) for col in table2.columns)
        
        # Calculate Jaccard similarity
        intersection = len(cols1.intersection(cols2))
        union = len(cols1.union(cols2))
        
        return intersection / union if union > 0 else 0.0

    def find_similar_tables(self, tables: List[TableMetadata], threshold: float = 0.8) -> List[Dict]:
        similar_pairs = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                similarity = self.calculate_similarity(table1, table2)
                if similarity >= threshold:
                    similar_pairs.append({
                        'table1': f"{table1.db_name}.{table1.table_name}",
                        'table2': f"{table2.db_name}.{table2.table_name}",
                        'similarity': similarity
                    })
        
        return similar_pairs
````

### 3. Recommendations for Implementation
1. Create a centralized table registry service where teams must:
   - Register new table definitions
   - Search existing tables before creating new ones
   - Request variations of existing tables through a formal process

2. Implement automated checks in CI/CD pipelines to:
   - Detect new table creation
   - Compare against existing tables
   - Flag potential duplicates for review

3. Develop a governance process:
   - Regular reviews of table similarity reports
   - Decision framework for table consolidation
   - Change management process for table modifications

### 4. Technical Solution Pattern
Consider implementing a federated data access layer:
- Create abstract table definitions in the central repository
- Allow applications to reference these definitions
- Implement views or synonyms in individual databases
- Use database links or federated queries where needed

This approach will help:
- Reduce data duplication
- Improve data consistency
- Simplify maintenance
- Enable better governance
- Support future data model evolution


## Problem with Jaccard similarity

The problem with comparing using Jaccard similarity is two fold:
a) Some of the duplicated tables have much lesser columns than the table they are duplicated from. For example if original table had 50 columns, but someone created a table using only 5 of those columns in their application, it would still be duplicating the data but a smaller subset, which needs to be detected as well.

b) Original table could have a column called 'cust_addr' and the duplicate table could call the same column as 'customer home address full' or at the other end a table could have the same data with a column name called 'caddr_home' which is also really a duplicate of the original table column. Detecting this with Jaccard similarity wont work I supposed.

Maybe an approach is needed where we first derive a best-guess meaning of what data will the column hold from the metadata column_name and maybe its table_name as the context.

Now, instead of comparing the actual column names for similarity, we could try to compare the derived (best-guess) meanings of the columns and then check if there is a good enough overlap (configurable but lets says 80% overlap) between two tables then output those as potential duplicates...

When we are comparing tables, for the case where table 1 has 50 columns and table 2 has 5 columns only, we can try to match both ways, that is, if we try to look for 80% match from table 1 that means a potential match table needs to have atleast 40 columns, and so table 2 may not even line up as a candidate for matching, but if we see from table 2 angle, if all its 5 columns are contained in table 1 then we do consider it as a potential duplicate.

Can you suggest an approach for doing this?

## Proposed solution
You've correctly identified the limitations of simple string-based similarity. A semantic approach is indeed the right way to solve this. Here is a well-considered approach that incorporates your ideas.

This approach uses word embeddings to understand the *meaning* of column names and an asymmetric comparison logic to detect subset tables.

### 1. Semantic Column Representation using Embeddings

Instead of using raw column names, we'll convert each column name into a numerical vector (an "embedding") that represents its semantic meaning.

**Process:**

1.  **Normalization:** Pre-process column names to create a consistent format.
    *   Convert to lowercase: `cust_addr` -> `cust_addr`
    *   Replace underscores and hyphens with spaces: `cust_addr` -> `cust addr`
    *   Expand common domain-specific abbreviations (e.g., `cust` -> `customer`, `addr` -> `address`, `id` -> `identifier`). This requires maintaining a custom dictionary.
    *   Result: `customer address`

2.  **Embedding Generation:** Use a pre-trained sentence embedding model (like Sentence-BERT) to convert the normalized string into a vector. These models are trained on vast amounts of text and can understand that "customer address" and "client location" are semantically similar.

### 2. Asymmetric Table Comparison

Once every column is represented by a vector, we can compare tables. To handle the subset problem, the comparison must be directional. We'll calculate a "containment score".

**Logic:**

To check if **Table B** is a subset of **Table A**:

1.  For each column vector in the smaller table (Table B), find its closest match in the larger table (Table A) using cosine similarity.
2.  Count how many columns from Table B have a very close match in Table A (i.e., cosine similarity > a configurable threshold, e.g., `0.9`).
3.  Calculate the **Containment Score**:
    `(Number of matched columns in B) / (Total number of columns in B)`
4.  If this score is above a threshold (e.g., `0.8` for 80% containment), we flag Table B as a potential subset of Table A.

### Suggested Implementation

Here is a Python example demonstrating this approach using the `sentence-transformers` library.

First, install the required library:
`pip install sentence-transformers`

````python
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch
import re

# This model is good for semantic similarity tasks.
# It should be initialized only once.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# A simple dictionary for expanding common abbreviations.
# This should be expanded based on your domain knowledge.
ABBREVIATION_MAP = {
    'cust': 'customer',
    'addr': 'address',
    'id': 'identifier',
    'num': 'number',
    'desc': 'description',
    'caddr': 'customer address'
}

def normalize_column_name(name: str) -> str:
    """Cleans and expands a column name for semantic analysis."""
    name = name.lower()
    name = re.sub(r'[_.-]', ' ', name)
    # A simple regex-based expansion
    words = name.split()
    expanded_words = [ABBREVIATION_MAP.get(word, word) for word in words]
    return ' '.join(expanded_words)

class SemanticTableMetadata:
    def __init__(self, db_name: str, table_name: str, columns: List[Dict]):
        self.db_name = db_name
        self.table_name = table_name
        # Store original column metadata
        self.columns = columns
        # Pre-compute normalized names and embeddings for efficiency
        self.column_normalized_names = [normalize_column_name(c['name']) for c in columns]
        self.column_embeddings = embedding_model.encode(self.column_normalized_names, convert_to_tensor=True)

class SemanticTableAnalyzer:
    def calculate_containment(self, table_subset: SemanticTableMetadata, table_superset: SemanticTableMetadata, similarity_threshold: float = 0.9) -> float:
        """
        Calculates how much of table_subset is semantically contained within table_superset.
        Returns a score between 0.0 and 1.0.
        """
        if not table_subset.columns:
            return 0.0

        # Find the best match for each column in the subset within the superset
        matched_columns = 0
        for subset_col_embedding in table_subset.column_embeddings:
            # Calculate cosine similarities between the subset column and all superset columns
            similarities = util.cos_sim(subset_col_embedding, table_superset.column_embeddings)
            
            # Check if the best match exceeds the threshold
            if torch.max(similarities) > similarity_threshold:
                matched_columns += 1
        
        return matched_columns / len(table_subset.columns)

    def find_duplicate_and_subset_tables(self, tables: List[SemanticTableMetadata], containment_threshold: float = 0.8) -> List[Dict]:
        """
        Finds tables that are potential duplicates or subsets of other tables.
        """
        potential_duplicates = []
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                # To handle subsets, we must check containment in both directions
                # Ensure we always check the smaller table against the larger one
                if len(table1.columns) > len(table2.columns):
                    smaller_table, larger_table = table2, table1
                else:
                    smaller_table, larger_table = table1, table2

                containment_score = self.calculate_containment(smaller_table, larger_table)
                
                if containment_score >= containment_threshold:
                    potential_duplicates.append({
                        'subset_table': f"{smaller_table.db_name}.{smaller_table.table_name}",
                        'superset_table': f"{larger_table.db_name}.{larger_table.table_name}",
                        'containment_score': containment_score,
                        'subset_size': len(smaller_table.columns),
                        'superset_size': len(larger_table.columns)
                    })
        
        return potential_duplicates

# --- Example Usage ---
# Assume you load your metadata into this structure
# tables_metadata = [
#     SemanticTableMetadata('db1', 'customers', [{'name': 'cust_id'}, {'name': 'cust_addr'}, ...50 total...]),
#     SemanticTableMetadata('db2', 'clients', [{'name': 'client_identifier'}, {'name': 'customer home address full'}]),
#     SemanticTableMetadata('db3', 'user_locations', [{'name': 'caddr_home'}, {'name': 'user_id'}])
# ]
#
# analyzer = SemanticTableAnalyzer()
# duplicates = analyzer.find_duplicate_and_subset_tables(tables_metadata, containment_threshold=1.0) # 100% containment
# for d in duplicates:
#     print(d)
````

### Summary of the Workflow

1.  **One-Time Setup:**
    *   Curate a domain-specific `ABBREVIATION_MAP`.
    *   Load the `sentence-transformer` model.

2.  **Preprocessing (Run once per table):**
    *   For each table in your central metadata store, create a `SemanticTableMetadata` object. This will pre-process and generate embeddings for all its columns.
    *   Store these embeddings alongside your metadata to avoid re-computing them.

3.  **Analysis (Run as needed):**
    *   Instantiate the `SemanticTableAnalyzer`.
    *   Pass the list of `SemanticTableMetadata` objects to `find_duplicate_and_subset_tables`.
    *   Review the output, which will now correctly identify subset tables and handle semantically similar column names.

## Problem with Abbreviation Map suggested in the solution

One of the issues I see with this approach is that the 'normalize_column_name' function that uses the ABBREVIATION_MAP is not going to work with production metadata column names that can have numerous variations since there is currently no standard column naming convention across teams. So this ABBREVIATION_MAP will not be easy to hand-build and maintain using just the domain knowledge by an SME. 

We will need a heuristic (best guess) approach to map abbreviated column names or plain english column names to their semantic meaning, where parts of the column name can be separated by underscore or dot or dash or other special characters that databases like MSSQL can allow. 

Also the semantic meaning of a column like ID could depend on whether the table is called customer_data or the table is store_locations or if the table is called product_details. In these cases the column name like ID would have completely different semantic meaning, like "customer identifier" in customer_data table and "product identifier" in the product_details table etc.

What will be a good AI model to use and what would be a good way to generate the embeddings in such cases so that when we do the cosine similarity the results accuracy is high.

It may mean that we may need to train a model with this context and tranining examples, but if it can be done with an out-of-box pre-trained model it would be great.

And if we do have to train a model, which model would be best suited, what training methodology would you suggest (LoRA or QLoRA or some other) for this use case.

## Proposed Solution

You've hit on the core challenge of applying semantic analysis to real-world, messy metadata. Providing context (like the table name) is crucial, and relying on a manually curated map is brittle.

Here is an advanced approach that addresses these points, starting with a powerful out-of-the-box method and then outlining a path to fine-tuning for maximum accuracy.

### 1. Improved Out-of-the-Box Approach: Context-Aware Embeddings

The key is to give the pre-trained model more context than just the column name. Instead of embedding `"customer address"`, we will embed a more descriptive string like `"table: customers column: customer address"`. A powerful sentence-transformer model can understand that the word "identifier" in the context of a "customers table" is semantically very close to "id" in the context of a "clients table".

This directly addresses your point about the meaning of `ID` changing based on the table context.

Here is the updated implementation:

````python
// ...existing code...
import re

# This model is good for semantic similarity tasks.
# It should be initialized only once.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# The abbreviation map is now a secondary helper, not the primary solution.
# It can be kept small for very common, unambiguous abbreviations.
ABBREVIATION_MAP = {
    'addr': 'address',
    'id': 'identifier',
    'num': 'number',
    'desc': 'description'
}

def normalize_name(name: str) -> str:
    """Cleans a name by lowercasing, replacing separators, and expanding common abbreviations."""
    name = name.lower()
    # Replace common separators with spaces
    name = re.sub(r'[_.-]', ' ', name)
    words = name.split()
    # Expand abbreviations if they exist in the map
    expanded_words = [ABBREVIATION_MAP.get(word, word) for word in words]
    return ' '.join(expanded_words)

class SemanticTableMetadata:
    def __init__(self, db_name: str, table_name: str, columns: List[Dict]):
        self.db_name = db_name
        self.table_name = table_name
        self.columns = columns
        
        # Create context-aware strings for embedding
        normalized_table_name = normalize_name(self.table_name)
        contextual_column_strings = []
        for c in columns:
            normalized_column_name = normalize_name(c['name'])
            # The input to the model now includes table context
            contextual_column_strings.append(f"table: {normalized_table_name} column: {normalized_column_name}")

        # Pre-compute embeddings for efficiency
        self.column_embeddings = embedding_model.encode(contextual_column_strings, convert_to_tensor=True)

class SemanticTableAnalyzer:
// ...existing code...
````

**Why this is better:**

*   **Context-Aware:** `ID` in `customers` becomes `"table: customers column: identifier"`, which is semantically distinct from `"table: products column: identifier"`.
*   **Reduces Reliance on `ABBREVIATION_MAP`:** The model can often infer meaning from context. For example, it might learn that `cust_id` in a `orders` table is related to a `customers` table, even if `cust` isn't in your map. The map becomes a tool for helping with only the most obscure or ambiguous abbreviations.

This approach is powerful and should be your first step. It may provide high enough accuracy for your needs without any model training.

### 2. Fine-Tuning for Ultimate Accuracy

If the out-of-the-box model still struggles with your specific enterprise jargon and abbreviation patterns, fine-tuning is the next step.

**Which Model to Fine-Tune?**

Stick with a **Sentence-Transformer** model (like `all-MiniLM-L6-v2` or a larger variant). These are specifically designed for creating embeddings for similarity comparison and are highly efficient. Using a massive generative LLM is unnecessary and computationally expensive for this task.

**Training Methodology: LoRA**

**LoRA (Low-Rank Adaptation)** is an excellent choice here.

*   **Why LoRA?** It is a form of Parameter-Efficient Fine-Tuning (PEFT). Instead of retraining the entire multi-million parameter model, you only train a very small number of new "adapter" weights. This makes training dramatically faster, requires significantly less GPU memory, and avoids "catastrophic forgetting" where the model loses its original powerful capabilities. QLoRA is a further optimization that quantizes the model to use even less memory, making it accessible on more modest hardware.

**How to Train (The Process):**

1.  **Generate a Training Dataset:** You don't need millions of examples. A few thousand high-quality examples can make a huge difference. The goal is to create pairs of column descriptions that *should* be considered identical.
    *   **Source:** A domain expert (SME) can create this dataset.
    *   **Format:** A simple CSV file with two columns: `anchor` and `positive`. Each row represents a pair of semantically equivalent columns.

    **Example `training_data.csv`:**
    ```csv
    anchor,positive
    "table: customers column: cust_id","table: clients column: client identifier"
    "table: customers column: cust_addr","table: customer_data column: customer home address full"
    "table: products column: id","table: inventory column: product_id"
    "table: store_locations column: id","table: shops column: store_identifier"
    ```
    The model will be trained to produce nearly identical embeddings for the `anchor` and `positive` in each row.

2.  **Choose a Loss Function:** Use `MultipleNegativesRankingLoss`. This is a standard and highly effective loss function for this type of training. It works by taking a batch of pairs and treating all other items in the batch as negative examples, efficiently teaching the model what is similar and what is not.

3.  **Run the Training:** The `sentence-transformers` library has excellent documentation and built-in utilities for loading a dataset, defining the loss function, and running the fine-tuning process with just a few lines of code.

By fine-tuning with LoRA on a dataset of your own domain-specific examples, you create a specialized model that deeply understands your company's unique data language, providing the highest possible accuracy for your table similarity analysis.

## Final Proposal

can you beef up the ABBREVIATION_MAP with some of the common database table column names seen in the field ?
That map will be significantly larger than the 4 entries above, so can you put that in a CSV file that can be used to load the abbreviations_map

Finally, could you generate the python code outlined in the conversation?

## Proposed Solution

I'll help create a more comprehensive abbreviation mapping system and provide the complete implementation.

First, let's create the abbreviations CSV file:

````python
abbreviation,full_form
addr,address
acct,account
alt,alternate
amt,amount
bal,balance
cat,category
cust,customer
desc,description
dept,department
dob,date of birth
dt,date
emp,employee
ext,extension
fname,first name
grp,group
id,identifier
img,image
lname,last name
loc,location
max,maximum
min,minimum
mth,month
nbr,number
num,number
org,organization
pct,percentage
ph,phone
pic,picture
pos,position
prd,period
prev,previous
qty,quantity
ref,reference
req,request
sal,salary
seq,sequence
ssn,social security number
stat,status
tel,telephone
temp,temperature
tot,total
tx,transaction
typ,type
usr,user
vol,volume
yr,year
zip,zip code
````

Now, let's create the complete implementation with all the improvements discussed:

````python
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
````

Finally, let's create a requirements file:

````python
pandas==2.0.3
sentence-transformers==2.2.2
torch>=2.0.0
````

To use this implementation:

1. Save the CSV file as `column_abbreviations.csv`
2. Save the Python code as `table_similarity_analyzer.py`
3. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the example:
```bash
python table_similarity_analyzer.py
```

The code now includes:
- A comprehensive abbreviation mapping system loaded from CSV
- Context-aware column comparison using table names
- Detailed matching information including column-level similarities
- Better error handling and documentation
- A working example demonstrating the analysis of two similar tables

You can extend the abbreviations list in the CSV file as you discover more common patterns in your specific domain.
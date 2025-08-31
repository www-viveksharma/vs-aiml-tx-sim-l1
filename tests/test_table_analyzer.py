from pathlib import Path
import sys
import unittest

## This test script:
""" 
Tests environment suffix detection by comparing identical tables with different suffixes
Tests context awareness by comparing tables with similar column names but different semantic contexts
Tests semantic similarity with asymmetric column counts and different naming patterns

Key features:

Uses Python's unittest framework for structured testing
Includes detailed assertions to verify expected behavior
Prints debug information during test execution
Tests both positive and negative cases
Verifies containment scores meet expected thresholds
Note that you might need to adjust the similarity and containment thresholds in the main code depending on how strict you want the matching to be. The current thresholds (0.9 for similarity and 0.8 for containment) are quite high and might need to be lowered for more flexible matching.
 """

# Add the parent directory to Python path to import the main script
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
from table_similarity_analyzer import SemanticTableAnalyzer

class TestTableAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SemanticTableAnalyzer()

    def test_environment_suffix_tables(self):
        """Test case 1: Tables that differ only by environment suffix"""
        print("\n=== Testing Environment Suffix Scenario ===")
        
        tables_metadata = [
            {
                'db_name': 'sales_db_dev',
                'table_name': 'customer_orders_dev',
                'columns': [
                    {'name': 'order_id'},
                    {'name': 'customer_id'},
                    {'name': 'order_date'},
                    {'name': 'total_amount'}
                ]
            },
            {
                'db_name': 'sales_db_prod',
                'table_name': 'customer_orders_prod',
                'columns': [
                    {'name': 'order_id'},
                    {'name': 'customer_id'},
                    {'name': 'order_date'},
                    {'name': 'total_amount'}
                ]
            }
        ]
        
        results = self.analyzer.find_duplicate_and_subset_tables(tables_metadata)
        self.assertTrue(len(results) > 0, "Should detect tables with environment suffixes as similar")
        if results:
            self.assertGreater(results[0]['containment_score'], 0.8)

    def test_different_context_tables(self):
        """Test case 2: Tables with different contexts but similar column names"""
        print("\n=== Testing Different Context Scenario ===")
        
        tables_metadata = [
            {
                'db_name': 'customer_db',
                'table_name': 'customer_data',
                'columns': [
                    {'name': 'id'},
                    {'name': 'name'},
                    {'name': 'address'},
                    {'name': 'phone'},
                    {'name': 'email'}
                ]
            },
            {
                'db_name': 'retail_db',
                'table_name': 'store_locations',
                'columns': [
                    {'name': 'id'},
                    {'name': 'store_name'},
                    {'name': 'address'},
                    {'name': 'phone'},
                    {'name': 'email'}
                ]
            }
        ]
        
        results = self.analyzer.find_duplicate_and_subset_tables(tables_metadata)
        self.assertLess(len(results), 1, "Should not detect these as similar tables despite column name overlap")

    def test_semantic_subset_tables(self):
        """Test case 3: Large table and small table with semantically similar columns"""
        print("\n=== Testing Semantic Subset Scenario ===")
        
        tables_metadata = [
            {
                'db_name': 'hr_db',
                'table_name': 'employee_master',
                'columns': [
                    {'name': 'employee_id'},
                    {'name': 'first_name'},
                    {'name': 'last_name'},
                    {'name': 'birth_date'},
                    {'name': 'hire_date'},
                    {'name': 'department'},
                    {'name': 'position'},
                    {'name': 'salary'},
                    {'name': 'manager_id'},
                    {'name': 'email'},
                    {'name': 'phone'},
                    {'name': 'address_line1'},
                    {'name': 'address_line2'},
                    {'name': 'city'},
                    {'name': 'state'},
                    {'name': 'country'},
                    {'name': 'postal_code'},
                    {'name': 'social_security'},
                    {'name': 'marital_status'},
                    {'name': 'emergency_contact'}
                ]
            },
            {
                'db_name': 'hr_db',
                'table_name': 'employee_basic_info',
                'columns': [
                    {'name': 'emp_num'},                    # semantically similar to employee_id
                    {'name': 'worker_name'},                # combines first_name + last_name
                    {'name': 'dept'},                       # similar to department
                    {'name': 'contact_details'},            # combines phone + email
                    {'name': 'location'}                    # combines address fields
                ]
            }
        ]
        
        results = self.analyzer.find_duplicate_and_subset_tables(tables_metadata)
        self.assertTrue(len(results) > 0, "Should detect semantic similarity despite different column counts")
        if results:
            # The small table should be identified as a subset
            subset_result = next((r for r in results if r['subset_table'].endswith('employee_basic_info')), None)
            self.assertIsNotNone(subset_result)
            self.assertGreater(subset_result['containment_score'], 0.6)

def run_tests():
    print("Running table similarity analysis tests...")
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()
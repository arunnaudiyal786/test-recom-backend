"""
Script to enrich test_plan_historical.csv with category labels from categories.json
and modify descriptions to include category-relevant context.
"""

import json
import csv
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CATEGORIES_PATH = BASE_DIR / "data" / "metadata" / "categories.json"
INPUT_CSV = BASE_DIR / "data" / "raw" / "test_plan_historical.csv"
OUTPUT_CSV = BASE_DIR / "data" / "raw" / "test_plan_historical.csv"

# Load categories
with open(CATEGORIES_PATH, 'r') as f:
    categories_data = json.load(f)

categories = {cat['id']: cat for cat in categories_data['categories']}

# Mapping rules based on test case content patterns
# Each test case type maps to primary and secondary categories
CATEGORY_MAPPING = {
    # Billing tests -> batch_enrollment_maintenance, membership, error_processing
    'BL-CORE': ['batch_enrollment_maintenance', 'membership', 'error_processing', 'file_processing'],

    # Enrollment tests -> membership, eligibility_feed, batch_enrollment_maintenance
    'EN-ENROLL': ['membership', 'eligibility_feed', 'batch_enrollment_maintenance', 'small_group_individual_maintenance'],

    # Claims tests -> error_processing, file_processing, limited_liability_cob
    'CL-CLAIMS': ['error_processing', 'file_processing', 'limited_liability_cob', 'membership'],

    # Premium tests -> product_addition_update, membership, batch_enrollment_maintenance
    'PR-PREMIUM': ['product_addition_update', 'membership', 'batch_enrollment_maintenance', 'reporting'],

    # Renewal tests -> batch_enrollment_maintenance, membership, eligibility_feed
    'RN-RENEWAL': ['batch_enrollment_maintenance', 'membership', 'eligibility_feed', 'contract_addition_update'],

    # Integration tests -> file_processing, technical_implementation, technical_implementation_dli_bm
    'IN-INTEGRATION': ['file_processing', 'technical_implementation', 'technical_implementation_dli_bm', 'eligibility_feed'],

    # Reporting tests -> reporting, letters_reporting, file_processing
    'RP-REPORTING': ['reporting', 'letters_reporting', 'file_processing', 'job_run_request'],

    # Customer Service tests -> membership, id_card_changes, pcp_maintenance
    'CS-CUSTSERV': ['membership', 'online_screen_changes', 'id_card_changes', 'pcp_maintenance'],

    # Security tests -> technical_implementation, testing, technical_implementation_table_update
    'SEC-SECURITY': ['technical_implementation', 'testing', 'technical_implementation_table_update', 'release_management'],

    # Performance tests -> testing, batch_online_abend, job_run_request
    'PF-PERFORMANCE': ['testing', 'batch_online_abend', 'job_run_request', 'scheduling_job_decommission']
}

# Category-specific description enrichment phrases
ENRICHMENT_PHRASES = {
    'batch_enrollment_maintenance': [
        "This involves batch processing of enrollment records.",
        "Bulk update validation is required.",
        "Mass maintenance operation verification needed.",
        "Batch job execution for multiple accounts.",
    ],
    'batch_online_abend': [
        "Monitor for potential batch job failures.",
        "Verify system recovery from processing errors.",
        "Check for abnormal job termination scenarios.",
        "Validate error handling during batch execution.",
    ],
    'contract_addition_update': [
        "Contract terms and service agreement validation.",
        "Vendor contract configuration verification.",
        "Client agreement update processing.",
        "Contract modification workflow testing.",
    ],
    'eligibility_feed': [
        "Eligibility transmission validation required.",
        "834 file processing verification.",
        "Enrollment feed data integrity check.",
        "Coverage status update validation.",
    ],
    'error_processing': [
        "Error queue handling validation.",
        "Exception processing verification.",
        "Data validation error scenarios.",
        "Error resolution workflow testing.",
    ],
    'file_processing': [
        "File transfer validation required.",
        "Data file parsing verification.",
        "SFTP file processing check.",
        "File import/export validation.",
    ],
    'id_card_changes': [
        "Member ID card update verification.",
        "Card reissue processing validation.",
        "Identification card information accuracy.",
        "Card printing workflow testing.",
    ],
    'job_run_request': [
        "Scheduled job execution validation.",
        "Batch job run request processing.",
        "Job scheduling verification.",
        "Process execution workflow testing.",
    ],
    'letters_reporting': [
        "Member correspondence generation.",
        "Document generation validation.",
        "Welcome letter processing verification.",
        "EOB generation testing.",
    ],
    'limited_liability_cob': [
        "COB determination validation.",
        "Coordination of benefits processing.",
        "Primary/secondary payer verification.",
        "Dual coverage handling testing.",
    ],
    'membership': [
        "Member record status validation.",
        "Enrollment status verification.",
        "Subscriber information accuracy.",
        "Member ID processing testing.",
    ],
    'migration': [
        "Data migration validation required.",
        "System migration verification.",
        "Platform conversion testing.",
        "Data cutover validation.",
    ],
    'online_screen_changes': [
        "UI screen update validation.",
        "Portal display verification.",
        "User interface testing.",
        "Frontend screen changes validation.",
    ],
    'overage_dependent': [
        "Overage dependent eligibility check.",
        "Age limit validation required.",
        "Dependent aging-out verification.",
        "Student dependent processing testing.",
    ],
    'pcp_maintenance': [
        "PCP assignment validation.",
        "Primary care provider update verification.",
        "Physician assignment testing.",
        "Provider assignment workflow validation.",
    ],
    'product_addition_update': [
        "Benefit plan configuration validation.",
        "Product setup verification.",
        "Plan update processing testing.",
        "Product definition validation.",
    ],
    'release_management': [
        "Release deployment validation.",
        "Version rollout verification.",
        "Go-live readiness testing.",
        "Production release validation.",
    ],
    'reporting': [
        "Dashboard data accuracy validation.",
        "Analytics report verification.",
        "KPI metrics testing.",
        "Data extract validation.",
    ],
    'scheduling_job_decommission': [
        "Job decommission validation.",
        "Schedule removal verification.",
        "Job sunset processing testing.",
        "Retired job cleanup validation.",
    ],
    'small_group_individual_maintenance': [
        "Small group account maintenance.",
        "Individual plan processing validation.",
        "Small employer update verification.",
        "Single account maintenance testing.",
    ],
    'technical_implementation': [
        "Technical configuration validation.",
        "System integration verification.",
        "Technical setup testing.",
        "Implementation workflow validation.",
    ],
    'technical_implementation_dli_bm': [
        "DLI to BM interface validation.",
        "Data load interface verification.",
        "Business module integration testing.",
        "DLI configuration validation.",
    ],
    'technical_implementation_route_code': [
        "Route code configuration validation.",
        "Routing logic verification.",
        "Route table update testing.",
        "Routing rules validation.",
    ],
    'technical_implementation_table_update': [
        "Database table update validation.",
        "Reference table modification verification.",
        "Schema change testing.",
        "Table data integrity validation.",
    ],
    'testing': [
        "QA verification required.",
        "UAT test case execution.",
        "Regression testing validation.",
        "Quality assurance verification.",
    ],
}


def get_label_code(label_id: str) -> str:
    """Get the team/area code from the Labels2 field."""
    for code in CATEGORY_MAPPING.keys():
        if code in label_id:
            return code
    return None


def assign_categories(row: dict) -> list:
    """Assign 1-3 categories based on the test case content."""
    label_code = row.get('Labels2', '')

    # Find matching category mapping
    for code, cats in CATEGORY_MAPPING.items():
        if code in label_code:
            # Randomly select 1-3 categories from the mapping
            num_cats = random.randint(1, min(3, len(cats)))
            selected = random.sample(cats, num_cats)
            return selected

    # Default fallback
    return ['testing', 'technical_implementation']


def enrich_description(description: str, assigned_categories: list) -> str:
    """Add category-relevant context to the description."""
    # Pick one enrichment phrase from the primary category
    primary_cat = assigned_categories[0]
    phrases = ENRICHMENT_PHRASES.get(primary_cat, [])

    if phrases:
        enrichment = random.choice(phrases)
        # Insert the enrichment at a natural position in the description
        if description:
            # Add before the first period or at the end
            if '. ' in description:
                parts = description.split('. ', 1)
                enriched = f"{parts[0]}. {enrichment} {parts[1]}"
            else:
                enriched = f"{description} {enrichment}"
            return enriched

    return description


def get_category_names(category_ids: list) -> str:
    """Convert category IDs to human-readable names, comma-separated."""
    names = []
    for cat_id in category_ids:
        if cat_id in categories:
            names.append(categories[cat_id]['name'])
    return ', '.join(names)


def main():
    print(f"Loading categories from {CATEGORIES_PATH}")
    print(f"Processing CSV: {INPUT_CSV}")

    # Read the CSV
    rows = []
    with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows")

    # Add new column to fieldnames
    if 'Category_Labels' not in fieldnames:
        # Insert Category_Labels after Labels column
        labels_idx = fieldnames.index('Labels') if 'Labels' in fieldnames else 2
        new_fieldnames = fieldnames[:labels_idx+1] + ['Category_Labels'] + fieldnames[labels_idx+1:]
    else:
        new_fieldnames = fieldnames

    # Process each row
    for i, row in enumerate(rows):
        # Assign categories
        assigned_cats = assign_categories(row)

        # Get category names
        cat_labels = get_category_names(assigned_cats)
        row['Category_Labels'] = cat_labels

        # Enrich description
        original_desc = row.get('Description', '')
        enriched_desc = enrich_description(original_desc, assigned_cats)
        row['Description'] = enriched_desc

        print(f"Row {i+1}: Assigned categories: {cat_labels}")

    # Write the updated CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated CSV saved to: {OUTPUT_CSV}")
    print(f"Added 'Category_Labels' column with values from 25 categories")
    print(f"Enriched descriptions with category-relevant keywords")

    # Print sample of categories used
    print("\nCategories used in dataset:")
    all_cats = set()
    for row in rows:
        cats = row.get('Category_Labels', '').split(', ')
        all_cats.update(cats)

    for cat in sorted(all_cats):
        if cat:
            print(f"  - {cat}")


if __name__ == "__main__":
    main()

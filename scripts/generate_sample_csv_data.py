"""
Generate sample historical tickets in CSV format.
Replaces JSON-based data generation with CSV format matching Excel structure.
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


# Domain-specific ticket templates
TICKET_TEMPLATES = {
    "MM": [
        {
            "issue_type": "Bug",
            "summary": "MM_ALDER service memory leak in event handler",
            "description": "Memory usage grows continuously in event handler until service crashes. Heap dump shows retention of event listener objects. Issue started after recent deployment.",
            "labels": "Code Fix,#MM_ALDER,#MMALDR",
            "resolution": "1. Analyze heap dump: jmap -dump:live,format=b,file=heap.bin <pid>\n2. Identify leak source in background worker\n3. Add proper resource cleanup in finally blocks\n4. Implement connection pooling with max age\n5. Deploy fix and monitor memory usage\n6. Set up alerts for memory threshold (>80%)",
            "it_team": "MM Core Team",
        },
        {
            "issue_type": "Performance",
            "summary": "MM service database connection pool exhaustion",
            "description": "Service hitting maximum database connection limit during peak hours. Connection pool metrics show sustained 100% utilization. Causing 504 Gateway Timeout errors for member eligibility lookups.",
            "labels": "Configuration Fix,#MM_ALDER",
            "resolution": "1. Increase connection pool size from 100 to 200\n2. Implement connection timeout of 30 seconds\n3. Add connection pool monitoring alerts\n4. Review slow queries and add indexes\n5. Deploy configuration changes\n6. Monitor connection pool metrics",
            "it_team": "MM Infrastructure",
        },
        {
            "issue_type": "Bug",
            "summary": "MM_ALDER batch processing job failures",
            "description": "Nightly batch job failing with OOM errors. Processing 500K+ records causing heap overflow. Job runs fine with smaller datasets but fails in production volume.",
            "labels": "Code Fix,#MM_ALDER,#MMALDR",
            "resolution": "1. Implement batch processing with 1000 record chunks\n2. Add pagination to database queries\n3. Clear processed records from memory\n4. Increase JVM heap size to 8GB\n5. Add job monitoring and alerting\n6. Test with production data volume",
            "it_team": "MM Batch Processing",
        },
    ],
    "CIW": [
        {
            "issue_type": "Bug",
            "summary": "CIW data validation errors in claims submission",
            "description": "CIW claims submission rejecting valid records with validation errors. Error message: 'Invalid reference'. Validation logic too strict and rejecting legitimate edge cases.",
            "labels": "Code Fix,#CIW_INTEGRATION",
            "resolution": "1. Review validation rules in member registration/validators.py\n2. Analyze rejected records to identify pattern\n3. Update validation regex to handle edge cases\n4. Add exception handling for null/empty values\n5. Implement validation bypass for admin users\n6. Update unit tests with edge case scenarios\n7. Deploy to staging and test with sample data\n8. Monitor error rates after production deployment",
            "it_team": "CIW Integration",
        },
        {
            "issue_type": "Integration Issue",
            "summary": "CIW API authentication failures with third-party provider",
            "description": "OAuth token refresh failing intermittently. Receiving 401 Unauthorized errors. Token cache not properly synchronized across multiple service instances.",
            "labels": "Configuration Fix,#CIW_INTEGRATION",
            "resolution": "1. Implement Redis-based token cache for shared state\n2. Add automatic token refresh before expiration\n3. Implement retry logic with exponential backoff\n4. Add token validation logging\n5. Deploy configuration changes\n6. Monitor authentication success rates",
            "it_team": "CIW API Team",
        },
        {
            "issue_type": "Data Issue",
            "summary": "CIW duplicate member records in enrollment database",
            "description": "Finding duplicate member records with different IDs but same demographics. Data sync job not properly checking for existing members before creating new records.",
            "labels": "Data Fix,#CIW_INTEGRATION",
            "resolution": "1. Run deduplication query to identify all duplicates\n2. Create mapping table for duplicate IDs\n3. Update foreign key references to canonical ID\n4. Delete duplicate records\n5. Implement unique constraint on member demographics\n6. Update sync job to check existing members first\n7. Add data quality monitoring",
            "it_team": "CIW Data Team",
        },
    ],
    "Specialty": [
        {
            "issue_type": "Feature Request",
            "summary": "Custom workflow automation for specialty provider onboarding",
            "description": "Business team needs automated workflow for specialty provider credentialing. Current manual process takes 2 weeks. Need to integrate with external credentialing API and automate document verification.",
            "labels": "Code Fix,#SPECIALTY_CUSTOM",
            "resolution": "1. Design workflow state machine with 5 stages\n2. Implement credentialing API client\n3. Build document upload and verification service\n4. Create automated notification system\n5. Add workflow tracking dashboard\n6. Conduct UAT with business team\n7. Deploy to production with phased rollout",
            "it_team": "Specialty Services",
        },
        {
            "issue_type": "Bug",
            "summary": "Specialty authorization rules not triggering correctly",
            "description": "Custom business rules for specialty authorizations failing to trigger. Rules engine evaluating conditions incorrectly due to data type mismatch. Causing manual review of cases that should auto-approve.",
            "labels": "Code Fix,#SPECIALTY_CUSTOM",
            "resolution": "1. Review rules engine evaluation logic\n2. Fix data type conversions in rule conditions\n3. Add validation for rule configuration\n4. Implement rule testing framework\n5. Update existing rules with correct data types\n6. Add monitoring for rule execution failures\n7. Deploy fix and monitor auto-approval rates",
            "it_team": "Specialty Rules Engine",
        },
        {
            "issue_type": "Configuration",
            "summary": "Custom reporting dashboard performance degradation",
            "description": "Specialty business intelligence dashboard loading slowly. Report queries taking 30+ seconds. Need to optimize data aggregation and add caching layer.",
            "labels": "Configuration Fix,#SPECIALTY_CUSTOM",
            "resolution": "1. Analyze slow queries with EXPLAIN ANALYZE\n2. Add database indexes on frequently filtered columns\n3. Implement Redis caching for aggregate data\n4. Schedule background job to pre-compute daily reports\n5. Add query result pagination\n6. Deploy optimizations\n7. Monitor query performance metrics",
            "it_team": "Specialty BI Team",
        },
    ],
}

# Priority distribution
PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS = [0.2, 0.4, 0.3, 0.1]

# Assignee pool
ASSIGNEES = [
    "john.doe@company.com",
    "jane.smith@company.com",
    "bob.johnson@company.com",
    "alice.williams@company.com",
    "charlie.brown@company.com",
    "diana.garcia@company.com",
]

# Reporter pool
REPORTERS = [
    "support.team@company.com",
    "ops.team@company.com",
    "qa.team@company.com",
    "product.team@company.com",
]


def generate_dates():
    """Generate created and closed dates with realistic resolution time."""
    created = datetime.now() - timedelta(days=random.randint(30, 365))
    resolution_hours = random.uniform(1.0, 48.0)
    closed = created + timedelta(hours=resolution_hours)
    return created, closed, resolution_hours


def generate_ticket(domain: str, ticket_num: int) -> dict:
    """Generate a single ticket for the given domain."""
    template = random.choice(TICKET_TEMPLATES[domain])
    created, closed, resolution_hours = generate_dates()
    priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0]

    # Map domain names to 2-letter codes for ticket keys
    domain_code_map = {
        "MM": "MM",
        "CIW": "CI",
        "Specialty": "SP"
    }
    domain_code = domain_code_map.get(domain, domain[:2])

    return {
        "key": f"JIRA-{domain_code}-{ticket_num:03d}",
        "closed date": closed.strftime("%Y-%m-%d %H:%M:%S"),
        "issue type": template["issue_type"],
        "assignee": random.choice(ASSIGNEES),
        "created": created.strftime("%Y-%m-%d %H:%M:%S"),
        "IT Team": template["it_team"],
        "Issue Priority": priority,
        "Labels": template["labels"],
        "Reporter": random.choice(REPORTERS),
        "Resolution": template["resolution"],
        "Summary": template["summary"],
        "Description": template["description"],
    }


def generate_historical_tickets_csv(num_tickets: int = 100, output_path: str = None):
    """
    Generate historical tickets CSV file.

    Args:
        num_tickets: Total number of tickets to generate
        output_path: Path to save CSV file (defaults to data/raw/historical_tickets.csv)
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "raw" / "historical_tickets.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Distribute tickets across domains
    domains = ["MM", "CIW", "Specialty"]
    tickets_per_domain = num_tickets // len(domains)
    remainder = num_tickets % len(domains)

    all_tickets = []

    for idx, domain in enumerate(domains):
        count = tickets_per_domain + (1 if idx < remainder else 0)
        for i in range(count):
            ticket = generate_ticket(domain, len(all_tickets) + 1)
            all_tickets.append(ticket)

    # Shuffle to mix domains
    random.shuffle(all_tickets)

    # Define CSV columns in the exact order specified
    fieldnames = [
        "key",
        "closed date",
        "issue type",
        "assignee",
        "created",
        "IT Team",
        "Issue Priority",
        "Labels",
        "Reporter",
        "Resolution",
        "Summary",
        "Description",
    ]

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tickets)

    print(f"✓ Generated {len(all_tickets)} historical tickets")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDomain distribution:")

    # Map domain names to their codes for counting
    domain_code_map = {
        "MM": "MM",
        "CIW": "CI",
        "Specialty": "SP"
    }

    for domain in domains:
        code = domain_code_map.get(domain, domain[:2])
        count = sum(1 for t in all_tickets if t["key"].startswith(f"JIRA-{code}"))
        print(f"  {domain}: {count} tickets")

    return output_path


if __name__ == "__main__":
    generate_historical_tickets_csv(num_tickets=100)

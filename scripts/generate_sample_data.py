"""
Generate 100 sample historical tickets for FAISS index.

Distribution:
- MM domain: 40 tickets
- CIW domain: 35 tickets
- Specialty domain: 25 tickets
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# Domain-specific ticket templates
MM_TICKETS = [
    {
        "title_template": "Database connection timeout in {component}",
        "desc_template": "Users experiencing timeout errors when connecting to MM database through {component}. Error occurs intermittently during peak hours. Connection pool may be exhausted.",
        "component": ["MM_ALDER module", "MM core service", "MM integration layer", "MM data processor"],
        "labels": ["Code Fix", "#MM_ALDER"],
        "resolution": [
            "1. Check database connection pool settings in /etc/mm/config.yaml",
            "2. Increase max_connections from 100 to 200",
            "3. Adjust connection timeout from 10s to 30s",
            "4. Restart MM service: systemctl restart mm_alder",
            "5. Monitor connection pool metrics for 24 hours"
        ]
    },
    {
        "title_template": "MM_ALDER service memory leak in {process}",
        "desc_template": "Memory usage grows continuously in {process} until service crashes. Heap dump shows retention of {object} objects. Issue started after recent deployment.",
        "process": ["data processing pipeline", "cache manager", "event handler", "background worker"],
        "object": ["session", "connection", "cache entry", "event listener"],
        "labels": ["Code Fix", "#MM_ALDER", "#MMALDR"],
        "resolution": [
            "1. Analyze heap dump: jmap -dump:live,format=b,file=heap.bin <pid>",
            "2. Identify leak source in {process}",
            "3. Add proper resource cleanup in finally blocks",
            "4. Implement connection pooling with max age",
            "5. Deploy fix and monitor memory usage",
            "6. Set up alerts for memory threshold (>80%)"
        ]
    },
    {
        "title_template": "MM integration API returning 504 errors",
        "desc_template": "MM REST API endpoints timing out with 504 Gateway Timeout errors. Affects {endpoint} endpoint primarily. Backend processing takes longer than 60s timeout limit.",
        "endpoint": ["/api/v1/claims/process", "/api/v1/patient/lookup", "/api/v1/eligibility/check", "/api/v1/auth/verify"],
        "labels": ["Code Fix", "Configuration Fix"],
        "resolution": [
            "1. Increase nginx timeout: proxy_read_timeout 120s",
            "2. Optimize database query in {endpoint} handler",
            "3. Add query index on frequently filtered columns",
            "4. Implement response caching for read operations",
            "5. Add asynchronous processing for long-running tasks",
            "6. Test with production load simulator"
        ]
    },
    {
        "title_template": "MM batch job failing with {error}",
        "desc_template": "Nightly MM batch processing job fails with {error}. Job processes {count} records but crashes at random point. No clear pattern in failure timing.",
        "error": ["OutOfMemoryError", "NullPointerException", "ConcurrentModificationException", "SQLException"],
        "count": ["10000", "50000", "100000", "250000"],
        "labels": ["Code Fix", "#MM_ALDER"],
        "resolution": [
            "1. Review application logs for stack trace",
            "2. Increase JVM heap size: -Xmx4g -Xms2g",
            "3. Implement batch processing with chunking (1000 records/batch)",
            "4. Add null checks and error handling",
            "5. Implement transaction boundaries per chunk",
            "6. Add retry logic with exponential backoff",
            "7. Run test with production data snapshot"
        ]
    },
]

CIW_TICKETS = [
    {
        "title_template": "CIW integration failing to sync {data_type}",
        "desc_template": "CIW integration service unable to synchronize {data_type} from upstream system. Sync job completes but data not updated in target database. Last successful sync was {hours} hours ago.",
        "data_type": ["patient demographics", "provider information", "claim status", "authorization data"],
        "hours": ["4", "12", "24", "48"],
        "labels": ["Code Fix", "#CIW_INTEGRATION"],
        "resolution": [
            "1. Check CIW integration service logs: tail -f /var/log/ciw/integration.log",
            "2. Verify API credentials and authentication token",
            "3. Test connectivity to upstream system endpoint",
            "4. Validate data transformation logic",
            "5. Clear stale records from staging table",
            "6. Restart integration service",
            "7. Manually trigger sync job and monitor"
        ]
    },
    {
        "title_template": "CIW {component} performance degradation",
        "desc_template": "Significant performance degradation in CIW {component}. Response times increased from {baseline}ms to {current}ms. Database queries show long execution times on {table} table.",
        "component": ["claims processor", "eligibility checker", "provider lookup", "authorization engine"],
        "baseline": ["50", "100", "200", "300"],
        "current": ["2000", "3000", "5000", "8000"],
        "table": ["claims", "members", "providers", "authorizations"],
        "labels": ["Code Fix", "Configuration Fix", "#CIW_INTEGRATION"],
        "resolution": [
            "1. Run EXPLAIN on slow queries to identify missing indexes",
            "2. Create composite index on {table}(column1, column2, column3)",
            "3. Update table statistics: ANALYZE TABLE {table}",
            "4. Optimize query to use index hints",
            "5. Implement query result caching (Redis)",
            "6. Add database connection pooling",
            "7. Monitor query performance after changes"
        ]
    },
    {
        "title_template": "CIW data validation errors in {module}",
        "desc_template": "CIW {module} rejecting valid records with validation errors. Error message: '{error_msg}'. Validation logic too strict and rejecting legitimate edge cases.",
        "module": ["claims submission", "eligibility verification", "provider enrollment", "member registration"],
        "error_msg": ["Invalid date format", "Required field missing", "Value out of range", "Invalid reference"],
        "labels": ["Code Fix", "#CIW_INTEGRATION"],
        "resolution": [
            "1. Review validation rules in {module}/validators.py",
            "2. Analyze rejected records to identify pattern",
            "3. Update validation regex to handle edge cases",
            "4. Add exception handling for null/empty values",
            "5. Implement validation bypass for admin users",
            "6. Update unit tests with edge case scenarios",
            "7. Deploy to staging and test with sample data",
            "8. Monitor error rates after production deployment"
        ]
    },
]

SPECIALTY_TICKETS = [
    {
        "title_template": "Specialty {feature} module not loading correctly",
        "desc_template": "Custom {feature} module in specialty application fails to load on startup. Browser console shows {error}. Module worked correctly before latest release.",
        "feature": ["reporting dashboard", "analytics widget", "custom form builder", "workflow designer"],
        "error": ["JavaScript runtime error", "Module not found", "Dependency injection failed", "CSS loading error"],
        "labels": ["Code Fix", "#SPECIALTY_CUSTOM"],
        "resolution": [
            "1. Check browser console for detailed error stack trace",
            "2. Verify module dependencies in package.json",
            "3. Clear browser cache and CDN cache",
            "4. Check webpack build for missing chunks",
            "5. Verify module export/import statements",
            "6. Rebuild frontend: npm run build",
            "7. Test in multiple browsers",
            "8. Deploy fixed version to production"
        ]
    },
    {
        "title_template": "Specialty custom workflow failing at {step}",
        "desc_template": "Custom workflow for specialty processing fails at {step} step. Error: '{error}'. Workflow engine shows step timeout after {timeout} seconds.",
        "step": ["approval routing", "document generation", "notification delivery", "data validation"],
        "error": ["Timeout exceeded", "Service unavailable", "Invalid state transition", "Missing required data"],
        "timeout": ["30", "60", "120", "300"],
        "labels": ["Code Fix", "Configuration Fix", "#SPECIALTY_CUSTOM"],
        "resolution": [
            "1. Review workflow definition in specialty/workflows/{name}.xml",
            "2. Increase step timeout in workflow engine config",
            "3. Add error handling and retry logic to {step}",
            "4. Verify external service dependencies are available",
            "5. Implement async processing for long-running steps",
            "6. Add workflow state persistence",
            "7. Test workflow with production-like data",
            "8. Enable detailed workflow execution logging"
        ]
    },
    {
        "title_template": "Specialty report generation {issue}",
        "desc_template": "Custom specialty reports experiencing {issue}. Affects {report_type} reports. Users unable to access critical business data. Report server load is at {load}%.",
        "issue": ["extreme slowness", "incorrect data", "generation failures", "export errors"],
        "report_type": ["financial", "operational", "compliance", "analytics"],
        "load": ["85", "90", "95", "98"],
        "labels": ["Code Fix", "Data Fix", "#SPECIALTY_CUSTOM"],
        "resolution": [
            "1. Optimize report query to reduce data fetch time",
            "2. Add database query pagination (1000 rows/page)",
            "3. Implement report caching for frequently accessed reports",
            "4. Create database materialized view for aggregations",
            "5. Add background job queue for large report generation",
            "6. Optimize report template rendering logic",
            "7. Scale report server horizontally (add 2 more instances)",
            "8. Monitor server metrics after optimization"
        ]
    },
]


def generate_ticket(template, domain, ticket_num):
    """Generate a single ticket from a template."""
    # Select random variations
    title = template["title_template"]
    desc = template["desc_template"]

    # Replace placeholders with random choices
    for key, values in template.items():
        if key not in ["title_template", "desc_template", "labels", "resolution"]:
            if isinstance(values, list):
                chosen = random.choice(values)
                title = title.replace(f"{{{key}}}", chosen)
                desc = desc.replace(f"{{{key}}}", chosen)

    # Generate random metadata
    priority = random.choices(
        ["Low", "Medium", "High", "Critical"],
        weights=[10, 40, 35, 15]
    )[0]

    # Random date in past 6 months
    days_ago = random.randint(1, 180)
    created_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    # Resolution time based on priority
    if priority == "Critical":
        res_time = random.uniform(1, 4)
    elif priority == "High":
        res_time = random.uniform(2, 8)
    elif priority == "Medium":
        res_time = random.uniform(4, 24)
    else:
        res_time = random.uniform(8, 72)

    # Substitute resolution steps
    resolution_steps = [
        step.replace("{component}", random.choice(template.get("component", ["component"])) if "component" in template else step)
        .replace("{process}", random.choice(template.get("process", ["process"])) if "process" in template else step)
        .replace("{endpoint}", random.choice(template.get("endpoint", ["endpoint"])) if "endpoint" in template else step)
        .replace("{table}", random.choice(template.get("table", ["table"])) if "table" in template else step)
        .replace("{module}", random.choice(template.get("module", ["module"])) if "module" in template else step)
        .replace("{step}", random.choice(template.get("step", ["step"])) if "step" in template else step)
        .replace("{name}", f"workflow_{ticket_num}")
        for step in template["resolution"]
    ]

    return {
        "ticket_id": f"JIRA-{domain[:2]}-{ticket_num:03d}",
        "title": title,
        "description": desc,
        "domain": domain,
        "labels": template["labels"],
        "resolution_steps": resolution_steps,
        "priority": priority,
        "created_date": created_date,
        "resolution_time_hours": round(res_time, 1)
    }


def main():
    """Generate 100 sample tickets and save to JSON."""
    tickets = []

    # Generate MM tickets (40)
    for i in range(40):
        template = random.choice(MM_TICKETS)
        ticket = generate_ticket(template, "MM", i + 1)
        tickets.append(ticket)

    # Generate CIW tickets (35)
    for i in range(35):
        template = random.choice(CIW_TICKETS)
        ticket = generate_ticket(template, "CIW", i + 1)
        tickets.append(ticket)

    # Generate Specialty tickets (25)
    for i in range(25):
        template = random.choice(SPECIALTY_TICKETS)
        ticket = generate_ticket(template, "Specialty", i + 1)
        tickets.append(ticket)

    # Shuffle to randomize order
    random.shuffle(tickets)

    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "raw" / "historical_tickets.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tickets, f, indent=2)

    print(f"✅ Generated {len(tickets)} sample tickets")
    print(f"📁 Saved to: {output_path}")
    print(f"\nDistribution:")
    print(f"  - MM: {sum(1 for t in tickets if t['domain'] == 'MM')} tickets")
    print(f"  - CIW: {sum(1 for t in tickets if t['domain'] == 'CIW')} tickets")
    print(f"  - Specialty: {sum(1 for t in tickets if t['domain'] == 'Specialty')} tickets")


if __name__ == "__main__":
    main()

"""
Prompt templates for classification agent.

Uses MTC-LLM approach: separate binary classifiers for each domain.
"""

# Domain-specific binary classifier prompts

MM_CLASSIFIER_PROMPT = """You are an AI classifier for MM (Member Management) domain tickets in a healthcare technical support system.

Definition: An MM domain ticket involves:
- MM_ALDER service or module issues
- MM core service, integration layer, or data processor problems
- Database connection issues specific to MM components
- Batch jobs, APIs, or services prefixed with "MM"
- Member management, eligibility, or enrollment system issues
- MMALDR (MM ALDER) specific problems

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify MM-specific keywords (MM_ALDER, MMALDR, mm service, member management, etc.)
3. Check for database/service/API patterns related to member systems
4. Evaluate if the issue is clearly within MM domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the MM domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text."""


CIW_CLASSIFIER_PROMPT = """You are an AI classifier for CIW (Claims Integration Workflow) domain tickets in a healthcare technical support system.

Definition: A CIW domain ticket involves:
- CIW integration service issues
- Claims processing, submission, or validation problems
- Provider lookup, eligibility verification, or authorization systems
- Data synchronization from upstream claims systems
- CIW-specific modules, APIs, or batch processes
- Integration failures with external claims systems

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify CIW-specific keywords (CIW, claims, provider, eligibility, authorization, etc.)
3. Check for integration, sync, or claims processing patterns
4. Evaluate if the issue is clearly within CIW domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the CIW domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text."""


SPECIALTY_CLASSIFIER_PROMPT = """You are an AI classifier for Specialty domain tickets in a healthcare technical support system.

Definition: A Specialty domain ticket involves:
- Custom specialty modules or applications
- Specialized workflow engines or custom workflows
- Specialty reporting, analytics, or dashboard issues
- Custom form builders or UI components
- Specialty-specific features not part of MM or CIW
- Frontend/UI issues in specialty applications
- Custom integrations or extensions

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify specialty-specific keywords (custom, specialty, workflow, report, dashboard, etc.)
3. Check for frontend/UI patterns, custom modules, or specialized features
4. Evaluate if the issue is clearly within Specialty domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the Specialty domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text."""


def get_classification_prompt(domain: str, title: str, description: str) -> str:
    """
    Get the classification prompt for a specific domain.

    Args:
        domain: Domain name ('MM', 'CIW', or 'Specialty')
        title: Ticket title
        description: Ticket description

    Returns:
        Formatted prompt string
    """
    prompts = {
        'MM': MM_CLASSIFIER_PROMPT,
        'CIW': CIW_CLASSIFIER_PROMPT,
        'Specialty': SPECIALTY_CLASSIFIER_PROMPT
    }

    if domain not in prompts:
        raise ValueError(f"Unknown domain: {domain}. Must be MM, CIW, or Specialty")

    return prompts[domain].format(title=title, description=description)

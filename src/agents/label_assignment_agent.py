"""
Label assignment agent with three-tier labeling:
1. Historical Labels - from similar historical tickets
2. Business Labels - AI-generated business perspective
3. Technical Labels - AI-generated technical perspective
"""
import asyncio
from typing import Dict, Any, List, Set
from src.utils.openai_client import get_openai_client
from src.utils.config import Config
from src.utils.helpers import calculate_label_distribution
from src.prompts.label_assignment_prompts import (
    get_label_assignment_prompt,
    get_business_label_generation_prompt,
    get_technical_label_generation_prompt
)
from src.models.state_schema import TicketState, AgentOutput


class LabelAssignmentAgent:
    """
    Agent for assigning labels to tickets using three methods:
    1. Historical Labels: Validated against similar historical tickets
    2. Business Labels: AI-generated from business perspective
    3. Technical Labels: AI-generated from technical perspective
    """

    def __init__(self):
        self.client = get_openai_client()
        self.model = Config.CLASSIFICATION_MODEL
        self.temperature = Config.CLASSIFICATION_TEMPERATURE
        self.confidence_threshold = Config.LABEL_CONFIDENCE_THRESHOLD
        self.label_generation_enabled = Config.LABEL_GENERATION_ENABLED
        self.generated_label_threshold = Config.GENERATED_LABEL_CONFIDENCE_THRESHOLD
        self.business_label_max = Config.BUSINESS_LABEL_MAX_COUNT
        self.technical_label_max = Config.TECHNICAL_LABEL_MAX_COUNT

    # =========================================================================
    # HISTORICAL LABEL METHODS (Existing functionality)
    # =========================================================================

    def extract_candidate_labels(self, similar_tickets: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract all unique labels from similar tickets as candidates.

        Args:
            similar_tickets: List of similar ticket dicts

        Returns:
            Set of unique label names
        """
        candidate_labels = set()

        for ticket in similar_tickets:
            labels = ticket.get('labels', [])
            candidate_labels.update(labels)

        return candidate_labels

    async def evaluate_historical_label(
        self,
        label_name: str,
        title: str,
        description: str,
        domain: str,
        similar_tickets: List[Dict[str, Any]],
        label_frequency: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single historical label using binary classifier.

        Args:
            label_name: Label to evaluate
            title: Ticket title
            description: Ticket description
            domain: Classified domain
            similar_tickets: List of similar tickets
            label_frequency: Label frequency distribution

        Returns:
            Classification result dict
        """
        try:
            # Generate prompt
            prompt = get_label_assignment_prompt(
                label_name=label_name,
                title=title,
                description=description,
                domain=domain,
                similar_tickets=similar_tickets,
                label_frequency=label_frequency
            )

            # Call OpenAI with JSON mode
            response = await self.client.chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are a label classification expert for technical support systems."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=500
            )

            return {
                "label": label_name,
                "assign": response.get('assign_label', False),
                "confidence": response.get('confidence', 0.0),
                "reasoning": response.get('reasoning', ''),
                "evidence": response.get('supporting_evidence', [])
            }

        except Exception as e:
            print(f"⚠️ Error evaluating historical label '{label_name}': {str(e)}")
            return {
                "label": label_name,
                "assign": False,
                "confidence": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "evidence": []
            }

    async def assign_historical_labels(
        self,
        title: str,
        description: str,
        domain: str,
        similar_tickets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assign labels based on similar historical tickets.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Classified domain
            similar_tickets: List of similar tickets

        Returns:
            Dict with historical_labels, confidence, and distribution
        """
        # Extract candidate labels from similar tickets
        candidate_labels = self.extract_candidate_labels(similar_tickets)

        if not candidate_labels:
            return {
                "historical_labels": [],
                "historical_label_confidence": {},
                "historical_label_distribution": {}
            }

        # Calculate label frequency distribution
        label_frequency = calculate_label_distribution(similar_tickets)

        # Evaluate all labels in parallel
        tasks = [
            self.evaluate_historical_label(
                label, title, description, domain, similar_tickets, label_frequency
            )
            for label in candidate_labels
        ]

        evaluation_results = await asyncio.gather(*tasks)

        # Filter by confidence threshold
        assigned_labels = []
        label_confidence = {}

        for result in evaluation_results:
            label = result['label']
            assign = result['assign']
            confidence = result['confidence']

            label_confidence[label] = confidence

            if assign and confidence >= self.confidence_threshold:
                assigned_labels.append(label)

        # Format label distribution for output
        label_distribution = {
            label: freq_info['formatted']
            for label, freq_info in label_frequency.items()
        }

        return {
            "historical_labels": assigned_labels,
            "historical_label_confidence": label_confidence,
            "historical_label_distribution": label_distribution
        }

    # =========================================================================
    # BUSINESS LABEL GENERATION (New AI-generated)
    # =========================================================================

    async def generate_business_labels(
        self,
        title: str,
        description: str,
        domain: str,
        priority: str,
        similar_tickets: List[Dict[str, Any]],
        existing_labels: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate business-oriented labels using AI.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Classified domain
            priority: Ticket priority
            similar_tickets: List of similar tickets
            existing_labels: Set of existing labels from historical tickets

        Returns:
            List of generated business labels with confidence and reasoning
        """
        try:
            # Generate prompt
            prompt = get_business_label_generation_prompt(
                title=title,
                description=description,
                domain=domain,
                priority=priority,
                similar_tickets=similar_tickets,
                existing_labels=existing_labels,
                max_labels=self.business_label_max
            )

            # Call OpenAI with JSON mode
            response = await self.client.chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are a business analyst expert in IT service management and ticket categorization."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.4,  # Slightly higher for creative label generation
                max_tokens=1000
            )

            # Extract and filter labels by confidence
            business_labels = response.get('business_labels', [])
            business_summary = response.get('business_summary', '')

            # Filter by confidence threshold
            filtered_labels = [
                {
                    "label": label.get('label', ''),
                    "confidence": label.get('confidence', 0.0),
                    "category": label.get('category', 'Unknown'),
                    "reasoning": label.get('reasoning', '')
                }
                for label in business_labels
                if label.get('confidence', 0.0) >= self.generated_label_threshold
            ]

            # Add summary to first label if exists
            if filtered_labels and business_summary:
                filtered_labels[0]['business_summary'] = business_summary

            return filtered_labels

        except Exception as e:
            print(f"⚠️ Error generating business labels: {str(e)}")
            return []

    # =========================================================================
    # TECHNICAL LABEL GENERATION (New AI-generated)
    # =========================================================================

    async def generate_technical_labels(
        self,
        title: str,
        description: str,
        domain: str,
        priority: str,
        similar_tickets: List[Dict[str, Any]],
        existing_labels: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate technical labels using AI.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Classified domain
            priority: Ticket priority
            similar_tickets: List of similar tickets
            existing_labels: Set of existing labels from historical tickets

        Returns:
            List of generated technical labels with confidence and reasoning
        """
        try:
            # Generate prompt
            prompt = get_technical_label_generation_prompt(
                title=title,
                description=description,
                domain=domain,
                priority=priority,
                similar_tickets=similar_tickets,
                existing_labels=existing_labels,
                max_labels=self.technical_label_max
            )

            # Call OpenAI with JSON mode
            response = await self.client.chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are a senior software engineer expert in system diagnostics and technical categorization."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.3,  # Lower for more precise technical analysis
                max_tokens=1000
            )

            # Extract and filter labels by confidence
            technical_labels = response.get('technical_labels', [])
            root_cause_hypothesis = response.get('root_cause_hypothesis', '')

            # Filter by confidence threshold
            filtered_labels = [
                {
                    "label": label.get('label', ''),
                    "confidence": label.get('confidence', 0.0),
                    "category": label.get('category', 'Unknown'),
                    "reasoning": label.get('reasoning', '')
                }
                for label in technical_labels
                if label.get('confidence', 0.0) >= self.generated_label_threshold
            ]

            # Add hypothesis to first label if exists
            if filtered_labels and root_cause_hypothesis:
                filtered_labels[0]['root_cause_hypothesis'] = root_cause_hypothesis

            return filtered_labels

        except Exception as e:
            print(f"⚠️ Error generating technical labels: {str(e)}")
            return []

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def __call__(self, state: TicketState) -> AgentOutput:
        """
        Main agent execution function for LangGraph.

        Executes three label assignment methods:
        1. Historical labels (from similar tickets)
        2. Business labels (AI-generated)
        3. Technical labels (AI-generated)

        Args:
            state: Current ticket state

        Returns:
            Partial state update with all label categories
        """
        try:
            title = state['title']
            description = state['description']
            priority = state.get('priority', 'Medium')
            domain = state.get('classified_domain')
            similar_tickets = state.get('similar_tickets', [])

            if not domain:
                raise ValueError("Domain not classified. Classification agent must run first.")

            if not similar_tickets:
                raise ValueError("No similar tickets found. Pattern recognition agent must run first.")

            print(f"\n🏷️  Label Assignment Agent - Three-Tier Label Analysis")
            print(f"   Domain: {domain} | Similar Tickets: {len(similar_tickets)}")

            # Extract existing labels for reference
            existing_labels = self.extract_candidate_labels(similar_tickets)
            print(f"   📋 Candidate historical labels: {', '.join(existing_labels) if existing_labels else 'None'}")

            # Run all three label methods in parallel
            if self.label_generation_enabled:
                print(f"   🔄 Running historical + AI label generation...")

                historical_task = self.assign_historical_labels(
                    title, description, domain, similar_tickets
                )
                business_task = self.generate_business_labels(
                    title, description, domain, priority, similar_tickets, existing_labels
                )
                technical_task = self.generate_technical_labels(
                    title, description, domain, priority, similar_tickets, existing_labels
                )

                historical_result, business_labels, technical_labels = await asyncio.gather(
                    historical_task, business_task, technical_task
                )
            else:
                print(f"   🔄 Running historical label assignment only...")
                historical_result = await self.assign_historical_labels(
                    title, description, domain, similar_tickets
                )
                business_labels = []
                technical_labels = []

            # Extract results
            historical_labels = historical_result['historical_labels']
            historical_label_confidence = historical_result['historical_label_confidence']
            historical_label_distribution = historical_result['historical_label_distribution']

            # Print results
            print(f"\n   📌 HISTORICAL LABELS ({len(historical_labels)}):")
            if historical_labels:
                for label in historical_labels:
                    conf = historical_label_confidence.get(label, 0)
                    dist = historical_label_distribution.get(label, 'N/A')
                    print(f"      ✅ {label}: {conf:.2%} confidence (historical: {dist})")
            else:
                print(f"      None assigned")

            print(f"\n   💼 BUSINESS LABELS ({len(business_labels)}):")
            if business_labels:
                for bl in business_labels:
                    print(f"      ✅ {bl['label']} [{bl['category']}]: {bl['confidence']:.2%}")
                    print(f"         └─ {bl['reasoning'][:80]}...")
            else:
                print(f"      None generated")

            print(f"\n   🔧 TECHNICAL LABELS ({len(technical_labels)}):")
            if technical_labels:
                for tl in technical_labels:
                    print(f"      ✅ {tl['label']} [{tl['category']}]: {tl['confidence']:.2%}")
                    print(f"         └─ {tl['reasoning'][:80]}...")
            else:
                print(f"      None generated")

            # Combine all labels for backward compatibility
            all_assigned_labels = list(historical_labels)
            all_label_confidence = dict(historical_label_confidence)

            # Add business labels to combined
            for bl in business_labels:
                label_name = f"[BIZ] {bl['label']}"
                all_assigned_labels.append(label_name)
                all_label_confidence[label_name] = bl['confidence']

            # Add technical labels to combined
            for tl in technical_labels:
                label_name = f"[TECH] {tl['label']}"
                all_assigned_labels.append(label_name)
                all_label_confidence[label_name] = tl['confidence']

            total_labels = len(historical_labels) + len(business_labels) + len(technical_labels)
            print(f"\n   ✅ Total labels assigned: {total_labels}")

            # Return state update
            return {
                # Historical labels
                "historical_labels": historical_labels,
                "historical_label_confidence": historical_label_confidence,
                "historical_label_distribution": historical_label_distribution,

                # AI-Generated labels
                "business_labels": business_labels,
                "technical_labels": technical_labels,

                # Combined (backward compatibility)
                "assigned_labels": all_assigned_labels,
                "label_confidence": all_label_confidence,
                "label_distribution": historical_label_distribution,

                # Status
                "status": "success",
                "current_agent": "Label Assignment Agent",
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"Assigned {len(historical_labels)} historical labels, "
                        f"{len(business_labels)} business labels, "
                        f"{len(technical_labels)} technical labels"
                    )
                }]
            }

        except Exception as e:
            print(f"   ❌ Label assignment error: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Label Assignment Agent",
                "error_message": f"Label assignment failed: {str(e)}"
            }


# Create singleton instance for use in LangGraph
label_assignment_agent = LabelAssignmentAgent()

"""
Resolution generation agent using Chain-of-Thought reasoning.

Generates comprehensive resolution plans based on similar historical tickets.
"""
from typing import Dict, Any, List
from src.utils.openai_client import get_openai_client
from src.utils.config import Config
from src.prompts.resolution_generation_prompts import get_resolution_generation_prompt
from src.models.state_schema import TicketState, AgentOutput


class ResolutionGenerationAgent:
    """Agent for generating resolution plans based on historical analysis."""

    def __init__(self):
        self.client = get_openai_client()
        self.model = Config.RESOLUTION_MODEL  # Use GPT-4o for better reasoning
        self.temperature = Config.RESOLUTION_TEMPERATURE
        self.max_tokens = Config.MAX_RESOLUTION_TOKENS

    async def generate_resolution_plan(
        self,
        title: str,
        description: str,
        domain: str,
        priority: str,
        labels: List[str],
        similar_tickets: List[Dict[str, Any]],
        avg_similarity: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive resolution plan.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Classified domain
            priority: Ticket priority
            labels: Assigned labels
            similar_tickets: List of similar tickets
            avg_similarity: Average similarity score

        Returns:
            Resolution plan dict
        """
        try:
            # Generate prompt
            prompt = get_resolution_generation_prompt(
                title=title,
                description=description,
                domain=domain,
                priority=priority,
                labels=labels,
                similar_tickets=similar_tickets,
                avg_similarity=avg_similarity
            )

            # Call OpenAI with JSON mode
            response = await self.client.chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are an expert technical support engineer with deep knowledge of healthcare IT systems."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response

        except Exception as e:
            print(f"⚠️ Error generating resolution: {str(e)}")
            # Return fallback resolution
            return {
                "summary": f"Error generating resolution plan: {str(e)}",
                "diagnostic_steps": [{
                    "step_number": 1,
                    "description": "Manual review required",
                    "commands": [],
                    "expected_output": "N/A",
                    "estimated_time_minutes": 0
                }],
                "resolution_steps": [{
                    "step_number": 1,
                    "description": "Escalate to senior engineer for manual resolution",
                    "commands": [],
                    "validation": "N/A",
                    "estimated_time_minutes": 0,
                    "risk_level": "low",
                    "rollback_procedure": None
                }],
                "additional_considerations": ["Automatic resolution generation failed"],
                "references": [],
                "total_estimated_time_hours": 0,
                "confidence": 0.0,
                "alternative_approaches": []
            }

    def validate_resolution_plan(self, resolution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enrich resolution plan.

        Args:
            resolution_plan: Raw resolution plan from LLM

        Returns:
            Validated and enriched plan
        """
        # Ensure all required fields exist
        defaults = {
            "summary": "No summary provided",
            "diagnostic_steps": [],
            "resolution_steps": [],
            "additional_considerations": [],
            "references": [],
            "total_estimated_time_hours": 0.0,
            "confidence": 0.5,
            "alternative_approaches": []
        }

        for key, default_value in defaults.items():
            if key not in resolution_plan:
                resolution_plan[key] = default_value

        # Calculate total time if not provided
        if resolution_plan["total_estimated_time_hours"] == 0:
            total_minutes = 0
            for step in resolution_plan.get("diagnostic_steps", []):
                total_minutes += step.get("estimated_time_minutes", 0)
            for step in resolution_plan.get("resolution_steps", []):
                total_minutes += step.get("estimated_time_minutes", 0)

            resolution_plan["total_estimated_time_hours"] = round(total_minutes / 60, 2)

        return resolution_plan

    async def __call__(self, state: TicketState) -> AgentOutput:
        """
        Main agent execution function for LangGraph.

        Args:
            state: Current ticket state

        Returns:
            Partial state update with resolution plan
        """
        try:
            title = state['title']
            description = state['description']
            domain = state.get('classified_domain')
            priority = state.get('priority', 'Medium')
            labels = state.get('assigned_labels', [])
            similar_tickets = state.get('similar_tickets', [])
            search_metadata = state.get('search_metadata', {})

            if not domain:
                raise ValueError("Domain not classified. Classification agent must run first.")

            if not similar_tickets:
                raise ValueError("No similar tickets found. Pattern recognition agent must run first.")

            avg_similarity = search_metadata.get('avg_similarity', 0.0)

            print(f"\n📝 Resolution Generation Agent - Creating resolution plan")

            # Generate the prompt first so we can capture it for transparency
            resolution_prompt = get_resolution_generation_prompt(
                title=title,
                description=description,
                domain=domain,
                priority=priority,
                labels=labels,
                similar_tickets=similar_tickets,
                avg_similarity=avg_similarity
            )

            # Generate resolution plan (uses the same prompt internally)
            resolution_plan = await self.generate_resolution_plan(
                title=title,
                description=description,
                domain=domain,
                priority=priority,
                labels=labels,
                similar_tickets=similar_tickets,
                avg_similarity=avg_similarity
            )

            # Validate and enrich
            resolution_plan = self.validate_resolution_plan(resolution_plan)

            # Extract resolution confidence
            resolution_confidence = resolution_plan.get('confidence', 0.5)

            print(f"   ✅ Generated resolution plan")
            print(f"   📊 Confidence: {resolution_confidence:.2%}")
            print(f"   ⏱️  Estimated time: {resolution_plan['total_estimated_time_hours']} hours")
            print(f"   🔍 Diagnostic steps: {len(resolution_plan.get('diagnostic_steps', []))}")
            print(f"   ⚙️  Resolution steps: {len(resolution_plan.get('resolution_steps', []))}")

            # Return state update with captured prompt for UI transparency
            return {
                "resolution_plan": resolution_plan,
                "resolution_confidence": resolution_confidence,
                "resolution_generation_prompt": resolution_prompt,  # Actual prompt sent to LLM
                "status": "success",
                "current_agent": "Resolution Generation Agent",
                "messages": [{
                    "role": "assistant",
                    "content": f"Generated resolution plan with {len(resolution_plan.get('resolution_steps', []))} steps (confidence: {resolution_confidence:.2%})"
                }]
            }

        except Exception as e:
            print(f"   ❌ Resolution generation error: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Resolution Generation Agent",
                "error_message": f"Resolution generation failed: {str(e)}"
            }


# Create singleton instance for use in LangGraph
resolution_generation_agent = ResolutionGenerationAgent()

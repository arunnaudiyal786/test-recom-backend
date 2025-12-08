"""
Session Manager for ticket processing runs.

Provides centralized session ID generation and path management
for storing outputs from each agent in isolated session directories.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionManager:
    """
    Manages session-based output storage for ticket processing.

    Each processing run gets a unique session ID in format ddmmyyyyhhmm,
    with outputs stored in isolated directories for audit trail and debugging.
    """

    def __init__(self, base_output_dir: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            base_output_dir: Base directory for all outputs.
                           Defaults to project_root/output/
        """
        if base_output_dir is None:
            from config.config import Config
            self.base_output_dir = Config.PROJECT_ROOT / "output"
        else:
            self.base_output_dir = Path(base_output_dir)

    def generate_session_id(self) -> str:
        """
        Generate a unique session ID based on current timestamp.

        Format: ddmmyyyyhhmm (e.g., 031220241535 = Dec 3, 2024 15:35)

        Returns:
            Session ID string
        """
        now = datetime.now()
        return now.strftime("%d%m%Y%H%M")

    def get_session_output_dir(self, session_id: str) -> Path:
        """
        Get the output directory path for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to session output directory
        """
        return self.base_output_dir / session_id

    def create_session_directory(self, session_id: str) -> Path:
        """
        Create the session output directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to created directory
        """
        session_dir = self.get_session_output_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def save_agent_output(
        self,
        session_id: str,
        agent_name: str,
        data: Dict[str, Any]
    ) -> Path:
        """
        Save an individual agent's output to the session directory.

        Args:
            session_id: Session identifier
            agent_name: Name of the agent (e.g., 'historical_match_agent')
            data: Agent output data to save

        Returns:
            Path to saved file
        """
        session_dir = self.create_session_directory(session_id)

        # Normalize agent name to snake_case filename
        filename = self._normalize_agent_filename(agent_name)
        output_path = session_dir / f"{filename}.json"

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path

    def save_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save session metadata (ticket info, timestamps, etc.).

        Args:
            session_id: Session identifier
            metadata: Metadata dictionary

        Returns:
            Path to saved file
        """
        session_dir = self.create_session_directory(session_id)
        output_path = session_dir / "session_metadata.json"

        # Add session_id and timestamp to metadata
        full_metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            **metadata
        }

        with open(output_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)

        return output_path

    def save_final_output(
        self,
        session_id: str,
        data: Dict[str, Any],
        filename: str = "ticket_resolution.json"
    ) -> Path:
        """
        Save the final accumulated output to the session directory.

        Args:
            session_id: Session identifier
            data: Final output data
            filename: Output filename

        Returns:
            Path to saved file
        """
        session_dir = self.create_session_directory(session_id)
        output_path = session_dir / filename

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path

    def update_latest_symlink(self, session_id: str) -> Optional[Path]:
        """
        Update the 'latest' symlink to point to the most recent session.

        Args:
            session_id: Session identifier to link to

        Returns:
            Path to symlink, or None if symlink creation failed
        """
        latest_link = self.base_output_dir / "latest"
        session_dir = self.get_session_output_dir(session_id)

        try:
            # Remove existing symlink if present
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                # If it's a regular directory (Windows fallback), remove it
                import shutil
                shutil.rmtree(latest_link)

            # Create new symlink (relative path for portability)
            latest_link.symlink_to(session_id, target_is_directory=True)
            return latest_link

        except OSError as e:
            # Symlink creation may fail on Windows without admin rights
            print(f"Warning: Could not create 'latest' symlink: {e}")
            return None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all available sessions with their metadata.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session info dicts, sorted by date (newest first)
        """
        sessions = []

        if not self.base_output_dir.exists():
            return sessions

        for item in self.base_output_dir.iterdir():
            # Skip non-directories and special entries
            if not item.is_dir() or item.name == "latest" or item.name.startswith("."):
                continue

            # Try to parse as session ID (ddmmyyyyhhmm format)
            if len(item.name) == 12 and item.name.isdigit():
                session_info = self._get_session_info(item.name)
                if session_info:
                    sessions.append(session_info)

        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return sessions[:limit]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Session info dict or None if not found
        """
        session_dir = self.get_session_output_dir(session_id)

        if not session_dir.exists():
            return None

        return self._get_session_info(session_id)

    def get_session_output(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the final output from a session.

        Args:
            session_id: Session identifier

        Returns:
            Output data dict or None if not found
        """
        output_path = self.get_session_output_dir(session_id) / "ticket_resolution.json"

        if not output_path.exists():
            return None

        with open(output_path, 'r') as f:
            return json.load(f)

    def get_agent_output(self, session_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific agent's output from a session.

        Args:
            session_id: Session identifier
            agent_name: Agent name

        Returns:
            Agent output data or None if not found
        """
        filename = self._normalize_agent_filename(agent_name)
        output_path = self.get_session_output_dir(session_id) / f"{filename}.json"

        if not output_path.exists():
            return None

        with open(output_path, 'r') as f:
            return json.load(f)

    def _normalize_agent_filename(self, agent_name: str) -> str:
        """
        Normalize agent name to a filesystem-safe filename.

        Args:
            agent_name: Agent name (e.g., 'historicalMatch', 'Historical Match Agent')

        Returns:
            Normalized filename (e.g., 'historical_match_agent')
        """
        # Mapping from API keys to filenames
        name_mapping = {
            "classification": "classification_agent",
            "historicalMatch": "historical_match_agent",
            "labelAssignment": "label_assignment_agent",
            "novelty": "novelty_detection_agent",
            "resolutionGeneration": "resolution_generation_agent",
            # Full names
            "Domain Classification Agent": "classification_agent",
            "Historical Match Agent": "historical_match_agent",
            "Label Assignment Agent": "label_assignment_agent",
            "Novelty Detection Agent": "novelty_detection_agent",
            "Resolution Generation Agent": "resolution_generation_agent",
            # Short names from node state
            "retrieval": "historical_match_agent",
            "labeling": "label_assignment_agent",
            "resolution": "resolution_generation_agent",
            # Backward compatibility
            "patternRecognition": "historical_match_agent",
            "Pattern Recognition Agent": "historical_match_agent",
        }

        if agent_name in name_mapping:
            return name_mapping[agent_name]

        # Fallback: convert to snake_case
        import re
        # Insert underscore before uppercase letters and convert to lowercase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', agent_name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower().replace(" ", "_").replace("__", "_")

    def _get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract session information from directory.

        Args:
            session_id: Session identifier

        Returns:
            Session info dict
        """
        session_dir = self.get_session_output_dir(session_id)

        if not session_dir.exists():
            return None

        # Parse timestamp from session ID (ddmmyyyyhhmm)
        try:
            created_at = datetime.strptime(session_id, "%d%m%Y%H%M")
        except ValueError:
            created_at = None

        # List available outputs
        available_outputs = []
        for f in session_dir.iterdir():
            if f.is_file() and f.suffix == ".json":
                available_outputs.append(f.stem)

        # Try to load session metadata
        metadata_path = session_dir / "session_metadata.json"
        ticket_id = None
        ticket_title = None

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    ticket_id = metadata.get("ticket_id")
                    ticket_title = metadata.get("title")
            except Exception:
                pass

        # Fallback: try to get ticket info from final output
        if not ticket_id:
            output_path = session_dir / "ticket_resolution.json"
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        output = json.load(f)
                        ticket_id = output.get("ticket_id")
                        ticket_title = output.get("title")
                except Exception:
                    pass

        return {
            "session_id": session_id,
            "created_at": created_at.isoformat() if created_at else None,
            "ticket_id": ticket_id,
            "ticket_title": ticket_title,
            "available_outputs": available_outputs,
            "path": str(session_dir)
        }


# Convenience function for quick access
def get_session_manager() -> SessionManager:
    """Get a SessionManager instance with default configuration."""
    return SessionManager()

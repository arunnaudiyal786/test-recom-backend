"""
Schema Configuration Loader

Loads and provides access to the schema configuration from YAML file.
This enables dynamic column mappings, domain definitions, and UI settings.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache


class SchemaConfig:
    """
    Configuration manager for dynamic schema definitions.

    Loads configuration from config/schema_config.yaml and provides
    typed access to column mappings, domain definitions, and UI settings.
    """

    _instance: Optional['SchemaConfig'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "schema_config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Schema configuration not found at {config_path}. "
                "Please create config/schema_config.yaml"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def reload(self):
        """Reload configuration from file (useful for hot-reloading)."""
        self._load_config()

    # =========================================================================
    # Data Source Configuration
    # =========================================================================

    @property
    def data_source_type(self) -> str:
        """Get the data source type (historical_tickets or test_plan)."""
        return self._config.get('data_source_type', 'historical_tickets')

    # =========================================================================
    # Column Mappings
    # =========================================================================

    @property
    def column_mappings(self) -> Dict[str, Optional[str]]:
        """Get column name mappings from internal names to CSV column names."""
        return self._config.get('column_mappings', {})

    def get_csv_column(self, internal_name: str) -> Optional[str]:
        """
        Get the CSV column name for an internal field name.

        Args:
            internal_name: Internal field name (e.g., 'ticket_id', 'title')

        Returns:
            CSV column name or None if not mapped
        """
        return self.column_mappings.get(internal_name)

    # =========================================================================
    # Label Configuration
    # =========================================================================

    @property
    def label_columns(self) -> List[str]:
        """Get list of columns that contain labels."""
        labels_config = self._config.get('labels', {})
        return labels_config.get('columns', [])

    @property
    def excluded_label_columns(self) -> List[str]:
        """Get list of columns to exclude from labels."""
        labels_config = self._config.get('labels', {})
        return labels_config.get('exclude_columns', [])

    # =========================================================================
    # Domain Configuration
    # =========================================================================

    @property
    def domain_extraction_column(self) -> str:
        """Get the column used to extract domain."""
        domains_config = self._config.get('domains', {})
        return domains_config.get('extraction_column', 'Labels2')

    @property
    def domain_map(self) -> Dict[str, str]:
        """Get mapping from CSV values to domain names."""
        domains_config = self._config.get('domains', {})
        return domains_config.get('domain_map', {})

    @property
    def domain_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get full domain definitions including prompts and display info."""
        domains_config = self._config.get('domains', {})
        return domains_config.get('definitions', {})

    @property
    def domain_names(self) -> List[str]:
        """Get list of all valid domain names."""
        return list(self.domain_definitions.keys())

    def get_domain_prompt(self, domain: str) -> str:
        """
        Get the classification prompt for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Classification prompt string
        """
        definitions = self.domain_definitions
        if domain in definitions:
            return definitions[domain].get('classification_prompt', '')
        return ''

    def get_domain_display_info(self, domain: str) -> Dict[str, Any]:
        """
        Get display information for a domain.

        Args:
            domain: Domain name

        Returns:
            Dict with full_name, description, color_scheme
        """
        definitions = self.domain_definitions
        if domain in definitions:
            return {
                'full_name': definitions[domain].get('full_name', domain),
                'description': definitions[domain].get('description', ''),
                'color_scheme': definitions[domain].get('color_scheme', 'slate'),
            }
        return {
            'full_name': domain,
            'description': '',
            'color_scheme': 'slate',
        }

    def extract_domain_from_value(self, value: str) -> str:
        """
        Extract domain name from a CSV column value.

        Args:
            value: Value from the domain extraction column

        Returns:
            Domain name or 'Unknown'
        """
        return self.domain_map.get(value, 'Unknown')

    # =========================================================================
    # Priority Configuration
    # =========================================================================

    @property
    def priority_config(self) -> Dict[str, Any]:
        """Get priority configuration."""
        return self._config.get('priority', {})

    @property
    def valid_priorities(self) -> List[str]:
        """Get list of valid priority values."""
        return self.priority_config.get('valid_values', ['Low', 'Medium', 'High', 'Critical'])

    def derive_priority_from_story_points(self, story_points: Any) -> str:
        """
        Derive priority level from story points.

        Args:
            story_points: Story points value (can be string or int)

        Returns:
            Priority level string
        """
        try:
            points = int(story_points)
            sp_map = self.priority_config.get('story_points_map', {})

            if points <= sp_map.get('low_max', 8):
                return 'Low'
            elif points <= sp_map.get('medium_max', 21):
                return 'Medium'
            elif points <= sp_map.get('high_max', 34):
                return 'High'
            else:
                return 'Critical'
        except (ValueError, TypeError):
            return 'Medium'

    # =========================================================================
    # UI Configuration
    # =========================================================================

    @property
    def ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get('ui', {})

    @property
    def color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Get color scheme definitions."""
        return self.ui_config.get('color_schemes', {})

    def get_domain_colors(self, domain: str) -> Dict[str, str]:
        """
        Get color classes for a domain.

        Args:
            domain: Domain name

        Returns:
            Dict with bg, text, border, icon color classes
        """
        display_info = self.get_domain_display_info(domain)
        color_scheme = display_info.get('color_scheme', 'slate')
        return self.color_schemes.get(color_scheme, self.color_schemes.get('slate', {}))

    @property
    def sample_ticket_placeholder(self) -> str:
        """Get sample ticket placeholder text for UI."""
        return self.ui_config.get('sample_ticket_placeholder', 'Enter ticket description...')

    # =========================================================================
    # Export for API
    # =========================================================================

    def get_frontend_config(self) -> Dict[str, Any]:
        """
        Get configuration formatted for frontend consumption.

        Returns:
            Dict with domains, colors, and UI settings
        """
        domains_for_frontend = {}
        for domain in self.domain_names:
            display_info = self.get_domain_display_info(domain)
            colors = self.get_domain_colors(domain)
            domains_for_frontend[domain] = {
                'full_name': display_info['full_name'],
                'description': display_info['description'],
                'colors': colors,
            }

        return {
            'domains': domains_for_frontend,
            'domain_list': self.domain_names,
            'priorities': self.valid_priorities,
            'sample_placeholder': self.sample_ticket_placeholder,
        }

    def get_classification_config(self) -> Dict[str, Any]:
        """
        Get configuration for the classification agent.

        Returns:
            Dict with domain names and their prompts
        """
        return {
            'domains': self.domain_names,
            'prompts': {
                domain: self.get_domain_prompt(domain)
                for domain in self.domain_names
            }
        }


# Singleton accessor
@lru_cache(maxsize=1)
def get_schema_config() -> SchemaConfig:
    """Get the singleton schema configuration instance."""
    return SchemaConfig()


# Convenience function to reload config
def reload_schema_config():
    """Reload schema configuration from file."""
    get_schema_config.cache_clear()
    return get_schema_config()

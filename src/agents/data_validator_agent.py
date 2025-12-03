"""
Data Validator Agent for the Data Preparation Pipeline.

Validates CSV ticket data, identifies missing critical fields,
and generates comprehensive data quality reports.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

from src.models.data_prep_state import DataPrepState, DataPrepAgentOutput
from config import Config


class DataValidatorAgent:
    """
    Agent responsible for validating incoming CSV ticket data.

    Responsibilities:
    - Load CSV file
    - Identify missing critical fields per row
    - Calculate completeness scores
    - Generate data quality report with null percentages
    """

    def __init__(self):
        """Initialize the Data Validator Agent."""
        # Define critical fields that should not be empty
        self.critical_fields = [
            "Summary",
            "Description",
            "Resolution",
            "Issue Priority",
            "Labels"
        ]

        # Define important (but not critical) fields
        self.important_fields = [
            "key",
            "created",
            "closed date",
            "issue type",
            "assignee",
            "IT Team",
            "Reporter"
        ]

        # All expected fields
        self.all_expected_fields = self.critical_fields + self.important_fields

    async def __call__(self, state: DataPrepState) -> DataPrepAgentOutput:
        """
        Main entry point for LangGraph workflow.

        Args:
            state: Current workflow state containing input_file_path

        Returns:
            Partial state update with validation results
        """
        try:
            print(f"[Data Validator Agent] Starting validation...")

            # Extract input file path from state
            input_file_path = state.get("input_file_path", "")
            if not input_file_path:
                raise ValueError("input_file_path not provided in state")

            # Load CSV file
            print(f"[Data Validator Agent] Loading CSV from: {input_file_path}")
            df = await self._load_csv(input_file_path)

            # Convert DataFrame to list of dicts for state storage
            raw_data = df.to_dict('records')
            total_rows = len(raw_data)
            column_names = list(df.columns)

            print(f"[Data Validator Agent] Loaded {total_rows} rows with {len(column_names)} columns")

            # Validate each row
            per_row_validation = await self._validate_rows(df)

            # Generate overall validation report
            validation_report = await self._generate_validation_report(
                df, per_row_validation
            )

            print(f"[Data Validator Agent] Validation complete. Score: {validation_report['overall_completeness_score']:.2%}")

            return {
                "status": "success",
                "current_agent": "Data Validator Agent",
                "processing_stage": "validation",
                "raw_data": raw_data,
                "total_rows": total_rows,
                "column_names": column_names,
                "validation_report": validation_report,
                "per_row_validation": per_row_validation,
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"Data Validator Agent completed. "
                        f"Processed {total_rows} rows. "
                        f"Overall completeness score: {validation_report['overall_completeness_score']:.2%}. "
                        f"Rows with issues: {validation_report['rows_with_issues']}."
                    )
                }]
            }

        except Exception as e:
            print(f"[Data Validator Agent] ERROR: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Data Validator Agent",
                "processing_stage": "validation",
                "error_message": f"Data validation failed: {str(e)}",
                "messages": [{
                    "role": "assistant",
                    "content": f"Data Validator Agent failed with error: {str(e)}"
                }]
            }

    async def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not path.suffix.lower() == '.csv':
            raise ValueError(f"File must be a CSV file, got: {path.suffix}")

        # Load CSV with flexible parsing
        # Use quotechar and handle multi-line fields properly
        df = pd.read_csv(
            path,
            encoding='utf-8',
            on_bad_lines='warn',  # Warn about bad lines but continue
            na_values=['', 'NA', 'N/A', 'null', 'None'],  # Treat these as NaN
            keep_default_na=True,
            quotechar='"',  # Handle quoted fields with newlines
            escapechar='\\',  # Handle escaped characters
            doublequote=True,  # Handle "" as escaped quote
            engine='python'  # Use Python engine for better multi-line support
        )

        return df

    async def _validate_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Validate each row in the DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation results per row
        """
        per_row_results = []

        for idx, row in df.iterrows():
            row_validation = {
                "row_index": int(idx),
                "is_valid": True,
                "missing_fields": [],
                "completeness_score": 0.0,
                "warnings": []
            }

            # Check for missing critical fields
            for field in self.critical_fields:
                if field in df.columns:
                    value = row.get(field)
                    if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                        row_validation["missing_fields"].append(field)
                        row_validation["is_valid"] = False
                else:
                    row_validation["missing_fields"].append(f"{field} (column missing)")
                    row_validation["is_valid"] = False

            # Check for missing important fields (warnings only)
            for field in self.important_fields:
                if field in df.columns:
                    value = row.get(field)
                    if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                        row_validation["warnings"].append(f"Missing optional field: {field}")
                else:
                    row_validation["warnings"].append(f"Column not present: {field}")

            # Calculate completeness score for this row
            row_validation["completeness_score"] = self._calculate_row_completeness(row, df.columns)

            per_row_results.append(row_validation)

        return per_row_results

    def _calculate_row_completeness(self, row: pd.Series, columns: pd.Index) -> float:
        """
        Calculate completeness score for a single row.

        TODO: You should implement this method to define your own weighting logic.

        Args:
            row: Single row from DataFrame
            columns: Available columns in DataFrame

        Returns:
            Completeness score between 0.0 and 1.0
        """
        # ========== YOUR IMPLEMENTATION HERE ==========
        # This is where you define how to weight different fields.
        #
        # Consider:
        # - Critical fields (Summary, Description, Resolution) should have higher weights
        # - Optional fields (Reporter, assignee) should have lower weights
        # - You might want to give partial credit for partial data
        #
        # Example implementation provided below - feel free to modify!

        total_weight = 0.0
        achieved_weight = 0.0

        # Critical field weights (higher importance)
        critical_weights = {
            "Summary": 0.25,
            "Description": 0.25,
            "Resolution": 0.20,
            "Issue Priority": 0.10,
            "Labels": 0.10
        }

        # Important field weights (lower importance)
        important_weights = {
            "key": 0.02,
            "created": 0.02,
            "closed date": 0.02,
            "issue type": 0.02,
            "assignee": 0.01,
            "IT Team": 0.005,
            "Reporter": 0.005
        }

        # Calculate score from critical fields
        for field, weight in critical_weights.items():
            total_weight += weight
            if field in columns:
                value = row.get(field)
                if not pd.isna(value) and (not isinstance(value, str) or value.strip() != ""):
                    achieved_weight += weight

        # Calculate score from important fields
        for field, weight in important_weights.items():
            total_weight += weight
            if field in columns:
                value = row.get(field)
                if not pd.isna(value) and (not isinstance(value, str) or value.strip() != ""):
                    achieved_weight += weight

        # Return normalized score
        if total_weight == 0:
            return 0.0

        return achieved_weight / total_weight

        # ========== END OF YOUR IMPLEMENTATION ==========

    async def _generate_validation_report(
        self,
        df: pd.DataFrame,
        per_row_validation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            df: Original DataFrame
            per_row_validation: Per-row validation results

        Returns:
            Overall validation report
        """
        total_rows = len(df)

        # Count valid rows
        valid_rows = sum(1 for v in per_row_validation if v["is_valid"])
        rows_with_issues = total_rows - valid_rows

        # Calculate overall completeness score
        if per_row_validation:
            overall_score = sum(v["completeness_score"] for v in per_row_validation) / len(per_row_validation)
        else:
            overall_score = 0.0

        # Calculate null percentages per column
        column_null_percentages = {}
        for col in df.columns:
            null_count = df[col].isna().sum()
            # Also count empty strings
            if df[col].dtype == object:
                null_count += (df[col] == "").sum()
            column_null_percentages[col] = (null_count / total_rows) * 100 if total_rows > 0 else 0

        # Identify critical fields that are missing entirely or have high null rates
        critical_fields_with_issues = []
        for field in self.critical_fields:
            if field not in df.columns:
                critical_fields_with_issues.append(f"{field} (column missing entirely)")
            elif column_null_percentages.get(field, 0) > 50:
                critical_fields_with_issues.append(
                    f"{field} ({column_null_percentages[field]:.1f}% null)"
                )

        # Count most common missing fields
        missing_field_counts = {}
        for validation in per_row_validation:
            for field in validation["missing_fields"]:
                missing_field_counts[field] = missing_field_counts.get(field, 0) + 1

        report = {
            "overall_completeness_score": overall_score,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "rows_with_issues": rows_with_issues,
            "column_null_percentages": column_null_percentages,
            "critical_fields_with_issues": critical_fields_with_issues,
            "missing_field_summary": missing_field_counts,
            "columns_present": list(df.columns),
            "columns_missing": [
                f for f in self.all_expected_fields if f not in df.columns
            ]
        }

        return report


# Create singleton instance for LangGraph workflow
data_validator_agent = DataValidatorAgent()

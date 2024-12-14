from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

class TimeFieldsModel(BaseModel):
    query: str = Field(..., description="Query string")  # Mandatory field
    workspace_id: str = Field(..., description="Workspace ID")  # Mandatory field
    days: Optional[int] = Field(None, ge=0, description="Number of days")
    hours: Optional[int] = Field(None, ge=0, description="Number of hours")
    minutes: Optional[int] = Field(None, ge=0, description="Number of minutes")
    seconds: Optional[int] = Field(None, ge=0, description="Number of seconds")
    start_date: Optional[datetime] = Field(None, description="Start date and time")
    end_date: Optional[datetime] = Field(None, description="End date and time")

    @model_validator(mode="after")
    def validate_time_fields(cls, model_instance):
        # Access attributes directly
        days = model_instance.days
        hours = model_instance.hours
        minutes = model_instance.minutes
        seconds = model_instance.seconds
        start_date = model_instance.start_date
        end_date = model_instance.end_date

        # Validate start_date and end_date
        if (start_date is None) != (end_date is None):  # XOR check for None
            raise ValueError("Both start_date and end_date must either have values or be None.")
        if start_date and end_date:
            if start_date >= end_date:
                raise ValueError("start_date must be less than end_date.")

        # Count non-None fields excluding start_date and end_date
        non_empty_fields = [field for field in [days, hours, minutes, seconds, start_date] if field is not None]

        # Ensure exactly one field has a value
        if not non_empty_fields:
            raise ValueError("At least one time-related field must have a value.")
        if len(non_empty_fields) > 1:
            raise ValueError("Only one time-related field can have a value at a time.")

        return model_instance






from typing import Optional

# Assuming TimeFieldsModel is already defined
class TimeFieldProcessor:
    def __init__(self):
        pass

    def process_time_fields(self, time_fields: "TimeFieldsModel") -> str:
        """Processes the validated time fields and performs operations."""
        # Process mandatory fields
        query = time_fields.query
        workspace_id = time_fields.workspace_id

        result = f"Query: {query}, Workspace ID: {workspace_id}. "

        # Process time fields
        if time_fields.days is not None:
            result += f"Processing {time_fields.days} days."
        elif time_fields.hours is not None:
            result += f"Processing {time_fields.hours} hours."
        elif time_fields.minutes is not None:
            result += f"Processing {time_fields.minutes} minutes."
        elif time_fields.seconds is not None:
            result += f"Processing {time_fields.seconds} seconds."
        elif time_fields.start_date and time_fields.end_date:
            duration = (time_fields.end_date - time_fields.start_date).total_seconds()
            result += f"Processing a time range: {duration} seconds."
        else:
            result += "No valid time fields to process."

        return result

    def convert_to_seconds(self, time_fields: "TimeFieldsModel") -> Optional[int]:
        """Converts time fields to seconds, if applicable."""
        # Ensure mandatory fields are present
        query = time_fields.query
        workspace_id = time_fields.workspace_id

        # Convert time fields to seconds
        if time_fields.days is not None:
            return time_fields.days * 24 * 3600
        elif time_fields.hours is not None:
            return time_fields.hours * 3600
        elif time_fields.minutes is not None:
            return time_fields.minutes * 60
        elif time_fields.seconds is not None:
            return time_fields.seconds
        elif time_fields.start_date and time_fields.end_date:
            return int((time_fields.end_date - time_fields.start_date).total_seconds())
        else:
            return None


from datetime import datetime

processor = TimeFieldProcessor()

# Example 1: Processing days
time_fields = TimeFieldsModel(query="fetch_logs", workspace_id="workspace123", days=5)
print(processor.process_time_fields(time_fields))  # Output includes query and workspace_id
print(processor.convert_to_seconds(time_fields))  # Converts to seconds

# Example 2: Processing a time range
time_fields = TimeFieldsModel(
    query="analyze_logs",
    workspace_id="workspace456",
    start_date=datetime(2024, 12, 14, 10, 0, 0),
    end_date=datetime(2024, 12, 15, 10, 0, 0)
)
print(processor.process_time_fields(time_fields))  # Output includes query, workspace_id, and duration
print(processor.convert_to_seconds(time_fields))  # Returns duration in seconds



dates = ["20241210", "20241211", "20241212"]
start_dates = [
    datetime.strptime(date, "%Y%m%d").replace(hour=10, minute=0, second=0) 
    for date in dates
]
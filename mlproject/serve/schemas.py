from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    """
    Request model for the prediction endpoint.

    Attributes:
        data (Dict[str, List]): A dictionary containing historical time-series data.
            - Keys: feature names, e.g., "date", "HUFL", "MUFL", "mobility_inflow", etc.
            - Values: Lists of values corresponding to each feature over time.

    Example:
        {
            "data": {
                "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
                "HUFL": [5.827, 5.8],
                "MUFL": [1.599, 1.492],
                "mobility_inflow": [1.234, 2.345]
            }
        }
    """

    # Sử dụng List[Any] để tránh warning về type hint
    data: Dict[str, List[Any]]

    # CẬP NHẬT: Dùng model_config thay cho class Config (Pydantic V2)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": {
                    "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
                    "HUFL": [5.827, 5.8],
                    "MUFL": [1.599, 1.492],
                    "mobility_inflow": [1.234, 2.345],
                }
            }
        }
    )

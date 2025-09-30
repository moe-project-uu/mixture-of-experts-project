import os
import csv
from typing import Dict, Any

class CSVLogger:
    """Simple CSV logger for tracking training metrics."""
    
    def __init__(self, log_path: str, fieldnames: list):
        """
        Initialize CSV logger.
        
        Args:
            log_path: Path to the CSV file
            fieldnames: List of column names for the CSV
        """
        self.log_path = log_path
        self.fieldnames = fieldnames
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics to CSV file.
        
        Args:
            metrics: Dictionary containing metric values
        """
        with open(self.log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(metrics)
    
    def read_logs(self):
        """Read all logged metrics from CSV file."""
        metrics = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    metrics.append(row)
        return metrics
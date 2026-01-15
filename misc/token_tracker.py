"""
Token usage tracking for OpenAI API calls.
Tracks input tokens, output tokens, and total consumption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("token_tracker")


class TokenTracker:
    """Track and log OpenAI API token usage."""
    
    def __init__(self, log_file: str = "output/token_usage.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()
    
    def _load_history(self):
        """Load token usage history from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load token history: {e}")
                self.history = {"usage_log": [], "totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        else:
            self.history = {"usage_log": [], "totals": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    
    def _save_history(self):
        """Save token usage history to file."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save token history: {e}")
    
    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model: str,
        campaign: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Record a token usage event.
        
        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            total_tokens: Total tokens used
            model: OpenAI model name (e.g., "gpt-4o-mini")
            campaign: Campaign name (optional)
            metadata: Additional metadata (optional)
        """
        usage_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "campaign": campaign,
            "metadata": metadata or {},
        }
        
        # Add to log
        self.history["usage_log"].append(usage_entry)
        
        # Update totals
        self.history["totals"]["input_tokens"] += input_tokens
        self.history["totals"]["output_tokens"] += output_tokens
        self.history["totals"]["total_tokens"] += total_tokens
        
        # Save to file
        self._save_history()
        
        # Log to console
        logger.info(
            f"Token usage recorded: {input_tokens:,} input + {output_tokens:,} output = {total_tokens:,} total "
            f"(Model: {model}, Campaign: {campaign or 'N/A'})"
        )
        logger.info(
            f"Cumulative totals: {self.history['totals']['input_tokens']:,} input + "
            f"{self.history['totals']['output_tokens']:,} output = "
            f"{self.history['totals']['total_tokens']:,} total"
        )
    
    def get_totals(self) -> Dict[str, int]:
        """Get cumulative token usage totals."""
        return self.history["totals"].copy()
    
    def get_recent_usage(self, n: int = 10) -> list:
        """Get the most recent n usage entries."""
        return self.history["usage_log"][-n:]
    
    def get_usage_by_campaign(self, campaign: str) -> Dict[str, int]:
        """Get token usage totals for a specific campaign."""
        campaign_entries = [
            entry for entry in self.history["usage_log"]
            if entry.get("campaign") == campaign
        ]
        
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for entry in campaign_entries:
            totals["input_tokens"] += entry["input_tokens"]
            totals["output_tokens"] += entry["output_tokens"]
            totals["total_tokens"] += entry["total_tokens"]
        
        return totals
    
    def reset_totals(self):
        """Reset cumulative totals (keeps history log)."""
        self.history["totals"] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._save_history()
        logger.info("Token usage totals reset")


# Global tracker instance
_tracker = None


def get_tracker() -> TokenTracker:
    """Get or create the global token tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def extract_token_usage(response) -> Optional[Dict[str, int]]:
    """
    Extract token usage from OpenAI API response.
    
    Args:
        response: OpenAI API response object (from chat.completions.create)
        
    Returns:
        Dict with 'input_tokens', 'output_tokens', 'total_tokens' or None if not available
    """
    try:
        if hasattr(response, "usage"):
            usage = response.usage
            return {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        elif isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")
    
    return None


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate token count for text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: OpenAI model name (used to select encoding)
        
    Returns:
        Estimated token count
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4
    
    try:
        # Map model names to encodings
        encoding_name = "cl100k_base"  # Default for gpt-4, gpt-4o-mini, etc.
        if "gpt-3.5" in model.lower():
            encoding_name = "cl100k_base"
        elif "gpt-4" in model.lower() or "gpt-4o" in model.lower():
            encoding_name = "cl100k_base"
        
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to estimate tokens with tiktoken: {e}, using fallback")
        return len(text) // 4

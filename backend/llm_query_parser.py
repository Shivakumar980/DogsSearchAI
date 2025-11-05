import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('config/.env')


class LLMQueryParser:
    """
    LLM-based query parser that extracts structured filters from natural language queries.
    
    Converts queries like:
    - "small furry friendly dog" → {size: 'small', temperament_required: ['friendly']}
    - "a dog for protection" → {temperament_required: ['protective', 'guard'], bred_for: 'protection'}
    - "apartment dog that's good with kids" → {apartment_suitable: True, good_with_kids: True}
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM query parser
        
        Args:
            model: OpenAI model to use for parsing (gpt-4o-mini is fast and cheap)
        """
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model = model
        
        # System prompt that defines the parsing task
        self.system_prompt = """You are an intelligent dog breed search query parser. Extract structured filters by intelligently interpreting adjectives, descriptive phrases, and natural language in queries about dog breeds.

AVAILABLE FILTER FIELDS:
- size: "tiny", "small", "medium", "large", or "giant" (based on weight)
- temperament_required: List of temperament traits (extract adjectives like "friendly", "playful", "protective", "loyal", "calm", "energetic", "gentle", etc.)
- temperament_avoid: List of traits to avoid (extract negative adjectives like "aggressive", "vocal", "barky", "stubborn", "destructive", etc.)
- weight_min_lbs: Minimum weight in pounds (extract from phrases like "over X lbs", "at least X", "heavy", "large")
- weight_max_lbs: Maximum weight in pounds (extract from phrases like "under X lbs", "less than X", "light weight", "lightweight", "small", "tiny")
- activity_level: "low", "moderate", or "high" (infer from "lazy", "calm", "energetic", "active", "high energy", etc.)
- apartment_suitable: true if user mentions apartment/condo/small space OR if size/weight suggests small living space
- good_with_kids: true if user mentions kids/children/family OR if temperament suggests family-friendly (gentle, patient, friendly)
- special_requirements: List of other needs (e.g., "hypoallergenic", "easy to train", "low maintenance", "independent")
- bred_for: What the dog was bred for (extract from context like "protection", "hunting", "companionship", "herding")

INTELLIGENT EXTRACTION RULES:
1. ADJECTIVES → FILTERS:
   - Size adjectives: "small", "tiny", "mini", "large", "big", "giant" → size OR weight_max_lbs/weight_min_lbs
   - Weight adjectives: "light", "lightweight", "heavy", "large" → weight_max_lbs or weight_min_lbs
   - Temperament adjectives: "friendly", "playful", "protective", "calm", "aggressive" → temperament_required or temperament_avoid
   - Energy adjectives: "energetic", "active", "lazy", "calm" → activity_level
   - Behavior adjectives: "quiet", "vocal", "barky", "independent" → temperament_required or temperament_avoid

2. CONTEXTUAL INFERENCE:
   - "light weight" / "lightweight" → weight_max_lbs: 25, size: "small"
   - "small dog" → size: "small" OR weight_max_lbs: 25
   - "family dog" → good_with_kids: true, temperament_required: ["friendly", "gentle"]
   - "apartment dog" → apartment_suitable: true, size: "small" OR weight_max_lbs: 25
   - "protection dog" / "guard dog" → bred_for: "protection", temperament_required: ["protective", "alert"]
   - "quiet" / "won't bark" → temperament_avoid: ["vocal", "barky"]
   - "energetic" → activity_level: "high" OR temperament_required: ["energetic", "active"]

3. PHRASE INTERPRETATION:
   - "dogs under X pounds" → weight_max_lbs: X
   - "dogs over X pounds" → weight_min_lbs: X
   - "first-time owner" → temperament_required: ["easy going", "trainable"], activity_level: "low" to "moderate"
   - "elderly person" → activity_level: "low", size: "small" to "medium", temperament_required: ["gentle", "calm"]

4. SMART BOOLEANS:
   - apartment_suitable: true if explicitly mentioned OR if size/weight suggests small space
   - good_with_kids: true if explicitly mentioned OR if temperament suggests family-friendly

Return ONLY a valid JSON object with these fields. Use null for missing fields, empty lists [] for missing lists, false for missing booleans.

EXAMPLES:
Query: "light weight dogs"
→ Extract: "light weight" = lightweight/small
→ Response: {"weight_max_lbs": 25, "size": "small", "apartment_suitable": false, "good_with_kids": false, ...}

Query: "small friendly dog for apartment"
→ Extract: "small" = size, "friendly" = temperament, "apartment" = living space
→ Response: {"size": "small", "temperament_required": ["friendly"], "apartment_suitable": true, ...}

Query: "energetic hiking companion good with kids"
→ Extract: "energetic" = activity, "hiking" = activity context, "good with kids" = family
→ Response: {"activity_level": "high", "temperament_required": ["energetic", "active"], "good_with_kids": true, ...}

Query: "quiet dog that won't bark at neighbors"
→ Extract: "quiet" = avoid vocal, "won't bark" = avoid vocal
→ Response: {"temperament_avoid": ["vocal", "barky"], ...}

Query: "protective guard dog"
→ Extract: "protective" = temperament, "guard" = bred_for
→ Response: {"temperament_required": ["protective", "alert"], "bred_for": "protection", ...}"""
    
    def parse(self, query: str, verbose: bool = False) -> Dict:
        """
        Parse a natural language query into structured filters
        
        Args:
            query: Natural language query about dog breeds
            verbose: Whether to print debug information
            
        Returns:
            Dictionary with extracted filters:
            - size: str | None
            - temperament_required: List[str]
            - temperament_avoid: List[str]
            - weight_min_lbs: int | None
            - weight_max_lbs: int | None
            - activity_level: str | None
            - apartment_suitable: bool
            - good_with_kids: bool
            - special_requirements: List[str]
            - bred_for: str | None
        """
        try:
            if verbose:
                print(f"  Parsing query: '{query}'")
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Parse this query: {query}"}
                ],
                temperature=0.1,  # Low temperature for consistent parsing
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content
            filters = json.loads(content)
            
            # Normalize the response
            filters = self._normalize_filters(filters)
            
            if verbose:
                print(f"  Extracted filters: {filters}")
            
            return filters
            
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ⚠️  Failed to parse JSON response: {e}")
                print(f"  Response was: {content}")
            # Return empty filters on parse error
            return {}
            
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error parsing query: {e}")
            return {}
    
    def _normalize_filters(self, filters: Dict) -> Dict:
        """Normalize filter values to expected types and formats"""
        normalized = {}
        
        # Size: must be one of the valid sizes
        valid_sizes = ["tiny", "small", "medium", "large", "giant"]
        if 'size' in filters and filters['size'] in valid_sizes:
            normalized['size'] = filters['size']
        
        # Temperament lists
        normalized['temperament_required'] = self._normalize_list(filters.get('temperament_required'))
        normalized['temperament_avoid'] = self._normalize_list(filters.get('temperament_avoid'))
        normalized['special_requirements'] = self._normalize_list(filters.get('special_requirements'))
        
        # Weight (must be positive numbers)
        if 'weight_min_lbs' in filters:
            try:
                weight_min = float(filters['weight_min_lbs'])
                if weight_min > 0:
                    normalized['weight_min_lbs'] = weight_min
            except (ValueError, TypeError):
                pass
        
        if 'weight_max_lbs' in filters:
            try:
                weight_max = float(filters['weight_max_lbs'])
                if weight_max > 0:
                    normalized['weight_max_lbs'] = weight_max
            except (ValueError, TypeError):
                pass
        
        # Activity level
        valid_activity_levels = ["low", "moderate", "high"]
        if 'activity_level' in filters and filters['activity_level'] in valid_activity_levels:
            normalized['activity_level'] = filters['activity_level']
        
        # Booleans
        normalized['apartment_suitable'] = bool(filters.get('apartment_suitable', False))
        normalized['good_with_kids'] = bool(filters.get('good_with_kids', False))
        
        # Bred for
        if 'bred_for' in filters and filters['bred_for']:
            normalized['bred_for'] = str(filters['bred_for'])
        
        return normalized
    
    def _normalize_list(self, value) -> list:
        """Normalize a value to a list"""
        if value is None:
            return []
        if isinstance(value, list):
            # Filter out None/empty values
            return [str(item).strip() for item in value if item and str(item).strip()]
        if isinstance(value, str):
            # Split comma-separated strings
            return [item.strip() for item in value.split(',') if item.strip()]
        return []


if __name__ == "__main__":
    # Test the parser
    parser = LLMQueryParser()
    
    test_queries = [
        "small furry friendly dog",
        "a dog for protection",
        "apartment dog that's good with kids",
        "large energetic dog for hiking",
        "dog that won't bark at neighbors but will alert me to intruders"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        filters = parser.parse(query, verbose=True)
        print(f"\nExtracted Filters:")
        for key, value in filters.items():
            if value:  # Only show non-empty values
                print(f"  {key}: {value}")
        print()


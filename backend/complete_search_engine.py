import os
import time
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from llm_query_parser import LLMQueryParser

load_dotenv('config/.env')


class CompleteSearchEngine:
    """
    Complete dog breed search engine with:
    1. LLM query understanding
    2. Query enhancement for better embeddings
    3. Vector search with metadata filtering
    4. Cross-encoder reranking
    5. Optional post-filtering
    6. Explanations
    """
    
    def __init__(
        self,
        use_llm_parser: bool = True,
        use_reranking: bool = True,
        use_post_filtering: bool = False,
        rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        """
        Args:
            use_llm_parser: Use LLM for query parsing (recommended)
            use_reranking: Use cross-encoder reranking (recommended)
            use_post_filtering: Use explicit post-filtering (optional)
            rerank_model: Cross-encoder model name
        """
        print("🚀 Initializing Complete Dog Breed Search Engine...")
        
        # Config
        self.use_llm_parser = use_llm_parser
        self.use_reranking = use_reranking
        self.use_post_filtering = use_post_filtering
        
        # Validate required API keys
        openai_api_key = os.getenv('OPENAI_API_KEY')
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'dog-breeds')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize LLM parser
        if use_llm_parser:
            print("  🤖 Loading LLM query parser...")
            self.llm_parser = LLMQueryParser()
        
        # Initialize cross-encoder
        if use_reranking:
            print(f"  📥 Loading cross-encoder: {rerank_model}...")
            self.cross_encoder = CrossEncoder(rerank_model)
        
        print("✅ Search engine ready!\n")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 50,
        verbose: bool = False
    ) -> Dict:
        """
        Complete search pipeline
        
        Args:
            query: Natural language query
            top_k: Number of final results
            rerank_top_n: Number of candidates for reranking
            verbose: Print detailed pipeline info
            
        Returns:
            Dict with 'results' and 'metadata' (filters, timing, etc.)
        """
        if verbose:
            print("\n" + "="*80)
            print("🔍 SEARCH PIPELINE")
            print("="*80)
            print(f"Query: '{query}'")
        
        start_time = time.time()
        pipeline_metadata = {
            'query': query,
            'stages': {}
        }
        
        # ====================================================================
        # STAGE 1: LLM Query Understanding
        # ====================================================================
        stage1_start = time.time()
        
        if self.use_llm_parser:
            if verbose:
                print(f"\n[Stage 1] LLM Query Understanding")
            
            llm_filters = self.llm_parser.parse(query, verbose=verbose)
            pipeline_metadata['llm_filters'] = llm_filters
        else:
            if verbose:
                print(f"\n[Stage 1] Simple Query Parsing")
            llm_filters = self._simple_parse(query)
        
        stage1_time = time.time() - stage1_start
        pipeline_metadata['stages']['query_parsing'] = {
            'duration': stage1_time,
            'method': 'llm' if self.use_llm_parser else 'regex'
        }
        
        if verbose:
            print(f"⏱️  Stage 1: {stage1_time:.3f}s")
        
        # ====================================================================
        # STAGE 2: Build Pinecone Filter (Hard Constraints)
        # ====================================================================
        pinecone_filter = self._build_pinecone_filter(llm_filters)
        pipeline_metadata['pinecone_filter'] = pinecone_filter
        
        if verbose and pinecone_filter:
            print(f"\nPinecone hard filters: {pinecone_filter}")
        
        # ====================================================================
        # STAGE 3: Enhance Query for Better Embedding
        # ====================================================================
        enhanced_query = self._enhance_query(query, llm_filters)
        
        if verbose and enhanced_query != query:
            print(f"\nEnhanced query: '{enhanced_query}'")
        
        # ====================================================================
        # STAGE 4: Vector Search
        # ====================================================================
        stage2_start = time.time()
        
        if verbose:
            print(f"\n[Stage 2] Vector Search + Metadata Filtering")
        
        query_embedding = self._generate_embedding(enhanced_query)
        
        # Build query params - only include filter if it's not empty
        query_params = {
            'vector': query_embedding,
            'top_k': rerank_top_n,
            'include_metadata': True
        }
        
        # Only add filter if it has conditions (empty dict means no filtering)
        if pinecone_filter:
            query_params['filter'] = pinecone_filter
        
        candidates = self.index.query(**query_params)
        
        stage2_time = time.time() - stage2_start
        pipeline_metadata['stages']['vector_search'] = {
            'duration': stage2_time,
            'candidates_retrieved': len(candidates['matches'])
        }
        
        if verbose:
            print(f"Retrieved: {len(candidates['matches'])} candidates")
            print(f"⏱️  Stage 2: {stage2_time:.3f}s")
        
        if not candidates['matches']:
            if verbose:
                print("\n⚠️  No results found")
            return {
                'results': [],
                'metadata': pipeline_metadata
            }
        
        # ====================================================================
        # STAGE 5: Cross-Encoder Reranking
        # ====================================================================
        if self.use_reranking:
            stage3_start = time.time()
            
            if verbose:
                print(f"\n[Stage 3] Cross-Encoder Reranking")
            
            reranked = self._rerank_with_cross_encoder(
                query,  # Use original query, not enhanced
                candidates['matches'],  # ALL candidates from semantic search (rerank_top_n = 72/100)
                top_k,  # Return top_k results after reranking all candidates
                verbose=verbose
            )
            
            stage3_time = time.time() - stage3_start
            pipeline_metadata['stages']['reranking'] = {
                'duration': stage3_time,
                'method': 'cross_encoder'
            }
            
            if verbose:
                print(f"⏱️  Stage 3: {stage3_time:.3f}s")
        else:
            reranked = candidates['matches'][:top_k * 2]
        
        # ====================================================================
        # STAGE 6: Optional Post-Filtering
        # ====================================================================
        if self.use_post_filtering:
            stage4_start = time.time()
            
            if verbose:
                print(f"\n[Stage 4] Post-Filtering with LLM Filters")
            
            filtered = self._post_filter(reranked, llm_filters, verbose=verbose)
            
            stage4_time = time.time() - stage4_start
            pipeline_metadata['stages']['post_filtering'] = {
                'duration': stage4_time,
                'before': len(reranked),
                'after': len(filtered)
            }
            
            if verbose:
                print(f"Filtered: {len(reranked)} → {len(filtered)} results")
                print(f"⏱️  Stage 4: {stage4_time:.3f}s")
        else:
            filtered = reranked
        
        # ====================================================================
        # STAGE 7: Format Results
        # ====================================================================
        final_results = self._format_results(
            {'matches': filtered[:top_k]},
            llm_filters
        )
        
        total_time = time.time() - start_time
        pipeline_metadata['total_duration'] = total_time
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ Total time: {total_time:.3f}s")
            print(f"📊 Returned: {len(final_results)} results")
            print(f"{'='*80}\n")
        
        return {
            'results': final_results,
            'metadata': pipeline_metadata
        }
    
    def _simple_parse(self, query: str) -> Dict:
        """Simple regex-based parser (fallback)"""
        import re
        
        filters = {}
        query_lower = query.lower()
        
        # Weight
        weight_match = re.search(r'under (\d+)\s*(lb|lbs|pound)', query_lower)
        if weight_match:
            filters['weight_max_lbs'] = int(weight_match.group(1))
        
        # Size
        if 'tiny' in query_lower:
            filters['size'] = 'tiny'
        elif 'small' in query_lower:
            filters['size'] = 'small'
        elif 'medium' in query_lower:
            filters['size'] = 'medium'
        elif 'large' in query_lower or 'big' in query_lower:
            filters['size'] = 'large'
        elif 'giant' in query_lower:
            filters['size'] = 'giant'
        
        return filters
    
    def _build_pinecone_filter(self, llm_filters: Dict) -> Dict:
        """
        Convert LLM filters to Pinecone metadata filters.
        Only uses fields that exist in Pinecone metadata (size, weight).
        Intent-based filters (apartment_suitable, good_with_kids) are handled
        via semantic search and query enhancement instead of hard filtering.
        """
        pinecone_filter = {}
        
        # Size - this field exists in metadata
        if 'size' in llm_filters:
            pinecone_filter['size_categories'] = {'$in': [llm_filters['size']]}
        
        # Weight - these fields exist in metadata
        if 'weight_max_lbs' in llm_filters:
            pinecone_filter['weight_max_lbs'] = {'$lte': llm_filters['weight_max_lbs']}
        
        if 'weight_min_lbs' in llm_filters:
            pinecone_filter['weight_min_lbs'] = {'$gte': llm_filters['weight_min_lbs']}
        
        # Note: apartment_suitable and good_with_kids are NOT stored in metadata
        # These are handled via semantic search and query enhancement instead
        # This prevents filtering out all results when these fields don't exist
        
        return pinecone_filter
    
    def _enhance_query(self, original_query: str, llm_filters: Dict) -> str:
        """Enhance query with LLM-extracted requirements for better embedding"""
        
        if not llm_filters:
            return original_query
        
        enhanced_parts = [original_query]
        
        # Add required temperament traits
        if 'temperament_required' in llm_filters:
            enhanced_parts.extend(llm_filters['temperament_required'])
        
        # Add special requirements
        if 'special_requirements' in llm_filters:
            # Limit to avoid too long query
            enhanced_parts.extend(llm_filters['special_requirements'][:3])
        
        # Add activity level
        if 'activity_level' in llm_filters:
            enhanced_parts.append(f"{llm_filters['activity_level']} energy")
        
        # Add weight context for better semantic matching
        if 'weight_max_lbs' in llm_filters:
            max_weight = llm_filters['weight_max_lbs']
            if max_weight <= 25:
                enhanced_parts.append("small lightweight")
            elif max_weight <= 50:
                enhanced_parts.append("medium weight")
        
        if 'weight_min_lbs' in llm_filters:
            min_weight = llm_filters['weight_min_lbs']
            if min_weight >= 50:
                enhanced_parts.append("large heavy")
        
        # Add apartment-friendly context (helps semantic search)
        if llm_filters.get('apartment_suitable'):
            enhanced_parts.append("apartment suitable small space")
            # If no size specified, infer small for apartment dogs
            if 'size' not in llm_filters:
                enhanced_parts.append("small")
        
        # Add family-friendly context
        if llm_filters.get('good_with_kids'):
            enhanced_parts.append("good with kids family friendly")
        
        # Add bred_for context if specified
        if 'bred_for' in llm_filters and llm_filters['bred_for']:
            enhanced_parts.append(llm_filters['bred_for'])
        
        return " ".join(enhanced_parts)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        verbose: bool = False
    ) -> List[Dict]:
        """Rerank using cross-encoder"""
        
        if not candidates:
            return []
        
        # Prepare pairs
        pairs = []
        for match in candidates:
            # Use structured_text for cross-encoder reranking (stored in metadata)
            doc_text = match['metadata'].get('structured_text', '')
            if not doc_text:
                # Fallback: construct structured text from metadata
                doc_text = self._construct_text_for_reranking(match['metadata'])
            pairs.append([query, doc_text])
        
        # Score with cross-encoder
        cross_encoder_scores = self.cross_encoder.predict(pairs)
        
        # Add scores
        for i, match in enumerate(candidates):
            match['cross_encoder_score'] = float(cross_encoder_scores[i])
            match['bi_encoder_score'] = match['score']
        
        # Sort by cross-encoder score
        reranked = sorted(
            candidates,
            key=lambda x: x['cross_encoder_score'],
            reverse=True
        )
        
        if verbose:
            print(f"Top 3 after reranking:")
            for i, match in enumerate(reranked[:3], 1):
                name = match['metadata']['name']
                ce_score = match['cross_encoder_score']
                bi_score = match['bi_encoder_score']
                print(f"  {i}. {name} (Cross: {ce_score:.3f}, Bi: {bi_score:.3f})")
        
        return reranked[:top_k]
    
    def _post_filter(
        self,
        results: List[Dict],
        llm_filters: Dict,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Explicit post-filtering using LLM filters
        Safety net to ensure temperament_avoid is enforced
        """
        
        if not llm_filters:
            return results
        
        filtered = []
        removed = []
        
        for match in results:
            metadata = match['metadata']
            temperament = metadata.get('temperament', '').lower()
            breed_group = metadata.get('breed_group', '').lower()
            
            should_include = True
            
            # Check temperament_avoid
            if 'temperament_avoid' in llm_filters:
                for avoid_trait in llm_filters['temperament_avoid']:
                    if avoid_trait.lower() in temperament:
                        should_include = False
                        removed.append((metadata['name'], f"has '{avoid_trait}'"))
                        break
                
                # Also check breed group (e.g., hounds are vocal)
                if 'vocal' in llm_filters['temperament_avoid'] or 'barky' in llm_filters['temperament_avoid']:
                    if breed_group == 'hound':
                        should_include = False
                        removed.append((metadata['name'], "breed group: hound (typically vocal)"))
            
            if should_include:
                filtered.append(match)
        
        if verbose and removed:
            print(f"  Removed by post-filtering:")
            for name, reason in removed[:5]:
                print(f"    - {name}: {reason}")
        
        return filtered
    
    def _construct_text_for_reranking(self, metadata: Dict) -> str:
        """Construct text from metadata"""
        parts = []
        
        parts.append(f"Breed: {metadata['name']}")
        
        if metadata.get('weight_max_lbs', 0) > 0:
            parts.append(
                f"Weight: {metadata['weight_min_lbs']:.0f}-{metadata['weight_max_lbs']:.0f} pounds"
            )
        
        parts.append(f"Size: {metadata.get('primary_size', 'unknown')}")
        
        if metadata.get('breed_group'):
            parts.append(f"Breed group: {metadata['breed_group']}")
        
        if metadata.get('bred_for'):
            parts.append(f"Bred for: {metadata['bred_for']}")
        
        if metadata.get('temperament'):
            parts.append(f"Temperament: {metadata['temperament']}")
        
        return ". ".join(parts) + "."
    
    def _find_category_boundaries(self, sorted_scores: List[float]) -> tuple:
        """
        Find category boundaries based on drastic score drops
        
        Returns:
            (excellent_threshold, good_threshold) - positions where categories change
        """
        if len(sorted_scores) < 3:
            # Too few results, use simple split
            return (1, len(sorted_scores))
        
        # Calculate score differences between consecutive positions
        score_drops = []
        for i in range(len(sorted_scores) - 1):
            drop = sorted_scores[i] - sorted_scores[i + 1]
            score_drops.append((i, drop))
        
        # Calculate average and standard deviation of drops
        drops_only = [drop for _, drop in score_drops]
        if not drops_only:
            return (max(1, len(sorted_scores) // 10), len(sorted_scores) // 2)
        
        avg_drop = sum(drops_only) / len(drops_only)
        variance = sum((d - avg_drop) ** 2 for d in drops_only) / len(drops_only)
        std_drop = variance ** 0.5
        
        # Find drastic drops (significantly larger than average)
        # A drop is "drastic" if it's > avg + 1.5 * std (or > 2x average if std is small)
        drastic_threshold = max(avg_drop * 1.5, avg_drop + 1.5 * std_drop)
        
        # Find positions with drastic drops
        drastic_drops = [(pos, drop) for pos, drop in score_drops if drop >= drastic_threshold]
        
        # Determine boundaries
        excellent_threshold = len(sorted_scores)
        good_threshold = len(sorted_scores)
        
        if drastic_drops:
            # Sort by position (ascending)
            drastic_drops.sort(key=lambda x: x[0])
            
            # First drastic drop: Excellent → Good boundary
            # Use position AFTER the drop (so items at drop position are still excellent)
            first_drop_pos = drastic_drops[0][0]
            excellent_threshold = first_drop_pos + 1
            
            # Second drastic drop (if exists): Good → Fair boundary
            if len(drastic_drops) > 1:
                second_drop_pos = drastic_drops[1][0]
                good_threshold = second_drop_pos + 1
            else:
                # If only one drastic drop, split remaining into good and fair
                # Use ~60% of remaining for good
                remaining = len(sorted_scores) - excellent_threshold
                good_threshold = excellent_threshold + max(remaining * 2 // 3, remaining // 2)
        else:
            # No drastic drops found, use fallback distribution
            excellent_threshold = max(2, len(sorted_scores) // 10)  # Top 10%
            good_threshold = excellent_threshold + len(sorted_scores) // 2  # Next 50%
        
        # Ensure thresholds are within bounds
        excellent_threshold = min(excellent_threshold, len(sorted_scores))
        good_threshold = min(good_threshold, len(sorted_scores))
        
        # Ensure at least 1 excellent and reasonable distribution
        if excellent_threshold < 1:
            excellent_threshold = 1
        if good_threshold <= excellent_threshold:
            good_threshold = excellent_threshold + max(1, (len(sorted_scores) - excellent_threshold) // 2)
        
        return (excellent_threshold, good_threshold)
    
    def _get_match_category(self, score: float, all_scores: List[float], excellent_threshold: int, good_threshold: int) -> Dict[str, any]:
        """
        Categorize match quality based on position relative to score drop boundaries
        
        Returns:
            Dict with 'category', 'label', 'color', and 'description'
        """
        if not all_scores:
            return {
                'category': 'unknown',
                'label': 'Unknown',
                'color': '#6E6E6E',
                'description': 'Match quality unknown'
            }
        
        sorted_scores = sorted(all_scores, reverse=True)
        
        # Find position of this score (handle duplicates by using average position)
        positions = [i for i, s in enumerate(sorted_scores) if s == score]
        if positions:
            # Use average position (0-indexed)
            avg_position = sum(positions) / len(positions)
        else:
            avg_position = len(sorted_scores) / 2  # Default if score not found
        
        # Categorize based on position relative to boundaries
        if avg_position < excellent_threshold:
            # Top results before first drastic drop - Excellent matches
            return {
                'category': 'excellent',
                'label': 'Excellent Match',
                'color': '#4ADE80',  # Green
                'description': 'Highly relevant to your search',
                'icon': '⭐'
            }
        elif avg_position < good_threshold:
            # Between first and second drop (or first drop to ~60% mark) - Good matches
            return {
                'category': 'good',
                'label': 'Good Match',
                'color': '#60A5FA',  # Blue
                'description': 'Good match for your needs',
                'icon': '✓'
            }
        else:
            # After second drop (or after ~60% mark) - Fair matches
            return {
                'category': 'fair',
                'label': 'Fair Match',
                'color': '#FBBF24',  # Amber
                'description': 'Relevant but may not be ideal',
                'icon': '○'
            }
    
    def _format_results(self, results, llm_filters: Dict) -> List[Dict]:
        """Format results with explanations"""
        formatted = []
        
        # Extract all scores for finding category boundaries based on score drops
        all_scores = []
        for match in results['matches']:
            score = match.get('cross_encoder_score', match.get('score', 0))
            all_scores.append(score)
        
        # Find category boundaries based on drastic score drops
        sorted_scores = sorted(all_scores, reverse=True)
        excellent_threshold, good_threshold = self._find_category_boundaries(sorted_scores)
        
        for match in results['matches']:
            metadata = match['metadata']
            
            # Identify matching traits
            matching_traits = []
            if 'temperament_required' in llm_filters:
                temp_lower = metadata.get('temperament', '').lower()
                for trait in llm_filters['temperament_required']:
                    if trait.lower() in temp_lower:
                        matching_traits.append(trait)
            
            # Get match category based on score drops
            score = match.get('cross_encoder_score', match.get('score', 0))
            match_category = self._get_match_category(score, all_scores, excellent_threshold, good_threshold)
            
            result = {
                'name': metadata['name'],
                'score': round(match.get('cross_encoder_score', match['score']), 3),
                'bi_encoder_score': round(match.get('bi_encoder_score', match['score']), 3),
                'cross_encoder_score': round(match.get('cross_encoder_score', 0), 3) if 'cross_encoder_score' in match else None,
                'match_category': match_category,
                'size': metadata['primary_size'],
                'weight': f"{metadata['weight_min_lbs']:.0f}-{metadata['weight_max_lbs']:.0f} lbs",
                'height': f"{metadata['height_min_inches']:.0f}-{metadata['height_max_inches']:.0f} in",
                'lifespan': metadata.get('life_span', ''),
                'breed_group': metadata.get('breed_group', ''),
                'bred_for': metadata.get('bred_for', ''),
                'temperament': metadata.get('temperament', ''),
                'matching_traits': matching_traits,
                'image_url': metadata.get('image_url', ''),
                'metadata': metadata
            }
            
            formatted.append(result)
        
        return formatted


# ============================================================================
# HELPER: Display Results with Explanations
# ============================================================================

def display_results(response: Dict, show_filters: bool = True):
    """
    Display search results with explanations
    
    Args:
        response: Dict with 'results' and 'metadata' from search()
        show_filters: Whether to show extracted filters
    """
    results = response['results']
    metadata = response['metadata']
    
    print("\n" + "="*80)
    print(f"🔍 SEARCH RESULTS")
    print("="*80)
    print(f"Query: '{metadata['query']}'")
    
    # Show extracted filters
    if show_filters and 'llm_filters' in metadata and metadata['llm_filters']:
        print(f"\n📋 Understood Requirements:")
        llm_filters = metadata['llm_filters']
        
        if 'temperament_required' in llm_filters:
            print(f"  ✅ Must have: {', '.join(llm_filters['temperament_required'])}")
        
        if 'temperament_avoid' in llm_filters:
            print(f"  ❌ Must avoid: {', '.join(llm_filters['temperament_avoid'])}")
        
        if 'size' in llm_filters:
            print(f"  📏 Size: {llm_filters['size']}")
        
        if 'activity_level' in llm_filters:
            print(f"  ⚡ Activity: {llm_filters['activity_level']}")
        
        if 'apartment_suitable' in llm_filters:
            print(f"  🏢 Apartment suitable: Yes")
        
        if 'good_with_kids' in llm_filters:
            print(f"  👶 Good with kids: Yes")
        
        if 'special_requirements' in llm_filters:
            for req in llm_filters['special_requirements'][:2]:
                print(f"  ℹ️  {req}")
    
    # Show timing
    if 'total_duration' in metadata:
        print(f"\n⏱️  Search completed in {metadata['total_duration']:.3f}s")
    
    # Show results
    print(f"\n{'='*80}")
    print(f"Found {len(results)} matches:")
    print("="*80)
    
    for i, breed in enumerate(results, 1):
        print(f"\n{i}. {breed['name']}")
        print(f"   Relevance: {breed['score']:.3f}", end="")
        
        if breed['cross_encoder_score']:
            print(f" (Bi: {breed['bi_encoder_score']:.3f}, Cross: {breed['cross_encoder_score']:.3f})")
        else:
            print()
        
        print(f"   Size: {breed['size']} | Weight: {breed['weight']} | Height: {breed['height']}")
        
        if breed['breed_group']:
            print(f"   Group: {breed['breed_group']}", end="")
            if breed['bred_for']:
                print(f" | Bred for: {breed['bred_for']}")
            else:
                print()
        
        if breed['lifespan']:
            print(f"   Lifespan: {breed['lifespan']}")
        
        if breed['temperament']:
            temp = breed['temperament']
            if len(temp) > 80:
                temp = temp[:77] + "..."
            print(f"   Temperament: {temp}")
        
        # Highlight matching traits
        if breed['matching_traits']:
            print(f"   ✅ Matches your needs: {', '.join(breed['matching_traits'])}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test the complete engine
    engine = CompleteSearchEngine(
        use_llm_parser=True,
        use_reranking=True,
        use_post_filtering=True
    )
    
    test_queries = [
        "dog that won't bark at neighbors but will alert me to intruders",
        "small apartment dog for first-time owner",
        "energetic hiking companion good with kids",
        "dog for elderly person with limited mobility",
        "I work long hours, need independent dog under 25 lbs"
    ]
    
    for query in test_queries:
        response = engine.search(query, top_k=5, verbose=True)
        display_results(response)
        print("\n" + "="*80 + "\n")
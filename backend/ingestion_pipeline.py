import os
import re
import json
import time
import requests
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load environment variables
load_dotenv('config/.env')


class DogBreedIngestionPipeline:
    """
    Simplified dog breed ingestion pipeline with only essential features:
    1. Fetch data from API
    2. Clean data
    3. Map size categories
    4. Generate embedding text
    5. Create embeddings
    6. Index to Pinecone
    """
    
    def __init__(self):
        # Configuration
        self.dogs_api_url = os.getenv('DOGS_API_URL')
        self.dogs_api_key = os.getenv('DOGS_API_KEY', '')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'dog-breeds')
        
        # Validate required API keys
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.dogs_api_url:
            raise ValueError("DOGS_API_URL environment variable is required")
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Size categories (only hard filter we need)
        self.size_categories = {
            "tiny": {"max": 10},
            "small": {"max": 25},
            "medium": {"min": 25, "max": 60},
            "large": {"min": 60, "max": 100},
            "giant": {"min": 100}
        }
    
    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "="*70)
        print("🐕 DOG BREED INGESTION PIPELINE")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        try:
            # Stage 1: Fetch data
            print("📥 [1/6] Fetching data from API...")
            raw_data = self._fetch_data()
            print(f"    ✅ Fetched {len(raw_data)} breeds\n")
            
            # Stage 2: Clean data
            print("🧹 [2/6] Cleaning data...")
            cleaned_data = self._clean_data(raw_data)
            print(f"    ✅ Cleaned {len(cleaned_data)} breeds\n")
            
            # Stage 3: Map size categories
            print("🗺️  [3/6] Mapping size categories...")
            enriched_data = self._map_size_categories(cleaned_data)
            print(f"    ✅ Mapped sizes for {len(enriched_data)} breeds\n")
            
            # Stage 4: Generate text for embedding
            print("📝 [4/6] Generating embedding text...")
            # Generate conversational text for embeddings
            texts = self._generate_texts(enriched_data)
            # Generate structured text for cross-encoder reranking
            structured_texts = self._generate_structured_texts(enriched_data)
            
            # Validate text lengths match
            if len(texts) != len(structured_texts):
                raise ValueError(
                    f"Mismatch: Generated {len(texts)} conversational texts "
                    f"but {len(structured_texts)} structured texts. "
                    f"Text formats must match in length."
                )
            
            if len(texts) != len(enriched_data):
                raise ValueError(
                    f"Mismatch: Generated {len(texts)} texts "
                    f"but have {len(enriched_data)} breeds. "
                    f"Texts and data must match in length."
                )
            
            # Add both text formats to each breed entry
            for breed, rich_text, structured_text in zip(enriched_data, texts, structured_texts):
                breed['rich_text'] = rich_text
                breed['structured_text'] = structured_text
            
            print(f"    ✅ Generated text for {len(texts)} breeds")
            
            # Safe average calculation (avoid division by zero)
            if texts:
                avg_conversational = sum(len(t) for t in texts) / len(texts)
                print(f"    📏 Avg length (conversational): {avg_conversational:.0f} chars")
            else:
                print(f"    ⚠️  No conversational texts generated")
            
            if structured_texts:
                avg_structured = sum(len(t) for t in structured_texts) / len(structured_texts)
                print(f"    📏 Avg length (structured): {avg_structured:.0f} chars")
            else:
                print(f"    ⚠️  No structured texts generated")
            print()
            
            # Stage 5: Generate embeddings
            print("🧠 [5/6] Generating embeddings...")
            embeddings = self._generate_embeddings(texts)
            
            # Validate embeddings match data length
            if len(embeddings) != len(enriched_data):
                raise ValueError(
                    f"Mismatch: Generated {len(embeddings)} embeddings "
                    f"but have {len(enriched_data)} breeds. "
                    f"Embeddings and data must match in length."
                )
            
            if len(texts) != len(embeddings):
                raise ValueError(
                    f"Mismatch: Generated {len(embeddings)} embeddings "
                    f"but have {len(texts)} texts. "
                    f"Texts and embeddings must match in length."
                )
            
            print(f"    ✅ Generated {len(embeddings)} embeddings\n")
            
            # Stage 6: Index to Pinecone
            print("📊 [6/6] Indexing to Pinecone...")
            self._index_to_pinecone(enriched_data, embeddings)
            print(f"    ✅ Indexed {len(enriched_data)} breeds\n")
            
            # Summary
            duration = time.time() - start_time
            print("="*70)
            print("✨ PIPELINE COMPLETED SUCCESSFULLY ✨")
            print(f"⏱️  Total time: {duration:.2f}s")
            print(f"📊 Breeds indexed: {len(enriched_data)}")
            print("="*70 + "\n")
            
            # Save data
            self._save_data(enriched_data)
            
            return enriched_data
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
    
    def _fetch_data(self) -> List[Dict]:
        """Fetch data from Dog API"""
        headers = {'x-api-key': self.dogs_api_key} if self.dogs_api_key else {}
        
        response = requests.get(self.dogs_api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def _clean_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Clean and normalize data"""
        cleaned = []
        
        for breed in raw_data:
            # Skip if missing essentials
            if not breed.get('id') or not breed.get('name'):
                continue
            
            # Clean weight
            weight = self._parse_range(breed.get('weight', {}).get('imperial', ''))
            
            # Clean height
            height = self._parse_range(breed.get('height', {}).get('imperial', ''))
            
            # Clean lifespan
            lifespan = self._parse_range(breed.get('life_span', ''))
            
            cleaned.append({
                'id': breed['id'],
                'name': breed['name'],
                'breed_group': breed.get('breed_group', 'Unknown'),
                'bred_for': breed.get('bred_for', 'Unknown'),
                'temperament': breed.get('temperament', ''),
                'life_span': breed.get('life_span', ''),
                'weight_min_lbs': weight[0],
                'weight_max_lbs': weight[1],
                'height_min_inches': height[0],
                'height_max_inches': height[1],
                'lifespan_min_years': lifespan[0],
                'lifespan_max_years': lifespan[1],
                'image_url': breed.get('image', {}).get('url', '')
            })
        
        return cleaned
    
    def _parse_range(self, text: str) -> tuple:
        """Parse numeric range from text"""
        # Clean up
        text = re.sub(r'up\s*-\s*', '0 - ', str(text))
        text = re.sub(r'NaN', '0', text)
        
        # Extract numbers
        numbers = re.findall(r'\d+\.?\d*', text)
        
        if len(numbers) >= 2:
            return (float(numbers[0]), float(numbers[1]))
        elif len(numbers) == 1:
            val = float(numbers[0])
            return (val, val)
        else:
            return (0.0, 0.0)
    
    def _map_size_categories(self, data: List[Dict]) -> List[Dict]:
        """Map size categories based on weight"""
        for breed in data:
            weight_min = breed['weight_min_lbs']
            weight_max = breed['weight_max_lbs']
            
            # Determine size categories
            sizes = []
            for size, bounds in self.size_categories.items():
                min_bound = bounds.get("min", 0)
                max_bound = bounds.get("max", 999)
                
                # Check overlap
                if weight_min <= max_bound and weight_max >= min_bound:
                    sizes.append(size)
            
            breed['size_categories'] = sizes if sizes else ['unknown']
            breed['primary_size'] = sizes[0] if sizes else 'unknown'
        
        return data
    
    def _generate_texts(self, data: List[Dict]) -> List[str]:
        """Generate rich conversational text for embedding"""
        texts = []
        
        for breed in data:
            # Start with breed name
            name = breed['name']
            description_parts = [f"The name of the breed is {name}"]
            
            # Physical characteristics - conversational format
            if breed['weight_max_lbs'] > 0:
                if breed['weight_min_lbs'] == breed['weight_max_lbs']:
                    weight_info = f"{breed['weight_min_lbs']:.0f} pounds"
                else:
                    weight_info = f"{breed['weight_min_lbs']:.0f}-{breed['weight_max_lbs']:.0f} pounds"
                description_parts.append(f"its weight is {weight_info}")
            
            if breed['height_max_inches'] > 0:
                if breed['height_min_inches'] == breed['height_max_inches']:
                    height_info = f"{breed['height_min_inches']:.0f} inches"
                else:
                    height_info = f"{breed['height_min_inches']:.0f}-{breed['height_max_inches']:.0f} inches"
                description_parts.append(f"its height is {height_info}")
            
            size = breed['primary_size']
            description_parts.append(f"its size is {size}")
            
            # Lifespan
            if breed['lifespan_max_years'] > 0:
                if breed['lifespan_min_years'] == breed['lifespan_max_years']:
                    lifespan = f"{breed['lifespan_min_years']:.0f} years"
                else:
                    lifespan = f"{breed['lifespan_min_years']:.0f}-{breed['lifespan_max_years']:.0f} years"
                description_parts.append(f"with a lifespan of {lifespan}")
            
            # Breed group
            if breed['breed_group'] and breed['breed_group'] != 'Unknown':
                description_parts.append(f"and the breed group is {breed['breed_group']}")
            
            # Bred for
            if breed['bred_for'] and breed['bred_for'] != 'Unknown':
                description_parts.append(f"it was bred for {breed['bred_for'].lower()}")
            
            # Temperament
            if breed['temperament']:
                description_parts.append(f"its temperament is {breed['temperament']}")
            
            # Join all parts into natural flowing text
            text = ", ".join(description_parts) + "."
            texts.append(text)
        
        return texts
    
    def _generate_structured_texts(self, data: List[Dict]) -> List[str]:
        """Generate structured text for cross-encoder reranking"""
        texts = []
        
        for breed in data:
            parts = []
            
            # Breed name
            parts.append(f"Breed: {breed['name']}")
            
            # Physical characteristics
            if breed['weight_max_lbs'] > 0:
                if breed['weight_min_lbs'] == breed['weight_max_lbs']:
                    weight_info = f"{breed['weight_min_lbs']:.0f} pounds"
                else:
                    weight_info = f"{breed['weight_min_lbs']:.0f}-{breed['weight_max_lbs']:.0f} pounds"
                parts.append(f"Weight: {weight_info}")
            
            if breed['height_max_inches'] > 0:
                if breed['height_min_inches'] == breed['height_max_inches']:
                    height_info = f"{breed['height_min_inches']:.0f} inches"
                else:
                    height_info = f"{breed['height_min_inches']:.0f}-{breed['height_max_inches']:.0f} inches"
                parts.append(f"Height: {height_info}")
            
            parts.append(f"Size: {breed['primary_size']}")
            
            # Lifespan
            if breed['lifespan_max_years'] > 0:
                if breed['lifespan_min_years'] == breed['lifespan_max_years']:
                    lifespan = f"{breed['lifespan_min_years']:.0f} years"
                else:
                    lifespan = f"{breed['lifespan_min_years']:.0f}-{breed['lifespan_max_years']:.0f} years"
                parts.append(f"Lifespan: {lifespan}")
            
            # Breed group
            if breed['breed_group'] and breed['breed_group'] != 'Unknown':
                parts.append(f"Breed group: {breed['breed_group']}")
            
            # Bred for
            if breed['bred_for'] and breed['bred_for'] != 'Unknown':
                parts.append(f"Bred for: {breed['bred_for']}")
            
            # Temperament
            if breed['temperament']:
                parts.append(f"Temperament: {breed['temperament']}")
            
            # Join with periods for structured format
            text = ". ".join(parts) + "."
            texts.append(text)
        
        return texts
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        embeddings = []
        batch_size = 100
        
        for i in tqdm(range(0, len(texts), batch_size), desc="    Embedding"):
            batch = texts[i:i + batch_size]
            
            response = self.openai_client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            time.sleep(0.1)  # Rate limiting
        
        return embeddings
    
    def _index_to_pinecone(self, data: List[Dict], embeddings: List[List[float]]):
        """Index vectors to Pinecone"""
        # Create index if needed
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"    Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)  # Wait for index to be ready
        
        index = self.pc.Index(self.index_name)
        
        # Prepare vectors
        vectors = []
        for breed, embedding in zip(data, embeddings):
            vectors.append({
                'id': f"breed_{breed['id']}",
                'values': embedding,
                'metadata': {
                    # Basic info
                    'name': breed['name'],
                    'breed_group': breed['breed_group'],
                    'bred_for': breed['bred_for'],
                    'temperament': breed['temperament'][:500],  # Limit size
                    'life_span': breed['life_span'],
                    
                    # Numeric (filterable)
                    'weight_min_lbs': breed['weight_min_lbs'],
                    'weight_max_lbs': breed['weight_max_lbs'],
                    'height_min_inches': breed['height_min_inches'],
                    'height_max_inches': breed['height_max_inches'],
                    'lifespan_min_years': breed['lifespan_min_years'],
                    'lifespan_max_years': breed['lifespan_max_years'],
                    
                    # Size categories (our only hard filter)
                    'size_categories': breed['size_categories'],
                    'primary_size': breed['primary_size'],
                    
                    # Media
                    'image_url': breed['image_url'],
                    
                    # Text formats for reranking
                    'structured_text': breed.get('structured_text', '')[:5000]  # Limit size for metadata
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in tqdm(range(0, len(vectors), batch_size), desc="    Upserting"):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        # Get stats
        stats = index.describe_index_stats()
        print(f"    📊 Index now has {stats['total_vector_count']} vectors")
    
    def _save_data(self, data: List[Dict]):
        """Save enriched data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'data/enriched_breeds_{timestamp}.json'
        
        os.makedirs('data', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved data to: {filepath}")


if __name__ == "__main__":
    pipeline = DogBreedIngestionPipeline()
    pipeline.run()
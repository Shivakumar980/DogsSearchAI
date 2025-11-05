"""
Dog Breed Search - Ingestion Pipeline
Run this to index dog breeds into Pinecone
"""

from ingestion_pipeline import DogBreedIngestionPipeline

if __name__ == "__main__":
    pipeline = DogBreedIngestionPipeline()
    breeds = pipeline.run()
    
    print(f"\n✅ Successfully indexed {len(breeds)} dog breeds!")
    print("\nYou can now build your search/query system.")
"""
Complete Interactive Dog Breed Search
With LLM query understanding + Cross-encoder reranking
"""

from complete_search_engine import CompleteSearchEngine, display_results


def main():
    print("\n" + "="*80)
    print("🐕 COMPLETE DOG BREED SEARCH ENGINE")
    print("="*80)
    print("\nPowered by:")
    print("  • LLM Query Understanding (GPT-4o-mini)")
    print("  • Semantic Vector Search (OpenAI + Pinecone)")
    print("  • Cross-Encoder Reranking (MS-MARCO)")
    print("\n" + "="*80)
    print("\nExample Queries:")
    print("  • 'dog that won't bark at neighbors'")
    print("  • 'small apartment dog for first-time owner'")
    print("  • 'energetic hiking companion good with kids'")
    print("  • 'dog for elderly person with limited mobility'")
    print("  • 'I work long hours, need independent dog'")
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'verbose' to toggle detailed pipeline info")
    print("="*80 + "\n")
    
    # Initialize engine
    engine = CompleteSearchEngine(
        use_llm_parser=True,
        use_reranking=True,
        use_post_filtering=True  # Extra safety
    )
    
    verbose = False
    
    while True:
        try:
            query = input("🔍 Search: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"\n{'✅' if verbose else '❌'} Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue
            
            # Search
            response = engine.search(query, top_k=10, verbose=verbose)
            
            # Display
            display_results(response, show_filters=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


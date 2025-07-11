from scripts.query_agent import query_movie_agent

def main():
    print("\n Welcome to the AI Movie Plot Agent ")
    print("Ask a movie-related question, or type 'exit' to quit.\n")

    while True:
        query = input("Your Question: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        results = query_movie_agent(query, top_k=3)
        print("\n Top Matching Results:\n")
        for i, (title, plot) in enumerate(results, 1):
            print(f"{i}. {title}\n{plot[:400]}...\n") 

if __name__ == "__main__":
    main()

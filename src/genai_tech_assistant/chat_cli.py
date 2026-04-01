from genai_tech_assistant.RAG.qa_pipeline import answer_question


def main() -> None:
    print("GenAI Tech Assistant CLI")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        result = answer_question(question, top_k=5)

        print("\nAssistant:")
        print(result.answer)
        print()

        if result.retrieved:
            print("Retrieved context:")
            for i, chunk in enumerate(result.retrieved, start=1):
                meta = chunk.metadata or {}
                source = meta.get("source_file", "unknown")
                page = meta.get("page_number", "unknown")
                snippet = chunk.text[:250].replace("\n", " ")
                print(
                    f"[{i}] distance={chunk.distance:.4f} \n file={source} \n page={page}")
                print(f"    {snippet} ...\n")
                print()
                print()

        print("-" * 80)


if __name__ == "__main__":
    main()

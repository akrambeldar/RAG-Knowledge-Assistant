from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from rag_chain import rag_chain, retriever

load_dotenv()

questions = [
    "What is the main topic of the documents?",
    "Summarise the key points covered.",
]

print("Running evaluation queries...")
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for q in questions:
    docs = retriever.invoke(q)
    answer = rag_chain.invoke(q)
    eval_data["question"].append(q)
    eval_data["answer"].append(answer)
    eval_data["contexts"].append([d.page_content for d in docs])
    eval_data["ground_truth"].append("")

dataset = Dataset.from_dict(eval_data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

print("\n--- RAGAs Evaluation Results ---")
print(result)

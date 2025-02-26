from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from retrieval import query_database
import torch
import gc
import atexit
import nltk
from nltk.corpus import wordnet

# Load tokenizer and model with optimized settings for CPU
MODEL_NAME = "facebook/opt-350m"  # Smaller model (~2GB)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load model in fp16 to reduce memory usage, but ensure it runs on CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Change FP16 to FP32 for CPU
    low_cpu_mem_usage=True,
    device_map="cpu"
)
device = torch.device("cpu")
app = Flask(__name__)


# Initialize NLP pipeline for paraphrasing (can be improved with a better model)
paraphrase_pipeline = pipeline("text2text-generation", model="t5-small")

def expand_query_with_lsca(question):
    nltk.download("wordnet")
    words = question.split()
    expanded_terms = []

    for word in words:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        expanded_terms.extend(synonyms)

    # Paraphrase the question using a small model (alternative: use OpenAI API)
    paraphrased_question = paraphrase_pipeline(question, max_length=50, num_return_sequences=1)[0]["generated_text"]

    # Combine original, synonyms, and paraphrased question
    expanded_query = list(set(words + expanded_terms))  # Remove duplicates
    return " ".join(expanded_query) + " " + paraphrased_question


# Cleanup function to release resources
def cleanup():
    global model, tokenizer
    del model, tokenizer
    gc.collect()

atexit.register(cleanup)

@app.route("/ask", methods=["POST"])
def ask():
    """Handles chatbot queries with database context."""

    if not request.is_json:
        return jsonify({"error": "Invalid request format. Expected JSON."}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in request body."}), 400

    question = data["question"]
    # Expand query using LSCA
    expanded_query = question

    # Retrieve relevant documents using the expanded query
    retrieved_docs = query_database(question)

    if not retrieved_docs:  # Ensure we have context
        context = "No relevant documents found."
    else:
        context = "\n".join(retrieved_docs)  # Format context properly

    # **Improved Prompt to Force Context Use**
    # Limit context length (optional: adjust number of sentences)
    relevant_sentences = context.split(". ")[:3]  # Keep only top 3 sentences
    truncated_context = ". ".join(relevant_sentences)
    print("Retrieval Done -----------------------")
    # Construct improved prompt
    input_text = (
        "You are an AI assistant that answers questions **strictly based on the provided context**. "
        "If the answer is not found in the context, say 'I don't know.'\n\n"
        "### Context:\n"
        f"{truncated_context}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:"
    )



    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {"input_ids": input_ids}

    # Ensure input is within model's token limit (2048 for OPT)
    max_length = min(len(inputs["input_ids"][0]), 2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # inputs = {k: v[:, :max_length] for k, v in inputs.items()}
    # Generate response (limit output length)
    print("Generating Response")
    with torch.no_grad():
        output = model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=False,  # Change to deterministic mode
    temperature=0.0,  
    repetition_penalty=1.5, 
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)


    # Decode response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    response_text = response_text.replace(input_text, "").strip()


    # Clean output to remove unnecessary text
    if "Answer (keep it short and relevant):" in response_text:
        response_text = response_text.split("Answer (keep it short and relevant):")[-1].strip()

    return jsonify({"answer": response_text})




if __name__ == "__main__":
    app.run(debug=False,threaded=False)

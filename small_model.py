from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrieval import query_database
import torch
import gc
import atexit

# Set model name
MODEL_NAME = "facebook/opt-350m"  # Adjust based on available GPU memory

# Set device (Prefer GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with optimized settings for GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
    device_map="auto"
).eval()  # Set model to evaluation mode

app = Flask(__name__)

# Cleanup function to release resources
def cleanup():
    global model, tokenizer
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    # Retrieve relevant documents
    retrieved_docs = query_database(question)
    context = "\n".join(retrieved_docs[:3]) if retrieved_docs else "No relevant documents found."

    # Construct optimized prompt
    input_text = (
        "You are an AI assistant that answers questions **strictly based on the provided context**. "
        "If the answer is not found in the context, say 'I don't know.'\n\n"
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        "### Answer:"
    )

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # Generate response using GPU
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Clean response
    response_text = response_text.replace(input_text, "").strip()

    return jsonify({"answer": response_text})

if __name__ == "__main__":
    app.run(debug=False, threaded=False)

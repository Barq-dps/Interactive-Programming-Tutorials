import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re

# Hugging Face API key (hidden for security)
hf_api_key = "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"

# Model name
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# Load tokenizer and model from Hugging Face with API key
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_api_key)

# Create a text generation pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


# Step 2: Extract Topics and Generate Coding Questions
def generate_questions_from_text(text):
    topics = re.findall(r'Chapter\s\d+:\s([A-Za-z\s]+[A-Za-z])', text)
    
    questions = [f"Write a Python function to implement {topic}." for topic in topics]
    return questions


def evaluate_user_code_with_llama(user_code, question):
    """Evaluate user code using Qwen2.5 1.5B and provide feedback."""
    prompt = f"Evaluate the following Python code to solve the task:\nTask: {question}\nCode:\n{user_code}\nProvide feedback if the solution is correct or not."
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    output = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response.strip()


# Main Program
if __name__ == "__main__":
    pdf_path = "data/sample_docs.pdf"  # Path to PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text:
        # Generate questions from extracted topics
        questions = generate_questions_from_text(extracted_text)
        
        # Ask user to answer questions
        for question in questions:
            print(f"\n{question}")
            user_code = input("Write your code here:\n")
            
            # Evaluate the user's code
            feedback = evaluate_user_code_with_llama(user_code, question)
            print(f"\nFeedback:\n{feedback}")
    else:
        print("Failed to extract text or PDF is empty.")

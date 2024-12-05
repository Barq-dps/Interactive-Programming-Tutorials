import spacy
import re
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def load_text_file(file_name):
    """Load content from a text file."""
    file_path = "data/sample_docs.txt"  # Combine the base directory and file name
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")


def tokenize_sentences(text):
    """Tokenize text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def extract_code_snippets(text):
    """Extract code snippets from text using regex."""
    code_snippets = re.findall(r"(?:^|\n)(?: {4}|\t|def |class |\w+\s?=\s?|print\().+?\n", text)
    return [snippet.strip() for snippet in code_snippets]


# Test the script
if __name__ == "__main__":
    file_path = "sample_docs.txt"  # Replace with your file path
    text = load_text_file(file_path)

    if text:
        print("\nSentences:")
        sentences = tokenize_sentences(text)
        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")

        print("\nCode Snippets:")
        code_snippets = extract_code_snippets(text)
        for snippet in code_snippets:
            print(snippet)

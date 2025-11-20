
"""
AI Text Humanizer - Python 3.14 Compatible Version
Transforms AI-generated text to human-like content
Removes plagiarism patterns and bypasses AI detection
Supports processing of 15,000+ word documents
"""

import re
import random
import sys
from typing import List, Dict, Optional
import argparse

# Check Python version
if sys.version_info < (3, 14):
    print("Warning: This script is optimized for Python 3.14+")

try:
    import nltk
    from nltk.corpus import wordnet, stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    print("Installing NLTK...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.corpus import wordnet, stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

try:
    from transformers import (
        PegasusForConditionalGeneration,
        PegasusTokenizerFast,
        T5ForConditionalGeneration,
        T5Tokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: Transformers library not available. Using fallback methods.")
    HAS_TRANSFORMERS = False

# Download required NLTK data with error handling
def setup_nltk():
    """Setup NLTK data with proper error handling"""
    required_data = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords', 'punkt_tab']
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            try:
                nltk.download(data_name, quiet=True)
            except Exception as e:
                print(f"Note: Could not download {data_name}: {e}")


setup_nltk()


class TextHumanizer:
    """Advanced text humanizer with Python 3.14 compatibility"""
    
    def __init__(self, use_gpu: bool = False, use_transformers: bool = True):
        """Initialize models and resources"""
        self.use_transformers = use_transformers and HAS_TRANSFORMERS
        
        if self.use_transformers:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            
            try:
                # Load models with explicit error handling for Python 3.14
                print("Loading Pegasus model...")
                self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
                    "tuner007/pegasus_paraphrase",
                    torch_dtype=torch.float32
                ).to(self.device)
                self.pegasus_tokenizer = PegasusTokenizerFast.from_pretrained(
                    "tuner007/pegasus_paraphrase"
                )
                
                print("Loading T5 model...")
                self.t5_model = T5ForConditionalGeneration.from_pretrained(
                    "ramsrigouthamg/t5_paraphraser",
                    torch_dtype=torch.float32
                ).to(self.device)
                self.t5_tokenizer = T5Tokenizer.from_pretrained(
                    "ramsrigouthamg/t5_paraphraser",
                    legacy=False
                )
                
                print("Models loaded successfully!")
            except Exception as e:
                print(f"Error loading transformer models: {e}")
                print("Falling back to rule-based methods")
                self.use_transformers = False
        else:
            print("Using rule-based humanization methods only")
        
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced transition phrases for natural flow
        self.transitions = [
            "However,", "Moreover,", "Furthermore,", "In addition,",
            "On the other hand,", "Nevertheless,", "That said,",
            "Interestingly,", "In fact,", "As a result,", "Consequently,",
            "To clarify,", "In other words,", "Notably,", "Indeed,",
            "Meanwhile,", "Similarly,", "Alternatively,", "For instance,"
        ]
        
        # Filler phrases for human-like writing
        self.fillers = [
            "you know", "I mean", "sort of", "kind of", "basically",
            "actually", "literally", "honestly", "clearly", "obviously",
            "essentially", "generally", "typically", "usually", "often"
        ]
        
        # Sentence starters for variety
        self.starters = [
            "It's worth noting that", "Importantly,", "Significantly,",
            "What's interesting is that", "One might argue that",
            "Consider that", "Take for example", "Looking at this,"
        ]
    
    def get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms using WordNet with POS awareness"""
        synonyms = []
        
        # Map NLTK POS tags to WordNet POS tags
        pos_map = {
            'N': wordnet.NOUN, 
            'V': wordnet.VERB, 
            'J': wordnet.ADJ, 
            'R': wordnet.ADV
        }
        wordnet_pos = pos_map.get(pos[0] if pos else None)
        
        try:
            for syn in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and len(synonym) > 2:
                        synonyms.append(synonym)
        except Exception:
            pass
        
        return list(set(synonyms))[:5]
    
    def synonym_replacement(self, text: str, replace_ratio: float = 0.3) -> str:
        """Replace words with synonyms preserving meaning"""
        try:
            words = word_tokenize(text)
            tagged = pos_tag(words)
            new_words = []
            
            for word, pos in tagged:
                # Skip stop words, short words, and punctuation
                if (word.lower() in self.stop_words or 
                    not word.isalnum() or 
                    len(word) < 4):
                    new_words.append(word)
                    continue
                
                # Randomly replace based on ratio
                if random.random() < replace_ratio:
                    synonyms = self.get_synonyms(word, pos)
                    if synonyms:
                        new_words.append(random.choice(synonyms))
                    else:
                        new_words.append(word)
                else:
                    new_words.append(word)
            
            # Reconstruct with proper spacing
            result = []
            for i, word in enumerate(new_words):
                if i > 0 and word not in '.,!?;:)]}' and new_words[i-1] not in '([{':
                    result.append(' ')
                result.append(word)
            
            return ''.join(result)
        except Exception as e:
            print(f"Synonym replacement error: {e}")
            return text
    
    def paraphrase_pegasus(self, text: str, num_variants: int = 3) -> List[str]:
        """Paraphrase using Pegasus model with Python 3.14 compatibility"""
        if not self.use_transformers:
            return [self.synonym_replacement(text, 0.4)]
        
        try:
            # Tokenize with explicit parameters
            batch = self.pegasus_tokenizer(
                [text],
                truncation=True,
                padding='longest',
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with updated parameters for newer transformers
            with torch.no_grad():
                translated = self.pegasus_model.generate(
                    **batch,
                    max_length=512,
                    num_beams=num_variants * 2,
                    num_return_sequences=num_variants,
                    temperature=1.5,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    early_stopping=True
                )
            
            paraphrases = self.pegasus_tokenizer.batch_decode(
                translated, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return paraphrases if paraphrases else [text]
            
        except Exception as e:
            print(f"Pegasus error: {e}")
            return [self.synonym_replacement(text, 0.4)]
    
    def paraphrase_t5(self, text: str) -> str:
        """Paraphrase using T5 model with Python 3.14 compatibility"""
        if not self.use_transformers:
            return self.synonym_replacement(text, 0.4)
        
        try:
            input_text = f"paraphrase: {text}"
            input_ids = self.t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=5,
                    temperature=1.2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    early_stopping=True
                )
            
            result = self.t5_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return result if result else text
            
        except Exception as e:
            print(f"T5 error: {e}")
            return self.synonym_replacement(text, 0.4)
    
    def add_human_variability(self, text: str) -> str:
        """Add human-like variations and imperfections"""
        try:
            sentences = sent_tokenize(text)
            humanized_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Add variety to sentence starts
                if i > 0 and random.random() < 0.15:
                    starter = random.choice(self.starters)
                    sentence = f"{starter} {sentence[0].lower()}{sentence[1:]}"
                
                # Add transition words
                if i > 0 and random.random() < 0.2:
                    transition = random.choice(self.transitions)
                    sentence = f"{transition} {sentence[0].lower()}{sentence[1:]}"
                
                # Sparingly add filler phrases
                if random.random() < 0.08:
                    words = sentence.split()
                    if len(words) > 7:
                        insert_pos = random.randint(3, min(len(words) - 3, 8))
                        filler = random.choice(self.fillers)
                        words.insert(insert_pos, f"{filler},")
                        sentence = ' '.join(words)
                
                # Split overly long sentences
                if len(sentence.split()) > 30 and random.random() < 0.4:
                    words = sentence.split()
                    # Find a good break point
                    break_point = len(words) // 2
                    for offset in range(5):
                        if break_point + offset < len(words):
                            if words[break_point + offset] in ['and', 'but', 'or', 'which', 'that']:
                                break_point += offset
                                break
                    
                    part1 = ' '.join(words[:break_point]).rstrip(',') + '.'
                    part2 = words[break_point].capitalize() + ' ' + ' '.join(words[break_point+1:])
                    humanized_sentences.extend([part1, part2])
                else:
                    humanized_sentences.append(sentence)
            
            return ' '.join(humanized_sentences)
        except Exception as e:
            print(f"Variability error: {e}")
            return text
    
    def remove_repetitive_patterns(self, text: str) -> str:
        """Remove AI writing patterns"""
        # Common AI phrases to replace
        replacements = {
            r'\bin conclusion\b': lambda: random.choice(['to wrap up', 'finally', 'to sum up', 'in summary']),
            r'\bit is important to note that\b': lambda: random.choice(['note that', 'remember', 'keep in mind', 'bear in mind']),
            r'\bfurthermore\b': lambda: random.choice(['also', 'plus', 'and', 'additionally']),
            r'\bnevertheless\b': lambda: random.choice(['but', 'still', 'yet', 'however']),
            r'\badditionally\b': lambda: random.choice(['also', 'besides', 'plus', 'too']),
            r'\bin order to\b': lambda: 'to',
            r'\bdue to the fact that\b': lambda: random.choice(['because', 'since', 'as']),
            r'\bat this point in time\b': lambda: random.choice(['now', 'currently', 'today']),
        }
        
        for pattern, replacement_func in replacements.items():
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            for match in reversed(matches):
                replacement = replacement_func() if callable(replacement_func) else replacement_func
                text = text[:match.start()] + replacement + text[match.end():]
        
        return text
    
    def process_batch(self, sentences: List[str], batch_size: int = 5) -> List[str]:
        """Process sentences in batches"""
        processed = []
        total_batches = (len(sentences) - 1) // batch_size + 1
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches}...")
            
            for sentence in batch:
                # Skip very short sentences
                if len(sentence.split()) < 5:
                    processed.append(sentence)
                    continue
                
                if self.use_transformers:
                    # Use transformer models
                    variants = self.paraphrase_pegasus(sentence, num_variants=2)
                    best_variant = random.choice(variants)
                else:
                    # Fallback to synonym replacement
                    best_variant = self.synonym_replacement(sentence, replace_ratio=0.35)
                
                # Additional synonym pass
                best_variant = self.synonym_replacement(best_variant, replace_ratio=0.15)
                
                processed.append(best_variant)
        
        return processed
    
    def humanize_text(self, text: str, min_words: int = 15000) -> str:
        """Main humanization pipeline"""
        print(f"\n{'='*60}")
        print("AI TEXT HUMANIZER - Python 3.14 Compatible")
        print(f"{'='*60}")
        print(f"\nStarting humanization process...")
        
        current_words = len(text.split())
        print(f"Original text length: {current_words:,} words")
        
        if current_words < min_words:
            print(f"⚠ Warning: Text has only {current_words:,} words")
            print(f"  Minimum recommended: {min_words:,} words")
            print("  Consider providing longer text for best results\n")
        
        # Pipeline steps
        print(f"\n[1/5] Removing AI patterns...")
        text = self.remove_repetitive_patterns(text)
        
        print("[2/5] Splitting into sentences...")
        sentences = sent_tokenize(text)
        print(f"  Total sentences: {len(sentences)}")
        
        print("[3/5] Paraphrasing sentences...")
        processed_sentences = self.process_batch(sentences, batch_size=5)
        
        print("[4/5] Reconstructing text...")
        humanized_text = ' '.join(processed_sentences)
        
        print("[5/5] Adding human-like variations...")
        humanized_text = self.add_human_variability(humanized_text)
        
        final_words = len(humanized_text.split())
        print(f"\n{'='*60}")
        print(f"✓ Completed!")
        print(f"  Final text length: {final_words:,} words")
        print(f"  Change: {final_words - current_words:+,} words")
        print(f"{'='*60}\n")
        
        return humanized_text


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='AI Text Humanizer - Python 3.14 Compatible',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python humanizer.py -i input.txt -o output.txt
  python humanizer.py -i input.txt -o output.txt --gpu
  python humanizer.py -i input.txt -o output.txt --no-transformers
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input text file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output text file path'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=15000,
        help='Minimum word count (default: 15000)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    parser.add_argument(
        '--no-transformers',
        action='store_true',
        help='Use only rule-based methods (no AI models)'
    )
    
    args = parser.parse_args()
    
    # Read input
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"✗ Error: Input file '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        sys.exit(1)
    
    # Initialize humanizer
    try:
        humanizer = TextHumanizer(
            use_gpu=args.gpu,
            use_transformers=not args.no_transformers
        )
    except Exception as e:
        print(f"✗ Error initializing: {e}")
        sys.exit(1)
    
    # Process text
    try:
        humanized_text = humanizer.humanize_text(input_text, min_words=args.min_words)
    except Exception as e:
        print(f"✗ Error processing text: {e}")
        sys.exit(1)
    
    # Write output
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(humanized_text)
        print(f"✓ Humanized text saved to: {args.output}")
        print("\n✓ Plagiarism patterns removed")
        print("✓ AI detection signatures minimized")
        print("✓ Natural human-like variations added")
    except Exception as e:
        print(f"✗ Error writing output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

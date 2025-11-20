
"""
Advanced AI Text Humanizer with Detection Bypass - Python 3.14
Fixed NLTK compatibility issues
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import sys
import re
import random
from typing import List, Optional, Tuple
from collections import Counter

# Fix NLTK data download and loading
try:
    import nltk
    from nltk.corpus import wordnet, stopwords, brown
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    from nltk.corpus import wordnet, stopwords, brown
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

try:
    from transformers import (
        PegasusForConditionalGeneration,
        PegasusTokenizerFast,
        T5ForConditionalGeneration,
        T5Tokenizer
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def setup_nltk():
    """Setup NLTK with Python 3.14 compatibility"""
    print("Setting up NLTK resources...")
    
    # List of required packages for Python 3.14
    required_packages = [
        'punkt',                              # Sentence tokenizer
        'punkt_tab',                          # New in NLTK 3.9+
        'averaged_perceptron_tagger',         # Generic tagger
        'averaged_perceptron_tagger_eng',     # English-specific (Python 3.14)
        'wordnet',                            # Synonym database
        'stopwords',                          # Common words
        'brown',                              # Text corpus
        'omw-1.4',                            # Open Multilingual Wordnet
    ]
    
    for package in required_packages:
        try:
            # Try to find the package
            if package in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{package}')
            elif 'tagger' in package:
                nltk.data.find(f'taggers/{package}')
            elif package in ['wordnet', 'stopwords', 'brown']:
                nltk.data.find(f'corpora/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
            print(f"✓ {package} already installed")
        except LookupError:
            try:
                print(f"⏳ Downloading {package}...")
                nltk.download(package, quiet=False)
                print(f"✓ {package} downloaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not download {package}: {e}")
                if 'tagger_eng' in package:
                    print("  Trying alternative download method...")
                    try:
                        # Alternative method for language-specific taggers
                        import ssl
                        ssl._create_default_https_context = ssl._create_unverified_context
                        nltk.download(package, quiet=False)
                        print(f"✓ {package} downloaded with alternative method")
                    except Exception as e2:
                        print(f"  Still failed: {e2}")
    
    print("NLTK setup complete!\n")


# Run NLTK setup immediately
setup_nltk()


class AdvancedTextHumanizer:
    """Advanced humanizer with perplexity and burstiness manipulation"""
    
    def __init__(self, use_gpu=False, use_transformers=True, progress_callback=None):
        self.use_transformers = use_transformers and HAS_TRANSFORMERS
        self.progress_callback = progress_callback
        
        if self.use_transformers:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            self._log(f"Using device: {self.device}")
            
            try:
                self._log("Loading advanced models...")
                
                self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
                    "tuner007/pegasus_paraphrase", torch_dtype=torch.float32
                ).to(self.device)
                self.pegasus_tokenizer = PegasusTokenizerFast.from_pretrained(
                    "tuner007/pegasus_paraphrase"
                )
                
                self.t5_model = T5ForConditionalGeneration.from_pretrained(
                    "ramsrigouthamg/t5_paraphraser", torch_dtype=torch.float32
                ).to(self.device)
                self.t5_tokenizer = T5Tokenizer.from_pretrained(
                    "ramsrigouthamg/t5_paraphraser", legacy=False
                )
                
                self._log("Models loaded successfully!")
            except Exception as e:
                self._log(f"Model loading error: {e}")
                self._log("Using rule-based methods")
                self.use_transformers = False
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self._log("Warning: Stopwords not available, using basic set")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Enhanced natural language patterns
        self.contractions = {
            'do not': "don't", 'does not': "doesn't", 'did not': "didn't",
            'is not': "isn't", 'are not': "aren't", 'was not': "wasn't",
            'were not': "weren't", 'have not': "haven't", 'has not': "hasn't",
            'had not': "hadn't", 'will not': "won't", 'would not': "wouldn't",
            'cannot': "can't", 'could not': "couldn't", 'should not': "shouldn't",
            'it is': "it's", 'that is': "that's", 'there is': "there's",
            'I am': "I'm", 'you are': "you're", 'we are': "we're",
            'they are': "they're", 'I have': "I've", 'you have': "you've",
            'we have': "we've", 'they have': "they've", 'I would': "I'd",
            'you would': "you'd", 'he would': "he'd", 'she would': "she'd"
        }
        
        self.casual_transitions = [
            "But", "And", "So", "Plus", "Also", "Though", "Still",
            "Yet", "Now", "Well", "Anyway", "Besides", "Meanwhile",
            "Then again", "That said", "On top of that", "What's more"
        ]
        
        self.natural_starters = [
            "Look,", "See,", "Thing is,", "Here's the thing:",
            "You know what?", "Honestly,", "Frankly,", "Truth is,",
            "The fact is", "Really,", "Actually,", "Basically,",
            "In reality,", "To be fair,", "Let's be honest,"
        ]
        
        self.hedging = [
            "might", "could", "possibly", "perhaps", "probably",
            "seems like", "appears to", "tends to", "often",
            "generally", "typically", "usually", "sometimes",
            "in some cases", "to some extent", "arguably"
        ]
        
        self.personal_phrases = [
            "I think", "I believe", "In my view", "From what I've seen",
            "In my experience", "It seems to me", "I'd say", "I reckon",
            "I figure", "My take is", "The way I see it"
        ]
        
        self.colloquial = [
            "a lot of", "pretty much", "kind of", "sort of",
            "more or less", "pretty", "quite", "really", "very",
            "super", "totally", "definitely", "absolutely"
        ]
    
    def _log(self, message):
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
    
    def calculate_perplexity_score(self, text: str) -> float:
        """Calculate text perplexity (higher = more human-like)"""
        try:
            words = word_tokenize(text.lower())
        except:
            words = text.lower().split()
        
        if len(words) < 2:
            return 0.0
        
        word_freq = Counter(words)
        total_words = len(words)
        
        entropy = 0
        for word, freq in word_freq.items():
            prob = freq / total_words
            if prob > 0:
                entropy -= prob * (prob ** 0.5)
        
        return min(entropy * 100, 100)
    
    def calculate_burstiness(self, text: str) -> float:
        """Calculate burstiness (sentence length variation)"""
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
        
        if len(sentences) < 2:
            return 0.0
        
        try:
            lengths = [len(word_tokenize(s)) for s in sentences]
        except:
            lengths = [len(s.split()) for s in sentences]
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        
        burstiness = (std_dev / avg_length) * 100 if avg_length > 0 else 0
        return min(burstiness, 100)
    
    def add_contractions(self, text: str) -> str:
        """Add natural contractions"""
        for formal, casual in self.contractions.items():
            if random.random() < 0.6:
                pattern = r'\b' + formal + r'\b'
                text = re.sub(pattern, casual, text, flags=re.IGNORECASE)
        return text
    
    def get_diverse_synonyms(self, word: str, pos_tag: str = None) -> List[str]:
        """Get less common, more diverse synonyms"""
        synonyms = []
        pos_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}
        wordnet_pos = pos_map.get(pos_tag[0] if pos_tag else None)
        
        try:
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            for syn in synsets[:3]:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and len(synonym) > 2:
                        synonyms.append(synonym)
        except:
            pass
        
        return list(set(synonyms))[:8]
    
    def increase_perplexity(self, text: str) -> str:
        """Increase perplexity by making word choices less predictable"""
        try:
            words = word_tokenize(text)
            tagged = pos_tag(words)
        except Exception as e:
            self._log(f"Tokenization fallback: {e}")
            words = text.split()
            tagged = [(w, 'NN') for w in words]
        
        new_words = []
        
        for i, (word, pos) in enumerate(tagged):
            if not word.isalnum() or word.lower() in self.stop_words:
                new_words.append(word)
                continue
            
            if random.random() < 0.4 and len(word) > 4:
                synonyms = self.get_diverse_synonyms(word, pos)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        result = []
        for i, word in enumerate(new_words):
            if i > 0 and word not in '.,!?;:)]}"\'' and new_words[i-1] not in '([{"\'':
                result.append(' ')
            result.append(word)
        
        return ''.join(result)
    
    def increase_burstiness(self, text: str) -> str:
        """Vary sentence lengths dramatically"""
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
            sentences = [s + '.' if not s.endswith('.') else s for s in sentences]
        
        new_sentences = []
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            try:
                words = word_tokenize(sentence)
            except:
                words = sentence.split()
            
            pattern = random.choice(['short', 'long', 'medium', 'combined'])
            
            if pattern == 'short' and len(words) > 12:
                mid = len(words) // 2
                part1 = ' '.join(words[:mid]).rstrip(',;') + '.'
                part2 = words[mid].capitalize() + ' ' + ' '.join(words[mid+1:])
                new_sentences.extend([part1, part2])
                
            elif pattern == 'long' and i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                connector = random.choice([', and', ', but', ', yet', '; however,', ' while'])
                combined = sentence.rstrip('.!?') + connector + ' ' + next_sentence[0].lower() + next_sentence[1:]
                new_sentences.append(combined)
                i += 1
                
            elif pattern == 'combined' and len(words) > 8:
                fragment_pos = random.randint(len(words)//2, len(words)-2)
                fragment = ' '.join(words[fragment_pos:]) + '.'
                main = ' '.join(words[:fragment_pos]) + '.'
                new_sentences.extend([main, fragment])
                
            else:
                new_sentences.append(sentence)
            
            i += 1
        
        return ' '.join(new_sentences)
    
    def add_human_imperfections(self, text: str) -> str:
        """Add natural human writing quirks"""
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
        
        humanized = []
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.12 and i > 0:
                starter = random.choice(self.natural_starters)
                sentence = f"{starter} {sentence[0].lower()}{sentence[1:]}"
            
            if random.random() < 0.15:
                words = sentence.split()
                if len(words) > 5:
                    hedge = random.choice(self.hedging)
                    insert_pos = random.randint(1, min(4, len(words)-2))
                    words.insert(insert_pos, hedge)
                    sentence = ' '.join(words)
            
            if random.random() < 0.10 and i > 0:
                personal = random.choice(self.personal_phrases)
                sentence = f"{personal}, {sentence[0].lower()}{sentence[1:]}"
            
            if random.random() < 0.18:
                words = sentence.split()
                if len(words) > 4:
                    for j, word in enumerate(words):
                        if word.lower() in ['very', 'extremely', 'highly']:
                            words[j] = random.choice(self.colloquial)
                            break
                sentence = ' '.join(words)
            
            if random.random() < 0.25 and i > 0:
                transition = random.choice(self.casual_transitions)
                sentence = f"{transition}, {sentence[0].lower()}{sentence[1:]}"
            
            humanized.append(sentence)
        
        return ' '.join(humanized)
    
    def remove_ai_markers(self, text: str) -> str:
        """Remove telltale AI writing patterns"""
        ai_patterns = {
            r'\bIt is important to note that\b': lambda: random.choice(['Note that', 'Keep in mind', 'Remember']),
            r'\bIt should be noted that\b': lambda: random.choice(['Worth noting:', 'Thing is,', 'Here\'s the deal:']),
            r'\bIt is worth mentioning that\b': lambda: random.choice(['Also,', 'Plus,', 'And']),
            r'\bIn conclusion\b': lambda: random.choice(['So', 'Bottom line', 'To wrap up', 'In the end']),
            r'\bIn summary\b': lambda: random.choice(['To sum it up', 'Long story short', 'Basically']),
            r'\bFurthermore\b': lambda: random.choice(['Also', 'Plus', 'And', 'On top of that']),
            r'\bMoreover\b': lambda: random.choice(['Besides', 'Plus', 'What\'s more', 'Also']),
            r'\bNevertheless\b': lambda: random.choice(['But', 'Still', 'Yet', 'Even so']),
            r'\bHowever\b': lambda: random.choice(['But', 'Though', 'Still', 'That said']) if random.random() < 0.5 else 'However',
            r'\bTherefore\b': lambda: random.choice(['So', 'Thus', 'That\'s why']),
            r'\bConsequently\b': lambda: random.choice(['So', 'As a result', 'Because of this']),
            r'\bAdditionally\b': lambda: random.choice(['Also', 'Plus', 'And', 'Too']),
            r'\bSubsequently\b': lambda: random.choice(['Then', 'Later', 'After that']),
            r'\bIn order to\b': lambda: 'to',
            r'\bDue to the fact that\b': lambda: random.choice(['because', 'since', 'as']),
            r'\bFor the purpose of\b': lambda: random.choice(['to', 'for']),
            r'\bWith regards to\b': lambda: random.choice(['about', 'regarding', 'on']),
            r'\bIn light of\b': lambda: random.choice(['given', 'considering', 'because of']),
            r'\bIt can be seen that\b': lambda: random.choice(['We see', 'You can see', 'Clearly']),
            r'\bIt is evident that\b': lambda: random.choice(['Clearly', 'Obviously', 'It\'s clear that']),
        }
        
        for pattern, replacement_func in ai_patterns.items():
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            for match in reversed(matches):
                replacement = replacement_func()
                text = text[:match.start()] + replacement + text[match.end():]
        
        return text
    
    def paraphrase_advanced(self, text: str, method='pegasus') -> str:
        """Advanced paraphrasing with multiple passes"""
        if not self.use_transformers:
            return self.increase_perplexity(text)
        
        try:
            if method == 'pegasus':
                batch = self.pegasus_tokenizer([text], truncation=True, padding='longest',
                                               max_length=512, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.pegasus_model.generate(
                        **batch,
                        max_length=512,
                        num_beams=8,
                        num_return_sequences=3,
                        temperature=1.7,
                        do_sample=True,
                        top_k=60,
                        top_p=0.93,
                        repetition_penalty=1.5,
                        early_stopping=True
                    )
                
                results = self.pegasus_tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                             clean_up_tokenization_spaces=True)
                return random.choice(results) if results else text
                
            elif method == 't5':
                input_text = f"paraphrase: {text}"
                input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt",
                                                     max_length=512, truncation=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        input_ids,
                        max_length=512,
                        num_beams=6,
                        temperature=1.6,
                        do_sample=True,
                        top_k=60,
                        top_p=0.91,
                        repetition_penalty=1.4,
                        early_stopping=True
                    )
                
                result = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
                return result if result else text
        except Exception as e:
            self._log(f"Paraphrase error: {e}")
            return self.increase_perplexity(text)
    
    def process_batch_advanced(self, sentences: List[str], batch_size: int = 4) -> List[str]:
        """Process with multiple techniques"""
        processed = []
        total_batches = (len(sentences) - 1) // batch_size + 1
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_num = i // batch_size + 1
            self._log(f"Processing batch {batch_num}/{total_batches}...")
            
            for sentence in batch:
                if len(sentence.split()) < 4:
                    processed.append(sentence)
                    continue
                
                result = sentence
                
                if self.use_transformers:
                    method = random.choice(['pegasus', 't5'])
                    result = self.paraphrase_advanced(result, method=method)
                else:
                    result = self.increase_perplexity(result)
                
                result = self.increase_perplexity(result)
                result = self.add_contractions(result)
                
                processed.append(result)
        
        return processed
    
    def humanize_text(self, text: str, min_words: int = 15000) -> Tuple[str, dict]:
        """Main humanization with metrics"""
        self._log("="*60)
        self._log("ADVANCED AI DETECTION BYPASS")
        self._log("="*60)
        
        original_words = len(text.split())
        self._log(f"Original: {original_words:,} words")
        
        initial_perplexity = self.calculate_perplexity_score(text)
        initial_burstiness = self.calculate_burstiness(text)
        
        self._log(f"Initial Perplexity: {initial_perplexity:.1f}/100")
        self._log(f"Initial Burstiness: {initial_burstiness:.1f}/100")
        
        self._log("\n[1/8] Removing AI markers...")
        text = self.remove_ai_markers(text)
        
        self._log("[2/8] Adding contractions...")
        text = self.add_contractions(text)
        
        self._log("[3/8] Splitting into sentences...")
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('. ')
        self._log(f"Total sentences: {len(sentences)}")
        
        self._log("[4/8] Advanced paraphrasing...")
        sentences = self.process_batch_advanced(sentences, batch_size=4)
        
        self._log("[5/8] Increasing perplexity...")
        text = ' '.join(sentences)
        text = self.increase_perplexity(text)
        
        self._log("[6/8] Increasing burstiness...")
        text = self.increase_burstiness(text)
        
        self._log("[7/8] Adding human imperfections...")
        text = self.add_human_imperfections(text)
        
        self._log("[8/8] Final polish...")
        text = self.add_contractions(text)
        
        final_words = len(text.split())
        final_perplexity = self.calculate_perplexity_score(text)
        final_burstiness = self.calculate_burstiness(text)
        
        metrics = {
            'original_words': original_words,
            'final_words': final_words,
            'initial_perplexity': initial_perplexity,
            'final_perplexity': final_perplexity,
            'initial_burstiness': initial_burstiness,
            'final_burstiness': final_burstiness,
            'perplexity_increase': final_perplexity - initial_perplexity,
            'burstiness_increase': final_burstiness - initial_burstiness
        }
        
        self._log("\n" + "="*60)
        self._log("✓ PROCESSING COMPLETE!")
        self._log(f"Final words: {final_words:,}")
        self._log(f"Perplexity: {initial_perplexity:.1f} → {final_perplexity:.1f} (+{metrics['perplexity_increase']:.1f})")
        self._log(f"Burstiness: {initial_burstiness:.1f} → {final_burstiness:.1f} (+{metrics['burstiness_increase']:.1f})")
        self._log("="*60)
        
        return text, metrics


# [Rest of the GUI code remains the same as before - include the full AdvancedHumanizerGUI class here]

class AdvancedHumanizerGUI(ctk.CTk):
    """Enhanced GUI with metrics display"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Advanced AI Text Humanizer - Detection Bypass")
        self.geometry("1400x900")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.humanizer = None
        self.processing = False
        self.metrics = {}
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create enhanced UI"""
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)
        
        # Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🛡️ AI Detection\nBypass Tool",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        # Version
        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text="Python 3.14 Compatible",
            font=ctk.CTkFont(size=9)
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Settings
        self.settings_label = ctk.CTkLabel(
            self.sidebar,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.settings_label.grid(row=2, column=0, padx=20, pady=(10, 10))
        
        # GPU
        self.gpu_var = ctk.BooleanVar(value=False)
        self.gpu_switch = ctk.CTkSwitch(
            self.sidebar,
            text="🚀 GPU Acceleration",
            variable=self.gpu_var
        )
        self.gpu_switch.grid(row=3, column=0, padx=20, pady=8)
        
        # Transformers
        self.transformers_var = ctk.BooleanVar(value=HAS_TRANSFORMERS)
        self.transformers_switch = ctk.CTkSwitch(
            self.sidebar,
            text="🤖 AI Models",
            variable=self.transformers_var,
            state="normal" if HAS_TRANSFORMERS else "disabled"
        )
        self.transformers_switch.grid(row=4, column=0, padx=20, pady=8)
        
        # Min words slider
        self.min_words_label = ctk.CTkLabel(
            self.sidebar,
            text="📏 Min Words: 15000",
            font=ctk.CTkFont(size=12)
        )
        self.min_words_label.grid(row=5, column=0, padx=20, pady=(15, 0))
        
        self.min_words_slider = ctk.CTkSlider(
            self.sidebar,
            from_=5000,
            to=30000,
            number_of_steps=25,
            command=self.update_min_words
        )
        self.min_words_slider.set(15000)
        self.min_words_slider.grid(row=6, column=0, padx=20, pady=8)
        
        # Metrics Display
        self.metrics_label = ctk.CTkLabel(
            self.sidebar,
            text="📊 Detection Metrics",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.metrics_label.grid(row=7, column=0, padx=20, pady=(20, 10))
        
        self.perplexity_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.perplexity_frame.grid(row=8, column=0, padx=20, pady=5)
        
        self.perplexity_label = ctk.CTkLabel(
            self.perplexity_frame,
            text="Perplexity: --",
            font=ctk.CTkFont(size=11)
        )
        self.perplexity_label.pack()
        
        self.burstiness_label = ctk.CTkLabel(
            self.perplexity_frame,
            text="Burstiness: --",
            font=ctk.CTkFont(size=11)
        )
        self.burstiness_label.pack(pady=5)
        
        self.detection_label = ctk.CTkLabel(
            self.perplexity_frame,
            text="🎯 Detection Risk: Unknown",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        )
        self.detection_label.pack(pady=10)
        
        # Buttons
        self.load_btn = ctk.CTkButton(
            self.sidebar,
            text="📂 Load File",
            command=self.load_file,
            font=ctk.CTkFont(size=13)
        )
        self.load_btn.grid(row=13, column=0, padx=20, pady=8)
        
        self.save_btn = ctk.CTkButton(
            self.sidebar,
            text="💾 Save Output",
            command=self.save_file,
            font=ctk.CTkFont(size=13)
        )
        self.save_btn.grid(row=14, column=0, padx=20, pady=8)
        
        self.clear_btn = ctk.CTkButton(
            self.sidebar,
            text="🗑️ Clear All",
            command=self.clear_all,
            fg_color="gray",
            font=ctk.CTkFont(size=13)
        )
        self.clear_btn.grid(row=15, column=0, padx=20, pady=(8, 20))
        
        # Main area
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)
        
        # Input
        self.input_label = ctk.CTkLabel(
            self.main_frame,
            text="📝 Input Text (AI Generated)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.input_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.input_text = ctk.CTkTextbox(
            self.main_frame,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.input_text.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="nsew")
        
        # Process button
        self.process_btn = ctk.CTkButton(
            self.main_frame,
            text="✨ HUMANIZE & BYPASS DETECTION",
            command=self.process_text,
            font=ctk.CTkFont(size=17, weight="bold"),
            height=55,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        self.process_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Output
        self.output_label = ctk.CTkLabel(
            self.main_frame,
            text="✅ Output Text (Humanized & Undetectable)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.output_label.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="w")
        
        self.output_text = ctk.CTkTextbox(
            self.main_frame,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.output_text.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Status bar
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=1, column=1, padx=20, pady=(0, 15), sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to humanize",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.word_count_label = ctk.CTkLabel(
            self.status_frame,
            text="Input: 0 words | Output: 0 words",
            font=ctk.CTkFont(size=11)
        )
        self.word_count_label.pack(side="right", padx=10, pady=5)
        
        # Progress
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.grid(row=2, column=1, padx=20, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0)
        
        # Console
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=3, column=1, padx=20, pady=(0, 15), sticky="ew")
        self.console_frame.grid_columnconfigure(0, weight=1)
        
        self.console_label = ctk.CTkLabel(
            self.console_frame,
            text="📋 Processing Log",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.console_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.console = ctk.CTkTextbox(
            self.console_frame,
            height=100,
            font=ctk.CTkFont(family="Courier", size=10)
        )
        self.console.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.input_text.bind("<<Modified>>", self.update_word_count)
        self.output_text.bind("<<Modified>>", self.update_word_count)
    
    def log(self, message):
        self.console.insert("end", f"{message}\n")
        self.console.see("end")
        self.update()
    
    def update_status(self, message):
        self.status_label.configure(text=message)
        self.update()
    
    def update_word_count(self, event=None):
        input_words = len(self.input_text.get("1.0", "end-1c").split())
        output_words = len(self.output_text.get("1.0", "end-1c").split())
        self.word_count_label.configure(
            text=f"Input: {input_words:,} words | Output: {output_words:,} words"
        )
    
    def update_min_words(self, value):
        self.min_words_label.configure(text=f"📏 Min Words: {int(value)}")
    
    def update_metrics_display(self, metrics):
        """Update detection risk metrics"""
        perp = metrics.get('final_perplexity', 0)
        burst = metrics.get('final_burstiness', 0)
        
        self.perplexity_label.configure(text=f"Perplexity: {perp:.1f}/100")
        self.burstiness_label.configure(text=f"Burstiness: {burst:.1f}/100")
        
        avg_score = (perp + burst) / 2
        
        if avg_score >= 60:
            risk = "🟢 LOW"
            color = "green"
        elif avg_score >= 40:
            risk = "🟡 MEDIUM"
            color = "orange"
        else:
            risk = "🔴 HIGH"
            color = "red"
        
        self.detection_label.configure(
            text=f"Detection Risk: {risk}",
            text_color=color
        )
    
    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.input_text.delete("1.0", "end")
                self.input_text.insert("1.0", content)
                self.log(f"Loaded: {filepath}")
                self.update_status(f"Loaded file")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load:\n{str(e)}")
    
    def save_file(self):
        output = self.output_text.get("1.0", "end-1c")
        
        if not output.strip():
            messagebox.showwarning("Warning", "No output to save!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(output)
                self.log(f"Saved: {filepath}")
                messagebox.showinfo("Success", "File saved!")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def clear_all(self):
        self.input_text.delete("1.0", "end")
        self.output_text.delete("1.0", "end")
        self.console.delete("1.0", "end")
        self.progress_bar.set(0)
        self.perplexity_label.configure(text="Perplexity: --")
        self.burstiness_label.configure(text="Burstiness: --")
        self.detection_label.configure(text="🎯 Detection Risk: Unknown", text_color="gray")
        self.log("Cleared")
        self.update_status("Ready")
    
    def process_text(self):
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return
        
        input_text = self.input_text.get("1.0", "end-1c").strip()
        
        if not input_text:
            messagebox.showwarning("Warning", "Enter text first!")
            return
        
        self.process_btn.configure(state="disabled", text="⏳ BYPASSING DETECTION...")
        self.processing = True
        self.progress_bar.set(0)
        
        thread = threading.Thread(target=self._process_thread, args=(input_text,))
        thread.daemon = True
        thread.start()
    
    def _process_thread(self, input_text):
        try:
            min_words = int(self.min_words_slider.get())
            
            self.humanizer = AdvancedTextHumanizer(
                use_gpu=self.gpu_var.get(),
                use_transformers=self.transformers_var.get(),
                progress_callback=self.log
            )
            
            self.progress_bar.set(0.1)
            humanized, metrics = self.humanizer.humanize_text(input_text, min_words=min_words)
            self.progress_bar.set(1.0)
            
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", humanized)
            
            self.metrics = metrics
            self.update_metrics_display(metrics)
            
            self.update_status("✓ Processing complete - Detection bypassed!")
            messagebox.showinfo("Success", "Text humanized successfully!\n\nCheck the metrics for detection risk assessment.")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed:\n{str(e)}")
        finally:
            self.processing = False
            self.process_btn.configure(state="normal", text="✨ HUMANIZE & BYPASS DETECTION")
            self.progress_bar.set(0)


def main():
    app = AdvancedHumanizerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

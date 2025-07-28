#!/usr/bin/env python3
"""
LLaVA v1.6 13B Cross-Cultural Meme Translation Test Script
Tests the model's ability to translate Chinese memes to US cultural context
"""

import os
import sys
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llava_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLaVAMemeTranslator:
    """LLaVA model wrapper for meme translation testing"""
    
    def __init__(self, model_name: str = None):
        """Initialize the LLaVA model and processor"""
        # Use your local cached model first, then fallback to online
        if model_name is None:
            # Your local cached model path
            local_cache_path = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/huggingface_cache/models--llava-hf--llava-v1.6-vicuna-13b-hf"
            
            # Check if local cache exists and has model files
            if os.path.exists(local_cache_path):
                # Look for the actual model directory inside the cache
                snapshots_dir = os.path.join(local_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Find the latest snapshot
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        # Use the first (and likely only) snapshot
                        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                        # Verify it has model files
                        model_files = [f for f in os.listdir(snapshot_path) if f.endswith(('.bin', '.safetensors', '.json'))]
                        if model_files:
                            self.model_name = snapshot_path
                            logger.info(f"Using local cached model: {self.model_name}")
                        else:
                            logger.warning(f"Local cache found but no model files detected, falling back to online")
                            self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
                    else:
                        logger.warning(f"Local cache directory empty, falling back to online")
                        self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
                else:
                    logger.warning(f"Local cache structure unexpected, falling back to online")
                    self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
            else:
                logger.warning(f"Local cache not found at {local_cache_path}, using online model")
                self.model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
        else:
            self.model_name = model_name
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing LLaVA model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU mode")
        
        # Load processor and model
        self.processor = None
        self.model = None
        self._load_model()
        
        # Define the system prompt
        self.system_prompt = """ROLE: You are a cross-cultural meme translation specialist focused on humor preservation. Please explain your reasoning step by step.
INPUT: [Meme image] + [Cultural description in any language] + [GIVEN LABEL]
OUTPUT FORMAT (all in English):
1. EMOTION ANALYSIS:
    "Primary Emotion: [joy/anger/sadness/fear/disgust]
    Intensity Level: [low/mid/high]
    Explanation: [1-2 sentences describing the emotional context and why this intensity level], infer emotion and intensity if they are not specified in the INPUT, otherwise use the INPUT description."
 
2. CULTURAL MAPPING (2-3 sentences):
    "In US culture, this humor/intent translates to [specific US cultural equivalent that maintains the same comedic effect]. Americans express this same sentiment through [US equivalent] while preserving the [original humor type: irony/sarcasm/wholesome/etc.]. This captures the original's [specific comedic mechanism],fits the target culture's humor and references perfectly."

3. VISUAL RECOMMENDATION (two separate options):
   
   OPTION A - CARTOON CHARACTER:
   "Use a clear and engaging image of [specific cartoon character] in [specific pose that mirrors original intent]. Visual elements should include [details that support the humor]. 
Template style: [meme format that preserves comedic timing]."
   
   OPTION B - REAL HUMAN FIGURE:
   "Use a clear and engaging image of [demographic] person with [expression/gesture that conveys same humor]. Person should be [appearance] in [setting that enhances the joke]. Use [style that maintains comedic effect]."

4. ADAPTED CAPTION (original English preserving humor):
   "[Write a new clear and engaging caption in English that would work for English culture. Ensure it maintains the same humor/intent as the original. Use appropriate English slang, references, or expressions], make the caption and image work together as a meme, delivering a clear emotional/humorous effect"

CONSTRAINTS:
- Choose exactly one primary emotion from: joy, anger, sadness, fear, disgust
- Choose exactly one intensity level: low, high
- Preserve original humor style and intent
- Visual recommendations should support both the emotion and humor
- Caption should reflect the identified emotional tone"""
    
    def _load_model(self):
        """Load the LLaVA model with optimized settings matching your deployment"""
        try:
            # Load processor
            logger.info(f"Loading LLaVA processor from: {self.model_name}")
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name, local_files_only=True)
            
            # Configure model loading parameters (matching your deployment script)
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": True,  # Using 8-bit quantization like your deployment
                "local_files_only": True,  # Force using local files only
            }
            
            # Disable flash attention if not available (from your script logic)
            try:
                import flash_attn
                logger.info("Flash attention detected and will be used")
            except ImportError:
                logger.info("Flash attention not available, using eager attention")
                model_kwargs["attn_implementation"] = "eager"
            
            # Load model
            logger.info(f"Loading LLaVA model from: {self.model_name}")
            logger.info("Using 8-bit quantization for memory efficiency...")
            logger.info("Loading from local cache (no download required)...")
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
            
            logger.info("Model loaded successfully from local cache!")
            logger.info(f"Model device: {self.model.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load model from local cache: {e}")
            logger.info("Attempting to fall back to online model with authentication...")
            
            # Fallback to online model if local fails
            try:
                # Get HuggingFace token if available
                token = os.getenv('HF_TOKEN')
                online_model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
                
                # Load processor with authentication
                if token:
                    self.processor = LlavaNextProcessor.from_pretrained(online_model_id, use_auth_token=token)
                else:
                    self.processor = LlavaNextProcessor.from_pretrained(online_model_id)
                
                # Configure model loading for online
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "load_in_8bit": True,
                }
                
                if token:
                    model_kwargs["use_auth_token"] = token
                
                # Disable flash attention if not available
                try:
                    import flash_attn
                except ImportError:
                    model_kwargs["attn_implementation"] = "eager"
                
                logger.info(f"Loading model online from: {online_model_id}")
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    online_model_id, **model_kwargs
                )
                
                logger.info("Model loaded successfully from online source!")
                
                # Update model_name for consistency
                self.model_name = online_model_id
                
            except Exception as fallback_error:
                logger.error(f"Both local and online loading failed:")
                logger.error(f"Local error: {e}")
                logger.error(f"Online error: {fallback_error}")
                logger.error("Please check:")
                logger.error("1. Local cache integrity")
                logger.error("2. HuggingFace token: export HF_TOKEN='your_token'")
                logger.error("3. Internet connection for fallback")
                sys.exit(1)
    
    def translate_meme(self, image_path: str, content: str, emotion: str, intensity: str) -> str:
        """Translate a single meme using the model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Construct the input prompt
            user_prompt = f"Cultural description: {content}\nGiven emotion: {emotion}\nGiven intensity: {intensity}"
            
            # Prepare inputs
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt + "\n\n" + user_prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Apply chat template and process
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            logger.debug(f"Generating response for image: {image_path}")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            prompt_length = len(self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
            generated_text = response[prompt_length:].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return f"ERROR: {str(e)}"

class MemeTestSuite:
    """Test suite for evaluating meme translation performance"""
    
    def __init__(self, dataset_path: str, images_dir: str, output_dir: str = "test_results", model_path: str = None):
        """Initialize test suite"""
        self.dataset_path = dataset_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        logger.info(f"Loaded dataset with {len(self.dataset)} entries")
        
        # Initialize model
        self.translator = LLaVAMemeTranslator(model_path)
        
        # Results storage
        self.results = []
    
    def _load_dataset(self) -> List[Dict]:
        """Load the CSV dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)
    
    def select_random_samples(self, n: int = 20, seed: int = 42) -> List[Dict]:
        """Select n random samples from the dataset"""
        random.seed(seed)
        if n >= len(self.dataset):
            logger.warning(f"Requested {n} samples but dataset only has {len(self.dataset)} entries")
            return self.dataset
        
        selected = random.sample(self.dataset, n)
        logger.info(f"Selected {len(selected)} random samples for testing")
        return selected
    
    def run_test_batch(self, test_samples: List[Dict]) -> List[Dict]:
        """Run translation tests on a batch of samples"""
        results = []
        total_samples = len(test_samples)
        
        logger.info(f"Starting batch test with {total_samples} samples...")
        
        for i, sample in enumerate(test_samples, 1):
            logger.info(f"Processing sample {i}/{total_samples}: {sample['filename']}")
            
            # Check if image file exists
            image_path = self.images_dir / sample['filename']
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                result = {
                    'filename': sample['filename'],
                    'original_content': sample['content'],
                    'original_emotion': sample['emotion'],
                    'original_intensity': sample['intensity'],
                    'translation': f"ERROR: Image file not found - {image_path}",
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                continue
            
            # Record start time
            start_time = time.time()
            
            # Perform translation
            translation = self.translator.translate_meme(
                str(image_path),
                sample['content'],
                sample['emotion'],
                sample['intensity']
            )
            
            # Record processing time
            processing_time = time.time() - start_time
            
            # Store result
            result = {
                'filename': sample['filename'],
                'original_content': sample['content'],
                'original_emotion': sample['emotion'],
                'original_intensity': sample['intensity'],
                'translation': translation,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            logger.info(f"Completed sample {i}/{total_samples} in {processing_time:.2f}s")
            
            # Log GPU memory usage periodically
            if torch.cuda.is_available() and i % 5 == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f}GB")
        
        return results
    
    def save_results(self, results: List[Dict], filename_suffix: str = "") -> str:
        """Save test results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"meme_translation_results_{timestamp}{filename_suffix}"
        
        # Save as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_filename}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save summary report
        self._generate_summary_report(results, base_filename)
        
        logger.info(f"Results saved to:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV: {csv_path}")
        
        return str(json_path)
    
    def _generate_summary_report(self, results: List[Dict], base_filename: str):
        """Generate a summary report of the test results"""
        report_path = self.output_dir / f"{base_filename}_summary.txt"
        
        total_samples = len(results)
        successful_translations = len([r for r in results if not r['translation'].startswith('ERROR')])
        error_count = total_samples - successful_translations
        
        avg_processing_time = sum(r['processing_time'] for r in results) / total_samples if total_samples > 0 else 0
        
        # Count emotion distribution
        emotion_dist = {}
        for result in results:
            emotion = result['original_emotion']
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # Count intensity distribution
        intensity_dist = {}
        for result in results:
            intensity = result['original_intensity']
            intensity_dist[intensity] = intensity_dist.get(intensity, 0) + 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("LLaVA MEME TRANSLATION TEST SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: LLaVA v1.6 13B\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Successful Translations: {successful_translations}\n")
            f.write(f"Errors: {error_count}\n")
            f.write(f"Success Rate: {successful_translations/total_samples*100:.1f}%\n")
            f.write(f"Average Processing Time: {avg_processing_time:.2f}s\n\n")
            
            f.write("EMOTION DISTRIBUTION:\n")
            for emotion, count in emotion_dist.items():
                f.write(f"  {emotion}: {count} ({count/total_samples*100:.1f}%)\n")
            
            f.write("\nINTENSITY DISTRIBUTION:\n")
            for intensity, count in intensity_dist.items():
                f.write(f"  {intensity}: {count} ({count/total_samples*100:.1f}%)\n")
            
            if error_count > 0:
                f.write("\nERRORS:\n")
                for result in results:
                    if result['translation'].startswith('ERROR'):
                        f.write(f"  {result['filename']}: {result['translation']}\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the test suite"""
    # Configuration
    DATASET_PATH = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/data/labeled_data.csv"
    IMAGES_DIR = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/image/test_meme"
    NUM_SAMPLES = 20
    SEED = 42
    
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Test LLaVA meme translation')
    parser.add_argument('--model', type=str, default=None, help='Path to LLaVA model (local path or HF identifier)')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Path to CSV dataset')
    parser.add_argument('--images', type=str, default=IMAGES_DIR, help='Path to images directory')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed for sample selection')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        logger.error(f"Images directory not found: {args.images}")
        sys.exit(1)
    
    # Initialize test suite
    logger.info("Initializing LLaVA Meme Translation Test Suite...")
    test_suite = MemeTestSuite(args.dataset, args.images, args.output, args.model)
    
    # Select random samples
    test_samples = test_suite.select_random_samples(args.samples, args.seed)
    
    # Log test configuration
    logger.info("="*50)
    logger.info("TEST CONFIGURATION")
    logger.info("="*50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Images Directory: {args.images}")
    logger.info(f"Number of Samples: {len(test_samples)}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Output Directory: {args.output}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*50)
    
    # Run tests
    try:
        start_time = time.time()
        results = test_suite.run_test_batch(test_samples)
        total_time = time.time() - start_time
        
        # Save results
        results_file = test_suite.save_results(results, f"_n{len(test_samples)}")
        
        # Final summary
        successful = len([r for r in results if not r['translation'].startswith('ERROR')])
        logger.info("="*50)
        logger.info("TEST COMPLETED!")
        logger.info("="*50)
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Average Time per Sample: {total_time/len(test_samples):.2f}s")
        logger.info(f"Success Rate: {successful/len(test_samples)*100:.1f}%")
        logger.info(f"Results saved to: {results_file}")
        logger.info("="*50)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
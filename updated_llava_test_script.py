#!/usr/bin/env python3
"""
LLaVA v1.6 13B Cross-Cultural Meme Translation Test Script
Updated with refined creative prompt for US cultural adaptation
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
        
        # Define the UPDATED system prompt with creative US adaptation focus
        self.system_prompt = """ROLE: You are a cross-cultural meme specialist who creates NEW US-appropriate memes that capture the same emotional essence as the original.

INPUT: [Meme image] + [Cultural description] + [Original emotion/intensity]

OUTPUT FORMAT (all in English):

1. EMOTION ANALYSIS:
"Primary Emotion: [joy/anger/sadness/fear/disgust]
Intensity Level: [low/high]
Explanation: [Describe the core emotional trigger and why this intensity fits]"

2. CULTURAL ESSENCE:
"The universal humor/feeling here is: [identify the core relatable human experience]. In US culture, this same feeling appears when [specific US scenario that evokes identical emotion]."

3. IMAGE GENERATION INSTRUCTIONS:

OPTION A - CARTOON STYLE:
"Create a cartoon image using [specific popular US cartoon character like SpongeBob, Tom and Jerry, Bugs Bunny, Homer Simpson, etc.] in [specific pose/action] with [facial expression]. The character should be [specific position/gesture]. Background: [setting description]. Style: [maintain original character's art style]. Mood: [lighting/color scheme that enhances the emotion]. DO NOT include any speech bubbles, text, or dialogue in the image."

OPTION B - PHOTOREALISTIC STYLE:
"Create a photo of [specific person type] in [specific pose/action] showing [expression]. Setting: [detailed environment]. The person should appear [age/appearance]. Camera angle: [perspective]. Lighting: [mood/atmosphere]. DO NOT include any text, signs, or speech elements in the image."

4. US MEME CAPTION:
[Write only the caption text here - no quotation marks, no explanations, no hashtags, no emojis, no additional commentary. Just the pure text that will be overlaid on the image.]

CONSTRAINTS:
- Primary emotion: joy, anger, sadness, fear, or disgust only
- Intensity: low or high only  
- Be creative - don't translate, create NEW content for US culture
- Image instructions must be specific enough for an image generator
- Image instructions must NOT include any dialogue, speech, or text elements
- Caption must be pure text only - absolutely NO emojis, hashtags, quotation marks, or explanations
- Caption section must contain ONLY the text to be overlaid - nothing else"""
    
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
    
    def _validate_output_format(self, translation: str, filename: str) -> Tuple[bool, str]:
        """Validate that the model output has correct format and clean sections"""
        
        # Check for required sections
        required_sections = [
            "1. EMOTION ANALYSIS:",
            "2. CULTURAL ESSENCE:",
            "3. IMAGE GENERATION INSTRUCTIONS:",
            "4. US MEME CAPTION:"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in translation:
                missing_sections.append(section)
        
        if missing_sections:
            return False, f"Missing sections: {missing_sections}"
        
        # Extract caption section
        try:
            caption_part = translation.split("4. US MEME CAPTION:")[-1].strip()
            
            # Check for corrupted caption (mixed with image instructions)
            corruption_indicators = [
                "Create a cartoon image",
                "Create a photo of",
                "Background:",
                "Style:",
                "Mood:",
                "Camera angle:",
                "Lighting:",
                "DO NOT include"
            ]
            
            has_corruption = any(indicator in caption_part for indicator in corruption_indicators)
            if has_corruption:
                return False, "Caption section contains image generation instructions"
            
            # Check for repetitive text (like "moodsky" repetition)
            words = caption_part.lower().split()
            if len(words) > 5:
                # Check if more than 30% of words are repetitive
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                max_repetition = max(word_counts.values()) if word_counts else 0
                if max_repetition > len(words) * 0.3:
                    return False, "Caption contains repetitive/corrupted text"
            
            # Check caption length (too short or too long)
            if len(caption_part.strip()) < 3:
                return False, "Caption too short or empty"
            
            # Check word count for meme suitability (12 words max)
            word_count = len(caption_part.strip().split())
            if word_count > 15:
                return False, f"Caption too long ({word_count} words, max 12 recommended for memes)"
                
        except Exception as e:
            return False, f"Failed to parse caption section: {str(e)}"
        
        return True, ""

    def translate_meme(self, image_path: str, content: str, emotion: str, intensity: str) -> str:
        """Translate a single meme using the model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Construct the input prompt with the new format
            user_prompt = f"""Now analyze this meme:
Image: [PROVIDED]
Description: {content}
Original Emotion: {emotion}
Original Intensity: {intensity}

Please follow the exact format with numbered sections and provide a clean caption in section 4."""
            
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
            
            # Generate response with updated parameters for longer responses
            logger.debug(f"Generating response for image: {image_path}")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1200,  # Increased for detailed image generation instructions
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,  # Added for better quality
                    repetition_penalty=1.1,  # Added to reduce repetitive text
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if "ASSISTANT:" in response:
                generated_text = response.split("ASSISTANT:")[-1].strip()
            else:
                # Fallback: try to extract after the prompt
                prompt_text = self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
                if prompt_text in response:
                    generated_text = response.replace(prompt_text, "").strip()
                else:  
                    generated_text = response.strip()
            
            # Validate output format
            is_valid, error_msg = self._validate_output_format(generated_text, image_path)
            if not is_valid:
                logger.warning(f"Output validation failed for {image_path}: {error_msg}")
                # Still return the output but log the issue
                return f"WARNING - Format issue ({error_msg}):\n\n{generated_text}"
            
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
    
    def _validate_sample(self, sample: Dict, sample_number: int) -> Tuple[bool, str]:
        """Validate that a sample has all required fields and the image exists"""
        required_fields = ['filename', 'content', 'emotion', 'intensity']
        
        # Check for missing required fields
        missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            logger.warning(f"Sample {sample_number} ({sample.get('filename', 'unknown')}): {error_msg}")
            return False, error_msg
        
        # Check if image file exists
        image_path = self.images_dir / sample['filename']
        if not image_path.exists():
            error_msg = f"Image file not found: {image_path}"
            logger.warning(f"Sample {sample_number} ({sample['filename']}): {error_msg}")
            return False, error_msg
        
        # Check if image is readable
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as e:
            error_msg = f"Invalid or corrupted image file: {str(e)}"
            logger.warning(f"Sample {sample_number} ({sample['filename']}): {error_msg}")
            return False, error_msg
        
        # Check if content description is meaningful (not just whitespace or too short)
        if len(sample['content'].strip()) < 10:
            error_msg = f"Content description too short or empty: '{sample['content']}'"
            logger.warning(f"Sample {sample_number} ({sample['filename']}): {error_msg}")
            return False, error_msg
        
        return True, ""

    def run_test_batch(self, test_samples: List[Dict]) -> List[Dict]:
        """Run translation tests on a batch of samples"""
        results = []
        total_samples = len(test_samples)
        valid_samples = 0
        skipped_samples = 0
        
        logger.info(f"Starting batch test with {total_samples} samples...")
        logger.info("Using UPDATED prompt for creative US cultural adaptation")
        logger.info("Validating samples before processing...")
        
        # First pass: validate all samples
        validation_results = []
        for i, sample in enumerate(test_samples, 1):
            is_valid, error_msg = self._validate_sample(sample, i)
            validation_results.append((is_valid, error_msg))
            if is_valid:
                valid_samples += 1
            else:
                skipped_samples += 1
        
        logger.info(f"Validation complete: {valid_samples} valid, {skipped_samples} skipped")
        
        # Second pass: process only valid samples
        for i, (sample, (is_valid, error_msg)) in enumerate(zip(test_samples, validation_results), 1):
            logger.info(f"Processing sample {i}/{total_samples}: {sample.get('filename', 'unknown')}")
            
            if not is_valid:
                # Record the skipped sample with error details
                result = {
                    'filename': sample.get('filename', 'unknown'),
                    'original_content': sample.get('content', 'N/A'),
                    'original_emotion': sample.get('emotion', 'N/A'),
                    'original_intensity': sample.get('intensity', 'N/A'),
                    'translation': f"SKIPPED: {error_msg}",
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'skipped'
                }
                results.append(result)
                logger.info(f"Skipped sample {i}: {error_msg}")
                continue
            
            # Record start time
            start_time = time.time()
            
            # Perform translation (only for valid samples)
            try:
                translation = self.translator.translate_meme(
                    str(self.images_dir / sample['filename']),
                    sample['content'],
                    sample['emotion'],
                    sample['intensity']
                )
                status = 'success'
            except Exception as e:
                translation = f"ERROR: Processing failed - {str(e)}"
                status = 'error'
                logger.error(f"Processing error for {sample['filename']}: {e}")
            
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
                'timestamp': datetime.now().isoformat(),
                'status': status
            }
            
            results.append(result)
            logger.info(f"Completed sample {i}/{total_samples} in {processing_time:.2f}s")
            
            # Preview first successful result for debugging
            if i == 1 and status == 'success':
                logger.info("="*60)
                logger.info("FIRST RESULT PREVIEW:")
                logger.info("="*60)
                logger.info(f"Original: {sample['content'][:100]}...")
                logger.info(f"Translation Preview: {translation[:200]}...")
                logger.info("="*60)
            
            # Log GPU memory usage periodically
            if torch.cuda.is_available() and i % 5 == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f}GB")
        
        return results
    
    def save_results(self, results: List[Dict], filename_suffix: str = "") -> str:
        """Save test results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"meme_translation_results_UPDATED_{timestamp}{filename_suffix}"
        
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
        successful_translations = len([r for r in results if r.get('status') == 'success'])
        skipped_samples = len([r for r in results if r.get('status') == 'skipped'])
        error_samples = len([r for r in results if r.get('status') == 'error'])
        
        # Calculate processing time only for successful samples
        successful_results = [r for r in results if r.get('status') == 'success']
        avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Count emotion distribution (only for processed samples)
        processed_results = [r for r in results if r.get('status') in ['success', 'error']]
        emotion_dist = {}
        for result in processed_results:
            emotion = result['original_emotion']
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # Count intensity distribution (only for processed samples)
        intensity_dist = {}
        for result in processed_results:
            intensity = result['original_intensity']
            intensity_dist[intensity] = intensity_dist.get(intensity, 0) + 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("LLaVA MEME TRANSLATION TEST SUMMARY (UPDATED PROMPT)\n")
            f.write("="*60 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: LLaVA v1.6 13B\n")
            f.write(f"Prompt Version: Creative US Cultural Adaptation (v2)\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Successful Translations: {successful_translations}\n")
            f.write(f"Skipped (Validation Failed): {skipped_samples}\n")
            f.write(f"Processing Errors: {error_samples}\n")
            f.write(f"Success Rate: {successful_translations/total_samples*100:.1f}% (of all samples)\n")
            f.write(f"Processing Success Rate: {successful_translations/(total_samples-skipped_samples)*100:.1f}% (of valid samples)\n" if (total_samples-skipped_samples) > 0 else "")
            f.write(f"Average Processing Time: {avg_processing_time:.2f}s (successful samples only)\n\n")
            
            f.write("PROMPT IMPROVEMENTS:\n")
            f.write("- Removed fallback examples and suggestions\n")
            f.write("- Focus on creative US cultural adaptation vs translation\n")
            f.write("- Specific cartoon character suggestions (SpongeBob, etc.)\n")
            f.write("- Detailed image generation instructions\n")
            f.write("- Original US meme caption creation\n")
            f.write("- Added validation for image-description pairs\n")
            f.write("- Enforced pure-text captions (no emojis/hashtags) for text overlay API\n")
            f.write("- Removed speech/dialogue from image generation instructions\n")
            f.write("- Added output format validation to prevent section mixing\n")
            f.write("- Limited caption length to 12 words max for optimal meme readability\n\n")
            
            if processed_results:
                f.write("EMOTION DISTRIBUTION (Processed Samples):\n")
                for emotion, count in emotion_dist.items():
                    f.write(f"  {emotion}: {count} ({count/len(processed_results)*100:.1f}%)\n")
                
                f.write("\nINTENSITY DISTRIBUTION (Processed Samples):\n")
                for intensity, count in intensity_dist.items():
                    f.write(f"  {intensity}: {count} ({count/len(processed_results)*100:.1f}%)\n")
            
            # Report skipped samples
            if skipped_samples > 0:
                f.write(f"\nSKIPPED SAMPLES ({skipped_samples}):\n")
                skipped_results = [r for r in results if r.get('status') == 'skipped']
                for result in skipped_results:
                    f.write(f"  {result['filename']}: {result['translation'].replace('SKIPPED: ', '')}\n")
            
            # Report processing errors
            if error_samples > 0:
                f.write(f"\nPROCESSING ERRORS ({error_samples}):\n")
                error_results = [r for r in results if r.get('status') == 'error']
                for result in error_results:
                    f.write(f"  {result['filename']}: {result['translation'].replace('ERROR: ', '')}\n")
            
            # Sample successful results
            if successful_results:
                f.write(f"\nSAMPLE SUCCESSFUL RESULTS (First 3):\n")
                for i, result in enumerate(successful_results[:3], 1):
                    f.write(f"\nSAMPLE {i}: {result['filename']}\n")
                    f.write(f"Original: {result['original_content'][:100]}...\n")
                    f.write(f"Translation: {result['translation'][:300]}...\n")
                    f.write("-" * 40 + "\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main function to run the test suite"""
    # Configuration
    DATASET_PATH = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/data/labeled_data.csv"
    IMAGES_DIR = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/image/test_meme"
    NUM_SAMPLES = 10  # Start with fewer samples to test the new prompt
    SEED = 42
    
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Test LLaVA meme translation with UPDATED prompt')
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
    logger.info("Initializing LLaVA Meme Translation Test Suite with UPDATED PROMPT...")
    test_suite = MemeTestSuite(args.dataset, args.images, args.output, args.model)
    
    # Select random samples
    test_samples = test_suite.select_random_samples(args.samples, args.seed)
    
    # Log test configuration
    logger.info("="*60)
    logger.info("TEST CONFIGURATION (UPDATED PROMPT VERSION)")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Images Directory: {args.images}")
    logger.info(f"Number of Samples: {len(test_samples)}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Output Directory: {args.output}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("PROMPT CHANGES:")
    logger.info("- Creative US adaptation vs translation")
    logger.info("- Specific cartoon character suggestions")
    logger.info("- Image generator ready instructions")
    logger.info("- No fallback examples")
    logger.info("="*60)
    
    # Run tests
    try:
        start_time = time.time()
        results = test_suite.run_test_batch(test_samples)
        total_time = time.time() - start_time
        
        # Save results
        results_file = test_suite.save_results(results, f"_n{len(test_samples)}")
        
        # Final summary
        successful = len([r for r in results if r.get('status') == 'success'])
        skipped = len([r for r in results if r.get('status') == 'skipped'])
        errors = len([r for r in results if r.get('status') == 'error'])
        
        logger.info("="*60)
        logger.info("TEST COMPLETED WITH UPDATED PROMPT!")
        logger.info("="*60)
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Total Samples: {len(test_samples)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Skipped (validation failed): {skipped}")
        logger.info(f"Processing errors: {errors}")
        logger.info(f"Overall Success Rate: {successful/len(test_samples)*100:.1f}%")
        if len(test_samples) - skipped > 0:
            logger.info(f"Processing Success Rate: {successful/(len(test_samples)-skipped)*100:.1f}% (of valid samples)")
        logger.info(f"Average Time per Successful Sample: {total_time/successful:.2f}s" if successful > 0 else "No successful samples")
        logger.info(f"Results saved to: {results_file}")
        logger.info("="*60)
        
        # Quick quality check on first successful result
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            first_result = successful_results[0]
            logger.info("QUICK QUALITY CHECK - First Successful Result:")
            logger.info(f"File: {first_result['filename']}")
            logger.info(f"Original Emotion: {first_result['original_emotion']}")
            if any(char in first_result['translation'] for char in ["SpongeBob", "Homer", "Bugs Bunny", "Tom and Jerry"]):
                logger.info("✅ GOOD: Using specific cartoon characters")
            else:
                logger.info("⚠️  May need adjustment: No specific cartoon characters detected")
            
            if "Create a cartoon image using" in first_result['translation']:
                logger.info("✅ GOOD: Following image generation format")
            else:
                logger.info("⚠️  May need adjustment: Image generation format unclear")
            
            # Check for emoji-free captions and proper format
            import re
            
            # Extract just the caption part after "4. US MEME CAPTION:"
            if '4. US MEME CAPTION:' in first_result['translation']:
                caption_full = first_result['translation'].split('4. US MEME CAPTION:')[-1].strip()
                # Remove any quotation marks or extra formatting
                caption_clean = caption_full.strip('"').strip("'").strip()
                
                # Check for format corruption
                corruption_indicators = [
                    "Create a cartoon image",
                    "Create a photo of", 
                    "Background:",
                    "moodsky",
                    "color/color"
                ]
                
                has_corruption = any(indicator in caption_clean for indicator in corruption_indicators)
                
                if has_corruption:
                    logger.info("❌ Caption section contains image generation instructions")
                else:
                    # Check for emojis
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001F900-\U0001F9FF"  # supplemental symbols
                        "]+", flags=re.UNICODE)
                    
                    # Check for hashtags
                    has_hashtags = '#' in caption_clean
                    
                    if emoji_pattern.search(caption_clean):
                        logger.info("⚠️  Caption contains emojis - needs cleaning for text overlay")
                    elif has_hashtags:
                        logger.info("⚠️  Caption contains hashtags - needs cleaning for text overlay")
                    else:
                        logger.info("✅ GOOD: Pure text caption (no emojis/hashtags)")
                    
                    # Check caption length
                    word_count = len(caption_clean.split())
                    if len(caption_clean) < 3:
                        logger.info("⚠️  Caption too short")
                    elif word_count > 12:
                        logger.info(f"⚠️  Caption too long ({word_count} words, recommended max 12 for memes)")
                    else:
                        logger.info(f"✅ GOOD: Caption length appropriate ({word_count} words)")
                        
                    logger.info(f"Caption preview: '{caption_clean[:60]}...'")
                
                # Check for speech patterns in image instructions
                has_speech_in_image = any(phrase in first_result['translation'] for phrase in [
                    'saying', 'says', 'talking', 'speaking', 'dialogue', 'speech bubble'
                ])
                
                if has_speech_in_image:
                    logger.info("⚠️  Image instructions contain speech elements - may confuse generator")
                else:
                    logger.info("✅ GOOD: Image instructions are speech-free")
            else:
                logger.info("⚠️  No caption section found in output")
        else:
            logger.warning("❌ No successful results to analyze")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

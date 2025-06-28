#!/usr/bin/env python3
"""
RunPod Serverless Worker for SmolVLM Vision-Language Model
=========================================================

This worker handles SmolVLM inference on RunPod serverless GPU infrastructure.
Optimized for geotechnical engineering document analysis.

Author: Generated for Geotechnical Engineering
Version: 1.0.0 (RunPod Serverless)
"""

import runpod
import torch
import base64
import io
import json
import time
from typing import Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
MODEL = None
PROCESSOR = None
DEVICE = None

def initialize_model():
    """Initialize SmolVLM model and processor"""
    global MODEL, PROCESSOR, DEVICE
    
    try:
        # Determine device
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {DEVICE}")
        
        # Model configuration
        model_name = "HuggingFaceTB/SmolVLM-Instruct"
        
        # Load processor
        logger.info("Loading SmolVLM processor...")
        PROCESSOR = AutoProcessor.from_pretrained(model_name)
        
        # Load model with appropriate settings
        logger.info("Loading SmolVLM model...")
        if DEVICE == "cuda":
            # Use bfloat16 for GPU
            torch_dtype = torch.bfloat16
            try:
                # Try with FlashAttention2
                MODEL = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    _attn_implementation="flash_attention_2"
                )
                logger.info("SmolVLM loaded with FlashAttention2")
            except Exception as e:
                logger.warning(f"FlashAttention2 not available: {e}")
                # Fallback to eager attention
                MODEL = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    _attn_implementation="eager"
                )
                logger.info("SmolVLM loaded with eager attention")
        else:
            # CPU fallback
            MODEL = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            MODEL = MODEL.to(DEVICE)
            logger.info("SmolVLM loaded on CPU")
        
        logger.info("SmolVLM initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize SmolVLM: {str(e)}")
        return False

def process_image_from_base64(image_data: str) -> Optional[Image.Image]:
    """Convert base64 image data to PIL Image"""
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def run_inference(image: Image.Image, query: str, **kwargs) -> Dict[str, Any]:
    """Run SmolVLM inference on image and query"""
    try:
        start_time = time.time()
        
        # Extract generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.3)
        do_sample = kwargs.get("do_sample", True)
        top_p = kwargs.get("top_p", 0.9)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        # Apply chat template
        input_text = PROCESSOR.apply_chat_template(messages, tokenize=False)
        
        # Process inputs
        inputs = PROCESSOR(
            text=input_text, 
            images=[image], 
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=PROCESSOR.tokenizer.eos_token_id
            )
        
        # Decode response
        response = PROCESSOR.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the input)
        if "Assistant:" in response:
            generated_text = response.split("Assistant:")[-1].strip()
        else:
            # Fallback extraction
            generated_text = response[len(input_text):].strip()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "response": generated_text,
            "processing_time": f"{processing_time:.2f}s",
            "model_info": {
                "model_name": "SmolVLM-Instruct",
                "device": str(DEVICE),
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "top_p": top_p
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processing_time": "0s"
        }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for SmolVLM inference
    
    Expected input format:
    {
        "image_data": "base64_encoded_image_string",
        "query": "text query for the image",
        "max_new_tokens": 512,  # optional
        "temperature": 0.3,     # optional
        "do_sample": True       # optional
    }
    """
    try:
        # Get job input
        job_input = job.get("input", {})
        
        # Validate required inputs
        if "image_data" not in job_input:
            return {
                "error": "Missing required field: image_data"
            }
        
        if "query" not in job_input:
            return {
                "error": "Missing required field: query"
            }
        
        # Extract inputs
        image_data = job_input["image_data"]
        query = job_input["query"]
        
        # Optional parameters
        generation_params = {
            "max_new_tokens": job_input.get("max_new_tokens", 512),
            "temperature": job_input.get("temperature", 0.3),
            "do_sample": job_input.get("do_sample", True),
            "top_p": job_input.get("top_p", 0.9)
        }
        
        # Log job info
        logger.info(f"Processing job {job.get('id', 'unknown')} with query: {query[:50]}...")
        
        # Process image
        image = process_image_from_base64(image_data)
        if image is None:
            return {
                "error": "Failed to process image data"
            }
        
        # Check model initialization
        if MODEL is None or PROCESSOR is None:
            logger.info("Model not initialized, initializing now...")
            if not initialize_model():
                return {
                    "error": "Failed to initialize SmolVLM model"
                }
        
        # Run inference
        result = run_inference(image, query, **generation_params)
        
        if result["success"]:
            logger.info(f"Job {job.get('id', 'unknown')} completed successfully in {result['processing_time']}")
            return {
                "response": result["response"],
                "processing_time": result["processing_time"],
                "model_info": result["model_info"],
                "image_size": f"{image.size[0]}x{image.size[1]}",
                "query_length": len(query)
            }
        else:
            logger.error(f"Job {job.get('id', 'unknown')} failed: {result['error']}")
            return {
                "error": result["error"]
            }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "error": f"Handler error: {str(e)}"
        }

# Health check endpoint
def health_check() -> Dict[str, Any]:
    """Health check for the worker"""
    try:
        return {
            "status": "healthy",
            "model_loaded": MODEL is not None,
            "processor_loaded": PROCESSOR is not None,
            "device": str(DEVICE) if DEVICE else "unknown",
            "torch_cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    # Initialize model on startup
    logger.info("Initializing SmolVLM worker...")
    
    if initialize_model():
        logger.info("SmolVLM worker ready for inference")
    else:
        logger.warning("SmolVLM worker starting without model (will initialize on first request)")
    
    # Start RunPod serverless worker
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True
        }
    )

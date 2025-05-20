# main.py

import base64
import json
import os
import requests
import time
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging
from dotenv import load_dotenv
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
load_dotenv()

# ------------------ MODEL & ANALYZER ------------------

class ObjectDetectionModule:
    def __init__(self,
                 yolo_model_path: str = "yolov9-e.pt",
                 detr_model_name: str = "facebook/detr-resnet-50",
                 use_deyo: bool = False):
        self.yolo_model_path = yolo_model_path
        self.detr_model_name = detr_model_name
        self.use_deyo = use_deyo
        self.yolo_model = None
        self.detr_model = None
        self.detr_processor = None
        self.deyo_model = None

    def _init_yolo(self):
        if self.yolo_model is None:
            from ultralytics import YOLO
            local_model_path = self.yolo_model_path
            self.yolo_model = YOLO(local_model_path)

    def _init_detr(self):
        if self.detr_model is None or self.detr_processor is None:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.detr_processor = DetrImageProcessor.from_pretrained(self.detr_model_name)
            self.detr_model = DetrForObjectDetection.from_pretrained(self.detr_model_name)

    def _init_deyo(self):
        if self.use_deyo and self.deyo_model is None:
            self._init_yolo()
            self._init_detr()
            self.deyo_model = {
                "yolo": self.yolo_model,
                "detr": (self.detr_model, self.detr_processor)
            }

    def detect_objects_yolo(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        self._init_yolo()
        results = self.yolo_model(image_path, conf=conf_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "model": "yolo"
                })
        return detections

    def detect_objects_detr(self, image_path: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        self._init_detr()
        image = Image.open(image_path).convert("RGB")
        inputs = self.detr_processor(images=image, return_tensors="pt")
        outputs = self.detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "box": [x1, y1, x2, y2],
                "class": self.detr_model.config.id2label[label.item()],
                "confidence": score.item(),
                "model": "detr"
            })
        return detections

    def detect_objects_deyo(self, image_path: str) -> List[Dict[str, Any]]:
        self._init_deyo()
        yolo_detections = self.detect_objects_yolo(image_path)
        detr_detections = self.detect_objects_detr(image_path)
        detections = yolo_detections + detr_detections
        for detection in detections:
            detection["model"] = "deyo"
        return detections

    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        if self.use_deyo:
            return self.detect_objects_deyo(image_path)
        else:
            yolo_detections = self.detect_objects_yolo(image_path)
            detr_detections = self.detect_objects_detr(image_path)
            return yolo_detections + detr_detections

class EnhancedFinancialImageAnalyzer:
    def __init__(self,
                 together_api_key: str,
                 vision_llm_model: str = "together/llama-3-70b-vision",
                 reasoning_llm_model: str = "Qwen/Qwen3-235B-A22B-fp8-tput",
                 text_llm_model: str = "nim/meta/llama-3.3-70b-instruct",
                 yolo_model_path: str = "yolov8n.pt",
                 detr_model_name: str = "facebook/detr-resnet-50",
                 use_deyo: bool = False,
                 use_cache: bool = True,
                 cache_dir: str = "cache",
                 parallel_execution: bool = True,
                 max_workers: int = 3):
        self.together_api_key = together_api_key
        self.vision_llm_model = vision_llm_model
        self.reasoning_llm_model = reasoning_llm_model
        self.text_llm_model = text_llm_model
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.object_detector = ObjectDetectionModule(
            yolo_model_path=yolo_model_path,
            detr_model_name=detr_model_name,
            use_deyo=use_deyo
        )
        self.taxonomy = self._create_taxonomy()

    def _create_taxonomy(self) -> Dict[str, List[str]]:
        return {
            "Color Palette": ["Dominant colors", "Brightness", "Warm vs cool tones", "Contrast level"],
            "Layout & Composition": ["Text-to-image ratio", "Left vs right alignment", "Symmetry", "Whitespace usage"],
            "Image Type": ["Image focus type", "Visual format", "Illustration vs photo"],
            "Elements": ["Number of products shown", "Number of people shown", "Design density"],
            "Presence of Text": ["Embedded text present", "Text language", "Font style"],
            "Theme": ["Festival/special occasion logo", "Festival name", "Logo size", "Logo placement"],
            "CTA": ["Call-to-action button present", "CTA placement", "CTA contrast"],
            "Object Detection": ["Objects visible", "Brand logo visible", "Brand logo size"],
            "Character": ["Emotion (if faces shown)", "Gender shown (if people shown)"],
            "Character Role": ["Employment type (if shown)"],
            "Context": ["Environment type", "Location hints"],
            "Offer": ["Offer text present", "Offer type", "Offer text size", "Offer text position"]
        }

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_cache_path(self, image_path: str) -> str:
        image_name = os.path.basename(image_path)
        cache_name = f"{os.path.splitext(image_name)[0]}_analysis.json"
        return os.path.join(self.cache_dir, cache_name)

    def _load_from_cache(self, image_path: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(image_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_to_cache(self, image_path: str, results: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path(image_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass

    def _call_together_api(self, model: str, messages: List[Dict[str, Any]],
                           max_tokens: int = 1024, temperature: float = 0.2) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    retry_delay = min(retry_delay * 2, 60)
                    time.sleep(retry_delay)
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        return {"error": f"API error: {response.status_code} - {response.text}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {"error": f"Request error: {str(e)}"}
        return {"error": "Max retries exceeded"}

    def _extract_json_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in response:
            return {"error": response["error"]}
        try:
            content = response["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except:
                json_match = re.search(r'({[\s\S]*})', content)
                if json_match:
                    return json.loads(json_match.group(1))
                return {"raw_content": content}
        except Exception as e:
            return {"error": f"Error extracting JSON: {str(e)}",
                    "raw_content": content if 'content' in locals() else "No content"}

    def _add_confidence_scores(self, result: Dict[str, Any], confidence: float = 0.9) -> Dict[str, Any]:
        result_keys = [key for key in list(result.keys()) if key not in ["error", "raw_content"]]
        for key in result_keys:
            result[f"{key}_confidence"] = confidence
        return result

    # ----------- PROMPT METHODS (ALL INCLUDED) -----------

    def _color_palette_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """You are a financial image analyzer. Analyze this image and provide ONLY the following specific tags:

1. Dominant colors: List the main colors (Red, Yellow, Blue, Green, Black, White, etc.)
2. Brightness: Categorize as Dark or Light
3. Warm vs cool tones: Categorize as Warm, Cool, or Neutral
4. Contrast level: Categorize as High, Medium, or Low

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the color palette of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _layout_composition_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following layout information:

1. Text-to-image ratio: Estimate as percentage (10%, 30%, 50%, etc.)
2. Left vs right alignment: Categorize as Left, Right, or Center
3. Symmetry: Categorize as Symmetrical or Asymmetrical
4. Whitespace usage: Categorize as Low, Medium, or High

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the layout and composition of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _image_type_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following image type information:

1. Image focus type: Categorize as Product, Lifestyle, or Mixed
2. Visual format: Categorize as Static, Animated, or Video
3. Illustration vs photo: Categorize as Illustration or Photograph

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the image type of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _elements_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following element information:

1. Number of products shown: Count as 1, 2, 3+
2. Number of people shown: Count as 0, 1, 2, 3+
3. Design density: Categorize as Minimal, Medium, or Crowded

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the elements in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _text_presence_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following text information:

1. Embedded text present: Answer Yes or No
2. Text language: Identify as English, Hindi, Marathi, etc.
3. Font style: Categorize as Bold, Serif, Sans-serif, or Script

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the text presence in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _theme_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following theme information:

1. Festival/special occasion logo: Answer Yes or No
2. Festival name: Identify as Diwali, Holi, Independence Day, or None
3. Logo size: Categorize as Small, Medium, or Large
4. Logo placement: Identify as Top Left, Top Right, Bottom Left, Bottom Right, or Center

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the theme of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _cta_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following CTA information:

1. Call-to-action button present: Answer Yes or No
2. CTA placement: Identify as Top, Center, or Bottom
3. CTA contrast: Categorize as High, Medium, or Low

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the CTA in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _object_detection_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following object information:

1. Objects visible: List visible objects (TV, Sofa, Person, Phone, Refrigerator, etc.)
2. Brand logo visible: Answer Yes or No
3. Brand logo size: Categorize as Small, Medium, or Large

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify objects in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _object_verification_prompt(self, image_base64: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detection_text = ""
        for i, det in enumerate(detections):
            detection_text += f"{i + 1}. Class: {det['class']}, Confidence: {det['confidence']:.2f}, Model: {det['model']}\n"
        return [
            {
                "role": "system",
                "content": """You are an expert in computer vision and object detection. 
                Analyze this image and verify the detected objects. For each detection:
                1. Confirm if the object is actually present
                2. Verify if the classification is correct
                3. Suggest any missed objects
                4. Provide confidence in your assessment (0-1)

                Format your response as a JSON with these keys:
                - "verified_detections": List of objects you confirm are present
                - "corrected_detections": List of objects with corrected classifications
                - "missed_objects": List of objects that were missed
                - "confidence": Your overall confidence in this assessment
                """
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Verify these object detections:\n{detection_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _character_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following character information:

1. Emotion (if faces shown): Categorize as Happy, Excited, Neutral, Angry, or None
2. Gender shown (if people shown): Categorize as Male, Female, Both, or Not Applicable

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the characters in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _character_role_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following character role information:

1. Employment type (if shown): Identify as Doctor, Student, Businessperson, or None

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the character roles in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _context_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following context information:

1. Environment type: Categorize as Indoor, Outdoor, Office, or Natural
2. Location hints: Identify as Kitchen, Park, Store, Living Room, or None

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the context of this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _offer_prompt(self, image_base64: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """Analyze this financial marketing image and provide ONLY the following offer information:

1. Offer text present: Answer Yes or No
2. Offer type: Categorize as Discount, Cashback, Freebie, Combo, or None
3. Offer text size: Categorize as Small, Medium, or Large
4. Offer text position: Identify as Top Left, Top Right, Center, or Bottom

Format your response as a JSON with exactly these keys and values from the options provided."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the offers in this financial marketing image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

    def _integration_reasoning_prompt(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": """You are an expert financial marketing analyst. Review the provided analysis results and:
1. Identify any inconsistencies or errors
2. Determine the overall marketing strategy and target audience
3. Assess the effectiveness of the visual communication

Format your response as a structured JSON."""
            },
            {
                "role": "user",
                "content": f"Integrate and reason about these analyses of a financial marketing image:\n{json.dumps(analysis_results, indent=2)}"
            }
        ]

    # ----------- ANALYSIS LOGIC -----------

    def _run_enhanced_object_detection(self, image_path: str) -> Dict[str, Any]:
        detections = self.object_detector.detect_objects(image_path)
        if not detections:
            return self._run_llm_object_detection(self._encode_image(image_path))
        image_base64 = self._encode_image(image_path)
        verification_prompt = self._object_verification_prompt(image_base64, detections)
        verification_response = self._call_together_api(self.vision_llm_model, verification_prompt)
        verification_result = self._extract_json_from_response(verification_response)
        verified_objects = []
        if "verified_detections" in verification_result:
            verified_objects.extend(verification_result["verified_detections"])
        if "corrected_detections" in verification_result:
            verified_objects.extend(verification_result["corrected_detections"])
        if "missed_objects" in verification_result:
            verified_objects.extend(verification_result["missed_objects"])
        if not verified_objects and detections:
            verified_objects = [det["class"] for det in detections]
        objects_str = ", ".join(set(
            obj if isinstance(obj, str) else str(obj)
            for obj in verified_objects
        )) if verified_objects else "None"
        has_logo = any("logo" in det["class"].lower() for det in detections)
        if not has_logo and "brand logo" in str(verification_result).lower():
            has_logo = True
        logo_size = "None"
        if has_logo:
            logo_detections = [det for det in detections if "logo" in det["class"].lower()]
            if logo_detections:
                img = cv2.imread(image_path)
                img_area = img.shape[0] * img.shape[1]
                logo_det = max(
                    logo_detections,
                    key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])
                )
                logo_area = (logo_det["box"][2] - logo_det["box"][0]) * (logo_det["box"][3] - logo_det["box"][1])
                logo_ratio = logo_area / img_area
                if logo_ratio < 0.05:
                    logo_size = "Small"
                elif logo_ratio < 0.15:
                    logo_size = "Medium"
                else:
                    logo_size = "Large"
            else:
                logo_size = "Small"
        result = {
            "Objects visible": objects_str,
            "Brand logo visible": "Yes" if has_logo else "No",
            "Brand logo size": logo_size,
        }
        confidence = verification_result.get("confidence", 0.95)
        return self._add_confidence_scores(result, confidence)

    def _run_llm_object_detection(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._object_detection_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_color_palette_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._color_palette_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_layout_composition_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._layout_composition_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_image_type_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._image_type_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_elements_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._elements_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_text_presence_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._text_presence_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_theme_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._theme_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.8)

    def _run_cta_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._cta_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _run_character_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._character_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_character_role_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._character_role_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.8)

    def _run_context_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._context_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.85)

    def _run_offer_analysis(self, image_base64: str) -> Dict[str, Any]:
        prompt = self._offer_prompt(image_base64)
        response = self._call_together_api(self.vision_llm_model, prompt)
        result = self._extract_json_from_response(response)
        return self._add_confidence_scores(result, 0.9)

    def _get_default_values_for_category(self, category: str) -> Dict[str, Any]:
        # (Default values as in your original code)
        # ... (omitted for brevity, see paste.txt for full dictionary)
        return {}

    def analyze_image(self, image_path: str, categories: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        image_base64 = self._encode_image(image_path)
        all_categories = list(self.taxonomy.keys())
        categories_to_analyze = categories if categories is not None else all_categories
        analysis_results = {}
        if self.parallel_execution:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for category in categories_to_analyze:
                    if category == "Color Palette":
                        futures[category] = executor.submit(self._run_color_palette_analysis, image_base64)
                    elif category == "Layout & Composition":
                        futures[category] = executor.submit(self._run_layout_composition_analysis, image_base64)
                    elif category == "Image Type":
                        futures[category] = executor.submit(self._run_image_type_analysis, image_base64)
                    elif category == "Elements":
                        futures[category] = executor.submit(self._run_elements_analysis, image_base64)
                    elif category == "Presence of Text":
                        futures[category] = executor.submit(self._run_text_presence_analysis, image_base64)
                    elif category == "Theme":
                        futures[category] = executor.submit(self._run_theme_analysis, image_base64)
                    elif category == "CTA":
                        futures[category] = executor.submit(self._run_cta_analysis, image_base64)
                    elif category == "Object Detection":
                        futures[category] = executor.submit(self._run_enhanced_object_detection, image_path)
                    elif category == "Character":
                        futures[category] = executor.submit(self._run_character_analysis, image_base64)
                    elif category == "Character Role":
                        futures[category] = executor.submit(self._run_character_role_analysis, image_base64)
                    elif category == "Context":
                        futures[category] = executor.submit(self._run_context_analysis, image_base64)
                    elif category == "Offer":
                        futures[category] = executor.submit(self._run_offer_analysis, image_base64)
                for category, future in futures.items():
                    try:
                        analysis_results[category] = future.result()
                    except Exception:
                        analysis_results[category] = self._get_default_values_for_category(category)
        else:
            for category in categories_to_analyze:
                try:
                    if category == "Color Palette":
                        analysis_results[category] = self._run_color_palette_analysis(image_base64)
                    elif category == "Layout & Composition":
                        analysis_results[category] = self._run_layout_composition_analysis(image_base64)
                    elif category == "Image Type":
                        analysis_results[category] = self._run_image_type_analysis(image_base64)
                    elif category == "Elements":
                        analysis_results[category] = self._run_elements_analysis(image_base64)
                    elif category == "Presence of Text":
                        analysis_results[category] = self._run_text_presence_analysis(image_base64)
                    elif category == "Theme":
                        analysis_results[category] = self._run_theme_analysis(image_base64)
                    elif category == "CTA":
                        analysis_results[category] = self._run_cta_analysis(image_base64)
                    elif category == "Object Detection":
                        analysis_results[category] = self._run_enhanced_object_detection(image_path)
                    elif category == "Character":
                        analysis_results[category] = self._run_character_analysis(image_base64)
                    elif category == "Character Role":
                        analysis_results[category] = self._run_character_role_analysis(image_base64)
                    elif category == "Context":
                        analysis_results[category] = self._run_context_analysis(image_base64)
                    elif category == "Offer":
                        analysis_results[category] = self._run_offer_analysis(image_base64)
                except Exception:
                    analysis_results[category] = self._get_default_values_for_category(category)
        for category in categories_to_analyze:
            if category not in analysis_results:
                analysis_results[category] = self._get_default_values_for_category(category)
        return analysis_results

    def compare_images(self, image_paths: List[str], categories: Optional[List[str]] = None) -> Dict[str, Any]:
        analyses = {}
        for image_path in image_paths:
            try:
                analyses[image_path] = self.analyze_image(image_path, categories)
            except Exception as e:
                analyses[image_path] = {"error": str(e)}
        comparison_prompt = self._integration_reasoning_prompt(analyses)
        response = self._call_together_api(self.reasoning_llm_model, comparison_prompt, max_tokens=2048)
        comparison = self._extract_json_from_response(response)
        return {
            "individual_analyses": analyses,
            "comparison": comparison
        }

    def visualize_detections(self, image_path: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_detected.jpg"
        detections = self.object_detector.detect_objects(image_path)
        img = cv2.imread(image_path)
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det["box"]]
            class_name = det["class"]
            confidence = det["confidence"]
            model_name = det["model"]
            color = (0, 255, 0)
            if model_name == "detr":
                color = (255, 0, 0)
            elif model_name == "deyo":
                color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(output_path, img)
        return output_path

# ------------------ FASTAPI APP ------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial-image-analyzer")

app = FastAPI(
    title="Enhanced Financial Image Analyzer API",
    description="API for analyzing financial marketing images with YOLO, DETR object detection and LLM analysis",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "ceeabe5b322559bd0fe8a21e18df89860cf31aa2af6e4b31cc60acd299f2d9c0")

analyzer = EnhancedFinancialImageAnalyzer(
    together_api_key=TOGETHER_API_KEY,
    vision_llm_model="Qwen/Qwen2.5-VL-72B-Instruct",
    reasoning_llm_model="Qwen/Qwen3-235B-A22B-fp8-tput",
    text_llm_model="meta-llama/Meta-Llama-Guard-3-8B",
    yolo_model_path="yolov8n.pt",
    detr_model_name="facebook/detr-resnet-50",
    use_deyo=False,
    use_cache=True,
    cache_dir="cache",
    parallel_execution=True,
    max_workers=3
)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Enhanced Financial Image Analyzer API is ready",
        "version": "1.1.0",
        "models": {
            "vision_llm": analyzer.vision_llm_model,
            "reasoning_llm": analyzer.reasoning_llm_model,
            "yolo": analyzer.object_detector.yolo_model_path,
            "detr": analyzer.object_detector.detr_model_name
        }
    }

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    start_time = time.time()
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )
    temp_file_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Processing image: {file.filename}")
        full_results = analyzer.analyze_image(temp_file_path)
        tags_only_results = {}
        for category, values in full_results.items():
            tags_only_results[category] = {k: v for k, v in values.items() if not k.endswith("_confidence")}
        processing_time = time.time() - start_time
        logger.info(f"Image processed in {processing_time:.2f} seconds")
        return JSONResponse(content={
            "results": tags_only_results,
            "filename": file.filename,
            "processing_time_seconds": round(processing_time, 2)
        })
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze-image/focused/")
async def focused_analyze_image(
        file: UploadFile = File(...),
        categories: str = "Color Palette,Layout & Composition,Object Detection"
) -> Dict[str, Any]:
    start_time = time.time()
    category_list = [cat.strip() for cat in categories.split(",")]
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )
    temp_file_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Processing image with focus on: {categories}")
        focused_results = analyzer.analyze_image(temp_file_path, categories=category_list)
        tags_only_results = {}
        for category, values in focused_results.items():
            tags_only_results[category] = {k: v for k, v in values.items() if not k.endswith("_confidence")}
        processing_time = time.time() - start_time
        logger.info(f"Focused analysis completed in {processing_time:.2f} seconds")
        return JSONResponse(content={
            "results": tags_only_results,
            "filename": file.filename,
            "categories_analyzed": category_list,
            "processing_time_seconds": round(processing_time, 2)
        })
    except Exception as e:
        logger.error(f"Error in focused analysis of {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/visualize-detections/")
async def visualize_detections(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPG, PNG, or BMP image."
        )
    timestamp = int(time.time())
    temp_file_path = f"temp_uploads/temp_{timestamp}_{file.filename}"
    output_filename = f"detected_{timestamp}_{file.filename}"
    output_path = f"visualizations/{output_filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Visualizing detections for: {file.filename}")
        visualization_path = analyzer.visualize_detections(temp_file_path, output_path)
        visualization_url = f"/visualizations/{os.path.basename(visualization_path)}"
        return JSONResponse(content={
            "filename": file.filename,
            "visualization_path": visualization_url
        })
    except Exception as e:
        logger.error(f"Error visualizing detections for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error visualizing detections: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/compare-images/")
async def compare_images(files: List[UploadFile] = File(...), categories: str = None) -> Dict[str, Any]:
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least two images to compare"
        )
    category_list = None
    if categories:
        category_list = [cat.strip() for cat in categories.split(",")]
    temp_file_paths = []
    try:
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file format for {file.filename}. Please upload only image files."
                )
            temp_path = f"temp_uploads/temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_path)
        logger.info(f"Comparing {len(files)} images")
        comparison_results = analyzer.compare_images(temp_file_paths, categories=category_list)
        filtered_analyses = {}
        for path, analysis in comparison_results["individual_analyses"].items():
            filename = os.path.basename(path)
            filtered_analyses[filename] = {}
            for category, values in analysis.items():
                if category != "error":
                    filtered_analyses[filename][category] = {
                        k: v for k, v in values.items() if not k.endswith("_confidence")
                    }
                else:
                    filtered_analyses[filename][category] = values
        return JSONResponse(content={
            "individual_analyses": filtered_analyses,
            "comparison": comparison_results["comparison"],
            "filenames": [file.filename for file in files]
        })
    except Exception as e:
        logger.error(f"Error comparing images: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error comparing images: {str(e)}")
    finally:
        for path in temp_file_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

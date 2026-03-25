"""
CAST Evaluation Script for HuatuoGPT-Vision-7B

Evaluates CAST on SLAKE and MIMIC datasets using HuatuoGPT-Vision-7B.

Usage:
    python evaluate_huatuo.py --dataset slake --mode auto \
        --model_path /path/to/HuatuoGPT-Vision-7B \
        --input_path /path/to/test.json --img_root /path/to/dataset \
        --huatuo_code_path /path/to/HuatuoGPT-Vision
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image

from cast import (
    AutoMaskProvider,
    GTMaskProvider,
    NullMaskProvider,
    MedSAM3ProposalGenerator,
    SimpleProposalGenerator,
    ProposalCache,
    ProbCFGLogitsProcessor,
)


# ============== Datasets ==============

class SLAKEDataset:
    """Dataset for SLAKE medical VQA format."""

    def __init__(self, annotation_file, vis_root, lang='en'):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        if lang:
            self.annotation = [ann for ann in annotations if ann.get('q_lang') == lang]
        else:
            self.annotation = annotations
        self.vis_root = vis_root
        self.img_ids = {}
        for ann in self.annotation:
            img_name = ann['img_name']
            if img_name not in self.img_ids:
                self.img_ids[img_name] = len(self.img_ids)

    def __len__(self):
        return len(self.annotation)

    def _load_detection(self, img_folder):
        detection_path = os.path.join(img_folder, 'detection.json')
        if os.path.exists(detection_path):
            with open(detection_path, 'r') as f:
                detections = json.load(f)
            if detections and len(detections) > 0:
                first_det = detections[0]
                obj_name = list(first_det.keys())[0]
                return obj_name, first_det[obj_name]
        return None, None

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_name = ann['img_name']
        img_folder = img_name.split('/')[0]
        img_folder_path = os.path.join(self.vis_root, 'imgs', img_folder)
        image_path = os.path.join(self.vis_root, 'imgs', img_name)
        image = Image.open(image_path).convert('RGB')

        mask_path = os.path.join(img_folder_path, 'mask.png')
        image_mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', image.size, 0)

        obj_name, bbox = self._load_detection(img_folder_path)
        if bbox:
            x, y, w, h = bbox
            bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))
        else:
            bbox_xyxy = (0.0, 0.0, 0.0, 0.0)

        return {
            "image": image, "image_mask": image_mask,
            "question": ann['question'], "answer": ann['answer'],
            "answer_type": ann.get('answer_type', 'OPEN').upper(),
            "image_id": self.img_ids[img_name],
            "image_path": image_path,
            "highlights": [obj_name] if obj_name else [],
            "attribute": "match", "bbox": bbox_xyxy,
        }


class MIMICDataset:
    """Dataset for MIMIC-Ext-VQA format."""

    def __init__(self, annotation_file, vis_root):
        with open(annotation_file, 'r') as f:
            self.annotation = json.load(f)
        self.vis_root = vis_root
        self.img_ids = {}
        for ann in self.annotation:
            img_id = ann['image_ids'][0]
            if img_id not in self.img_ids:
                self.img_ids[img_id] = len(self.img_ids)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann['image_path'])
        image = Image.open(image_path).convert('RGB')

        bbox_dict = ann['template_arguments'].get('bbox', {})
        bbox = tuple(map(float, bbox_dict['0'])) if bbox_dict and '0' in bbox_dict else (0.0, 0.0, 0.0, 0.0)

        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        if bbox != (0.0, 0.0, 0.0, 0.0):
            x1, y1, x2, y2 = map(int, bbox)
            mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
        image_mask = Image.fromarray(mask, mode='L')

        question = ann['question']
        answer = ann['answer']
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        return {
            "image": image, "image_mask": image_mask,
            "question": question, "answer": answer,
            "answer_type": ann.get('answer_type', 'open').upper(),
            "image_id": self.img_ids[ann['image_ids'][0]],
            "image_path": image_path,
            "highlights": list(ann['template_arguments'].get('object', {}).values()),
            "attribute": "match", "bbox": bbox,
        }


# ============== HuatuoGPT utilities ==============

IMAGE_TOKEN_INDEX = -200


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids
                     for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def build_prompt(question):
    return f"<|user|>\n<image>\n{question}\n<|assistant|>\n"


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_image(image, image_processor):
    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    return image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


def load_huatuo_model(model_path, device):
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    print(f"Loading HuatuoGPT-Vision-7B from {model_path}...")
    model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(
        model_path, init_vision_encoder_from_ckpt=True,
        output_loading_info=True, torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(model_path)
    vision_tower.to(dtype=torch.bfloat16, device=device)
    image_processor = vision_tower.image_processor

    model.config.tokenizer_padding_side = 'left'
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, tokenizer, image_processor


# ============== Mask utilities ==============

def roi_mask_to_image_token_mask(roi_mask, patch_grid=24):
    if roi_mask is None or roi_mask.max() == 0:
        return np.zeros(patch_grid * patch_grid, dtype=np.int64)
    roi_tensor = torch.from_numpy(roi_mask).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(roi_tensor, size=(patch_grid, patch_grid), mode='nearest')
    return (resized.squeeze() > 0.5).long().numpy().flatten()


def build_combined_highlight_mask(input_ids_list, image_token_mask, patch_grid=24):
    combined_mask = []
    for token_id in input_ids_list:
        if token_id == IMAGE_TOKEN_INDEX:
            combined_mask.extend(image_token_mask.tolist())
        else:
            combined_mask.append(0)
    return np.array(combined_mask, dtype=np.int64)


# ============== Scoring ==============

def compute_recall(prediction, answer):
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    pred_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 1.0 if not pred_tokens else 0.0
    return len(answer_tokens & pred_tokens) / len(answer_tokens)


def compute_score(prediction, answer, answer_type):
    if answer_type == "CLOSED":
        return 1 if answer.lower() in prediction.lower() else 0
    return compute_recall(prediction, answer)


# ============== Counterfactual Selector ==============

class HuatuoCounterfactualSelector:
    def __init__(self, model, tokenizer, image_processor,
                 occlusion_modes=("blur", "mean"),
                 lambda_area=0.1, mu_consistency=0.1,
                 T_score=32, T_draft=32, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.occlusion_modes = occlusion_modes
        self.lambda_area = lambda_area
        self.mu_consistency = mu_consistency
        self.T_score = T_score
        self.T_draft = T_draft
        self.device = device

    def _prepare_inputs(self, image, question):
        prompt = build_prompt(question)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)
        image_tensor = process_image(image, self.image_processor).unsqueeze(0).to(dtype=torch.bfloat16, device=self.device)
        return input_ids, image_tensor

    def draft_answer(self, image, question, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.T_draft
        input_ids, image_tensor = self._prepare_inputs(image, question)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, images=image_tensor,
                do_sample=False, max_new_tokens=max_new_tokens,
                num_beams=1, use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def loglikelihood(self, image, question, answer, T=None):
        T = T or self.T_score
        prompt = build_prompt(question)
        full_text = prompt + answer
        input_ids = tokenizer_image_token(full_text, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.device)
        image_tensor = process_image(image, self.image_processor).unsqueeze(0).to(dtype=torch.bfloat16, device=self.device)
        prompt_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt')
        prompt_length = prompt_ids.shape[0]
        outputs = self.model(input_ids=input_ids, images=image_tensor)
        logits = outputs.logits
        num_image_placeholders = (input_ids[0] == IMAGE_TOKEN_INDEX).sum().item()
        expanded_prompt_length = prompt_length + num_image_placeholders * (576 - 1)
        shift_logits = logits[:, expanded_prompt_length - 1:-1, :]
        answer_tokens = self.tokenizer(answer, add_special_tokens=False, return_tensors='pt')
        shift_labels = answer_tokens['input_ids'].to(self.device)
        actual_T = min(T, shift_labels.size(1), shift_logits.size(1))
        shift_logits = shift_logits[:, :actual_T, :]
        shift_labels = shift_labels[:, :actual_T]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum().item()

    def occlude(self, image_np, mask, mode="blur"):
        import cv2
        occluded = image_np.copy()
        mask_bool = mask.astype(bool)
        if mode == "mean":
            occluded[mask_bool] = image_np.mean(axis=(0, 1))
        elif mode == "blur":
            blurred = cv2.GaussianBlur(image_np, (51, 51), 30)
            occluded[mask_bool] = blurred[mask_bool]
        else:
            raise ValueError(f"Unknown occlusion mode: {mode}")
        return occluded

    def score(self, image, question, mask, answer=None):
        if answer is None or answer == "":
            answer = self.draft_answer(image, question)
        if answer == "":
            return 0.0
        img_np = np.array(image)
        ll_full = self.loglikelihood(image, question, answer)
        deltas = []
        for mode in self.occlusion_modes:
            img_occ = self.occlude(img_np, mask, mode=mode)
            ll_occ = self.loglikelihood(Image.fromarray(img_occ), question, answer)
            deltas.append(ll_full - ll_occ)
        mean_delta = np.mean(deltas)
        area_penalty = self.lambda_area * mask.mean()
        consistency = -np.var(deltas) if len(deltas) > 1 else 0.0
        return mean_delta - area_penalty + self.mu_consistency * consistency

    def prefilter(self, image, question, proposals, topK=16):
        if len(proposals) <= topK:
            return proposals
        scores = [np.exp(-((m.mean() - 0.25) ** 2) / (2 * 0.15 ** 2)) for m in proposals]
        indices = np.argsort(scores)[::-1][:topK]
        return [proposals[i] for i in indices]


# ============== ARCD Evaluation ==============

@torch.no_grad()
def evaluate_arcd(model, tokenizer, image_processor, eval_dataset, mask_provider,
                  config, device, disable_tqdm=False):
    """CAST evaluation via spatial CFG."""
    patch_grid = config['patch_grid']
    save_info_list = []
    open_scores, closed_scores = [], []

    for i in tqdm(range(len(eval_dataset)), desc="CAST", disable=disable_tqdm):
        try:
            example = eval_dataset[i]
            image = example['image']
            question = example['question']
            answer = example['answer'][0] if isinstance(example['answer'], list) else example['answer']
            answer_type = example.get('answer_type', 'OPEN').upper()
            bbox = example.get('bbox', (0.0, 0.0, 0.0, 0.0))

            meta = {
                'image_id': example.get('image_id', i),
                'image_path': example.get('image_path'),
                'image_mask': example.get('image_mask'),
                'bbox': bbox, 'answer': answer,
            }

            roi_mask = mask_provider.get_mask(image, question, meta)
            image_token_mask = roi_mask_to_image_token_mask(roi_mask, patch_grid)

            prompt = build_prompt(question)
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)
            image_tensor = process_image(image, image_processor).unsqueeze(0).to(dtype=torch.bfloat16, device=device)

            input_ids_list = input_ids[0].tolist()
            combined_mask = build_combined_highlight_mask(input_ids_list, image_token_mask, patch_grid)

            (
                _, position_ids, attention_mask, _, inputs_embeds, _
            ) = model.prepare_inputs_labels_for_multimodal_new(
                input_ids, None,
                torch.ones_like(input_ids, dtype=torch.long),
                None, None, image_tensor
            )

            seq_len = inputs_embeds.shape[1]
            combined_mask_tensor = torch.tensor(
                combined_mask[:seq_len], dtype=torch.long, device=device
            ).unsqueeze(0)

            if combined_mask_tensor.shape[1] < seq_len:
                pad = torch.zeros(1, seq_len - combined_mask_tensor.shape[1],
                                  dtype=torch.long, device=device)
                combined_mask_tensor = torch.cat([combined_mask_tensor, pad], dim=1)
            elif combined_mask_tensor.shape[1] > seq_len:
                combined_mask_tensor = combined_mask_tensor[:, :seq_len]

            cfg_inputs_embeds = inputs_embeds.repeat(2, 1, 1)
            attn_cond = attention_mask.clone()
            attn_uncond = attention_mask.clone()
            attn_uncond[combined_mask_tensor == 1] = 0
            cfg_attention_mask = torch.cat([attn_cond, attn_uncond], dim=0)
            cfg_position_ids = position_ids.repeat(2, 1) if position_ids is not None else None

            from transformers.generation.utils import GenerationMixin
            generated_ids = GenerationMixin.generate(
                model,
                inputs_embeds=cfg_inputs_embeds,
                attention_mask=cfg_attention_mask,
                position_ids=cfg_position_ids,
                max_new_tokens=config['max_new_tokens'],
                num_beams=config['num_beams'],
                do_sample=False, use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[ProbCFGLogitsProcessor(
                    guidance_scale=config['cfg_scale'], use_log=True
                )],
            )

            prediction = tokenizer.batch_decode(
                generated_ids[:1], skip_special_tokens=True
            )[0].strip().strip('.')

            score = compute_score(prediction, answer, answer_type)

            if answer_type == 'CLOSED':
                closed_scores.append(score)
            else:
                open_scores.append(score)

            save_info_list.append({
                'image_id': example.get('image_id', i),
                'question': question, 'answer': answer,
                'answer_type': answer_type, 'prediction': prediction, 'score': score,
                'roi_mask_coverage': float(roi_mask.mean()) if roi_mask is not None else 0.0,
            })

            if i < 5 or (i + 1) % 50 == 0:
                print(f"\n[{i+1}/{len(eval_dataset)}] Q: {question[:60]}...")
                print(f"  GT: {answer} | Pred: {prediction[:60]}...")
                print(f"  Score: {score:.2f} | ROI coverage: {roi_mask.mean():.2%}")

        except Exception as e:
            import traceback
            print(f"\nERROR on sample {i}: {e}")
            traceback.print_exc()
            save_info_list.append({
                'image_id': i, 'question': '', 'answer': '', 'answer_type': 'OPEN',
                'prediction': f'ERROR: {e}', 'score': 0.0, 'roi_mask_coverage': 0.0,
            })
            open_scores.append(0.0)
            torch.cuda.empty_cache()

    return save_info_list, open_scores, closed_scores


# ============== Mask Provider ==============

def create_mask_provider(mode, config, model=None, tokenizer=None, image_processor=None, device='cuda'):
    if mode == 'null':
        print("Using NullMaskProvider (baseline)")
        return NullMaskProvider()
    elif mode == 'gt':
        print("Using GTMaskProvider (ground truth)")
        return GTMaskProvider()
    elif mode == 'auto':
        print("Using AutoMaskProvider (automatic ROI)")

        proposal_gen = None
        try:
            proposal_gen = MedSAM3ProposalGenerator(
                config_path=config['medsam3_config'],
                weights_path=config['medsam3_weights'],
                medsam3_path=config.get('medsam3_path'),
                detection_threshold=0.1, device=device
            )
        except Exception as e:
            print(f"  Warning: MedSAM3 init failed: {e}")
            proposal_gen = SimpleProposalGenerator()

        cache = ProposalCache(cache_dir=config['cache_dir'], prefix=config['cache_prefix'])

        selector = None
        if model is not None and tokenizer is not None:
            selector = HuatuoCounterfactualSelector(
                model=model, tokenizer=tokenizer, image_processor=image_processor,
                occlusion_modes=config.get('occlusion_modes', ['blur', 'mean']),
                lambda_area=config.get('lambda_area', 0.1),
                mu_consistency=config.get('mu_consistency', 0.1),
                T_score=config.get('T_score', 32),
                T_draft=config.get('T_draft', 32),
                device=device
            )

        fallback_gen = SimpleProposalGenerator(n_segments=64, compactness=10.0)

        return AutoMaskProvider(
            proposal_generator=proposal_gen,
            fallback_generator=fallback_gen,
            roi_selector=selector,
            proposal_cache=cache,
            K=config.get('K', 16), device=device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============== Main ==============

def parse_args():
    parser = argparse.ArgumentParser(description='CAST Evaluation with HuatuoGPT-Vision-7B')

    parser.add_argument('--dataset', type=str, required=True, choices=['slake', 'mimic'])
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'gt', 'null'])

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--huatuo_code_path', type=str, default=None,
                        help='Path to HuatuoGPT-Vision code directory')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--output_file', type=str, default=None)

    parser.add_argument('--medsam3_config', type=str, default=None)
    parser.add_argument('--medsam3_weights', type=str, default=None)
    parser.add_argument('--medsam3_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='cache')

    parser.add_argument('--cfg_scale', type=float, default=5.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--skip_proposal_gen', action='store_true')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup HuatuoGPT code path
    huatuo_path = args.huatuo_code_path or os.environ.get('HUATUO_PATH', '')
    if huatuo_path and huatuo_path not in sys.path:
        sys.path.insert(0, huatuo_path)

    cache_prefix = 'slake' if args.dataset == 'slake' else 'mimic'
    cache_dir = os.path.join(args.cache_dir, f'{args.dataset}_proposals')

    config = {
        'num_beams': args.num_beams,
        'max_new_tokens': args.max_new_tokens,
        'cfg_scale': args.cfg_scale,
        'K': args.K,
        'cache_dir': cache_dir,
        'cache_prefix': cache_prefix,
        'medsam3_config': args.medsam3_config,
        'medsam3_weights': args.medsam3_weights,
        'medsam3_path': args.medsam3_path,
        'skip_proposal_gen': args.skip_proposal_gen,
        'patch_grid': 24,
        'num_image_tokens': 576,
        'T_score': 32, 'T_draft': 32,
        'lambda_area': 0.1, 'mu_consistency': 0.1,
        'occlusion_modes': ['blur', 'mean'],
    }

    device = torch.device("cuda:0")

    print(f"\n{'=' * 60}")
    print(f"HuatuoGPT-Vision-7B | CAST | Dataset: {args.dataset.upper()} | Mode: {args.mode.upper()}")
    print(f"{'=' * 60}\n")

    # Load dataset
    if args.dataset == 'slake':
        eval_dataset = SLAKEDataset(annotation_file=args.input_path, vis_root=args.img_root)
    else:
        eval_dataset = MIMICDataset(annotation_file=args.input_path, vis_root=args.img_root)

    if args.debug:
        class SubsetDataset:
            def __init__(self, dataset, n=20):
                self.dataset = dataset
                self.n = min(n, len(dataset))
            def __len__(self):
                return self.n
            def __getitem__(self, i):
                return self.dataset[i]
        eval_dataset = SubsetDataset(eval_dataset, 20)
        print("DEBUG MODE: Using only 20 samples")

    print(f"Loaded {len(eval_dataset)} examples")

    # Load model
    model, tokenizer, image_processor = load_huatuo_model(args.model_path, device)

    # Create mask provider
    mask_provider = create_mask_provider(
        args.mode, config, model, tokenizer, image_processor, str(device)
    )

    # Evaluate
    save_info_list, open_scores, closed_scores = evaluate_arcd(
        model, tokenizer, image_processor, eval_dataset, mask_provider,
        config, device, disable_tqdm=args.no_tqdm
    )

    # Results
    open_recall = np.mean(open_scores) * 100 if open_scores else 0.0
    closed_acc = np.mean(closed_scores) * 100 if closed_scores else 0.0
    total_samples = len(open_scores) + len(closed_scores)
    overall = (sum(open_scores) + sum(closed_scores)) / total_samples * 100 if total_samples > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Results - {args.dataset.upper()} / HuatuoGPT / CAST / {args.mode.upper()}")
    print(f"{'=' * 60}")
    print(f"Open Questions (Recall):   {open_recall:.2f}% ({len(open_scores)} samples)")
    print(f"Closed Questions (Acc):    {closed_acc:.2f}% ({len(closed_scores)} samples)")
    print(f"Overall:                   {overall:.2f}% ({total_samples} samples)")
    print(f"{'=' * 60}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = args.output_file or os.path.join(
        args.save_dir, f'output_{args.dataset}_huatuo_cast_{args.mode}.json'
    )

    with open(save_path, 'w') as f:
        json.dump({
            'model': 'HuatuoGPT-Vision-7B',
            'method': 'cast', 'mode': args.mode,
            'dataset': args.dataset,
            'metrics': {
                'open_recall': open_recall, 'open_count': len(open_scores),
                'closed_accuracy': closed_acc, 'closed_count': len(closed_scores),
                'overall': overall, 'total_count': total_samples,
            },
            'config': {
                'cfg_scale': config['cfg_scale'],
                'num_beams': config['num_beams'],
            },
            'sample_info': save_info_list,
        }, f, indent=2)

    print(f"Results saved to: {save_path}")


if __name__ == '__main__':
    main()

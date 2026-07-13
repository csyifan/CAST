"""
CAST Evaluation Script for MIMIC Dataset (Phi-3.5V-Med)

Evaluates CAST (Contrastive Anatomical Spatial-Temporal decoding)
on the MIMIC-CXR-JPG medical VQA dataset.

Usage:
    python evaluate_mimic.py --mode auto
    python evaluate_mimic.py --mode gt      # Use GT masks (upper bound)
    python evaluate_mimic.py --mode null    # No mask (baseline)
"""

import os
import json
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

from cast import (
    AutoMaskProvider,
    GTMaskProvider,
    NullMaskProvider,
    CounterfactualSelector,
    MedSAM3ProposalGenerator,
    SimpleProposalGenerator,
    ProposalCache,
    ProbCFGLogitsProcessor,
)


# ============== MIMIC Dataset ==============

class MIMICDataset:
    """Dataset for MIMIC-Ext-VQA format with region information."""

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

    def _create_bbox_mask(self, image_size, bbox):
        w, h = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255
        return Image.fromarray(mask, mode='L')

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann['image_path'])
        image = Image.open(image_path).convert('RGB')

        bbox_dict = ann['template_arguments'].get('bbox', {})
        if bbox_dict and '0' in bbox_dict:
            bbox = tuple(map(float, bbox_dict['0']))
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

        image_mask = self._create_bbox_mask(image.size, bbox)
        mask_value = 255 if bbox != (0.0, 0.0, 0.0, 0.0) else 0

        question = ann['question']
        answer = ann['answer']
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        answer_type = ann.get('answer_type', 'open').upper()

        obj_dict = ann['template_arguments'].get('object', {})
        cat_dict = ann['template_arguments'].get('category', {})
        highlights = list(obj_dict.values()) + list(cat_dict.values())
        if not highlights:
            highlights = [answer] if answer else []

        return {
            "image": image, "image_mask": image_mask,
            "question": question, "answer": answer,
            "answer_type": answer_type,
            "image_id": self.img_ids[ann['image_ids'][0]],
            "image_path": image_path, "highlights": highlights,
            "attribute": "match", "mask_value": mask_value,
            "bbox": bbox,
        }


# ============== Utilities ==============

def insert_separator(X, sep_list):
    if len(X) > len(sep_list):
        sep_list.append([])
    return [ele for sublist in zip(X, sep_list) for ele in sublist]


def generate_bbox_mask(mask_crop, n_rows, n_cols, block_dims=(12, 12)):
    """Generate 1D mask sequence for CAST attention."""
    if mask_crop.dim() != 3:
        raise ValueError(f"Expected [B, H, W], got {mask_crop.dim()} dims")

    B, H, W = mask_crop.shape
    device = mask_crop.device
    block_h, block_w = block_dims

    target_h_local = n_rows * block_h
    target_w_local = n_cols * block_w

    local_mask = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(target_h_local, target_w_local), mode='nearest'
    ).squeeze(1)

    newline_local = torch.zeros((B, target_h_local, 1), device=device, dtype=torch.long)
    local_with_newlines = torch.cat([local_mask.long(), newline_local], dim=2)
    local_seq = local_with_newlines.view(B, -1)

    global_mask = F.interpolate(
        mask_crop.unsqueeze(1).float(),
        size=(block_h, block_w), mode='nearest'
    ).squeeze(1)

    newline_global = torch.zeros((B, block_h, 1), device=device, dtype=torch.long)
    global_with_newlines = torch.cat([global_mask.long(), newline_global], dim=2)
    global_seq = global_with_newlines.view(B, -1)

    separator = torch.zeros((B, 1), device=device, dtype=torch.long)
    return torch.cat([local_seq, separator, global_seq], dim=1)


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
    else:
        return compute_recall(prediction, answer)


# ============== Mask Provider ==============

def create_mask_provider(mode, config, model=None, processor=None, device='cuda'):
    if mode == 'null':
        print("Using NullMaskProvider (baseline, no ROI guidance)")
        return NullMaskProvider()
    elif mode == 'gt':
        print("Using GTMaskProvider (ground truth masks)")
        return GTMaskProvider()
    elif mode == 'auto':
        print("Using AutoMaskProvider (automatic ROI discovery)")

        proposal_gen = None
        try:
            proposal_gen = MedSAM3ProposalGenerator(
                config_path=config['medsam3_config'],
                weights_path=config['medsam3_weights'],
                medsam3_path=config.get('medsam3_path'),
                detection_threshold=0.3, device=device
            )
            print("  - MedSAM3 proposal generator initialized")
        except Exception as e:
            print(f"  - Warning: Could not initialize MedSAM3: {e}")
            proposal_gen = SimpleProposalGenerator()

        cache = ProposalCache(cache_dir=config['cache_dir'], prefix='mimic')
        print(f"  - Proposal cache: {config['cache_dir']}")

        selector = None
        if model is not None and processor is not None:
            selector = CounterfactualSelector(
                model=model, processor=processor,
                occlusion_modes=config.get('occlusion_modes', ['blur', 'mean']),
                lambda_area=config.get('lambda_area', 0.1),
                mu_consistency=config.get('mu_consistency', 0.1),
                T_score=config.get('T_score', 32),
                T_draft=config.get('T_draft', 32),
                device=device
            )
            print("  - Counterfactual selector initialized")

        return AutoMaskProvider(
            proposal_generator=proposal_gen,
            roi_selector=selector,
            proposal_cache=cache,
            K=config.get('K', 16), device=device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============== Evaluation ==============

@torch.no_grad()
def evaluate(model, processor, eval_dataset, mask_provider, config, device, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    model.eval()

    sqrt_num = int(np.sqrt(config['num_crops']))
    n_rows, n_cols = sqrt_num, sqrt_num

    save_info_list = []
    open_scores = []
    closed_scores = []

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    total = len(eval_dataset)
    indices = list(range(rank, total, world_size))

    print(f"\n[Rank {rank}/{world_size}] Evaluating {len(indices)}/{total} examples...")

    for i in tqdm(indices, disable=(rank != 0) or disable_tqdm):
        example = eval_dataset[i]
        image = example['image']
        question = example['question']
        answer = example['answer'] if isinstance(example['answer'], str) else example['answer'][0]
        answer_type = example.get('answer_type', 'OPEN').upper()
        bbox = example.get('bbox', (0.0, 0.0, 0.0, 0.0))

        meta = {
            'image_id': example.get('image_id', i),
            'image_path': example.get('image_path'),
            'image_mask': example.get('image_mask'),
            'bbox': bbox, 'answer': answer
        }

        roi_mask = mask_provider.get_mask(image, question, meta)
        mask_value = mask_provider.get_mask_value(roi_mask, bbox)

        highlights = example.get('highlights', [])
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=prompt, images=[image],
            images_mask=[roi_mask], bboxes=[bbox],
            mask_values=[mask_value],
            attribute=[example.get('attribute', 'match')],
            qs_highlighted_parts=highlights,
            return_tensors='pt'
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        bbox_attention_mask_crop = inputs['bbox_attention_mask']
        bbox_attention_mask = generate_bbox_mask(bbox_attention_mask_crop, n_rows, n_cols).tolist()
        highlight_attention_mask = inputs['highlight_attention_mask']

        prompt_chunks = inputs['prompt_chunks']
        image_ids_pad = inputs['image_ids_pad']

        input_ids = []
        combined_highlight_mask = []

        for tokens, mask in zip(
            insert_separator(prompt_chunks, image_ids_pad),
            insert_separator(highlight_attention_mask, bbox_attention_mask)
        ):
            input_ids.extend(tokens)
            combined_highlight_mask.extend(mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = (input_ids > -1000000).to(torch.long)
        combined_highlight_mask = torch.tensor(
            combined_highlight_mask, dtype=torch.long, device=device
        ).unsqueeze(0)

        hl_mask_ = combined_highlight_mask.clone().float()
        hl_mask_[combined_highlight_mask == 1] = config['perturb_weight']
        hl_mask_[combined_highlight_mask == 0] = config['attn_weight']

        cfg_batched_input = input_ids.repeat(2, 1)
        pixel_values = inputs['pixel_values'].repeat(2, 1, 1, 1, 1)
        image_sizes = inputs['image_sizes'].repeat(2, 1)

        processors = [ProbCFGLogitsProcessor(guidance_scale=config['cfg_scale'], use_log=True)]

        generated_ids = model.generate(
            input_ids=cfg_batched_input,
            pixel_values=pixel_values,
            attention_mask=torch.cat([attention_mask, hl_mask_], dim=0),
            image_sizes=image_sizes,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=config['max_new_tokens'],
            num_beams=config['num_beams'],
            logits_processor=processors,
        )

        prediction = processor.batch_decode(
            generated_ids[:1, input_ids.size(1):],
            skip_special_tokens=True
        )[0].strip().strip('.')

        score = compute_score(prediction, answer, answer_type)

        if answer_type == 'CLOSED':
            closed_scores.append(score)
        else:
            open_scores.append(score)

        save_info_list.append({
            'image_id': example.get('image_id', i),
            'question': question, 'answer': answer,
            'answer_type': answer_type, 'prediction': prediction,
            'score': score,
            'roi_mask_coverage': float(roi_mask.mean()) if roi_mask is not None else 0.0
        })

        if i < 5 or (i + 1) % 50 == 0:
            print(f"\n[{i+1}/{len(eval_dataset)}] Q: {question[:60]}...")
            print(f"  GT: {answer} | Pred: {prediction[:60]}...")
            print(f"  Score: {score:.2f} | ROI coverage: {roi_mask.mean():.2%}")

    return save_info_list, open_scores, closed_scores


def parse_args():
    parser = argparse.ArgumentParser(description='CAST Evaluation on MIMIC (Phi-3.5V-Med)')

    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'gt', 'null'])

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--output_file', type=str, default=None)

    parser.add_argument('--medsam3_config', type=str, default=None)
    parser.add_argument('--medsam3_weights', type=str, default=None)
    parser.add_argument('--medsam3_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='cache/mimic_proposals')

    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_crops', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--attn_weight', type=float, default=3.0)
    parser.add_argument('--perturb_weight', type=float, default=0.01)
    parser.add_argument('--K', type=int, default=16)

    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        'num_crops': args.num_crops,
        'num_beams': args.num_beams,
        'max_new_tokens': args.max_new_tokens,
        'cfg_scale': args.cfg_scale,
        'attn_weight': args.attn_weight,
        'perturb_weight': args.perturb_weight,
        'K': args.K,
        'cache_dir': args.cache_dir,
        'medsam3_config': args.medsam3_config,
        'medsam3_weights': args.medsam3_weights,
        'medsam3_path': args.medsam3_path,
        'T_score': 32, 'T_draft': 32,
        'lambda_area': 0.1, 'mu_consistency': 0.1,
        'occlusion_modes': ['blur', 'mean'],
    }

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')

    device = torch.device(f"cuda:{local_rank}")

    print(f"\n{'=' * 60}")
    print(f"CAST Evaluation on MIMIC")
    print(f"Mode: {args.mode.upper()}")
    print(f"{'=' * 60}\n")

    # Load dataset
    eval_dataset = MIMICDataset(
        annotation_file=args.input_path, vis_root=args.img_root
    )

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
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, num_crops=args.num_crops
    )
    processor.tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    )

    if args.lora_path:
        model.load_adapter(args.lora_path)
        print(f"Loaded LoRA from: {args.lora_path}")

    model = model.to(device)
    model.eval()

    mask_provider = create_mask_provider(args.mode, config, model, processor, device)

    save_info_list, open_scores, closed_scores = evaluate(
        model, processor, eval_dataset, mask_provider,
        config, device, disable_tqdm=args.no_tqdm
    )

    save_info_list = gather_object(save_info_list)
    open_scores = gather_object(open_scores)
    closed_scores = gather_object(closed_scores)

    if rank == 0:
        open_recall = np.mean(open_scores) * 100 if open_scores else 0.0
        closed_acc = np.mean(closed_scores) * 100 if closed_scores else 0.0
        total_samples = len(open_scores) + len(closed_scores)
        overall = (sum(open_scores) + sum(closed_scores)) / total_samples * 100 if total_samples > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(f"Results - MIMIC / {args.mode.upper()}")
        print(f"{'=' * 60}")
        print(f"Open Questions (Recall):   {open_recall:.2f}% ({len(open_scores)} samples)")
        print(f"Closed Questions (Acc):    {closed_acc:.2f}% ({len(closed_scores)} samples)")
        print(f"Overall:                   {overall:.2f}% ({total_samples} samples)")
        print(f"{'=' * 60}\n")

        os.makedirs(args.save_dir, exist_ok=True)
        save_path = args.output_file or os.path.join(args.save_dir, f'output_mimic_cast_{args.mode}.json')

        with open(save_path, 'w') as f:
            json.dump({
                'mode': args.mode,
                'metrics': {
                    'open_recall': open_recall, 'open_count': len(open_scores),
                    'closed_accuracy': closed_acc, 'closed_count': len(closed_scores),
                    'overall': overall, 'total_count': total_samples
                },
                'config': {
                    'cfg_scale': config['cfg_scale'],
                    'num_beams': config['num_beams'],
                    'attn_weight': config['attn_weight'],
                    'perturb_weight': config['perturb_weight'],
                    'K': config['K'],
                },
                'sample_info': save_info_list,
            }, f, indent=2)

        print(f"Results saved to: {save_path}")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

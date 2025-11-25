import os
import argparse
import torch
from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="체크포인트 step (예: 3000)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="029.American_Crow_result",
        help="체크포인트(.safetensors)들이 들어있는 디렉토리",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="029.American_Crow_result_emb_checkpoint",
        help="체크포인트 파일 공통 prefix (prefix_{step}.safetensors 형식)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ti_bird_outputs",
        help="이미지 출력 디렉토리",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=25,
        help="Diffusion num_inference_steps",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 체크포인트 경로 만들기
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{args.step}.safetensors")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")

    print(f"[INFO] 사용 step: {args.step}")
    print(f"[INFO] 체크포인트 로드: {ckpt_path}")

    # 파이프라인 로드
    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16).to("cuda")

    # 텍스트 인버전 토큰
    ti_tokens = [f"<{args.checkpoint_dir.split('/')[-1].split('.')[-1]}>"]
    instance_token = f"<{args.checkpoint_dir.split('/')[-1].split('.')[-1]}>"
    print(f"[INFO] >>>> {args.checkpoint_dir.split('/')[-1].split('.')[-1]}")

    # 체크포인트 로드
    state_dict = load_file(ckpt_path)

    # CLIP(TI)
    pipe.load_textual_inversion(state_dict["clip_l"], token=ti_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

    # T5(TI) 있으면 로드
    if "t5" in state_dict:
        pipe.load_textual_inversion(state_dict["t5"], token=ti_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

    prompt = f"a photo of {instance_token} bird"

    print(f"[INFO] 이미지 생성 중... (step={args.step})")
    image = pipe(prompt=prompt, num_inference_steps=args.num_inference_steps).images[0]

    out_name = f"ti_bird_{args.step}_v2.png"
    out_path = os.path.join(args.output_dir, out_name)
    image.save(out_path)

    print(f"[INFO] 저장 완료: {out_path}")


if __name__ == "__main__":
    main()

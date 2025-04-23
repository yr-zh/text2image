from flask import Flask, request, Response, jsonify, abort
import requests
import os
import json
import logging
from typing import Dict
import re
# from strategy_extension import Painter
from strategy_workflow import Painter


app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='#日志 %(asctime)s - %(levelname)s - %(message)s')
logging.info('===== startup.py 启动成功 ======')

app_ready = False
logging.info('===== 加载模型中 ======')

model_repo_dir = "/root"

MODEL_TYP = os.getenv("MODEL_TYP", 'flux')
UNET = os.getenv("UNET", 'flux1-schnell-Q2_K.gguf')
CLIP = os.getenv("CLIP", "t5-v1_1-xxl-encoder-Q3_K_S.gguf")

src_lang = os.getenv('SRC_LANG', 'zh')
to_lang = os.getenv("TO_LANG", "en")

if MODEL_TYP == 'sdxl':
    painter = Painter(
        model_typ=MODEL_TYP, 
        unet=f"sd_xl_turbo_1.0_fp16.safetensors")
    
elif MODEL_TYP == 'flux':
    painter = Painter(
        model_typ=MODEL_TYP,
        unet=UNET, 
        clip=CLIP)

logging.info('===== 模型加载完成 ======')
app_ready = True


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/ready", methods=["GET"]) 
def ready() -> Dict[str, bool]:  
    if app_ready:  
        return {"ready": True}  
    else:  
        abort(503)


@app.route("/api/v1/images/text2img", methods=["POST"])
def chat():
    body = request.get_json()
    prompt = body["prompt"]
    size = body["size"]
    logging.info(f"提示词长度: {len(prompt)}字")
    logging.info(f"图片大小: {size}")
    base64image = painter.generate(src_prompt=prompt, src_lang_code=src_lang)
    response = {
        "data": [
            {
                "content": base64image
            }
        ]
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run("0.0.0.0", 8083)

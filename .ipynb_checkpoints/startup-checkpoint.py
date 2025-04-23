from flask import Flask, request, Response, jsonify, abort
import requests
import os
import json
import logging
from typing import Dict
import re
from utils import Painter

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='#日志 %(asctime)s - %(levelname)s - %(message)s')
logging.info('===== startup.py 启动成功 ======')

app_ready = False
logging.info('===== 加载模型中 ======')

# model_repo_dir = "/mnt/data/zhaoyiru/text2image"
model_repo_dir = "/root"

model = 'flux' # sd3 or kolors or flux
src_lang = os.getenv('SRC_LANG', 'zh')
to_lang = os.getenv("TO_LANG", "en")

if model == 'sd3':
    painter = Painter(
        text2image_model_id=f"{model_repo_dir}/stable-diffusion-3-medium-diffusers", 
        model=model)
elif model == 'kolors':
    painter = Painter(
        text2image_model_id=f"{model_repo_dir}/Kolors-diffusers/",  
        model=model)
else:
    painter = Painter(
        text2image_model_id=f"{model_repo_dir}/FLUX.1-dev/",  
        model=model)

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
    app.run("0.0.0.0", 80)
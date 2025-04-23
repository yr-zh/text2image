import os
import json
import torch
from diffusers import StableDiffusion3Pipeline, KolorsPipeline, FluxPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from PIL import Image
import io
import base64
import gc 

# from comfy_script.runtime import *
# load()
# from comfy_script.runtime.nodes import *
torch.cuda.set_per_process_memory_fraction(0.09, 7)

from comfy_script.runtime.real import *
load()
from comfy_script.runtime.real.nodes import *
from torchvision.transforms import ToPILImage

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 512))
# lang_map = {
#     "zh": "Chinese",
#     "en": "English"
# }
lang_map = {
    "zh": "中文",
    "en": "英语",
    "it": "意大利语",
    "el": "希腊语",
    "de": "德语",
    "pl": "波兰语",
    "nl": "荷兰语",
    "tr": "土耳其语",
    "th": "泰语",
    "pt": "葡萄牙语",
    "vi": "越南语",
    "es": "西班牙语",
    "ja": "日语",
    "id": "印尼语",
    "he": "希伯来语",
    "ru": "俄语",
    "fr": "法语",
    "fa": "波斯语",
    "ar": "阿拉伯语"
}
try:
    model_info = json.loads(os.getenv("LEADERBOARD_MODELHUB_KEY2INFO"))
    token = model_info["token"]
    base_url = model_info["entrypoint"]
    modelId = model_info["model_key2info"]["llm"]["modelId"]
except:
    token = "2ba4b9416b194d129c42c8becb41a9ba"
    base_url = "http://modelhub.4pd.io/learnware/models/openai/4pd/api/v1"
    modelId = "public/qwen2-72b-instruct-awq@main"
print("token: ", token)
client = OpenAI(base_url=base_url, api_key=token)


class Translator:
#     PROMPT_TEMPLATE = """Below is a Text-to-Image prompt written in {src_lang}, please translate it into {tgt_lang}:
# \"{src_prompt}\".You must only return the {tgt_lang} prompt. You must not generate more than 77 tokens. """
    PROMPT_TEMPLATE = """你是一个专业的翻译官。可以流利的将{src_lang}翻译为 {tgt_lang}，现在你需要为我翻译一段文生图prompt，\
        prompt中可能包含主体、环境、动作以及图片风格等信息，你需要在保证整体翻译质量的前提下，尽量准确的翻译。\
        注意：只需给出翻译结果，不需要解释也不需要说任何多余的话，只需给出翻译结果，且翻译结果不超过77个token。原prompt： \"{src_prompt}\"，现在请你直接给出结果，翻译结果："""
    
    TEMPLATE_V2 = """You are a Text-to-Image AI assistant，now you will get a prompt for image generation. You need to generate a series of short sentences that can improve the quality of image generation based on it, such as: 8k, RAW photo, best quality, masterpiece. \
    You can use () to increase the weight, [] to decrease the weight. \
    You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
    but make sure there is a correlation between the input and output.You must not generate more than 77 tokens. \n\
        ### Input: {raw_prompt}\n### Output:"""
    
    def __init__(self, model_id=modelId):
        self.model_id = model_id
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_id
        # ).to(self.device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        ...

    def t(self, src_prompt, src_lang_code, tgt_lang_code):
        trans_prompt = self.PROMPT_TEMPLATE.format(
            src_lang=lang_map[src_lang_code],
            tgt_lang=lang_map[tgt_lang_code],
            src_prompt=src_prompt
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": trans_prompt}
        ]
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 128
        }
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def bp(self, src_prompt): # prompt has translated to English
        trans_prompt = self.TEMPLATE_V2.format(
            raw_prompt=src_prompt
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": trans_prompt}
        ]
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 128
        }
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
        

class BeautifulPropmt:
    def __init__(self, model_path='../pai-bloom-1b1-text2prompt-sd-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).eval().cuda()
        self.TEMPLATE_V2 = 'Converts a simple image description into a prompt. \
        Prompts are formatted as multiple related tags separated by commas, plus you can use () to increase the weight, [] to decrease the weight, \
        or use a number to specify the weight. You should add appropriate words to make the images described in the prompt more aesthetically pleasing, \
        but make sure there is a correlation between the input and output.\n\
        ### Input: {raw_prompt}\n### Output:'

    def generate(self, raw_prompt):
        input = self.TEMPLATE_V2.format(raw_prompt=raw_prompt)
        input_ids = self.tokenizer.encode(input, return_tensors='pt').cuda()
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=77,
            do_sample=True,
            temperature=1.1,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1)

        prompts = self.tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
        prompts = [p.strip() for p in prompts]
        return prompts[0]


class Painter:
    def __init__(self, model_typ, unet='flux1-schnell-Q2_K.gguf', clip='t5-v1_1-xxl-encoder-Q3_K_S.gguf'):
        self.translator = Translator()
        self.to_pil = ToPILImage()
        self.model_typ = model_typ

        if self.model_typ == 'sdxl':
            with Workflow():
                self.model, self.clip, self.vae = CheckpointLoaderSimple(unet)

        elif self.model_typ == 'flux':
            with Workflow():
                # self.model = UnetLoaderGGUF(unet)
                self.model = UnetLoaderGGUF('flux.1-lite-8B-alpha.gguf')
                self.clip = DualCLIPLoaderGGUF('clip_l.safetensors', clip, 'flux')


    def _translate(self, src_prompt, src_lang_code, tgt_lang_code):
        tgt_prompt = self.translator.t(src_prompt, src_lang_code, tgt_lang_code)
        return tgt_prompt
    
    def _beauty(self, src_prompt):
        tgt_prompt = self.translator.bp(src_prompt)
        return tgt_prompt
    
    def _image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode()

    def generate(self, src_prompt, src_lang_code="zh", tgt_lang_code="en", height=1024, width=1024):
        tgt_prompt = self._translate(src_prompt, src_lang_code, tgt_lang_code)
        # print("src prompt: ", src_prompt)
        node_cache = {}
        cache = lambda node: node_cache.setdefault(node, {})

        if self.model_typ == 'sdxl':
            with Workflow():
                negative_prompt = "ow quality, worst quality:1.4, bad_prompt:0.8, monochrome:1.1, greyscale"
                conditioning = CLIPTextEncode(tgt_prompt, self.clip)
                conditioning2 = CLIPTextEncode(tgt_prompt, self.clip)
                latent = EmptyLatentImage(IMAGE_SIZE, IMAGE_SIZE, 1)
                latent = KSampler(self.model, 156680208700286, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
                image_tensor = VAEDecode(latent, self.vae)[0].permute(2,0,1)
                image = self.to_pil(image_tensor)

        elif self.model_typ == 'flux':
            with Workflow(cache=cache):
                conditioning = CLIPTextEncodeFlux(self.clip, tgt_prompt, tgt_prompt, 3.5)
                del node_cache['CLIPTextEncodeFlux']

                conditioning2 = ConditioningZeroOut(conditioning)
                del node_cache['ConditioningZeroOut']

                latent = EmptyLatentImage(IMAGE_SIZE, IMAGE_SIZE, 1)
                del node_cache['EmptyLatentImage']
                # print("****************")
                # print(*node_cache.keys())

                latent = KSampler(self.model, 425948886010830, 20, 1, 'euler', 'simple', conditioning, conditioning2, latent, 1)
                vae = VAELoader('flux_ae.safetensors')
                image_tensor = VAEDecode(latent, vae)[0].permute(2,0,1)
                image = self.to_pil(image_tensor)
            torch.cuda.empty_cache()
        del cache
        # gc.collect()
        return self._image_to_base64(image)
    
if __name__ == '__main__':

    LANGUAGE_FROM = 'zh'
    LANGUAGE_TO = 'en'
    
    prompt = "一只狗和两只猫"
    translate_prompt = f"""The following is a prompt written in {LANGUAGE_FROM}, please translate it into {LANGUAGE_TO}:
    {prompt}
    You can only return an English prompt.
    """
    # painter = Painter('sdxl', 'sd_xl_turbo_1.0_fp16.safetensors')
    painter = Painter('flux', unet='flux1-schnell-Q2_K.gguf', clip='t5-v1_1-xxl-encoder-Q3_K_S.gguf')
    base64image = painter.generate(src_prompt=prompt, src_lang_code=LANGUAGE_FROM)

    print(base64image)
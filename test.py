import requests
import json

# 目标URL
url = 'http://127.0.0.1:8083/api/v1/images/text2img'

data = {
    "prompt": "一只红色的猫。",
    "size": "1024*1024"
}
for i in range(10):
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("请求成功！")
        dic = response.json()
        # print(dic)
    else:
        print(f"请求失败，状态码：{response.status_code}")
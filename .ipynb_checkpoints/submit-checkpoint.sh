curl --location --request POST 'http://contest.4pd.io:8080/submit' \
--header 'Authorization: Bearer 37eb7e2a16ba5ac5ba77bc8c89bb94b9' \
--form-string 'benchmark=text2img' \
--form-string 'contributors=zhuangqinyu,lvjinglan,chengxi' \
--form-string 'description=basemodel:qwen2_7-sd3' \
--form-string 'product_avaliable=1' \
--form-string 'source_code=https://gitlab.4pd.io/zhuangqinyu' \
--form 'config_file=@"/mnt/data/zhuangqinyu/repos/text2image/config.yaml"'

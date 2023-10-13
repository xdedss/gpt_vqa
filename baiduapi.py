# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
import os
from hashlib import md5

# Set your own appid/appkey.
appid = os.environ.get('BAIDU_API_ID', '')
appkey = os.environ.get('BAIDU_API_KEY', '')


def translate(text, from_lang, to_lang):
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    query = text

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign, 'action': 1}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # {
    #     "from": "en",
    #     "to": "zh",  
    #     "trans_result": [
    #         {
    #             "src": "Hello World! This is 1st paragraph.",
    #             "dst": "你好，世界！这是第一段。"
    #         },
    #         {
    #             "src": "This is 2nd paragraph.",
    #             "dst": "这是第2段。"
    #         }
    #     ]
    # }

    trans_result = result['trans_result']
    join_result = '\n'.join([d['dst'] for d in trans_result])
    return join_result



if __name__ == '__main__':
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = 'en'
    to_lang =  'zh'

    query = 'Where are the objects located on the soil segment?\nThis is 2nd paragraph.'

    print(translate(query, from_lang, to_lang))

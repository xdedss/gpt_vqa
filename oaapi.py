
import openai
import os
import time
import logging
import base64
from PIL import Image
from io import BytesIO

logger = logging.getLogger('oaapi')

openai.api_key = os.environ.get('OPENAI_API_KEY')
openai.base_url = os.environ.get('OPENAI_API_BASE')

def completion_once(prompt: str, base_url: str, model_name: str, stop=None) -> str:

    logger.info(f"using model {model_name}")

    client = openai.OpenAI(
        base_url=base_url,
        api_key="token-abc123",
    )

    for i in range(20):
        try:
            
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=1024,
                stop=stop,
            )
            text = completion.choices[0].text
            return (text)

        except openai.RateLimitError:
            # we should try again later
            wait_time = min(2 ** i, 30)
            logger.warn(f'openai rate limit, retry #{i+1} after {wait_time} s')
            time.sleep(wait_time)


def ask_once_with_image(system, user_question: str, image_path: str, model_name: str = 'gpt-4o-mini', image_resize=(512, 512), api_key=None, base_url=None) -> str:

    logger.info(f"using model {model_name}")

    # Open and resize the image
    with Image.open(image_path) as img:
        img = img.resize(image_resize)

        # Convert the resized image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if (api_key is not None and base_url is not None):
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = openai

    for i in range(20):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": user_question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{image_data}",
                                "detail": "low"
                            },
                        },
                    ]},
                ],
                temperature=0.7,
            )
            text = response.choices[0].message.content
            return (text)

        except openai.RateLimitError:
            # we should try again later
            wait_time = min(2 ** i, 30)
            logger.warn(f'openai rate limit, retry #{i+1} after {wait_time} s')
            time.sleep(wait_time)

def ask_once(system, user_question: str, model_name: str = 'gpt-3.5-turbo', api_key=None, base_url=None) -> str:

    logger.info(f"using model {model_name}")

    if (api_key is not None and base_url is not None):
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = openai

    for i in range(20):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_question},
                ],
                temperature=0.7,
            )
            text = response.choices[0].message.content
            return (text)

        except openai.RateLimitError:
            # we should try again later
            wait_time = min(2 ** i, 30)
            logger.warn(f'openai rate limit, retry #{i+1} after {wait_time} s')
            time.sleep(wait_time)


def ask_once_image(system, image_path: str, user_question: str, model_name: str = 'gpt-3.5-turbo', api_key=None, base_url=None) -> str:

    logger.info(f"using model {model_name}")

    if (api_key is not None and base_url is not None):
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    else:
        client = openai

    for i in range(20):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_question},
                ],
                temperature=0.7,
            )
            text = response.choices[0].message.content
            return (text)

        except openai.RateLimitError:
            # we should try again later
            wait_time = min(2 ** i, 30)
            logger.warn(f'openai rate limit, retry #{i+1} after {wait_time} s')
            time.sleep(wait_time)

def generate_desc():
#     system = '''You will act as 3 experts in remote sensing recognition. Each one is asked to describe the content of an image.

# You may chose any of following aspects if available:
# Object identification: Describe the objects present in the image and provide specific details about them. For example, if the objects are buildings, you can mention their architectural style, size, or purpose.
# Spatial arrangement: Discuss how the objects are positioned in relation to each other. Are they clustered together, evenly distributed, or arranged in a specific pattern?
# Object characteristics: Provide information about the physical attributes of the objects. For instance, if the objects are vehicles, you can mention their types, colors, or sizes.
# Object interactions: Describe any observable relationships or interactions between the objects. For instance, if there are vehicles on a road, you can mention their relative positions or movements.
# Object density: Discuss the density or concentration of the objects in different areas of the image. Are there densely populated regions or sparse areas?
# Object context: Consider the surrounding environment of the objects. Are they located in an urban setting, a rural landscape, or a natural habitat?
# Object significance: Highlight any notable or significant objects in the image. This could include landmarks, historical sites, or unique features that stand out.

# Note that only one sentence is allowed for each expert. Directly tell the answer and do not write any unnecessary explainations.

# '''
    system = '''You will act as 3 experts in remote sensing recognition. Each one is asked to describe the content of an image.

The user will give you informations. You may chose any of following aspects if available:
Object identification, Object position, Spatial arrangement, Object Count, Object characteristics, Object interactions, Object density, Object context, Object significance.

Note that only one sentence is generated for each expert.

'''
    system = '''You are a helpful assistant.'''

    user = '''here are objects detected on an remote sensing image, in x, y, w, h format.
The size of the image is 512 by 512. The origin point is at the top left corner.

[354, 425, 110, 104] KC-135
[318, 564, 109, 106] KC-135
[377, 274, 122, 119] E-3
[283, 704, 110, 95] KC-135
[423, 143, 109, 106] KC-135
[457, 2, 109, 108] KC-135
'''
#     user = '''here are objects detected in the image, in x, y, w, h format.
# The size of the image is 512 by 512. The origin point is at the top left corner.

# [263, 85, 109, 63] TU-22
# [275, 528, 107, 76] TU-22
# [276, 680, 106, 90] TU-22
# [262, 219, 110, 91] TU-22
# [275, 389, 109, 63] TU-22
# '''
    objs = '''
[263, 85, 109, 63] TU-22
[275, 528, 107, 76] TU-22
[276, 680, 106, 90] TU-22
[262, 219, 110, 91] TU-22
[275, 389, 109, 63] TU-22
'''
    objs = '''
[354, 425, 110, 104] KC-135
[318, 564, 109, 106] KC-135
[377, 274, 122, 119] E-3
[283, 704, 110, 95] KC-135
[423, 143, 109, 106] KC-135
[457, 2, 109, 108] KC-135
'''
    user = f'''here are objects detected on an remote sensing image, in x, y, w, h format.
The size of the image is 512 by 512. The origin point is at the top left corner.
{objs.strip()}
you will act as 10 remote sensing experts that are asked to describe the objects in the image with one sentence.
You may chose any of following aspects if available:
Object identification, Object position, Spatial arrangement, Object Count, Object characteristics, Object interactions, Object density, Object context, Object significance.
When mentioning the position of objects, use descriptive references instead of raw coordinates.'''
    res = ask_once(system, user)
    print(res)

if __name__ == '__main__':
    generate_desc()
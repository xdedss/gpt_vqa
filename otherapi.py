import anthropic
import base64
import os
import requests
from PIL import Image
from io import BytesIO

def send_claude(system, user, image_path, model="claude-3-haiku-20240307", image_resize=(400, 400)):
    # Resize image to 200x200 and encode as base64
    with Image.open(image_path) as img:
        img = img.resize(image_resize)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_data = base64.standard_b64encode(buffered.getvalue()).decode("utf-8")

    # Define the media type for JPEG
    image_media_type = "image/jpeg"

    # Initialize the Claude client (requires your Anthropic API key)
    client = anthropic.Anthropic(
        api_key=os.getenv('CLAUDE_API_KEY'),
        base_url=os.getenv('CLAUDE_API_BASE'),
    )

    try:
        # Send request to Claude
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": user,
                        }
                    ],
                }
            ],
        )
        print(message)
        return message.content[0].text  # Extract and return text response
    except Exception as e:
        return f"An error occurred: {e}"


def send_gemini(system, user, image_path, model_name="gemini-1.5-flash", image_resize=(768, 768)):
    # Open and resize the image
    with Image.open(image_path) as img:
        img = img.resize(image_resize)

        # Convert the resized image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    api_key = os.getenv('GEMINI_API_KEY')
    api_base = os.getenv('GEMINI_API_BASE')

    # Prepare the payload for the API request
    payload = {
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [{
            "parts": [
                {"text": user},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ]
        }]
    }

    # API endpoint URL
    url = f"{api_base.rstrip('/')}/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Set headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send POST request
    response = requests.post(url, headers=headers, json=payload)

    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Return the response data
    return response.json()['candidates'][0]['content']['parts'][0]['text']


if __name__ == '__main__':
    # print(send_claude('You are a helpful assistant.', 'How many buildings are there?', r'D:\LZR\Downloads\documents\RescuNet\val-org-img\10781.jpg'))
    # print(send_gemini('You are a helpful assistant. Answer with a number or yes/no', 'How many slightly damaged buildings are there?', r'D:\LZR\Downloads\documents\RescuNet\val-org-img\10781.jpg'))
    import oaapi
    print(oaapi.ask_once_with_image('You are a helpful assistant.', 'Describe the image', r'D:\LZR\Downloads\documents\RescuNet\val-org-img\10781.jpg'))


from openai import OpenAI

base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def simple_chat(messages, use_stream=False):
    response = client.chat.completions.create(
        model="internlm2_5-7b-chat",
        messages=messages,
        stream=use_stream,
        max_tokens=1024,
        temperature=0.95,
        presence_penalty=1.2,
        top_p=0.95,
    )
    if response:
        if use_stream:
            complete_message = ""
            return response
            # 控制台输出
            for chunk in response:
                delta_content = chunk.choices[0].delta.content
                complete_message += delta_content
                print(delta_content, end='')  # 输出当前块的内容
            print("\nComplete message:", complete_message)
        else:
            # print(response)
            return response
    else:
        print("Error:", response.status_code)
        return False


# if __name__ == "__main__":
    # simple_chat([
    #     {
    #         "role": "system",
    #         "content": "你是一个无知、爱发脾气、甩锅的助手，不会解决任何用户问题",
    #     },
    #     {
    #         "role": "user",
    #         "content": "你是谁"
    #     }
    # ], use_stream=True)
import openai
import pandas as pd
import time
import logging

# Your API key
openai.api_key = "sk-"
openai.base_url = "https://api.gpt.ge/v1/"
openai.default_headers = {"x-foo": "true"}
#assign experts for target



target_role_map = {
    "萝卜快跑": "自动驾驶与交通分析专家",
    "预制菜": "食品科学家",
    "iphone15": "科技分析师",
    "不婚主义": "社会学家",
    "裸辞": "职业顾问"
}


def load_csv_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python')
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read {file_path} with any of the encodings: {', '.join(encodings)}")

def get_completion_with_role(role, instruction, content):
    max_retries = 100000
    for i in range(max_retries):
        # try:
        messages = [
            {"role": "system", "content": f"你是一名 {role}."},
            {"role": "user", "content": f"{instruction}\n{content}"}
        ]
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
 
def get_completion(prompt):
    max_retries = 100000

    for i in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout,openai.error.APIConnectionError,openai.error.InvalidRequestError,openai.error.AuthenticationError):
            if i < max_retries - 1:
                time.sleep(2)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"
            
def linguist_analysis(tweet):
    instruction = "下面是由​​​符号“[SEP]”拼接的一段对话，请准确、简洁地解释句子中的语言要素以及这些要素如何影响意义，包括语法结构、时态和词形变化、虚拟语音、修辞手法、词汇选择等。无需做其他任何事情。"
    return get_completion_with_role("语言学家", instruction, tweet)

def expert_analysis(tweet, target):
    role = target_role_map.get(target, "专家")
    instruction = f"下面是由​​​符号“[SEP]”拼接的一段对话，请准确、简洁地解释引言中包含的关键元素，例如人物、事件、党派、宗教等。还要解释他们与{target}（如果存在）的关系。无需执行其他操作。"
    return get_completion_with_role(role, instruction, tweet)

def user_analysis(tweet):
    instruction = "下面是由​​​符号“[SEP]”拼接的一段对话，请分析下面的句子，重点关注内容、主题标签、网络俚语和口语、情绪基调、隐含含义等。无需做其他事情。"
    return get_completion_with_role("资深社交媒体用户", instruction, tweet)

def stance_analysis(tweet, ling_response, expert_response, user_response, target, stance):
    role = target_role_map.get(target, "专家")
    return get_completion(f"'''{tweet}'''\n <<<{ling_response}>>>\n [[[{expert_response}]]]\n---{user_response}---\n\
                        ''' ''' 所包围的句子，是由​​​符号“[SEP]”拼接的一段对话。\
                          您认为''' ''' 所包围的对话的最后一句背后的态度是{target}的{stance}。 \
                          <<< >>> 内的内容表示语言分析。[[[ ]]] 内的内容表示{role}的分析。 \
                          --- ---所包围的内容代表对资深社交媒体用户的分析。从中找出最能支持您观点的三条证据，并为您的观点进行论证。")


def final_judgement(tweet, favor_response, against_response, target):
    judgement=get_completion(f"确定该句子是否支持或反对{target}，或者与{target}无关。\n \
                             句子：{tweet}\n根据以下论点判断：\n\
                                支持态度的论据: {favor_response}\n\
                                    反对该态度的论点： {against_response}\n\
                                            选择:\n A: 反对\nB: 支持\nC: 中立\n 约束：只回答上面最准确的选项，不用回答其他的，你只需要输出 A/B/C")
    print(judgement)
    return judgement
    

def add_predictions_sequential(data, filname):
    results = []  # To store the results
    
    for index, row in data.iterrows():
        label = row['label']
        target = row['Target']
        tweet = row['Tweet']

        # Step 1: Linguist analysis
        ling_response = linguist_analysis(tweet)

        # Step 2: Expert analysis
        expert_response = expert_analysis(tweet, target)

        # Step 3: Heavy social media user analysis
        user_response = user_analysis(tweet)

        # Step 4: Debate
        favor_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "支持")
        against_response = stance_analysis(tweet, ling_response, expert_response, user_response, target, "反对")

        # Step 5: Final judgement
        final_response = final_judgement(tweet, favor_response, against_response, target)

        # Construct the result for the current tweet and add it to the results list
        result = {
            'label': label,
            'Tweet': tweet,
            'Target': target,
            'Linguist Analysis': ling_response,
            'Expert Analysis': expert_response,
            'User Analysis': user_response,
            'In Favor': favor_response,
            'Against': against_response,
            'Final Judgement': final_response
        }
        results.append(result)

        resss = pd.DataFrame(results)
        file_path = f"/home/yangyi/project/yangyi/data/WCSD/COLA/res/{filname}.csv"
        resss.to_csv(file_path, index=False)

    # for idx, res in enumerate(results):
    #     for key, value in res.items():
    #         data.at[idx, key] = value


datas = ['不婚主义', '萝卜快跑', '裸辞', '预制菜', 'iphone15']

data = load_csv_data(f"/home/yangyi/project/yangyi/data/WCSD/COLA/data/{datas[0]}.csv")

add_predictions_sequential(data, datas[0])

# nohup python /home/yangyi/project/yangyi/COLA/cola.py > /home/yangyi/project/yangyi/COLA/log/cola/不婚主义.log 2>&1 &

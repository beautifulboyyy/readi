import sys, os
import datetime
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import json
from sentence_transformers import util
from utils.freebase_func import *
import openai
import time


def readjson(file_name):
    """
    读取JSON文件
    
    输入:
        file_name: JSON文件路径
        
    输出:
        data: 从JSON文件中加载的数据
    """
    with open(file_name, encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl(file_path):
    """
    读取JSONL文件（每行一个JSON对象）
    
    输入:
        file_path: JSONL文件路径
        
    输出:
        data: 包含所有JSON对象的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def savejson(file_name, new_data):
    """
    将数据保存为JSON文件
    
    输入:
        file_name: 要保存的JSON文件路径
        new_data: 要保存的数据
        
    输出:
        无
    """
    with open(file_name, mode='w',encoding='utf-8') as fp:
        json.dump(new_data, fp, indent=4, sort_keys=False,ensure_ascii=False)


def get_openai_embedding(input_message, openai_api_keys):
    """
    获取OpenAI嵌入向量
    
    输入:
        input_message: 输入文本
        openai_api_keys: OpenAI API密钥
        
    输出:
        response['data']: 嵌入向量数据
    """
    ok = False
    openai.api_key = openai_api_keys
    openai.api_base = "https://use.52apikey.cn/v1"
    while not ok:
        try:
            response = openai.Embedding.create(engine="text-embedding-ada-002",
                                               input=input_message)
            ok = True
        except Exception as e:
            print(e)
            print('stuck in here get_openai_embedding')

    return response['data']


def run_llm(prompt, temperature, max_tokens, openai_api_keys, engine="gpt-3.5-turbo"):
    """
    运行LLM模型
    
    输入:
        prompt: 提示文本
        temperature: 温度参数
        max_tokens: 最大token数
        openai_api_keys: OpenAI API密钥
        engine: 模型引擎，默认为"gpt-3.5-turbo"
        
    输出:
        result: LLM生成的结果
    """
    messages = []
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0
    result = [{"content": ""}]

    openai.api_key = openai_api_keys
    openai.api_base = "https://use.52apikey.cn/v1"
    while(f <= 5):
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )

            result = response["choices"][0]['message']['content'].strip()
            if len(result) == 0:
                f += 1
                continue
            break

        except Exception as e:
            print("error: ", e)
            print("openai error, retry")

            f += 1
            # trim the input according to the model's max token limit
            if "gpt-4" in engine:
                messages[-1] = {"role":"user","content": prompt[:32767]}
                time.sleep(10)
            else:
                messages[-1] = {"role":"user","content": prompt[:16384]}
                time.sleep(5)

    return result


def get_ent_one_hop_rel(entity_id, pre_relations=[], pre_head=-1, literal=False):
    """
    获取实体的一跳关系
    
    输入:
        entity_id: 实体ID
        pre_relations: 先前的关系列表
        pre_head: 先前的头实体标识
        literal: 是否包含文字类型
        
    输出:
        total_relations: 实体的所有一跳关系列表
    """
    if entity_id.startswith("m.") == False and entity_id.startswith("g.")==False:
        return []
    
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = table_result_to_list(execute_sparql(sparql_relations_extract_head))

    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = table_result_to_list(execute_sparql(sparql_relations_extract_tail))

    if head_relations!=[]:
      head_relations=head_relations['relation']
      # each relation starts from ns:
      head_relations = [x.replace("http://rdf.freebase.com/ns/", "") for x in head_relations if 'http://rdf.freebase.com/ns' in x]

    # tail_relations = table_result_to_list(execute_sparql(sparql_relations_extract_tail))
    if tail_relations != []:
      tail_relations = tail_relations['relation']
      tail_relations = [x.replace("http://rdf.freebase.com/ns/", "") for x in tail_relations if 'http://rdf.freebase.com/ns' in x]

    remove_unnecessary_rel = True
    if remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    if len(pre_relations) != 0 and pre_head != -1:
        head_relations = [rel for rel in pre_relations if not pre_head and rel not in head_relations]
        tail_relations = [rel for rel in pre_relations if pre_head and rel not in tail_relations]

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = list(set(head_relations+tail_relations))
    total_relations.sort()  # make sure the orders in prompts are always the same

    return total_relations


def entity_search(entity, relation, head=True):
    """
    根据实体和关系搜索相关实体
    
    输入:
        entity: 实体
        relation: 关系
        head: 是否作为头实体搜索，默认为True
        
    输出:
        new_entity: 搜索到的相关实体列表
    """
    if head:
        if "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in relation:
            tail_entities_extract = sparql_tail_entities_extract_with_type% (entity)
            entities = table_result_to_list(execute_sparql(tail_entities_extract))
        else:
            tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
            entities = table_result_to_list(execute_sparql(tail_entities_extract))
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = table_result_to_list(execute_sparql(head_entities_extract))

    if entities != []:
        entities = entities['tailEntity']
        entities = [x.replace("http://rdf.freebase.com/ns/", "") for x in entities if 'http://rdf.freebase.com/ns' in x]

    new_entity = [entity for entity in entities if entity.startswith("m.")]

    return new_entity


def path_to_string(path: list) -> str:
    """
    将路径列表转换为字符串
    
    输入:
        path: 路径列表，每个元素为(h, r, t)三元组
        
    输出:
        result: 路径字符串
    """
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"

    return result.strip()

def string_to_path(path_string):
    """
    将路径字符串转换为列表
    
    输入:
        path_string: 路径字符串
        
    输出:
        result: 路径列表
    """
    result = []
    if type(path_string) == list:
        path_string = path_string[0]
        
    path_array = path_string.split("->")
    path_array = path_array[1:]
    for lines in path_array:
        result.append(lines.strip())
    return result


def similar_search_list(question, relation_list, options):
    """
    使用OpenAI嵌入向量根据问题过滤相似关系
    在大规模知识图谱中，relation_list可能非常大并使LLM混淆，因此进行此操作
    
    可以使用缓存嵌入向量进行优化
    建议为知识图谱中的所有关系和所有问题使用缓存嵌入向量以节省token
    出于政策原因，我们不公开嵌入向量。您可以使用get_openai_embedding获取嵌入向量，
    在data/openai_embeddings中创建缓存文件并修改此函数
    
    输入:
        question: 问题文本
        relation_list: 关系列表
        options: 提供openai_api_keys的配置选项
        
    输出:
        sorted_relation_list: 与问题相似的关系列表
    """
    question_embedding = get_openai_embedding(question, options.openai_api_keys)[0]['embedding']
    relation_embeddings = []
    # read cache file (if applicable)
    cache_file_path = "data/openai_embeddings/fb_relation_embed.json"
    if os.path.exists(cache_file_path):
        r_embedding_map = readjson(cache_file_path)
    else:
        r_embedding_map = {}

    for rel in relation_list:
        if rel in r_embedding_map.keys():
            relation_embeddings.append(r_embedding_map[rel])
        else:
            relation_embeddings.append(get_openai_embedding(rel, options.openai_api_keys)[0]['embedding'])
             
    # calculate similarity between the question and relations
    similarities = util.pytorch_cos_sim(question_embedding, relation_embeddings)

    # sort relation list by similarity 
    sorted_relations = [(relation, score) for relation, score in zip(relation_list, similarities.tolist()[0])]
    sorted_relations = sorted(sorted_relations, key=lambda x: x[1], reverse=True)

    sorted_relation_list = [relation[0] for relation in sorted_relations]
    return sorted_relation_list

def get_timestamp():
    """
    获取当前时间戳
    
    输入:
        无
        
    输出:
        timestamp字符串，格式为"月_日_时_分"
    """
    now = datetime.datetime.now()
    return now.strftime(r"%m_%d_%H_%M")


def jsonl_to_json(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为JSON文件
    
    输入:
        jsonl_file_path: JSONL文件路径
        json_file_path: 要保存的JSON文件路径
        
    输出:
        无
    """
    data = []
    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False,ensure_ascii=False)

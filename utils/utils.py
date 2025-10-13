import sys, os
import datetime
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import json
from sentence_transformers import util
from utils.freebase_func import *
import time
from dashscope import Generation, TextEmbedding
import dashscope
import threading

# 用于线程安全的API密钥轮询
_api_key_list = []
_current_key_index = 0
_key_lock = threading.Lock()


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
    获取Qwen嵌入向量
    
    输入:
        input_message: 输入文本
        openai_api_keys: Qwen API密钥
        
    输出:
        embeddings: 嵌入向量数据
    """
    # 支持多个API密钥
    global _api_key_list, _current_key_index
    if ',' in openai_api_keys:
        _api_key_list = openai_api_keys.split(',')
        # 去除空格
        _api_key_list = [key.strip() for key in _api_key_list]
    else:
        _api_key_list = [openai_api_keys]
    
    # 尝试使用Qwen的文本嵌入API
    for i in range(len(_api_key_list) * 2):  # 多次尝试
        with _key_lock:
            api_key = _api_key_list[_current_key_index]
            _current_key_index = (_current_key_index + 1) % len(_api_key_list)
        
        try:
            dashscope.api_key = api_key
            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v1,
                input=input_message
            )
            if response.status_code == 200:
                # 适配原来的返回格式
                embeddings = []
                for item in response.output['embeddings']:
                    embeddings.append({'embedding': item['embedding']})
                return embeddings
            else:
                print(f"Error in get_openai_embedding: {response.message}")
        except Exception as e:
            print(e)
            print('stuck in here get_openai_embedding')
    
    # 如果所有尝试都失败了，抛出异常
    raise Exception("Failed to get embeddings from Qwen API")


def run_llm(prompt, temperature, max_tokens, openai_api_keys, engine="qwen-plus"):
    """
    运行Qwen模型
    
    输入:
        prompt: 提示文本
        temperature: 温度参数
        max_tokens: 最大token数
        openai_api_keys: API密钥
        engine: 模型引擎，默认为"qwen-plus"
        
    输出:
        result: LLM生成的结果
    """
    # 支持多个API密钥
    global _api_key_list, _current_key_index
    if ',' in openai_api_keys:
        _api_key_list = openai_api_keys.split(',')
        # 去除空格
        _api_key_list = [key.strip() for key in _api_key_list]
    else:
        _api_key_list = [openai_api_keys]
    
    # Qwen模型映射
    qwen_model_mapping = {
        "gpt-3.5-turbo": "qwen-plus",
        "gpt-4-turbo": "qwen-max",
        "gpt-4-0613": "qwen-max",
        "gpt-4o": "qwen-max",
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max"
    }
    
    # 获取对应的Qwen模型
    if engine in qwen_model_mapping:
        qwen_model = qwen_model_mapping[engine]
    else:
        # 如果没有映射关系，默认使用qwen-plus
        qwen_model = "qwen-plus"
    
    messages = [
        {"role":"user","content":prompt}
    ]
    f = 0
    
    # 尝试使用Qwen API
    for i in range(len(_api_key_list) * 2):  # 多次尝试
        with _key_lock:
            api_key = _api_key_list[_current_key_index]
            _current_key_index = (_current_key_index + 1) % len(_api_key_list)
        
        try:
            dashscope.api_key = api_key
            response = Generation.call(
                model=qwen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format='message'
            )
            
            if response.status_code == 200:
                result = response.output.choices[0].message.content.strip()
                if len(result) > 0:
                    return result
            else:
                print(f"Qwen API error: {response.message}")
                
        except Exception as e:
            print("Qwen error: ", e)
            print("qwen error, retry")
    
    # 如果所有尝试都失败了，抛出异常
    raise Exception("Failed to get response from Qwen API")


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

import os
import utils
from utils import *
from config import *
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.faiss import AutoQueryEncoder
from collections import deque

# load hybried searcher for relation binding
query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
corpus = LuceneSearcher(os.path.join(CONTRIEVER_PATH, "contriever_fb_relation/index_relation_fb"))
bm25_searcher = LuceneSearcher(os.path.join(CONTRIEVER_PATH, 'contriever_fb_relation/index_relation_fb'))
contriever_searcher = FaissSearcher(os.path.join(CONTRIEVER_PATH, 'contriever_fb_relation/freebase_contriever_index'), query_encoder)
hsearcher = HybridSearcher(contriever_searcher, bm25_searcher)

def similar_relation_from_question(question, topk=5):
    """
    根据问题搜索相似的关系，用于处理损坏的推理路径
    
    输入:
        question: 问题文本
        topk: 返回的相似关系数量，默认为5
        
    输出:
        result: 检索到的相似关系列表
    """
    result = []
    hits = hsearcher.search(question, k=1000)[:topk]
    for hit in hits:
        result.append(json.loads(corpus.doc(str(hit.docid)).raw())['rel_ori'])
    return result

def grounding_relations(relation, topk=5):
    """
    将自然语言关系绑定到知识图谱关系候选
    
    输入:
        relation: 自然语言关系描述
        topk: 返回的候选关系数量，默认为5
        
    输出:
        result_no_q: 知识图谱关系候选列表
    """
    result_no_q = []
    relation_tokens = relation.replace("."," ").replace("_", " ").strip()
    hits = hsearcher.search(relation_tokens.replace("  "," ").strip(), k=1000)[:topk]
    for hit in hits:
        result_no_q.append(json.loads(corpus.doc(str(hit.docid)).raw())['rel_ori'])

    return result_no_q

def relation_binding(reasoning_path_LLM_init, topk=5):
    """
    将推理路径中的所有关系绑定到知识图谱关系候选
    
    输入:
        reasoning_path_LLM_init: 每个主题实体生成的推理路径字典
        topk: 每个关系绑定的候选数量，默认为5
        
    输出:
        grounded_relations: 每个推理路径的已绑定关系字典
    """
    predicted_reasoning_path = []
    for keys in reasoning_path_LLM_init.keys():
        if type(reasoning_path_LLM_init[keys]) == str:
            predicted_reasoning_path.append(utils.string_to_path(reasoning_path_LLM_init[keys]))
        elif len(reasoning_path_LLM_init[keys])>0:
            predicted_reasoning_path.append(utils.string_to_path(reasoning_path_LLM_init[keys][0]))
                
    # bind all relations to KG relation candidates（faiss）
    grounded_relations = {}
    for r in predicted_reasoning_path:
        for rel in r:
            relations_no_q = grounding_relations(rel, topk=topk)
            if rel not in grounded_relations.keys():
                grounded_relations[rel] = relations_no_q
            else:
                grounded_relations[rel] = list(set(grounded_relations[rel]+relations_no_q))

    return grounded_relations


def bfs_for_each_path(entity_id, target_path, grounded_reasoning_set, options, max_que = 300):
    """
    为每个推理路径进行路径连接，本质上是对推理路径中每个关系的BFS搜索
    在每一层BFS搜索中，检查当前节点的邻居是否与候选关系有交集，如果有则当前关系实例化成功
    如果实例化失败，返回有用的结构化信息（包括当前实例化的实例和失败点的候选关系）用于编辑
    
    输入:
        entity_id: 当前推理路径的主题实体ID
        target_path: 当前推理路径
        grounded_reasoning_set: 推理路径中每个位置的已绑定关系候选列表
        options: 解析后的参数
        max_que: 每层的最大队列大小，默认为300
        
    输出:
        result_paths: 实例化的推理路径（如果实例化失败则为空）
        grounded_knowledge_current: BFS期间存储的所有实例（长度从0开始）
        ungrounded_neighbor_relation_dict: 如果实例化失败，存储一些关系作为编辑候选
    """
    result_paths = []
    current_position = 0
    queue = deque([(entity_id, [], current_position)])  # BFS queue. Container for entities, path instances and current position on path
    grounded_knowledge_current = []
    ungrounded_neighbor_relation_dict = {}

    while queue:
        size = len(queue)
        ungrounded_neighbor_relation_dict.clear()

        if options.verbose:
            print("current layer size when BFS", size)

        while size > 0:
            size -= 1
            current_node, current_path, current_position = queue.popleft()

            # push current grounded path to grounded_knowledge_current.
            # Note that grounded_knowledge_current stores all instantiated path (including length from 0 to current path length
            grounded_knowledge_current.append((current_node, current_path, current_position))

            if current_position == len(target_path) and current_path not in result_paths and len(current_path) > 0:
                result_paths.append(current_path)

            # continue instantiation if current path is shorter than predicted target_path
            if current_position < len(target_path):
                # get edges (relations) around current node (except previous relations)
                pre_relations = [rel[1] for rel in current_path]
                edge_set = utils.get_ent_one_hop_rel(current_node, pre_relations=pre_relations)
                if len(edge_set) == 0:
                    continue

                # take intersection
                list1 = grounded_reasoning_set[current_position]  # grounded relations for current position
                list2 = edge_set                                  # relations around current node
                list3 = pre_relations

                intersection_grounded = list(set(list1) & set(list2))    
                intersection_have_grounded = list(set(intersection_grounded) & set(list3))    
                intersection = list(set(intersection_grounded) - set(intersection_have_grounded))

                # no intersection for current node (something goes wrong), store some relations as candidates for editing
                if len(intersection) <= 0:
                    ungrounded_neighbor_relation_dict[utils.id2entity_name_or_type_en(current_node)] = edge_set
                    continue
                # grounded success. add to queue for next loop
                else:
                    for relation in intersection:
                        if len(queue) >= max_que:
                            break
                        # forward and backward relations
                        neighbors_with_relation = [neighbor for neighbor in utils.entity_search(current_node, relation, True) + utils.entity_search(current_node, relation, False)]
                        for neighbor in neighbors_with_relation:
                            if len(queue) < max_que:
                                queue.append((neighbor, current_path + [(utils.id2entity_name_or_type_en(current_node), relation, utils.id2entity_name_or_type_en(neighbor))], current_position + 1))
                            else:
                                break
                            
        if len(ungrounded_neighbor_relation_dict.keys()) == 0 and len(grounded_knowledge_current) == 1 and grounded_knowledge_current[-1][-1] == 0:
            ungrounded_neighbor_relation_dict[utils.id2entity_name_or_type_en(grounded_knowledge_current[-1][0])] = utils.get_ent_one_hop_rel(grounded_knowledge_current[-1][0])

    return result_paths, grounded_knowledge_current, ungrounded_neighbor_relation_dict